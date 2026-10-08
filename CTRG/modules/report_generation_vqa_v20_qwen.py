import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, BertTokenizer, AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.bert.modeling_bert import BertConfig, BertModel
import math
import torch.nn.functional as F


from CTRG.modules import objectives, m3ae_utils
from transformers import LlamaForCausalLM, LlamaTokenizer
from CTRG.modules.m3ae_utils import init_weights
from CTRG.modules.language_encoders.bert_model import BertCrossLayer

from CTRG.modules.RadFM.vit_3d import ViT

from CTRG.modules.models.med import BertConfig, BertLMHeadModel

from peft import get_peft_model, LoraConfig, TaskType

import matplotlib.pyplot as plt

import re


#######################################
from transformers import AutoConfig, AutoTokenizer, AutoModel, pipeline
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead

from positional_encodings.torch_encodings import PositionalEncodingPermute3D

# from transformer_maskgit import CTViT
# from ctvit import CTViT

from transformers import LogitsProcessor, LogitsProcessorList

from .patch_selection import select_features_network_mmr_variable_k_gpu_purefeature_finetune, DefaultSelector_2, spatially_aware_feature_pooling, spatially_aware_feature_pooling_v2, spatially_aware_feature_pooling_v3, spatially_aware_feature_pooling_v5, spatially_aware_feature_pooling_fast, spatially_aware_feature_pooling_optimized, spatially_aware_feature_pooling_optimized_2, spatially_aware_feature_pooling_optimized_4, simple_masked_fps_selection

# import torchprofile
# spcially pooling based on v2.
# add optimized prompt and input construct strategy, like v6.
# v14 try with mask

import torch
import torch.nn as nn
import torch.nn.functional as F

from .organ_qformer import OrganFeatureRefiner, OrganFeatureRefiner_v2_LRMR_dual, OrganFeatureRefiner_v2_LRMR_dual_v3, OrganFeatureRefiner_v2_LRMR_dual_v2, OrganFeatureRefiner_v2_LRMR_dual_v2_new


def compute_token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute entropy per token from logits.
    Args:
        logits: (..., vocab_size)
    Returns:
        entropy: (...) — same shape minus last dim
    """
    probs = F.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)



class SimpleMLP(nn.Module):
    def __init__(self, indim, outdim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(in_features=indim, out_features=indim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=indim, out_features=outdim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

class RelevanceScorer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        # 这是一个轻量级的评分网络
        # 它学习判断：给定的“视觉-问题融合特征”是否重要
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 输出一个标量分数
        )

    def forward(self, fused_features):
        """
        Args:
            fused_features: (B, N, D) 已经融合了问题信息的视觉特征
        Returns:
            scores: (B, N)
        """
        # 映射到 (B, N, 1) -> squeeze -> (B, N)
        scores = self.net(fused_features).squeeze(-1)
        return scores

class CTRG_3D_lmae_vqa_v20_qwen(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.momentum = 0.99
        self.queue_size = 512
        self.temp = 0.07
        self.aligned_length = 128
        self.selected_patch_n = config["selected_patch"]
        self.set_model()

    def set_model(self, stage=None):
        
        #########################################

        resolution_after = self.config['image_size']

        self.multi_modal_language_proj = nn.Linear(self.config['input_text_embed_size'], self.config['hidden_size'])
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(self.config['input_image_embed_size'], self.config['hidden_size'])
        self.multi_modal_vision_proj.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, self.config["hidden_size"])
        self.modality_type_embeddings.apply(init_weights)


        self.multi_modal_vision_proj_forllm = SimpleMLP(self.config['hidden_size'], self.config["llm_dim"]) # 2560 for 4b, 2048 for 1.8b, 1024for.5b, 1536 for 2 1.5b 896 # 2048 for llama 3.2 1b 3072 for 3b
        self.multi_modal_vision_proj_forllm.apply(init_weights)
        
        # == End  : 1. Build Models ==

        # == Begin: 1.5 
        
        textmodelpath = self.config["text_model"] # /apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        bert_config = AutoConfig.from_pretrained(textmodelpath)
        
        self.vision_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        
        self.text_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(1)])
        
        self.contrastive_proj_image = nn.Linear(self.config["hidden_size"], 512)
        # self.contrastive_proj_image2 = nn.Linear(512, config["hidden_size"])
        self.contrastive_proj_text = nn.Linear(self.config["hidden_size"], 512)
        self.obver_norm_class = nn.Linear(768, 1)


        self.linear_proj = nn.Linear(2560, 768)
        # self.OrganFeatureRefiner = OrganFeatureRefiner_v2_LRMR_dual_v2_new(num_tokens = 32)
        self.OrganFeatureRefiner = OrganFeatureRefiner_v2_LRMR_dual_v3(
            num_tokens=16,
            feat_dim=768,
            num_layers=4,
            top_k=32,
        )
        

        self.organ_cls = nn.Linear(768, 10)
        # class token
        
        self.expert_image = nn.Embedding(10, self.config["hidden_size"])
        self.expert_image.apply(init_weights)

        #self.expert_text = nn.Embedding(10, self.config["hidden_size"])
        #self.expert_text.apply(init_weights)

        m3ae_utils.set_metrics(self)

        ###############################################################################

        textmodelpath = "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert"
        self.text_model = BertModel.from_pretrained(textmodelpath)

        # ck = torch.load(config["ct_clip_ckpoint"], map_location = "cpu")
        # nck = {}
        # nck = {k.replace("text_transformer.", ""):v for k,v in ck.items() if "text_transformer." in k}
        # self.text_model.load_state_dict(nck)

        self.tokenizer_text_enc = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)



        path = self.config["decoder_path"]
        llm_dim = self.config["llm_dim"]
        # path = "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-3.2-1B"
        # path = "/apdcephfs_cq10/share_1290796/lh/dataset/llava_med"
        # path = "/jizhicfs/datalh/dataset/qwen/qwen7b"
 

        self.text_decoder = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.float16,
                device_map="cuda",
                # load_in_8bit=True
            )
        

        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id + 1

        # self.selector = DefaultSelector_2(768)

        self.embed_tokens = self.text_decoder.get_input_embeddings()

        if True:
            
            if "phi" in self.config['decoder_path'] or "Qwen" in self.config["decoder_path"]:
                peft_config_ = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=32, lora_alpha=32, lora_dropout=0.05, target_modules=find_all_linear_names(self.text_decoder)
                )
            elif "Qwen" in self.config['decoder_path']:
                peft_config_ = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=64, lora_alpha=64, lora_dropout=0.05,target_modules=["W_pack", "o_proj"]
                )
            else:
                peft_config_ = LoraConfig(
                    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=128, lora_alpha=16, lora_dropout=0.05,target_modules=["W_pack", "o_proj"]
                )

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.0, target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"
                    ], use_rslora=False)
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=16, lora_dropout=0.0, use_rslora=False)

            self.text_decoder = get_peft_model(self.text_decoder, peft_config_)
            self.text_decoder.print_trainable_parameters()

            if False:
                trainable, total = 0, 0
                for n, p in self.text_decoder.named_parameters():
                    total  += p.numel()
                    if p.requires_grad:
                        trainable += p.numel()
                        print(n, p.shape)          # 把这行打开，一眼看出谁被漏掉
                print(f"\ntrainable : {trainable:,}")
                print(f"total     : {total:,}")
                print(f"ratio     : {trainable/total:.4%}")

            print('Loading LLAMA LoRA Done') 
        
        self.hparams.config["vocab_size"] = self.tokenizer.vocab_size
    
    def infer_text(self, text_model, question, tokenlizer, device):
        
        all_encoding=tokenlizer(question, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt")

        with torch.no_grad():
            
            local_text = text_model(all_encoding.input_ids.to(device), attention_mask = all_encoding.attention_mask.to(device))[0][:, 0:1, :] # .permute(1,0,2)

        return local_text


        
    def infer_image_organ(self, expert, query, uni_modal_image_feats, vision_extract_layer, organ_mask = None, target_organ = None, M = 200, k_cur = 15):
        
        region_list = ["trachea and bronchie", "heart", "lung", "esophagus", "pleura", "bone", "thyroid", "breast", "abdomen", "others"]
        all_idx = []
        for l in range(len(target_organ)):
            c_organ = eval(target_organ[l])
            c_idxs = []
            for organ in c_organ:
                organ = organ.split("/")[0]
                c_idx = region_list.index(organ)
                c_idxs.append(c_idx)
                break # 暂时就用第一个吧，
            all_idx.append(c_idxs)
        all_idx = torch.tensor(all_idx)


        b, D, H, W, C = uni_modal_image_feats.shape
        # cmask = cmask.reshape(b, 10, n)
        # cmask = F.interpolate(cmask, size=(D, H, W), mode='nearest')
        #print(cmask.shape)

        uni_modal_image_feats = uni_modal_image_feats.reshape(b, -1, C)

        uni_modal_image_feats = self.linear_proj(uni_modal_image_feats)

        # print(uni_modal_image_feats.shape, organ_mask.shape)

        b, n, C = uni_modal_image_feats.shape
        organ_mask = organ_mask.reshape(b, 10, n).permute(0,2,1)
        

        final_mask = []
        for l in range(organ_mask.shape[0]):
            c_mask = None
            for c_idx in all_idx[l]:
                if c_mask is None:
                    c_mask = organ_mask[l:l+1, :,  c_idx]
                else:
                    c_mask += organ_mask[l:l+1, :, c_idx]
            
            final_mask.append(c_mask)

        final_mask = torch.cat(final_mask, 0)

        #@ print(final_mask.shape, ifea.shape, all_idx.shape)
        # print(y.shape, attention_map.shape, query.shape) # query 2 1 768
        # with torch.no_grad():
        # print(ifea.shape)
        # all_final_feats, all_global_tokens, all_abnormal_tokens, total_aux_loss, metrics = self.OrganFeatureRefiner(ifea, mask = organ_mask.permute(0,2,1), abnorm_label=torch.zeros((2,10)).to(organ_mask.device), multi_organ_mode = True)
        with torch.no_grad():
            all_final_feats, all_global_tokens, all_abnormal_tokens, total_aux_loss, metrics = self.OrganFeatureRefiner(uni_modal_image_feats, final_mask, all_idx.squeeze())

            
        # print(final_feat.shape, global_token.shape)
        # selected_patch = spatially_aware_feature_pooling_optimized(y, attention_map, organ_pred, query, self.fps_fusion, self.SimpleSemanticPositionalEmbedder)
        # print(all_final_feats.shape, all_global_tokens.shape, all_abnormal_tokens.shape)
        final_feat = torch.cat([all_final_feats, all_global_tokens.unsqueeze(1), all_abnormal_tokens.unsqueeze(1)], 1)
        return final_feat, total_aux_loss


    def prompt_wrap(self, img_embeds, atts_img, question=None):
        batch_size = img_embeds.shape[0]
        device = img_embeds.device

        # 1. Handle question list
        if question is not None:
            if isinstance(question, str):
                questions = [question] * batch_size
            else:
                questions = question
        else:
            if isinstance(self.prompt, str):
                questions = [self.prompt] * batch_size
            else:
                questions = self.prompt

        # 2. p_before: Add BOS token explicitly (assuming LLaMA/Vicuna style '<s>')
        # Check your specific tokenizer's BOS token (e.g., tokenizer.bos_token)
        bos = self.tokenizer.bos_token if self.tokenizer.bos_token else ''
        p_before_str = f'{bos}Answer the question based on given image: <Img>'
        
        # 3. p_after
        p_after_list = [f'</Img> {q} \n Answer:' for q in questions]

        # 4. Tokenize p_before
        p_before_tokens = self.tokenizer(
            p_before_str,
            return_tensors="pt",
            add_special_tokens=False 
        ).to(device)
        
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_before_attn = p_before_tokens.attention_mask.expand(batch_size, -1)

        # 5. Tokenize p_after with LEFT PADDING and DYNAMIC LENGTH
        # Save original padding side to restore later
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'  # <--- CRITICAL FIX

        p_after_tokens = self.tokenizer(
            p_after_list,
            return_tensors="pt",
            add_special_tokens=False,
            padding='longest',       # <--- Efficiency FIX (was 'max_length')
            truncation=True,
            max_length=256,          # Increased safety limit
        ).to(device)
        
        # Restore padding side (good practice)
        self.tokenizer.padding_side = original_padding_side

        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids)
        p_after_attn = p_after_tokens.attention_mask

        # 6. Concatenate
        # Ensure atts_img is the same dtype as tokenizer masks (usually long/int64)
        atts_img = atts_img.to(p_before_attn.dtype)

        wrapped_img_embeds = torch.cat([
            p_before_embeds, 
            img_embeds, 
            p_after_embeds
        ], dim=1)

        wrapped_atts_img = torch.cat([
            p_before_attn, 
            atts_img, 
            p_after_attn
        ], dim=1)

        return wrapped_img_embeds, wrapped_atts_img


    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=False,
            unimodal=False
    ):
        ret = dict()

        # == Begin: Fetch the inputs ==
        img = batch["image"]
        device = img.device

        # == Begin: Image Encoding ==
        selected_patch = img.squeeze(1)

        batch_size, d,h,w, l = selected_patch.shape
        # selected_patch = self.multi_modal_vision_proj_forllm(selected_patch)

        # retrieval
        question = batch["question"]
        # print(batch["mask"].shape)

        local_text_emb = self.infer_text(self.text_model, question, self.tokenizer_text_enc, device)


        uni_modal_image_feats = batch["all_image"].cuda().squeeze(1)

        parsed_organs = batch["parsed_organs"]
        # print(parsed_organs)

        selected_patch, aux_loss = self.infer_image_organ(self.expert_image, local_text_emb, uni_modal_image_feats, self.vision_extract_layer, batch["mask"], target_organ = parsed_organs, M = 200, k_cur = 10)

        # print(selected_patch.shape)

        image_masks = torch.ones((selected_patch.size(0), selected_patch.size(1)), dtype=torch.long,
                                 device=device)
        selected_patch = self.multi_modal_vision_proj_forllm(selected_patch)
        # 1. Wrap Prompt (Images + Question)
        # Assumption: prompt_wrap uses LEFT PADDING as discussed previously
        img_embeds, atts_img = self.prompt_wrap(selected_patch, image_masks, batch["question"])

        # 2. Prepare Answers (Targets)
        answer = batch["answer"]
        
        # Ensure EOS token exists
        eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "</s>"
        text_input = [str(ans) + eos_token for ans in answer]

        # CRITICAL: Switch to RIGHT padding for the answer part
        # We want: [Answer, EOS, PAD, PAD] -> Model predicts Answer+EOS, ignores PAD
        self.tokenizer.padding_side = "right"

        to_regress_tokens = self.tokenizer(
            text_input,
            return_tensors="pt",
            padding="longest",       # Optimized: Pad to longest in batch, not fixed 50
            truncation=True,
            max_length=128,          # Safety cap for very long answers
            add_special_tokens=False 
        ).to(device)

        # 3. Create Targets (Labels)
        # Replace PAD with -100 (Ignore Index)
        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )

        # 4. Create Empty Targets for Prompt/Image part
        # These are all -100 because we don't calculate loss on the prompt/image
        empty_targets = torch.full(
            (atts_img.shape[0], atts_img.shape[1]), 
            -100, 
            dtype=torch.long, 
            device=device
        )

        # 5. Concatenate Targets: [-100, ..., -100, Answer_IDs, EOS, -100, -100]
        targets = torch.cat([empty_targets, targets], dim=1)

        # 6. Get Answer Embeddings
        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)

        # 7. Concatenate Embeddings: [Prompt_Embeds, Answer_Embeds]
        inputs_embeds = torch.cat([img_embeds, to_regress_embeds], dim=1)

        # 8. Concatenate Attention Masks
        # Ensure atts_img is LongTensor (int64) to match tokenizer mask
        atts_img = atts_img.to(torch.long)
        attention_mask = torch.cat([atts_img, to_regress_tokens.attention_mask], dim=1)

        # 前向 + loss
        decoder_output = self.text_decoder(
            inputs_embeds=inputs_embeds.to(device),
            attention_mask=attention_mask.to(device),
            return_dict=True,
            labels=targets.to(device),
        )


        # == Begin: == Output Multi-Modal Features ==
        if not self.training:
          
            outputs = self.text_decoder.generate(
                inputs_embeds = img_embeds,
                attention_mask = atts_img,
                num_beams=1,
                do_sample=False,
                # min_new_tokens=0,
                max_new_tokens=50,
                return_dict_in_generate=True, 
                eos_token_id=self.tokenizer.eos_token_id,  
                pad_token_id=self.tokenizer.pad_token_id, # 128001  # 或者使用其他合适的pad_token_id
                output_scores=True
                # output_attentions=True,
            )            

            scores = outputs.scores
            if not scores:
                raise ValueError("No tokens generated!")

            logits_generated = torch.stack(scores, dim=1)  # (B, N, V)
            batch_size, max_gen_len = logits_generated.shape[:2]

            # Precompute token-level entropy for all generated tokens
            token_entropy_all = compute_token_entropy(logits_generated)  # (B, N)

            captions = []
            final_entropies = []

            for i in range(batch_size):
                full_seq = outputs.sequences[i]
                prompt_len = full_seq.size(0) - max_gen_len
                generated_ids = full_seq[prompt_len:]  # (N,)

                # 1. Find EOS position in generated part
                eos_positions = (generated_ids == self.tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                eos_idx = eos_positions[0].item() if eos_positions.numel() > 0 else max_gen_len

                # 2. Find first high-entropy token BEFORE EOS
                high_entropy_idx = None
                entropy_threshold = 5.0
                for j in range(eos_idx):  # only check before EOS
                    if token_entropy_all[i, j].item() > entropy_threshold:
                        high_entropy_idx = j
                        break

                # 3. Determine how many generated tokens to keep
                keep_gen_len = high_entropy_idx if high_entropy_idx is not None else eos_idx

                # 4. Compute entropy ONLY over kept tokens
                if keep_gen_len == 0:
                    caption_entropy = 0.0
                    truncated_seq = full_seq[:prompt_len]
                else:
                    kept_entropy = token_entropy_all[i, :keep_gen_len]  # (keep_len,)
                    caption_entropy = kept_entropy.mean().item()
                    truncated_seq = full_seq[:prompt_len + keep_gen_len]

                # 5. Decode and post-process
                caption = self.tokenizer.decode(truncated_seq, skip_special_tokens=True)
                caption = caption.split("CLII")[0].strip()

                captions.append(caption)
                final_entropies.append(caption_entropy)
            
        else:
            captions = ["I am fxxking man."] * batch_size # for test 
            final_entropies = None
            token_entropy_all = None

        ret.update({
            "images": img,
            "decoder_output": decoder_output,
            "data_name": batch["data_name"],
            "text_ori": text_input,
            "captions": captions,
            "question": batch["question"],
            "question_type": batch["question_type"],
            "source_file": batch["source_file"],
            "entropy": final_entropies,
            "entropy_token": token_entropy_all,
            "aux_loss": aux_loss
        })

        return ret

        

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm2(self, batch))

        # Pre-Training: Masked Image Modeling
        if "mim" in self.current_tasks:
            ret.update(objectives.compute_mim(self, batch))

        # Pre-Training: Image Text Matching
        if "itm" in self.current_tasks:
            ret.update(objectives.compute_itm(self, batch))

        # Fine-Tuning: Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch, test=test))

        # Fine-Tuning: Image-Text Classification
        if "cls" in self.current_tasks:
            ret.update(objectives.compute_cls(self, batch, test=test))

        # Fine-Tuning: Image Retrieval and Text Retrieval
        if "irtr" in self.current_tasks:
            ret.update(objectives.compute_irtr(self, batch, test))
        
        if "rg" in self.current_tasks:
            ret.update(objectives.compute_rg_blue(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)

        output = self(batch)
        # print(self.hparams.config)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])
        # print(output, total_loss)
        return total_loss

    '''
    def training_step_end(self, batch_parts):

        print(batch_parts)

        return torch.mean(batch_parts)
    '''

    def on_training_epoch_end(self):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch, test=True)

    def on_test_epoch_end(self, outs):
        m3ae_utils.epoch_wrapup(self, test=True)

    def configure_optimizers(self):
        return m3ae_utils.set_schedule(self)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feats, text_feats, obverlabels):
        # gather keys before updating queue
        # print(image_feats.shape, text_feats.shape)
        for idx in range(obverlabels.shape[0]):
            obverlabel = obverlabels[idx]
            image_feat = image_feats[idx]
            text_feat = text_feats[idx]

            ttidx = []
            for iidx in range(obverlabel.shape[0]):
                if obverlabel[iidx]:
                    ttidx.append(iidx)
            #obverlabel = torch.cat([obverlabel, torch.zeros(1, device = obverlabel.device)], 0)
            batch_size = image_feat.shape[0]
            unnormsize = obverlabel.sum()
            image_feat = image_feat[ttidx]
            text_feat = text_feat[ttidx]

            ptr = int(self.queue_ptr)
            # assert self.queue_size % batch_size == 0  # for simplicity

            # replace the keys at ptr (dequeue and enqueue)
            if ptr + unnormsize > self.queue_size:
                image_feat = image_feat[:self.queue_size-ptr]
                text_feat = text_feat[:self.queue_size-ptr]
            
            self.image_queue[:, ptr:ptr + unnormsize] = image_feat.T.detach()
            self.text_queue[:, ptr:ptr + unnormsize] = text_feat.T.detach()
            ptr = (ptr + unnormsize) % self.queue_size  # move pointer

            self.queue_ptr[0] = ptr 

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    