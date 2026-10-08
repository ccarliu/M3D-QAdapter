import pytorch_lightning as pl
import torch
import torch.nn as nn

from transformers.models.bert.modeling_bert import BertConfig
from transformers import RobertaConfig, RobertaModel, BertTokenizer, BertModel

import math
import torch.nn.functional as F

import numpy as np


from CTRG.modules import objectives, m3ae_utils
from CTRG.modules.language_encoders.bert_model import BertCrossLayer
from CTRG.modules.m3ae_utils import init_weights
# from CTRG.modules.vision_encoders import swin_transformer as swin
# from CTRG.modules.vision_encoders.clip_model import build_model, adapt_position_encoding
# from CTRG.modules.vision_encoders.swin_helpers import swin_adapt_position_encoding

from CTRG.modules.models.med import BertConfig, BertLMHeadModel
from ct_clip import CTCLIP

#######################################
from transformers import AutoConfig, AutoTokenizer, AutoModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertOnlyMLMHead

from positional_encodings.torch_encodings import PositionalEncodingPermute3D

from CTRG.modules.RadFM.vit_3d import ViT, ViT_expert, ViT_reg

# from transformer_maskgit import CTViT
from ctvit import CTViT

# v14 with multi image per batch
# v15 with only itc
# v17 with change the text encoder
# v18, simple code
# v19: new patch selector back
# v21: v20 with a nagetive sample selective module to improve the quanity of nagetive pool.

# v24: with big resolution

# further version of v 24

class SparseAlignLoss(nn.Module):
    def __init__(self, dim, consistency_weight=0.1, neg_weight=1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.neg_weight = neg_weight
        
        # === 1. 可学习的投影层 (Adapter) ===
        # 负责将图像特征对齐到冻结的文本空间
        self.img_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # === 2. 可学习的温度系数 (Logit Scale) ===
        # 初始值设为 np.log(1/0.07) ≈ 2.65，这是对比学习的标准设定
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, raw_feat, query_emb, query_mask):
        """
        raw_feat: (B, T, L) 原始图像特征
        query_emb: (B, N, L) 冻结的文本特征
        query_mask: (B, N, T)
        """
        # 1. 先过投影层
        feat = self.img_projector(raw_feat)
        
        # 2. 归一化
        feat_norm = F.normalize(feat, dim=-1)
        query_norm = F.normalize(query_emb, dim=-1)
        
        # 3. 计算 Cosine 相似度
        # (B, N, 1, L) * (B, 1, T, L) -> (B, N, T)
        sim = F.cosine_similarity(query_norm.unsqueeze(2), feat_norm.unsqueeze(1), dim=-1)
        
        # 4. === 关键点：应用可学习的 Scale ===
        # 限制最大值防止梯度爆炸，通常 clamp 到 100 以内
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = sim * logit_scale  # 此时 logits 的范围被拉伸了，比如从 [-1,1] 变成 [-20, 20]
        
        # 5. 使用 Sigmoid 将其变为概率 (0~1)
        probs = torch.sigmoid(logits)
        
        # === Loss 计算 (BCE 风格) ===
        
        # 正样本 Loss: 希望 probs 接近 1
        # 只计算 mask=1 的部分
        pos_loss = -torch.log(probs + 1e-8) * query_mask
        pos_loss = pos_loss.sum() / (query_mask.sum() + 1e-8)
        
        # 负样本 Loss: 希望 probs 接近 0
        # 引入 "Safe Negative Mining" (忽略高置信度的背景)
        query_mask = query_mask.float() 
        bg_mask = 1.0 - query_mask
        
        # 动态阈值：如果背景的预测概率 > 0.8，我们认为它是漏标的，不惩罚
        # 注意：这里不需要手动调 threshold 了，因为 logits 是可学习的，
        # 模型会自己调整 scale 让正样本 > 0.5，负样本 < 0.5
        with torch.no_grad():
            # 这是一个启发式策略：忽略那些模型非常确信是正样本的背景区域
            potential_pos = (probs > 0.8).float()
            final_neg_mask = bg_mask * (1.0 - potential_pos)
            
        neg_loss = -torch.log(1 - probs + 1e-8) * final_neg_mask
        neg_loss = neg_loss.sum() / (final_neg_mask.sum() + 1e-8)
        
        
        return pos_loss + self.neg_weight * neg_loss

    @torch.no_grad()
    def evaluate(self, raw_feat, query_emb, gt_mask, threshold=0.5):
        """
        执行推理并计算定量指标验证特征质量。
        
        Args:
            raw_feat: (B, T, L) 原始图像特征
            query_emb: (B, N, L) 文本特征
            gt_mask: (B, N, T) 真实的 Ground Truth (0或1)
            threshold: 二值化阈值，默认 0.5
            
        Returns:
            dict: 包含 mIoU, mAP, Accuracy, Pos_Conf, Neg_Conf 等指标
        """
        # 1. 切换到评估模式 (影响 LayerNorm/Dropout)
        was_training = self.training
        self.eval()
        
        # === 推理逻辑 (与 Forward 保持一致) ===
        feat = self.img_projector(raw_feat)
        
        # 归一化
        feat_norm = F.normalize(feat, dim=-1)
        query_norm = F.normalize(query_emb, dim=-1)
        
        # 计算相似度 (B, N, T)
        sim = F.cosine_similarity(query_norm.unsqueeze(2), feat_norm.unsqueeze(1), dim=-1)
        
        # 应用 Scale 并转为概率
        scale = self.logit_scale.exp().clamp(max=100)
        probs = torch.sigmoid(sim * scale)
        
        # === 指标计算 ===
        # 准备数据
        probs_flat = probs.flatten().cpu().numpy()
        gt_flat = gt_mask.flatten().cpu().numpy()
        
        # 1. mAP (Mean Average Precision) - 最重要的排序指标
        # 处理全0或全1导致的报错
        try:
            if len(np.unique(gt_flat)) > 1:
                mAP = average_precision_score(gt_flat, probs_flat)
            else:
                mAP = 0.0
        except:
            mAP = 0.0

        # 2. 二值化指标 (IoU, Accuracy, F1)
        pred_mask = (probs > threshold).float()
        
        # Intersection & Union
        intersection = (pred_mask * gt_mask).sum()
        union = (pred_mask + gt_mask).clamp(max=1.0).sum()
        iou = intersection / (union + 1e-8)
        
        # Accuracy
        correct = (pred_mask == gt_mask).float().sum()
        accuracy = correct / gt_mask.numel()
        
        # Precision & Recall
        true_pos = intersection
        pred_pos = pred_mask.sum()
        actual_pos = gt_mask.sum()
        
        precision = true_pos / (pred_pos + 1e-8)
        recall = true_pos / (actual_pos + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        # 3. 分离度统计 (检查 Logit Scale 是否合适)
        # 正样本的平均预测概率 (越接近1越好)
        pos_conf = probs[gt_mask == 1].mean().item() if (gt_mask == 1).any() else 0.0
        # 负样本的平均预测概率 (越接近0越好)
        neg_conf = probs[gt_mask == 0].mean().item() if (gt_mask == 0).any() else 0.0

        # 恢复原本的训练状态
        self.train(was_training)
        
        return {
            "mAP": float(mAP),           # 综合排序能力 (推荐主要看这个)
            "mIoU": iou.item(),          # 区域重合度
            "F1_Score": f1.item(),       # 精确率和召回率的调和平均
            "Accuracy": accuracy.item(), # 像素级准确率
            "Pos_Conf": pos_conf,        # 正样本置信度 (应 > 0.8)
            "Neg_Conf": neg_conf,        # 负样本置信度 (应 < 0.2)
            "Scale_Val": scale.item()    # 当前学习到的温度系数
        }

class SparseAlignLossBaseline(nn.Module):
    def __init__(self, dim, consistency_weight=0.1, neg_weight=1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.neg_weight = neg_weight
        
        # === 1. 可学习的投影层 (保持不变) ===
        self.img_projector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
        # === 2. 可学习的温度系数 (保持不变) ===
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, raw_feat, query_emb, query_mask):
        """
        raw_feat: (B, T, L)
        query_emb: (B, N, L)
        query_mask: (B, N, T)
        """
        # 1. 投影与归一化
        feat = self.img_projector(raw_feat)
        feat_norm = F.normalize(feat, dim=-1)
        query_norm = F.normalize(query_emb, dim=-1)
        
        # 2. 计算相似度
        sim = F.cosine_similarity(query_norm.unsqueeze(2), feat_norm.unsqueeze(1), dim=-1)
        
        # 3. Scale & Sigmoid
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = sim * logit_scale
        probs = torch.sigmoid(logits)
        
        # === Loss 计算 ===
        
        # 确保 mask 是 float 类型
        query_mask = query_mask.float()
        
        # --- 正样本 Loss (保持不变) ---
        pos_loss = -torch.log(probs + 1e-8) * query_mask
        pos_loss = pos_loss.sum() / (query_mask.sum() + 1e-8)
        
        # --- 负样本 Loss (修改部分) ---
        # 这里的逻辑变简单了：只要 mask 不是 1，就全是负样本 (Hard Negative)
        bg_mask = 1.0 - query_mask
        
        # [移除] 这里去掉了原版中计算 potential_pos 和 final_neg_mask 的逻辑
        # 直接惩罚所有背景区域，要求它们的 probs 接近 0
        neg_loss = -torch.log(1 - probs + 1e-8) * bg_mask
        neg_loss = neg_loss.sum() / (bg_mask.sum() + 1e-8)
        
        return pos_loss + self.neg_weight * neg_loss

def attention_with_norm(tensor1, tensor2):
    """
    计算两个形状为 (b, n, l) 的张量之间的注意力图，并在计算前进行层归一化。
    
    参数:
    - tensor1: 形状为 (b, n, l) 的张量
    - tensor2: 形状为 (b, n, l) 的张量
    
    返回:
    - attention: 形状为 (b, n, n) 的注意力图
    """
    b, n, l = tensor1.shape
    
    # 应用层归一化
    tensor1 = F.normalize(tensor1, p=2, dim=-1)
    tensor2 = F.normalize(tensor2, p=2, dim=-1)
    
    # 计算相似度矩阵
    similarity = torch.bmm(tensor1, tensor2.transpose(1, 2))  # 形状为 (b, n, n)
    
    # 计算注意力图
    attention = F.softmax(similarity, dim=-1)
    
    return attention

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=4096, bottleneck_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, bottleneck_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        # out = self.decoder(z)
        return z

def mask_features_torch(uni_modal_image_feats, mask_rate=0.5):
    b, n, l = uni_modal_image_feats.size()
    n_half = int(n * (1-mask_rate))

    # 为每个样本生成一个随机的索引序列
    indices = torch.randperm(n)

    # 选择前 n_half 个索引并排序以保持顺序
    selected_indices = indices[:n_half].sort().values

    # 根据选择的索引提取特征
    masked_feats = uni_modal_image_feats[:, selected_indices, :]

    return masked_feats

class CTRG_pretrain_v29_textemb_perorgan_mask_lesion(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.momentum = 0.99
        self.queue_size = 10000
        self.temp = 0.07
        self.num_expert = 10
        # self.embedding_length = 1024
        self.aligned_length = 512
        # self.mask_rate = config['mask_rate']
      
        self.bs = config['batch_size']
        mask_ori = (torch.ones(self.num_expert, self.num_expert) * 0.1).fill_diagonal_(1).unsqueeze(0)
        mask1 = mask_ori.repeat(self.bs*self.bs, 1, 1).reshape(self.bs, self.bs, self.num_expert, self.num_expert).permute(0,2,1,3).reshape(self.bs*self.num_expert, self.bs*self.num_expert)
        mask2 = mask_ori.repeat(self.bs*self.queue_size // self.num_expert, 1, 1).reshape(self.bs, self.queue_size // self.num_expert, self.num_expert, self.num_expert).permute(0,2,3,1).reshape(self.bs*self.num_expert, self.num_expert*self.queue_size // self.num_expert)
        
        
        self.imt_mask = torch.cat([mask1, mask2], 1)

        print(self.imt_mask.shape)


        loaded_data = np.load(config.get("text_latent_feature_path", "text_latent_feature.npz"))
        # print(loaded_data.shape)
        # 访问加载的数据
        loaded_names = loaded_data['names']
        self.test_text_features = F.normalize(torch.tensor(loaded_data['features']), dim = -1) # [300:800]) # F.normalize(torch.tensor(loaded_data['features'][300:800]), dim = -1)
        print(self.test_text_features.shape)
        self.is_clip = ('swin' not in config['vit'])

   
        textmodelpath = config["text_model"] #/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model"
        bert_config = AutoConfig.from_pretrained(textmodelpath)

        self.lesion_align = SparseAlignLossBaseline(768) #SparseAlignLoss(768)
        # == vision encoder ==
        self.vision_enc_type = 1

        if self.vision_enc_type == 1:

            self.vision_encoder = CTViT(
                dim = config["hidden_size"],
                codebook_size = 8192,
                image_size = 480,
                patch_size = 30,
                temporal_patch_size = 15,
                spatial_depth = 6,
                temporal_depth = 6,
                dim_head = 32,
                heads = 8
            )
        elif self.vision_enc_type == 2:
            self.vision_encoder = ViT(
                image_size = (480, 480),          # image size
                frames = 240,               # max number of frames
                image_patch_size = 32,     # image patch size
                frame_patch_size = 16,      # frame patch size
                dim = 768,
                depth = 12,
                heads = 8,
                mlp_dim = 2048,
                dropout = 0.1,
                channels = 1,
                emb_dropout = 0.1
            )
        else:
            self.vision_encoder = ViT(
                in_channels=1,
                img_size=[32,256,256],
                patch_size=[8,16,16],
                hidden_size=768,
                mlp_dim=3072,#3072,
                num_layers=12,
                num_heads=12,
                pos_embed="perceptron",
                dropout_rate=0,
                spatial_dims=3,
                classification=False,
            )

        ck = torch.load(config["ct_clip_ckpoint"], map_location = "cpu")
        nck = {}
        nck = {k.replace("visual_transformer.", ""):v for k,v in ck.items() if "visual_transformer." in k}

        # self.vision_encoder.load_state_dict(nck, strict = False)

        textmodelpath = config["text_tokenlizer_path"]

        self.tokenizer = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)

        ##########################################
        # BERT used to encode the report text (configurable via
        # ``text_encoder_path`` / env ``CTRG_TEXT_ENCODER_PATH``).
        textmodelpath = config.get("text_encoder_path", config["text_tokenlizer_path"])
        # self.tokenizer = AutoTokenizer.from_pretrained(textmodelpath)
        # self.text_model = AutoModel.from_pretrained(textmodelpath, config=config_m)
        self.tokenizer_text_enc = BertTokenizer.from_pretrained(textmodelpath, do_lower_case=True)

        self.text_model = BertModel.from_pretrained(textmodelpath)
        nck = {}
        nck = {k.replace("text_transformer.", ""):v for k,v in ck.items() if "text_transformer." in k}
        self.text_model.load_state_dict(nck)

        ###############################

        self.contrastive_proj_image = nn.Linear(768, self.aligned_length, bias = False)
        # self.contrastive_proj_text = nn.Linear(4096, self.aligned_length, bias = False)

        # for local
        self.contrastive_proj_image2 = nn.Linear(config["hidden_size"], 768, bias = False)
        self.contrastive_proj_text2 = nn.Linear(768, self.aligned_length, bias = False)
        self.obver_norm_class = nn.Linear(768, 1)

        ###############################


        self.hparams.config["vocab_size"] = self.tokenizer.vocab_size
        self.organ_cls = nn.Linear(768, 10)
        
        #########################################

        resolution_after = config['image_size']

        
        self.expert_image = nn.Embedding(self.num_expert, 768)
        self.expert_image.apply(init_weights)

        self.expert_text = nn.Embedding(self.num_expert, 768)
        self.expert_text.apply(init_weights)

        self.vision_extract_layer = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(2)])
        
        self.leision_pre = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(1)])
        # self.copy_params()
        # create the queue
        self.register_buffer("text_queue_sim", torch.ones(self.queue_size) * 10000)
        self.register_buffer("text_queue", torch.randn(self.aligned_length, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(self.num_expert, dtype=torch.long)) 

        # self.p_enc_3d = PositionalEncodingPermute3D(config["hidden_size"]) 

        # == End


        m3ae_utils.set_metrics(self)

    def sparse_negative_entropy_loss(self, attention_map, cmask, tau=0.001, eps=1e-8):
        """
        attention_map: (B, T, S) 已经归一化到 [0,1]
        cmask:         (B, T, S) 0 表示需要监督的位置
        tau:           阈值，只惩罚 > tau 的位置
        """
        # 1. 只关注 mask==0 的区域
        inverse_mask = (cmask == 0)

        # 2. 再筛选出 attention > tau 的位置
        hard_mask = (attention_map > tau) & inverse_mask

        # 3. 如果没有满足条件的位置，直接返回 0
        num_hard = hard_mask.sum()
        if num_hard == 0:
            return torch.tensor(0.0, device=attention_map.device, requires_grad=True)

        # 4. 只在这些位置上计算 -log(1-p)
        attention_map = torch.clamp(attention_map, min=eps, max=1 - eps)
        loss = -torch.log(1 - attention_map) / 0.1 * hard_mask.float()

        # 5. 平均
        return loss.sum() / num_hard
    
    def pointwise_cross_entropy_loss(self, logits, target_onehot):
        """
        计算每个样本中 4096 个点的 10 分类交叉熵损失。

        Args:
            logits: Tensor of shape (B, 10, 4096) —— 模型输出的 logits
            target_onehot: Tensor of shape (B, 10, 4096) —— one-hot 标签

        Returns:
            loss: scalar tensor (mean over all points and batch)
        """
        # 1. 将 one-hot target 转为类别索引 (B, 4096)
        # 假设 target_onehot 在类别维度上是 one-hot（只有一个1）
        target_indices = torch.argmax(target_onehot, dim=1)  # shape: (B, 4096)

        # 2. 计算交叉熵
        # CrossEntropyLoss expects:
        #   input: (B, 10, 4096)
        #   target: (B, 4096) with dtype=torch.long
        loss = F.cross_entropy(logits, target_indices, reduction='mean')
        return loss
    
    def infer_text__(self, text_model, query, tokenlizer, device):
        # print(query)
        all_encoding=tokenlizer(query, padding="max_length",
                truncation=True,
                max_length=200,
                return_special_tokens_mask=True,
                return_tensors="pt")

        with torch.no_grad():
            
            local_text = text_model(all_encoding.input_ids.to(device), attention_mask = all_encoding.attention_mask.to(device))[0][:, 0:1, :] # .permute(1,0,2)

        return local_text

    def infer_image(self, vision_encoder, vision_extract_layer, vision_contrastive_proj, contrastive_proj_image2, expert, img, cmask, text_query = None, lesion_mask = None):

        #print(img.shape)
        if self.vision_enc_type == 1:
            uni_modal_image_feats = vision_encoder(img.unsqueeze(1), return_encoded_tokens=True)  # _, c, h, w, d # 240 480 480
            b, x1,x2,x3, d = uni_modal_image_feats.shape
            uni_modal_image_feats = uni_modal_image_feats.reshape(b, -1, d)
        else:
            # print(img.shape)
            all_images = []
            for l in range(6):
                # print(img.shape)
                uni_modal_image_feats, _ = vision_encoder(img.unsqueeze(1)[:, :, l*32:(l+1)*32])  # _, c, h, w, d
                # print(uni_modal_image_feats.shape)
                all_images.append(uni_modal_image_feats)
            uni_modal_image_feats = torch.cat(all_images, 1)

            bad = torch.isnan(uni_modal_image_feats) | torch.isinf(uni_modal_image_feats)
            if bad.sum() > 0:
                # 填充 0 并 detach，梯度畅通
                uni_modal_image_feats = torch.where(bad, torch.zeros_like(uni_modal_image_feats).detach(), uni_modal_image_feats)
                # 打印提示（只打印一次 / 每 step，可自行加 rank 过滤）
                print(f"[WARN] contains NaN/Inf - {bad.sum().item()} elements ")
        # print(uni_modal_image_feats.shape)
        b, n, d = uni_modal_image_feats.shape

        if self.training and False:
            uni_modal_image_feats = mask_features_torch(contrastive_proj_image2(uni_modal_image_feats))
        else:
            uni_modal_image_feats = contrastive_proj_image2(uni_modal_image_feats)
                
        x_attentions = []
        y_attentions = []

        # == Begin: extract key image feature
        x, y = expert.weight.unsqueeze(0), uni_modal_image_feats
        for layer_idx, (extract_layer) in enumerate(vision_extract_layer):
            # extract_layer.eval()
            x1 = extract_layer(x, y, output_attentions = True)
            y1 = extract_layer(y, x, output_attentions = True)

            x, y = x1[0], y1[0]
            x = x1[0]
            x_attention = x1[1:]
            y_attention = y1[1:]
            
            x_attentions.extend(x_attention[1])
            y_attentions.extend(y_attention[1])
        # p rint(y)
        # attention_map, _ = x_attention[1].max(1) #.softmax(1)
        attention_map = x_attention[1].mean(1) #.softmax(1)
        cmask = cmask.reshape(b, 10, n)
        # print(cmask.shape, attention_map.shape)
        # print(attention_map.max(), attention_map.min())
        ################
        if text_query is not None:

            # lesion_loss = self.sparse_mask_alignment_loss(y, text_query, lesion_mask)
            lesion_loss = self.lesion_align(y, text_query, lesion_mask)
        else:
            lesion_loss = 0
        # 只监督 mask == 0 的位置
        loss =  self.sparse_negative_entropy_loss(attention_map[:, :9], cmask[:, :9])
        # mask = mask.reshape(b, 10, n)
        organ_pred = self.organ_cls(y)
        loss_mask = self.pointwise_cross_entropy_loss(organ_pred.permute(0,2,1), cmask)
        ################
        loss = loss + loss_mask + lesion_loss

        _, selected_index = torch.sort(attention_map, 2)
        # print(attention_map.sum(2))
        # print((attention_map[0]>=0.003).sum(1))
        # for expert_i in range(10):
          #   print(attention_map[0, expert_i, selected_index[0, expert_i, -9:]])

        #print(x.shape)
        
        selected_patch = []
        for ii in range(x.shape[0]):
            # for expert_i in range(10):
             #   print(feature_similarity_matrix(y[ii, selected_index[ii, expert_i, -9:], :]))
            selected_patch.append(torch.cat([torch.cat([x[ii:ii+1, expert_i:expert_i + 1, :], y[ii:ii+1, selected_index[ii, expert_i, -9:], :]], 1) for expert_i in range(10)], 1)) # use y or orginal uni_modal_image_feats
        selected_patch = torch.cat(selected_patch, 0) 
        # print(x.shape)
        local_image = vision_contrastive_proj(x).reshape(-1, self.aligned_length)
        bad = torch.isnan(local_image) | torch.isinf(local_image)
        if bad.sum() > 0:
            # 填充 0 并 detach，梯度畅通
            local_image = torch.where(bad, torch.zeros_like(local_image).detach(), local_image)
            print("prelocalimage bad", bad.sum())

        selected_patch = torch.cat([selected_patch, x], 1)

        return selected_patch, local_image, uni_modal_image_feats, x_attention[1].mean(1), x_attentions, y_attentions, loss, cmask
 


    def infer_text(self, text_embedding, device):
        
        with torch.no_grad():
            local_text = text_embedding

        local_text = local_text.reshape(-1, self.aligned_length)

        return local_text, text_embedding #local_text_ori


    def _per_organ(self, sims, bs = 1):
        new_length = bs * self.num_expert
        sim_text_now = sims[:, :new_length].reshape(bs, self.num_expert, bs, self.num_expert)
        sim_text_queue = sims[:, new_length:].reshape(bs, self.num_expert, self.num_expert, self.queue_size // self.num_expert)
        
        total_text = []
        for l in range(self.num_expert):
            total_text.append(torch.cat([sim_text_now[:, l, :, l], sim_text_queue[:, l, l, :]], 1).unsqueeze(0))
        total_text = torch.cat(total_text, 0)

        return total_text

    def sparse_mask_alignment_loss(self, feat, query_emb, query_mask, temperature=0.07, consistency_weight=0.1):
        """
        专为稀疏 mask 设计的对齐损失（正样本极少场景）
        
        Args:
            feat: (B, T=4096, L) —— CT token features
            query_emb: (B, N, L) —— 文本 query embedding
            query_mask: (B, N, T) —— 二值 mask，1 表示该 token 属于病灶
            temperature: 相似度温度系数
            consistency_weight: 正样本内部一致性正则的权重（可选）
        """
        B, T, L = feat.shape
        _, N, _ = query_emb.shape

        # L2 normalize
        feat_norm = F.normalize(feat, dim=-1)          # (B, T, L)
        query_norm = F.normalize(query_emb, dim=-1)    # (B, N, L)

        # 扩展维度以便对齐
        feat_exp = feat_norm.unsqueeze(1)              # (B, 1, T, L)
        query_exp = query_norm.unsqueeze(2)            # (B, N, 1, L)
        mask_exp = query_mask.unsqueeze(-1)            # (B, N, T, 1)

        # 计算每个 query 与其对应正 token 的相似度（只在 mask=1 处有效）
        # 先计算所有相似度
        sim = F.cosine_similarity(query_exp, feat_exp, dim=-1)  # (B, N, T)

        # === 主损失：最大化 query 与 正区域平均特征 的相似度 ===
        # 计算正 token 的平均特征（按 mask 加权）
        masked_feat = feat_exp * mask_exp                      # (B, N, T, L)
        pos_sum = masked_feat.sum(dim=2)                       # (B, N, L)
        pos_count = query_mask.sum(dim=2, keepdim=True)        # (B, N, 1)
        
        # 防止无正样本（理论上应保证至少一个，但保险起见）
        valid = (pos_count > 0).float()
        pos_count = torch.clamp(pos_count, min=1.0)
        pos_mean = pos_sum / pos_count                         # (B, N, L)

        # query 与 pos_mean 的相似度（希望接近 1）
        align_sim = F.cosine_similarity(query_norm, pos_mean, dim=-1)  # (B, N)
        align_loss = (1 - align_sim) * valid.squeeze(-1)               # (B, N)
        align_loss = align_loss.sum() / (valid.sum() + 1e-8)

        # === 可选：正样本内部一致性正则（鼓励 mask 内特征相似）===
        # 对每个正样本 token，使其接近 pos_mean
        if consistency_weight > 0:
            # (B, N, T, L) - (B, N, 1, L) -> 广播
            diff = masked_feat - pos_mean.unsqueeze(2)
            consistency_loss = (diff.norm(dim=-1) ** 2) * mask_exp.squeeze(-1)  # (B, N, T)
            consistency_loss = consistency_loss.sum() / (query_mask.sum() + 1e-8)
        else:
            consistency_loss = 0.0

        total_loss = align_loss + consistency_weight * consistency_loss
        return total_loss

    def infer(
            self,
            batch,
            mask_text=False,
            mask_image=False,
            image_token_type_idx=1,
            img=None,
            output_attentions=False,
            unimodal=False,
            early_quit = False,
    ):
        ret = dict()

        # == Begin: Fetch the inputs ==
        if img is None:
            if f"image_{image_token_type_idx - 1}" in batch:
                img_key = f"image_{image_token_type_idx - 1}"
            else:
                img_key = "image"
            img = batch[img_key]
        bs = img.shape[0]
        if self.vision_enc_type == 3:
            img = batch["image_downsample"]

        text_ids = batch[f"text_ids"]
        #text_labels = batch[f"text_labels{do_mlm}"]
        text_masks = batch[f"text_masks"]
        text_seg_index = batch[f"seg_index"]

        obver_label = batch[f"obver_label"]

        organmask = batch[f"mask"]

        device = text_ids.device

        local_text, local_text_word = self.infer_text(batch["text_embedding"], device)

        ret['local_text_ori'] = local_text_word

        # print(batch['text_query'])
        # print(batch['lesion_mask'].shape)
        # The lesion-alignment branch is optional: it is only active when the
        # dataset actually provides `lesion_mask` / `text_query` (the plain
        # CTRGDataset_RATE_hr_sim pipeline). The qwen-feature dataset does not,
        # so we skip it instead of raising KeyError.
        lesion_mask_batch = batch.get("lesion_mask", None)
        has_lesion = (
            torch.is_tensor(lesion_mask_batch)
            and lesion_mask_batch.ndim >= 4          # (b, d, h, w)
            and lesion_mask_batch.max() != 0
        )
        if has_lesion and not early_quit:
            # print(batch['text_query'])
            text_query = []
            target_mask = []

            b, d, h, w = lesion_mask_batch.shape
            lesion_mask = batch['lesion_mask'].reshape(b, d // 15, 15, h // 30, 30, w // 30, 30).permute(0,1,3,5,2,4,6).reshape(b, 16, 16, 16, -1)
            for l in range(lesion_mask.max().int()):
                text_query.append(batch['text_query'][str(l)][0])
                cmask = (lesion_mask == (l+1)).float().sum(-1)
                # print(cmask.shape)
                target_mask.append(cmask >= 10)
            target_mask = torch.cat(target_mask, 0).reshape(len(text_query), -1).unsqueeze(0)

            local_text_emb = self.infer_text__(self.text_model, text_query, self.tokenizer_text_enc, device)
        else:
            local_text_emb = None
            target_mask = None

        # print(local_text_emb.shape)
        selected_patch, local_image, uni_modal_image_feats, attention_map, x_all_attn, y_all_attn, mask_loss, cmask = self.infer_image(self.vision_encoder, self.vision_extract_layer, self.contrastive_proj_image, self.contrastive_proj_image2, self.expert_image, img.cuda(), organmask, local_text_emb, target_mask)
        
        #print(local_image.shape)
        ret["attention_map"] = attention_map
        ret['local_image'] = F.normalize(local_image, dim = -1) # , eps=1e-8)
        ret['local_text'] = F.normalize(local_text, dim = -1) # , eps=1e-8)
        ret['img'] = img
        ret['mask'] = cmask
        ret['obver_label'] = obver_label
        ret['selected_patch'] = selected_patch
        ret['image_feature'] = uni_modal_image_feats
        ret['x_all_attn'] = x_all_attn
        ret['y_all_attn'] = y_all_attn 
        ret['bs'] = bs
        ret['mask_loss'] = mask_loss
        if early_quit:
            return ret

        with torch.no_grad():
            local_text_m = local_text.detach()
            local_text_m = F.normalize(local_text_m, dim = -1)
            local_text_m_all = torch.cat([local_text_m.t(),self.text_queue.clone().detach()],dim=1)

        ret['local_image_word'] = self.all_gather(uni_modal_image_feats, sync_grads = True)
        ret['local_text_word'] = self.all_gather(local_text_word, sync_grads=True)
        ret['local_text_m_all'] = local_text_m_all
        length = text_seg_index[0][-2]
        ret['length'] = self.all_gather(length)
        
        ret["norm_class_label"] = obver_label[0].unsqueeze(0).float()
        

        # 计算文↔文相似度矩阵
        sim_text = local_text_m @ local_text_m_all
        sim_targets = torch.zeros(sim_text.size()).to(img.device)
        sim_targets.fill_diagonal_(1)       

        total_text = self._per_organ(sim_text, bs)

        soft_target = F.softmax(total_text / self.temp, dim=2)
        # soft_target = F.softmax(sim_text / self.temp, dim=1)  # 归一化后的软目标
        # print(soft_target.shape) # 20 10020

        sim_targets = self._per_organ(sim_targets, bs)

        alpha=1

        

        sim_i2t_targets = alpha * soft_target + (1 - alpha) * sim_targets

        ret["sim_i2t_targets"] = sim_i2t_targets

        self._dequeue_and_enqueue2(self.all_gather(local_text_m), self.all_gather(sim_text[:, self.num_expert * selected_patch.shape[0]:]))

        ret.update({
            "text_ids": text_ids,
            "text_masks": text_masks,
        })

        return ret

    def forward(self, batch, test=False):
        ret = dict()

        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Pre-Training: Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm5(self, batch))
        
        if "rg" in self.current_tasks:
            ret.update(objectives.compute_rg(self, batch))

        return ret

    def training_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)
        total_loss = sum([v * self.hparams.config["loss_names"][k.replace("_loss", "")]
                          for k, v in output.items() if "loss" in k])
        return total_loss


    def on_train_epoch_end(self):
        m3ae_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch)

    def on_validation_epoch_end(self):
        m3ae_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        m3ae_utils.set_task(self)
        output = self(batch, test=True)

    def on_test_epoch_end(self):
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
    def _dequeue_and_enqueue2(self, text_feats, sim):
        # gather keys before updating queue
       
        # print(sim.shape, text_feats.shape) # numver-gpu / bs*10 / 5040, number-gpu / bs * 10 / 128
        text_feats = text_feats.reshape(-1, self.num_expert, self.aligned_length)
        sim = sim.reshape(-1, self.num_expert, self.num_expert, self.queue_size // self.num_expert).mean(-1)
        sim = torch.cat([sim[:, l : l+1, l : l+1] for l in range(self.num_expert)], 1).squeeze()
        # print(text_feats, sim.shape)
        step = self.queue_size // 10
        for idx in range(self.num_expert):
            #print(self.text_queue[:, idx * step : (idx+1) * step].T.shape)
            #print(text_feats.shape)
            cqueue, cscore = self._update_queue(self.text_queue[:, idx * step : (idx+1) * step].T, self.text_queue_sim[idx * step : (idx+1) * step], text_feats[:, idx], sim[:, idx])
            
            #p#rint(cqueue.shape)
            self.text_queue[:, idx * step : (idx+1) * step] = cqueue.T
            self.text_queue_sim[idx * step : (idx+1) * step] = cscore


    def _update_queue(self, text_queue, scores, x, x_scores):
        """
        更新队列，将新特征插入到队列中，并根据分数进行排序和截断。

        参数:
        text_queue (torch.Tensor): 当前的特征队列，维度为 [queue_length, c]。
        scores (torch.Tensor): 当前的分数数组，维度为 [queue_length]。
        x (torch.Tensor): 新的特征，维度为 [b, c]。
        x_scores (torch.Tensor): 新的分数，维度为 [b].

        返回:
        (torch.Tensor, torch.Tensor): 更新后的特征队列和分数数组。
        """
        #print(text_queue.shape, x.shape)
        #print(scores.shape, x_scores.shape)
        # 合并队列和新特征
        combined_features = torch.cat((text_queue, x), dim=0)
        combined_scores = torch.cat((scores, x_scores), dim=0)
        #print(combined_features.shape)
        # 根据分数进行排序
        sorted_indices = torch.argsort(combined_scores, descending=False)
        # print(sorted_indices)
        sorted_features = combined_features[sorted_indices]
        sorted_scores = combined_scores[sorted_indices]
        # print(sorted_features.shape)
        # print(text_queue.size())
        # 截断队列
        updated_text_queue = sorted_features[:text_queue.size(0)]
        updated_scores = sorted_scores[:scores.size(0)]

        return updated_text_queue, updated_scores