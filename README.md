# M3D-QAdapter — Pretraining + VQA Finetuning Code

> **Paper**: *M3D-QAdapter: 3D Medical VQA with Lesion-Level Finding-Segmentation
> Alignment and Query-Driven Adaptive Token Reduction*
>
> This repository contains the paper's two core mechanisms:
> **Lesion-Level Finding-Segmentation Alignment** and
> **Query-Driven Adaptive Token Reduction**.

**Stage-1 pretraining**
- Entry point: `main_3D.py` (sacred, named config `task_pretrain_m3ae_3D`)
- Model: `CTRG_pretrain_v29_textemb_perorgan_mask_lesion`
- Vision encoder: CTViT (`ctvit/`, patch 30×30, temporal patch 15 → input 240×480×480)
- Text encoder: CXR-BERT (`text_encoder_path`) + report text embeddings
- Data: CT-RATE (`MedicatDataModule_3D_RATE_hr*`)

**Stage-2 VQA finetuning**
- Entry point: `main_report_gen_vqa.py` (sacred, named config `task_finetune_vqa`)
- Model: `CTRG_3D_lmae_vqa_v18` (freezes the Stage-1 query experts, applies
  **question-guided Masked-FPS token reduction**, then feeds an LLM decoder finetuned with LoRA)
- LLM decoder: Llama-3.2-3B by default (`decoder_path`; can be replaced with e.g. Qwen3-4B,
  with the matching `llm_dim`)
- Data: CT-RATE VQA (`MedicatDataModule_3D_RATE_hr_fea_vqa` → `VQAdataset`, using the
  Stage-1 pre-extracted `image_feature` (4096×768) / `selected_patch` + 16³ organ masks)
- An alternative model, `CTRG_3D_lmae_vqa_v20_qwen` (Qwen3-VL spatial features +
  `OrganFeatureRefiner`), is kept in the repo, but the entry point uses v18 by default.

---

## 1. Layout

```
CTRG_pretrain_standalone/
├── main_3D.py                     # Stage-1 pretraining entry point (sacred automain)
├── main_report_gen_vqa.py         # Stage-2 VQA finetuning entry point (sacred automain)
├── text_latent_feature.npz        # cached text features loaded when the pretraining model is initialized
├── run_pretrain.sh                # Stage-1 launch script
├── run_vqa.sh                     # Stage-2 launch script
├── requirements.txt
├── ctvit/                         # CTViT (local package, originally at the root of the source repo)
│   ├── ctvit.py  attention.py
└── CTRG/
    ├── config.py                  # sacred config (all paths can be overridden via env vars / CLI)
    ├── datamodules/               # base_datamodule + pretraining_medicat_datamodule
    ├── datasets/                  # pretraining_ctrg_dataset (incl. VQAdataset*) + data_loader
    ├── transforms/                # clip transform / randaug
    ├── gadgets/my_metrics.py      # torchmetrics metrics (BLEU implemented locally, see below)
    └── modules/
        ├── report_generation_pretrain_v29_textemb_perorgan_mask_lesion.py   # Stage-1 pretraining model
        ├── report_generation_vqa_v18.py        # Stage-2 VQA model (CTRG_3D_lmae_vqa_v18, in use)
        ├── report_generation_vqa_v20_qwen.py   # Stage-2 alternative VQA model (Qwen features + OrganFeatureRefiner)
        ├── patch_selection.py                  # token selector for query-driven token reduction (FPS / pooling)
        ├── organ_qformer.py                    # OrganFeatureRefiner (query adaptation before the LLM)
        ├── objectives.py  m3ae_utils.py  dist_utils.py  online_utils.py
        ├── language_encoders/bert_model.py     # BertCrossLayer (used for token reduction)
        ├── models/med.py                       # BertLMHeadModel
        ├── RadFM/vit_3d.py, position_encoding.py
        └── gloria_loss/gloria_loss.py
```

Not copied (unrelated to the two stages of this repo): `CTRG/eval`, `CTRG/generation_api`
(pycocoevalcap + 20 jars), the other 140+ VQA variants / set-prediction / GRPO trainers
under `modules/`, `datasets/set_prediction_dataset*.py`, and the `set_prediction` configs.
(The v12–v31 models imported by the original `main_report_gen_vqa.py` were never actually
used; only `CTRG_3D_lmae_vqa_v20_qwen`, which is used, is kept.)

## 2. Installation

```bash
pip install -r requirements.txt

# ct_clip is not on PyPI; install it from the CT-CLIP source tree
# (--no-deps avoids pulling in unrelated dependencies such as wilds)
pip install -e /apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/CT_CLIP --no-deps

# nltk tokenizer data (the dataset uses sent_tokenize)
python -c "import nltk; nltk.download('punkt')"
```


## 3. Running

### 3.1 Stage-1 pretraining

```bash
# 3-GPU training
bash run_pretrain.sh

# Equivalent command
CUDA_VISIBLE_DEVICES=0,1,2 python main_3D.py \
    with task_pretrain_m3ae_3D \
    num_gpus=3 num_nodes=1 per_gpu_batchsize=2 test_only=False

# Enable the paper's lesion-level finding-segmentation alignment branch
... with_lesion=True \
    lesion_mask_path=/path/to/mask_preprocessed \
    lesion_mask_json=/path/to/ReXGroundingCT/dataset.json

# Smoke test (one train + val batch on 1 or 2 GPUs)
CUDA_VISIBLE_DEVICES=0 python main_3D.py with task_pretrain_m3ae_3D \
    num_gpus=1 per_gpu_batchsize=2 fast_dev_run=True strategy=auto \
    train_limit=20 val_limit=5
```

Common options:

- `resume_from=<ckpt>`: resume training from a checkpoint
- `test_only=True test_ckpt_path=<ckpt>`: test only
- `datamodule=rate_hr` (default; raw CT volumes fed to CTViT) / `rate_hr_fea_qwen`
  (pre-extracted Qwen3-VL features 80×8×8×2560; requires a feature-based encoder and is
  not compatible with the current model)
- `with_lesion=True`: enable the lesion alignment branch (requires ReXGroundingCT masks);
  off by default (same as the original snapshot)
- `num_expert` / `topk_struct`: structural hyperparameters of query-driven token reduction
- `strategy=auto`: for single-GPU / CPU debugging; multi-GPU defaults to
  `ddp_find_unused_parameters_true`

### 3.2 Stage-2 VQA finetuning

```bash
# Launch script (pass the Stage-1 pretrained checkpoint)
PRETRAIN_CKPT=/path/to/stage1_pretrain.ckpt bash run_vqa.sh

# Equivalent command
CUDA_VISIBLE_DEVICES=0 python main_report_gen_vqa.py \
    with task_finetune_vqa \
    num_gpus=1 num_nodes=1 test_only=False \
    pretrain_path=/path/to/stage1_pretrain.ckpt

# Smoke test (1 GPU, 1 batch, strategy=auto)
CUDA_VISIBLE_DEVICES=0 python main_report_gen_vqa.py with task_finetune_vqa \
    num_gpus=1 fast_dev_run=True strategy=auto
```

Common VQA options:

- `pretrain_path=<ckpt>`: load the Stage-1 pretrained encoder (non-strict; strongly recommended)
- `test_only=True test_ckpt_path=<vqa_ckpt>`: test with a finetuned VQA checkpoint
- `decoder_path=/path/to/LLM text_tokenlizer_path=/path/to/LLM llm_dim=<dim>`: switch the
  LLM decoder (default Qwen3-4B / `llm_dim=2560`, requires `transformers>=4.50`)
- `selected_patch=9`: number of tokens kept per organ (strength of query-driven token reduction)
- Frozen modules: the code automatically freezes `vision_encoder` and `expert`
  (the Stage-1 CTViT + query experts)


## 4. Citation

> If you use this code, please cite *M3D-QAdapter: 3D Medical VQA with Lesion-Level
> Finding-Segmentation Alignment and Query-Driven Adaptive Token Reduction*.

```bibtex
@inproceedings{m3d-qadapter,
  title   = {M3D-QAdapter: 3D Medical VQA with Lesion-Level Finding-Segmentation
             Alignment and Query-Driven Adaptive Token Reduction},
  author  = {TODO},   % fill in from the published version of the paper
  booktitle = {TODO},
  year    = {TODO}
}
```
