# M3D-QAdapter — 预训练 + VQA 微调代码

> **对应论文**：*M3D-QAdapter: 3D Medical VQA with Lesion-Level Finding-Segmentation
> Alignment and Query-Driven Adaptive Token Reduction*
>
> 包含论文两个核心机制 ——
> **病灶级 finding–分割对齐（Lesion-Level Finding-Segmentation Alignment）** 与
> **查询驱动的自适应 token 缩减（Query-Driven Adaptive Token Reduction）**。

**Stage-1 预训练**
- 入口：`main_3D.py`（sacred，named config `task_pretrain_m3ae_3D`）
- 模型：`CTRG_pretrain_v29_textemb_perorgan_mask_lesion`
- 视觉编码：CTViT（`ctvit/`，patch 30×30、temporal patch 15 → 输入 240×480×480）
- 文本编码：CXR-BERT（`text_encoder_path`）+ 报告文本 embedding
- 数据：CT-RATE（`MedicatDataModule_3D_RATE_hr*`）

**Stage-2 VQA 微调**
- 入口：`main_report_gen_vqa.py`（sacred，named config `task_finetune_vqa`）
- 模型：`CTRG_3D_lmae_vqa_v18`（冻结 Stage-1 的 query experts，经**问题引导的
  Masked-FPS token 缩减**后接 LLM 解码器，LoRA 微调）
- LLM 解码器：默认 Llama-3.2-3B（`decoder_path`，可换 Qwen3-4B 等，需对应 `llm_dim`）
- 数据：CT-RATE VQA（`MedicatDataModule_3D_RATE_hr_fea_vqa` → `VQAdataset`，
  Stage-1 预提取的 `image_feature`(4096×768) / `selected_patch` + 16³ 器官 mask）
- 另有备选模型 `CTRG_3D_lmae_vqa_v20_qwen`（用 Qwen3-VL 空间特征 + `OrganFeatureRefiner`），
  文件保留但入口默认使用 v18。

---

## 1. 目录结构

```
CTRG_pretrain_standalone/
├── main_3D.py                     # Stage-1 预训练入口（sacred automain）
├── main_report_gen_vqa.py         # Stage-2 VQA 微调入口（sacred automain）
├── text_latent_feature.npz        # 预训练模型初始化时加载的缓存文本特征
├── run_pretrain.sh                # Stage-1 启动脚本
├── run_vqa.sh                     # Stage-2 启动脚本
├── requirements.txt
├── ctvit/                         # CTViT（本地包，原仓库根目录下）
│   ├── ctvit.py  attention.py
└── CTRG/
    ├── config.py                  # sacred 配置（所有路径支持环境变量/CLI 覆盖）
    ├── datamodules/               # base_datamodule + pretraining_medicat_datamodule
    ├── datasets/                  # pretraining_ctrg_dataset（含 VQAdataset*）+ data_loader
    ├── transforms/                # clip transform / randaug
    ├── gadgets/my_metrics.py      # torchmetrics 指标（BLEU 已本地化，见下）
    └── modules/
        ├── report_generation_pretrain_v29_textemb_perorgan_mask_lesion.py   # Stage-1 预训练模型
        ├── report_generation_vqa_v18.py        # Stage-2 VQA 模型（CTRG_3D_lmae_vqa_v18，当前使用）
        ├── report_generation_vqa_v20_qwen.py   # Stage-2 VQA 备选模型（qwen 特征 + OrganFeatureRefiner）
        ├── patch_selection.py                  # 查询驱动 token 缩减的 token 选择器（FPS/pooling）
        ├── organ_qformer.py                    # OrganFeatureRefiner（送入 LLM 前的 query 适配）
        ├── objectives.py  m3ae_utils.py  dist_utils.py  online_utils.py
        ├── language_encoders/bert_model.py     # BertCrossLayer（token reduction 用）
        ├── models/med.py                       # BertLMHeadModel
        ├── RadFM/vit_3d.py, position_encoding.py
        └── gloria_loss/gloria_loss.py
```

未拷贝（与本仓库两阶段无关）：`CTRG/eval`、`CTRG/generation_api`（pycocoevalcap + 20 个 jar）、
`modules/` 下其余 140+ 个 VQA 变体 / set-prediction / GRPO trainer、
`datasets/set_prediction_dataset*.py`、`set_prediction` 相关配置。
（原 `main_report_gen_vqa.py` import 的 v12~v31 等模型均未实际使用，只保留实际用到的
`CTRG_3D_lmae_vqa_v20_qwen`。）

## 2. 安装

```bash
pip install -r requirements.txt

# ct_clip 不在 PyPI 上，从 CT-CLIP 源码树安装（--no-deps 避免拉入 wilds 等无关依赖）
pip install -e /apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/CT_CLIP --no-deps

# nltk 分词数据（数据集里用到 sent_tokenize）
python -c "import nltk; nltk.download('punkt')"
```


## 3. 运行

### 3.1 Stage-1 预训练

```bash
# 3 卡训练
bash run_pretrain.sh

# 等价命令
CUDA_VISIBLE_DEVICES=0,1,2 python main_3D.py \
    with task_pretrain_m3ae_3D \
    num_gpus=3 num_nodes=1 per_gpu_batchsize=2 test_only=False

# 打开论文的病灶级 finding–分割对齐分支
... with_lesion=True \
    lesion_mask_path=/path/to/mask_preprocessed \
    lesion_mask_json=/path/to/ReXGroundingCT/dataset.json

# 冒烟测试（1 卡 / 2 卡各跑 1 个 train+val batch）
CUDA_VISIBLE_DEVICES=0 python main_3D.py with task_pretrain_m3ae_3D \
    num_gpus=1 per_gpu_batchsize=2 fast_dev_run=True strategy=auto \
    train_limit=20 val_limit=5
```

常用开关：

- `resume_from=<ckpt>` 断点续训
- `test_only=True test_ckpt_path=<ckpt>` 只测试
- `datamodule=rate_hr`（默认，原始 CT volume，喂 CTViT）/ `rate_hr_fea_qwen`
  （预提取 Qwen3-VL 特征 80×8×8×2560，需要 feature 版编码器，当前模型不兼容）
- `with_lesion=True` 打开病灶对齐分支（需要 ReXGroundingCT mask），默认关闭（与原快照一致）
- `num_expert` / `topk_struct`：查询驱动 token 缩减的结构超参
- `strategy=auto` 单卡/CPU 调试用；多卡默认 `ddp_find_unused_parameters_true`

### 3.2 Stage-2 VQA 微调

```bash
# 启动脚本（指定 Stage-1 预训练 ckpt）
PRETRAIN_CKPT=/path/to/stage1_pretrain.ckpt bash run_vqa.sh

# 等价命令
CUDA_VISIBLE_DEVICES=0 python main_report_gen_vqa.py \
    with task_finetune_vqa \
    num_gpus=1 num_nodes=1 test_only=False \
    pretrain_path=/path/to/stage1_pretrain.ckpt

# 冒烟测试（1 卡 1 batch，strategy=auto）
CUDA_VISIBLE_DEVICES=0 python main_report_gen_vqa.py with task_finetune_vqa \
    num_gpus=1 fast_dev_run=True strategy=auto
```

VQA 常用开关：

- `pretrain_path=<ckpt>` 加载 Stage-1 预训练编码器（非 strict；强烈建议）
- `test_only=True test_ckpt_path=<vqa_ckpt>` 用微调后的 VQA ckpt 测试
- `decoder_path=/path/to/LLM text_tokenlizer_path=/path/to/LLM llm_dim=<dim>` 换 LLM 解码器
  （默认 Qwen3-4B / `llm_dim=2560`，需 `transformers>=4.50`）
- `selected_patch=9` 每器官保留的 token 数（查询驱动 token 缩减强度）
- 冻结项：代码自动冻结 `vision_encoder` 与 `expert`（Stage-1 的 CTViT + query experts）


## 8. 引用

> 若使用本代码，请引用 *M3D-QAdapter: 3D Medical VQA with Lesion-Level
> Finding-Segmentation Alignment and Query-Driven Adaptive Token Reduction*。

```bibtex
@inproceedings{m3d-qadapter,
  title   = {M3D-QAdapter: 3D Medical VQA with Lesion-Level Finding-Segmentation
             Alignment and Query-Driven Adaptive Token Reduction},
  author  = {TODO},   % 请按论文正式发表信息补全
  booktitle = {TODO},
  year    = {TODO}
}
```
