# M3D-QAdapter — 预训练代码（独立可运行）

> **对应论文**：*M3D-QAdapter: 3D Medical VQA with Lesion-Level Finding-Segmentation
> Alignment and Query-Driven Adaptive Token Reduction*
>
> 本仓库是从 `M3AE-master/CTRG_code/main_3D.py` 剥离出来的**论文预训练阶段（Stage-1）
> 独立可运行实现**：CT-RATE 3D CT 报告预训练，包含论文两个核心机制 ——
> **病灶级 finding–分割对齐（Lesion-Level Finding-Segmentation Alignment）** 与
> **查询驱动的自适应 token 缩减（Query-Driven Adaptive Token Reduction）**。

- 训练入口：`main_3D.py`（sacred，named config `task_pretrain_m3ae_3D`）
- 预训练模型：`CTRG.modules.report_generation_pretrain_v29_textemb_perorgan_mask_lesion`
  （`CTRG_pretrain_v29_textemb_perorgan_mask_lesion`）
- 视觉编码：CTViT（`ctvit/`，patch 30×30、temporal patch 15 → 输入 240×480×480）
- 文本编码：CXR-BERT（`text_encoder_path`）+ 报告文本 embedding
- 数据：CT-RATE（`MedicatDataModule_3D_RATE_hr*`）

> 说明：论文的 **3D Medical VQA 微调/推理**部分在原仓库的 VQA / set-prediction / GRPO
> 系列 trainer 中，不在本仓库范围内（本仓库只保留预训练链路及其依赖）。

---

## 1. 论文贡献 → 代码映射

### 1.1 Lesion-Level Finding-Segmentation Alignment（病灶级 finding–分割对齐）

把 ReXGroundingCT 的**病灶分割 mask**与**对应 finding 的文本描述**在视觉 token 网格上
做稀疏对齐监督，使每个病灶区域的特征与其 finding 文本 embedding 对齐。

| 论文字段 | 代码位置 |
|---|---|
| 病灶 mask + finding 文本读取 | `CTRG/datasets/pretraining_ctrg_dataset.py::CTRGDataset_RATE_hr_sim.get_lesion_mask()`（读 `lesion_mask_path/*.npz` 与 `lesion_mask_json`，多通道 mask 压成 `final_lesion_mask`，同时取该例 `findings` 作为 `text_query`） |
| mask 与视觉 token 网格对齐 | `..._lesion.py::infer()`：把 `lesion_mask` reshape 成 `d//15, h//30, w//30` → 16×16×16 patch 网格，`cmask >= 10` 作为正样本 |
| finding 文本编码 | `..._lesion.py::infer_text__()`（`self.text_model` + `tokenizer_text_enc`，参数冻结） |
| 对齐损失 | `..._lesion.py::SparseAlignLoss / SparseAlignLossBaseline`（`self.lesion_align`）：可学习投影 + 可学习 logit scale + sigmoid，pos/neg BCE + safe-negative mining（忽略 prob>0.8 的疑似漏标背景）；另有 `sparse_mask_alignment_loss()` |
| 病灶条件化融合 | `..._lesion.py::self.leision_pre`（1 层 `BertCrossLayer`）与 `infer_image()` 中的 `lesion_loss` |
| 总损失 | `infer_image()`：`loss = sparse_negative_entropy_loss(attn, cmask) + pointwise_cross_entropy_loss(organ_pred, cmask) + lesion_align(...)` |
| 开关 | `with_lesion=True`（默认 False，与原快照一致）+ `lesion_mask_path` / `lesion_mask_json` |

### 1.2 Query-Driven Adaptive Token Reduction（查询驱动的自适应 token 缩减）

用一小组**可学习 query（expert token）**与全量视觉 token 做双向 cross-attention，
按 attention 排序只保留每个 query 最相关的少数 token，从而把 3D CT 的海量 patch token
压缩成极小的 token 集合再送入后续模块。

| 论文字段 | 代码位置 |
|---|---|
| 可学习 query / expert token | `self.expert_image`、`self.expert_text`（`nn.Embedding(num_expert=10, 768)`，`num_expert=10`） |
| Query ↔ token 双向 cross-attention | `self.vision_extract_layer`（2 层 `BertCrossLayer`，`infer_image()` 中 `extract_layer(x, y)` / `extract_layer(y, x)`） |
| Attention 引导的 token 选择 | `infer_image()`：`torch.sort(attention_map, 2)` → 每个 expert 取 top-9（`selected_index[ii, expert_i, -9:]`），拼成 `selected_patch`，再拼回 expert token `x`：4096 patch token → 10×(9+1) 个 token |
| 稀疏性 / 器官监督 | `sparse_negative_entropy_loss(attention_map[:, :9], cmask[:, :9])`（器官 mask 约束 expert attention）、`self.organ_cls` + `pointwise_cross_entropy_loss` |
| 结构超参 | `num_expert=10`、`topk_struct`（config）、`hidden_size=768` |
| 队列式对比学习 | `text_queue` / `text_queue_sim`（10000×512 动量队列）、`_dequeue_and_enqueue2()`、`sim_i2t_targets` 软目标 |
| 注意力图输出 | `infer_image()` 返回 `attention_map = x_attention[1].mean(1)`，可用于病灶/器官可视化 |

---

## 2. 目录结构

```
CTRG_pretrain_standalone/
├── main_3D.py                     # 训练入口（sacred automain）
├── text_latent_feature.npz        # 模型初始化时加载的缓存文本特征
├── run_pretrain.sh                # 启动脚本
├── requirements.txt
├── ctvit/                         # CTViT（本地包，原仓库根目录下）
│   ├── ctvit.py  attention.py
└── CTRG/
    ├── config.py                  # sacred 配置（所有路径支持环境变量/CLI 覆盖）
    ├── datamodules/               # base_datamodule + pretraining_medicat_datamodule
    ├── datasets/                  # pretraining_ctrg_dataset + data_loader
    ├── transforms/                # clip transform / randaug
    ├── gadgets/my_metrics.py      # torchmetrics 指标（BLEU 已本地化，见下）
    └── modules/
        ├── report_generation_pretrain_v29_textemb_perorgan_mask_lesion.py   # 预训练模型（论文 Stage-1）
        ├── objectives.py  m3ae_utils.py  dist_utils.py  online_utils.py
        ├── language_encoders/bert_model.py     # BertCrossLayer（token reduction 用）
        ├── models/med.py                       # BertLMHeadModel
        ├── RadFM/vit_3d.py, position_encoding.py
        └── gloria_loss/gloria_loss.py
```

未拷贝（与预训练无关）：`CTRG/eval`、`CTRG/generation_api`（pycocoevalcap + 20 个 jar）、
`modules/` 下 150+ 个 VQA / set-prediction / GRPO trainer、`datasets/set_prediction_dataset*.py`、
`set_prediction` 相关配置。

## 3. 安装

```bash
pip install -r requirements.txt

# ct_clip 不在 PyPI 上，从 CT-CLIP 源码树安装（--no-deps 避免拉入 wilds 等无关依赖）
pip install -e /apdcephfs_cq10/share_1290796/lh/M3AE-master/CT-CLIP-main/CT_CLIP --no-deps

# nltk 分词数据（数据集里用到 sent_tokenize）
python -c "import nltk; nltk.download('punkt')"
```

## 4. 路径配置

`CTRG/config.py` 里所有数据/权重路径都可以用**环境变量**或 **sacred CLI** 覆盖
（`with text_path_train=/xxx` 优先级最高）。默认值是原机器上的路径，仅作参考。

| 环境变量 | 配置项 | 说明 |
|---|---|---|
| `CTRG_TEXT_MODEL` | `text_model` | BERT config 目录 |
| `CTRG_TEXT_ENCODER_PATH` | `text_encoder_path` | 报告文本编码器（CXR-BERT 目录） |
| `CTRG_TOKENIZER_PATH` | `text_tokenlizer_path` | BertTokenizer 目录 |
| `CTRG_CT_CLIP_CKPT` | `ct_clip_ckpoint` | CT_CLIP_zeroshot.pt（加载 visual/text transformer 权重） |
| `CTRG_TEXT_PATH_TRAIN/TEST` | `text_path_*` | 报告 csv |
| `CTRG_LABEL_PATH_TRAIN/TEST` | `label_path_*` | 多异常标签 csv |
| `CTRG_IMAGE_PATH_TRAIN/TEST` | `image_path_*` | 预处理后的 CT volume（.npz） |
| `CTRG_TEXT_EMB_TRAIN/TEST` | `text_emb_path_*` | `*.npz_text_feature.npy` |
| `CTRG_MASK_TRAIN/TEST` | `mask_path_*` | 器官 mask |
| `CTRG_MASK_FAST_TRAIN/TEST` | `mask_fast_path_*` | fast 器官 mask（.npy） |
| `CTRG_IMGFEA_TRAIN/TEST` | `imgfea_path_*` | 预提取图像特征（qwen dm 用） |
| `CTRG_ABN_TEXT_EMB` | `abnormality_text_embedding_path` | 每例异常文本 embedding |
| `CTRG_NORMALIZED_LABEL_TRAIN/VAL` | `normalized_label_path_*` | region-report csv |
| `CTRG_JSON_PATH_TRAIN/TEST` | `json_path_*` | report hierarchy tree |
| `CTRG_LESION_MASK_PATH/JSON` | `lesion_mask_*` | **ReXGroundingCT 病灶 mask（论文病灶对齐用，见 §1.1）** |
| `CTRG_TEXT_LATENT_FEATURE` | `text_latent_feature_path` | `text_latent_feature.npz` |

## 5. 运行

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

## 6. 相对原仓库的改动

1. **裁剪 import 爆炸**：`CTRG/modules/__init__.py` 置空（原文件 import 100+ 个 trainer，
   会拉入 peft/trl 等全栈依赖）；`datasets/__init__.py`、`datamodules/__init__.py`
   只保留用到的类；`gloria_loss/__init__.py` 置空；数据集文件删掉
   Deeptumor / pillar / VQAdataset 等未使用类（3347 → 2059 行）。
2. **去 `generation_api`**：`gadgets/my_metrics.py` 不再 import
   `generation_api.metrics`（pycocoevalcap + jar），改为文件内纯 Python 实现的
   `compute_scores`（BLEU-1..4 + ROUGE-L；METEOR/CIDER 记 0，预训练任务不用）。
3. **路径全部可配置**：模型里写死的 `BiomedVLP_cxr_bert`、数据集里写死的
   `json_path` / `abnormality_text_embedding_path` / `normalized_label_path_*` /
   mask 路径 / `store_path` 跳过逻辑，全部改为 config 项（带环境变量回退）。
   模型初始化时读的 `./text_latent_feature.npz` 改为 `text_latent_feature_path`。
4. **`main_3D.py` 瘦身**：删掉 7 个未使用的模型 import、`if False:` 里的硬编码
   `/jizhicfs` 分支；`max_epochs=20`、`grad_steps//grad_steps` 等调试残留改为
   `max_epoch_cap` / 正常计算；`test_only` 分支的硬编码 ckpt 改为 `test_ckpt_path`。
5. **健壮性修复**（不改算法语义）：
   - `infer()` 里 `batch["lesion_mask"]` 改为 `batch.get(...)`，且只在 mask 是
     `(b,d,h,w)` 4 维张量时进入病灶对齐分支 —— 原代码在 qwen 数据集（无 lesion_mask）
     上直接 KeyError，在 `with_lesion=False` 时因 `lesion_mask=1` 触发 unpack 报错。
   - 数据集病灶分支由 `if False:` 改为 `if config.get("with_lesion", False):`，
     使论文的 Lesion-Level Alignment 可以按需开启。
6. `ct_clip` 未 vendor 进仓库，按需求写在 `requirements.txt` 里（本地 editable 安装）。

## 7. 已验证

在 V100 × 1 与 V100 × 2 上，`fast_dev_run=True`、`per_gpu_batchsize=2` 均可完整跑通
1 个 train batch + 1 个 val batch（模型 456M 参数，347M 可训练）并正常结束。

已知约束：**`per_gpu_batchsize` 必须 ≥ 2**。动量队列更新
`_dequeue_and_enqueue2` 里 `sim.squeeze()` 在 batch=1 时会把 `(1,10,1)` 压成 `(10,)`，
导致 `sim[:, idx]` 越界；这是原实现就有的边界问题，未改动。

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
