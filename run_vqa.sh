#!/bin/bash
# M3D-QAdapter VQA fine-tuning (Stage-2) on CT-RATE VQA.
# Model: CTRG_3D_lmae_vqa_v18 (query-driven masked-FPS token reduction).
# Loads a Stage-1 pretrained query encoder (frozen) and fine-tunes an LLM
# decoder (LoRA). Default LLM is Llama-3.2-3B (works with transformers>=4.30);
# switch to Qwen3-4B (llm_dim=2560, transformers>=4.50) to match the paper ckpt.
set -e

NUM_GPUS=${NUM_GPUS:-1}
GPUS=${GPUS:-0}
PRETRAIN_CKPT=${PRETRAIN_CKPT:-""}   # Stage-1 pretrain checkpoint (strongly recommended)

CUDA_VISIBLE_DEVICES=${GPUS} python main_report_gen_vqa.py \
    with task_finetune_vqa \
    num_gpus=${NUM_GPUS} num_nodes=1 \
    test_only=False \
    pretrain_path=${PRETRAIN_CKPT}
# 仅测试（加载微调后的 VQA ckpt）:
#   ... test_only=True test_ckpt_path=/path/to/vqa_finetuned.ckpt
# 换用其它 LLM 解码器:
#   ... decoder_path=/path/to/LLM text_tokenlizer_path=/path/to/LLM llm_dim=<dim>
# 换 Stage-1 特征目录（image_feature/selected_patch 所在目录）:
#   ... imgfea_path_train=/path/to/feat imgfea_path_test=/path/to/feat

