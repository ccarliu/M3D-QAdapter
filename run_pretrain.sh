#!/bin/bash
# 3D CT-RATE pretraining (single node, multi GPU).
# All data / checkpoint paths can be overridden here or via env vars
# (see README.md -> "路径配置").
set -e

NUM_GPUS=${NUM_GPUS:-3}
PER_GPU_BATCHSIZE=${PER_GPU_BATCHSIZE:-2}
GPUS=${GPUS:-0,1,2}

CUDA_VISIBLE_DEVICES=${GPUS} python main_3D.py \
    with task_pretrain_m3ae_3D \
    data_root=data/pretrain_arrows/ \
    num_gpus=${NUM_GPUS} num_nodes=1 \
    per_gpu_batchsize=${PER_GPU_BATCHSIZE} \
    test_only=False
# 断点续跑:
#   ... resume_from=result/task_pretrain_m3ae-seed0-from_/version_X/checkpoints/last.ckpt
# 仅测试:
#   ... test_only=True test_ckpt_path=/path/to/epoch=x-step=y.ckpt
