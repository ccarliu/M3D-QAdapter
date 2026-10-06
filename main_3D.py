"""M3D-QAdapter -- 3D CT-RATE pretraining entry point (Stage-1).

Paper: "M3D-QAdapter: 3D Medical VQA with Lesion-Level Finding-Segmentation
Alignment and Query-Driven Adaptive Token Reduction".

This script trains `CTRG_pretrain_v29_textemb_perorgan_mask_lesion`, which
contains the two core mechanisms of the paper:
  * Lesion-Level Finding-Segmentation Alignment  -> `SparseAlignLoss(Baseline)`
    + ReXGroundingCT lesion masks (`with_lesion=True`).
  * Query-Driven Adaptive Token Reduction        -> `expert` query embeddings +
    BertCrossLayer attention-based top-k patch selection in `infer_image()`.

Usage (see run_pretrain.sh):
    python main_3D.py with task_pretrain_m3ae_3D \
        num_gpus=3 per_gpu_batchsize=2 \
        data_root=data/pretrain_arrows/
"""
import copy
import inspect
import os
import resource
import shutil

import pytorch_lightning as pl
import torch

from CTRG.config import ex
from CTRG.datamodules.pretraining_medicat_datamodule import (
    MedicatDataModule_3D_RATE_hr,
    MedicatDataModule_3D_RATE_hr_fea_qwen,
)

# The pretrain module below encodes volumes with CTViT
# (patch 30x30, temporal patch 15 => 240x480x480 raw CT), so by default it is
# fed RAW volumes. `rate_hr_fea_qwen` loads pre-extracted Qwen3-VL spatial
# features (80x8x8x2560) instead, which requires a feature-based encoder.
DATAMODULES = {
    "rate_hr": MedicatDataModule_3D_RATE_hr,
    "rate_hr_fea_qwen": MedicatDataModule_3D_RATE_hr_fea_qwen,
}
from CTRG.modules.report_generation_pretrain_v29_textemb_perorgan_mask_lesion import (
    CTRG_pretrain_v29_textemb_perorgan_mask_lesion,
)

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def save_code(model, dm, save_root_dir):
    """Snapshot the source files that define the current experiment."""
    code_dir = os.path.join(save_root_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    copied_files = set()

    def _copy_source_from_obj(obj, alias_name=None):
        if obj is None:
            return None

        src_path = None

        try:
            src_path = inspect.getsourcefile(obj)
        except Exception as e:
            print(f"[save_code] getsourcefile(obj) failed for {alias_name}: {e}")

        if src_path is None:
            try:
                src_path = inspect.getsourcefile(obj.__class__)
            except Exception as e:
                print(f"[save_code] getsourcefile(obj.__class__) failed for {alias_name}: {e}")

        if not src_path or not os.path.isfile(src_path):
            print(f"[save_code] source file not found for {alias_name}: {src_path}")
            return None

        abs_src_path = os.path.abspath(src_path)
        if abs_src_path in copied_files:
            return abs_src_path
        copied_files.add(abs_src_path)

        file_name = os.path.basename(abs_src_path)
        if alias_name:
            file_name = f"{alias_name}_{file_name}"
        dst_path = os.path.join(code_dir, file_name)
        shutil.copy2(abs_src_path, dst_path)
        return abs_src_path

    copied = {}
    copied["model"] = _copy_source_from_obj(model, "model")

    try:
        import CTRG.modules.objectives as objectives
        copied["objective"] = _copy_source_from_obj(objectives, "objective")
    except Exception as e:
        print(f"[save_code] failed to save objective: {e}")

    organ_refiner = getattr(model, "OrganFeatureRefiner", None)
    copied["organ_refiner"] = _copy_source_from_obj(organ_refiner, "organ_refiner")

    dataset_obj = None
    if hasattr(dm, "dataset_cls"):
        dataset_obj = dm.dataset_cls
    elif hasattr(dm, "train_dataset") and dm.train_dataset is not None:
        dataset_obj = dm.train_dataset
    copied["dataset"] = _copy_source_from_obj(dataset_obj, "dataset")

    print(f"[save_code] target_dir: {code_dir}")
    print(f"[save_code] copied_sources: {copied}")


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Module resolution / batch composition
    _config["image_size"] = _config["train_image_size"]
    _config["image_depth"] = _config["image_depth"]
    _config["batch_size"] = _config["per_gpu_batchsize"]

    dm_name = _config["datamodule"]
    if dm_name not in DATAMODULES:
        raise ValueError(f"unknown datamodule {dm_name!r}, expected one of {list(DATAMODULES)}")
    dm = DATAMODULES[dm_name](_config)

    model = CTRG_pretrain_v29_textemb_perorgan_mask_lesion(_config)

    # optionally resume from a pretrained checkpoint
    if _config.get("pretrain_path") and _config["pretrain_path"] not in ("xxx", ""):
        ck = torch.load(_config["pretrain_path"], map_location="cpu")["state_dict"]
        model.load_state_dict(ck, strict=False)

    # freeze the text encoder
    for name, param in model.named_parameters():
        if "text_model" in name or "text_decoder_ref" in name:
            param.requires_grad = False

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    loggers = [tb_logger]

    save_code(model, dm, tb_logger.log_dir)

    # Callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=3,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=True,
        save_weights_only=True if "finetune" in exp_name else False,
    )
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    callbacks = [checkpoint_callback, lr_callback]

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = _config["max_epoch"] if max_steps is None else _config.get("max_epoch_cap", 20)
    print(grad_steps, max_epochs)

    trainer = pl.Trainer(
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=False,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        strategy=_config["strategy"],
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=30,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        if "finetune" in exp_name:
            trainer.test(ckpt_path="best", datamodule=dm)
    else:
        ckpath = _config["test_ckpt_path"]
        if not ckpath:
            raise ValueError("test_only=True requires `test_ckpt_path` (path to a checkpoint)")
        ck = torch.load(ckpath, map_location="cpu")["state_dict"]
        model.load_state_dict(ck)

        trainer.test(model, datamodule=dm)
