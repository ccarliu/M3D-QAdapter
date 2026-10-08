"""M3D-QAdapter -- 3D Medical VQA fine-tuning entry point (Stage-2).

Paper: "M3D-QAdapter: 3D Medical VQA with Lesion-Level Finding-Segmentation
Alignment and Query-Driven Adaptive Token Reduction".

Loads the Stage-1 pretrained query (expert) token-reduction encoder (frozen),
attaches an LLM decoder (LoRA) and fine-tunes on CT-RATE VQA.

Model : CTRG_3D_lmae_vqa_v18  (query-driven masked-FPS token reduction)
Data  : MedicatDataModule_3D_RATE_hr_fea_vqa (pretrain image_feature, 4096x768)

Usage (see run_vqa.sh):
    python main_report_gen_vqa.py with task_finetune_vqa \
        num_gpus=1 per_gpu_batchsize=1 \
        pretrain_path=/path/to/stage1_pretrain.ckpt
"""
import copy
import inspect
import os
import resource
import shutil
import warnings

import pytorch_lightning as pl
import torch
import torch.multiprocessing

from CTRG.config import ex
from CTRG.datamodules.pretraining_medicat_datamodule import (
    MedicatDataModule_3D_RATE_hr,
    MedicatDataModule_3D_RATE_hr_fea_vqa,
)
from CTRG.modules.report_generation_vqa_v18 import CTRG_3D_lmae_vqa_v18

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

warnings.filterwarnings("ignore", category=UserWarning)


def save_code(model, dm, save_root_dir):
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
        shutil.copy2(abs_src_path, os.path.join(code_dir, file_name))
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

    # number of query-reduced patch tokens kept per organ expert
    _config["selected_patch"] = _config.get("selected_patch", 9)

    # Finalise config BEFORE building the model: v18 builds its submodules in
    # `.setup()` and reads image_size / batch there.
    _config["image_size"] = _config["train_image_size"]
    _config["image_depth"] = _config["image_depth"]
    _config["batch_size"] = 1
    _config["per_gpu_batchsize"] = 1

    # Datamodule (v18 consumes the Stage-1 `image_feature` features: 4096x768)
    if _config["use_feature"]:
        dm = MedicatDataModule_3D_RATE_hr_fea_vqa(_config)
    else:
        _config["num_workers"] = 0
        dm = MedicatDataModule_3D_RATE_hr(_config)

    # Build model. v18 uses the Lightning `setup()` hook for construction, so
    # call it manually here to materialise the parameters before loading the
    # Stage-1 checkpoint (the guard inside prevents a second rebuild when
    # Lightning calls setup() again at fit/test time).
    model = CTRG_3D_lmae_vqa_v18(_config)
    model.setup()
    _config["model_name"] = type(model).__name__

    # Load the Stage-1 pretrained encoder (non-strict: the LLM decoder / LoRA /
    # projection heads are freshly initialised here).
    if _config.get("pretrain_path") and _config["pretrain_path"] not in ("xxx", ""):
        ck = torch.load(_config["pretrain_path"], map_location="cpu")["state_dict"]
        model.load_state_dict(ck, strict=False)

    # where generation dumps go (consumed by objectives.compute_rg_blue)
    model.ckpath = _config["vqa_report_dump_path"]

    # In test mode, load the fine-tuned VQA checkpoint before testing.
    test_ckpt = _config["test_ckpt_path"]
    if _config["test_only"]:
        if not test_ckpt:
            raise ValueError("test_only=True requires `test_ckpt_path` (a fine-tuned VQA ckpt)")
        ck = torch.load(test_ckpt, map_location="cpu")["state_dict"]
        model.load_state_dict(ck, strict=False)
        model.ckpath = test_ckpt + "temp"

    model.save_hyperparameters()

    # Freeze the frozen Stage-1 components (vision encoder + query experts).
    for name, param in model.named_parameters():
        if "vision_encoder" in name or "expert" in name:
            param.requires_grad = False

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    save_code(model, dm, tb_logger.log_dir)
    loggers = [tb_logger]

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

    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = _config.get("max_epoch_cap", 20) if max_steps is None else max_steps
    max_epochs = _config["max_epoch"] if max_steps is None else max_epochs
    # VQA always runs batch_size==per_gpu*gpus => grad accumulation of 1
    grad_steps = 1
    print("grad_steps/max_epochs/max_steps:", grad_steps, max_epochs, max_steps)

    trainer = pl.Trainer(
        devices=num_gpus,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        benchmark=True,
        deterministic=False,
        max_epochs=max_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        callbacks=callbacks,
        logger=loggers,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"],
        strategy=_config["strategy"],
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm, ckpt_path=_config["resume_from"])
        if "finetune" in exp_name:
            trainer.test(ckpt_path="best", datamodule=dm)
    else:
        trainer.test(model, datamodule=dm, ckpt_path=test_ckpt)


if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    ex.run_commandline()
