from .base_datamodule import BaseDataModule
from ..datasets import (
    CTRGDataset_RATE_hr_sim,
    CTRGDataset_RATE_hr_sim_fea,
    CTRGDataset_RATE_hr_sim_fea_qwen,
    VQAdataset,
    VQAdataset_qwen,
)
from transformers import AutoTokenizer, BertTokenizer


class MedicatDataModule_3D_RATE_hr(BaseDataModule):
    """Raw CT-volume datamodule (240x480x480) -- feeds the CTViT encoder."""

    def __init__(self, _config):
        super().__init__(_config)

        self._config = _config
        path = _config["text_tokenlizer_path"]
        self.tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)

        self.vocab_size = self.tokenizer.vocab_size

        self.setup_flag = False

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE_hr_sim

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(split="train", config=self._config)

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(split="val", config=self._config)

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(split="test", config=self._config)

    @property
    def dataset_name(self):
        return "medicat"


class MedicatDataModule_3D_RATE_hr_fea(BaseDataModule):
    def __init__(self, _config):
        super().__init__(_config)

        self._config = _config
        path = _config["text_tokenlizer_path"]

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
            self.tokenizer.pad_token_id = 0
        except Exception:
            self.tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=True)

        self.setup_flag = False

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE_hr_sim_fea

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(split="train", config=self._config)

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(split="val", config=self._config)

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(split="test", config=self._config)

    @property
    def dataset_name(self):
        return "medicat"


class MedicatDataModule_3D_RATE_hr_fea_qwen(MedicatDataModule_3D_RATE_hr_fea):
    def __init__(self, _config):
        super().__init__(_config)

    @property
    def dataset_cls(self):
        return CTRGDataset_RATE_hr_sim_fea_qwen


class MedicatDataModule_3D_RATE_hr_fea_vqa(BaseDataModule):
    """VQA fine-tuning datamodule (pre-extracted image features + LLM tokenizer)."""

    def __init__(self, _config):
        super().__init__(_config)

        path = _config["text_tokenlizer_path"]
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token_id = 0

        self._config = _config
        self.setup_flag = False

    @property
    def dataset_cls(self):
        return VQAdataset

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(split="train", config=self._config)

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(split="val", config=self._config)

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(split="test", config=self._config)

    @property
    def dataset_name(self):
        return "medicat"


class MedicatDataModule_3D_RATE_hr_fea_vqa_qwen(MedicatDataModule_3D_RATE_hr_fea_vqa):
    """M3D-QAdapter VQA fine-tuning datamodule (Qwen3-VL spatial features)."""

    def __init__(self, _config):
        super().__init__(_config)

    @property
    def dataset_cls(self):
        return VQAdataset_qwen
