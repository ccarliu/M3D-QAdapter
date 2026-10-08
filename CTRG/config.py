import os

from sacred import Experiment

ex = Experiment("METER", save_git_info=False)


def _env(key, default=""):
    """Path helper.

    Every dataset / checkpoint path below can be overridden either by an
    environment variable (``export CTRG_TEXT_PATH_TRAIN=/your/path``) or on the
    sacred command line (``with text_path_train=/your/path``).  The defaults are
    the paths used on the original machine and are kept only as a reference.
    """
    return os.environ.get(key, default)


def _loss_names(d):
    ret = {
        "mlm": 0,
        "mim": 0,
        "itm": 0,
        "vqa": 0,
        "cls": 0,
        "irtr": 0,
        "rg": 0,
    }
    ret.update(d)
    return ret


@ex.config
def config():
    exp_name = "meter"
    seed = 0
    datasets = ["medicat", "roco"]
    loss_names = _loss_names({"itm": 1, "mlm": 1})
    batch_size = 4096

    # Image setting
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    image_size = 224
    image_depth = 240
    patch_size = 32
    draw_false_image = 1
    image_only = False

    # Text Setting
    vqa_label_size = 3129
    mlc_label_size = 14
    max_text_len = 40
    tokenizer = "bert-base-uncased"
    vocab_size = 30522
    whole_word_masking = True
    mlm_prob = 0.15
    draw_false_text = 0

    # Transformer Setting
    num_top_layer = 6
    input_image_embed_size = 768
    input_text_embed_size = 768
    vit = 'ViT-B/32'
    hidden_size = 768
    num_heads = 12
    num_layers = 6
    mlp_ratio = 4
    drop_rate = 0.1

    # MIM decoder Setting
    mim_prob = 0.75
    mim_decoder_hidden_size = 384
    mim_decoder_num_layers = 4
    mim_decoder_num_heads = 6
    norm_pix_loss = True
    mim_layer = -1

    # Optimizer Setting
    optim_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 0.01
    decay_power = 1
    max_epoch = 100
    # hard cap on epochs when max_steps is set (trainer stops on either)
    max_epoch_cap = 20
    max_steps = 400000
    warmup_steps = 10000
    end_lr = 0
    lr_multiplier_head = 5  # multiply lr for prediction heads
    lr_multiplier_multi_modal = 5  # multiply lr for the multi-modal module

    # Downstream Setting
    get_recall_metric = False

    # PL Trainer Setting
    resume_from = None
    fast_dev_run = False
    val_check_interval = 1.0
    test_only = False
    default_root_dir = "checkpoints"
    # PL strategy; use "auto" for a single GPU / CPU smoke test
    strategy = "ddp_find_unused_parameters_true"
    # which datamodule to build: "rate_hr" (raw CT volumes) or
    # "rate_hr_fea_qwen" (pre-extracted Qwen3-VL spatial features)
    datamodule = "rate_hr"
    # checkpoint used when test_only=True
    test_ckpt_path = ""
    # input resolution used by the 3D module (overrides image_size in main_3D)
    train_image_size = [480, 480]

    # below params varies with the environment
    data_root = ""
    log_dir = "result"
    per_gpu_batchsize = 0
    num_gpus = 8
    num_nodes = 1
    load_path = ""
    num_workers = 3
    precision = 16

    label_column_name = ""

    use_feature = True

    topk_struct = 4

    # ── Data / checkpoint paths (all overridable via env vars or the CLI) ──
    text_model = _env("CTRG_TEXT_MODEL", "/apdcephfs_cq10/share_1290796/lh/dataset/CTRG/model")
    ct_clip_ckpoint = _env("CTRG_CT_CLIP_CKPT", "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert/CT_CLIP_zeroshot.pt")

    text_emb_path_train = _env("CTRG_TEXT_EMB_TRAIN", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/text_embedding/ipmiversion")
    text_emb_path_test = _env("CTRG_TEXT_EMB_TEST", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/M3AE-master/text_embedding/ipmiversion")

    mask_path_train = _env("CTRG_MASK_TRAIN", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/train")
    mask_fast_path_train = _env("CTRG_MASK_FAST_TRAIN", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/train_fast")

    mask_path_test = _env("CTRG_MASK_TEST", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/valid")
    mask_fast_path_test = _env("CTRG_MASK_FAST_TEST", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/valid_fast")

    text_path_train = _env("CTRG_TEXT_PATH_TRAIN", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_reports.csv")
    image_path_train = _env("CTRG_IMAGE_PATH_TRAIN", "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/train_preprocessed")
    label_path_train = _env("CTRG_LABEL_PATH_TRAIN", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_train_predicted_labels.csv")

    text_path_test = _env("CTRG_TEXT_PATH_TEST", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radiology_text_reports_validation_reports.csv")
    image_path_test = _env("CTRG_IMAGE_PATH_TEST", "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/valid_preprocessed")
    label_path_test = _env("CTRG_LABEL_PATH_TEST", "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_multi_abnormality_labels_valid_predicted_labels.csv")

    # finetune / feature paths
    text_tokenlizer_path = _env("CTRG_TOKENIZER_PATH", "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert")
    # BERT used to encode the report text (was hard-coded inside the module).
    text_encoder_path = _env("CTRG_TEXT_ENCODER_PATH", "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert")
    decoder_path = _env("CTRG_DECODER_PATH", "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-2-7b-chat-hf")
    llm_dim = 3072
    pretrain_path = "xxx"

    imgfea_path_train = _env("CTRG_IMGFEA_TRAIN", "/jizhicfs/datalh/M3AE/img_feature/qwen3_4B_vl_multigpu")
    imgfea_path_test = _env("CTRG_IMGFEA_TEST", "/jizhicfs/datalh/M3AE/img_feature/qwen3_4B_vl_multigpu")

    # lesion masks (ReXGroundingCT)
    lesion_mask_path = _env("CTRG_LESION_MASK_PATH", "/jizhicfs/datalh/dataset/CTRATE/data_volumes/dataset/mask_preprocessed")
    lesion_mask_json = _env("CTRG_LESION_MASK_JSON", "/apdcephfs_cq8/private_carlohliu/CTRATE/ReXGroundingCT/dataset.json")

    # per-volume abnormality text embeddings (.npz)
    abnormality_text_embedding_path = _env(
        "CTRG_ABN_TEXT_EMB",
        "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/text_impression_embedding",
    )
    # region-report csv used to build the per-volume abnormality dict
    normalized_label_path_train = _env(
        "CTRG_NORMALIZED_LABEL_TRAIN",
        "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/train_region_report_abnormality_exacted_normalized_c.csv",
    )
    normalized_label_path_val = _env(
        "CTRG_NORMALIZED_LABEL_VAL",
        "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/dataset_radgenome_files_validation_region_report_abnormalities_extracted_normalized_c.csv",
    )
    # report hierarchy trees (only used by the raw-volume dataset variant)
    json_path_train = _env(
        "CTRG_JSON_PATH_TRAIN",
        "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/report_hierarchy_tree/train",
    )
    json_path_test = _env(
        "CTRG_JSON_PATH_TEST",
        "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/report_hierarchy_tree/valid",
    )

    # cached latent text features loaded by the pretrain module
    text_latent_feature_path = _env("CTRG_TEXT_LATENT_FEATURE", "text_latent_feature.npz")

    # ── VQA fine-tuning (M3D-QAdapter Stage-2) ──
    # VQA question/answer csv (CT-RATE VQA).
    vqa_data_train_path = _env(
        "CTRG_VQA_TRAIN",
        "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/VQA/all_data_merged_train_with_targetorgan_llm.csv",
    )
    vqa_data_test_path = _env(
        "CTRG_VQA_TEST",
        "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/VQA/all_data_merged_test_with_targetorgan_llm.csv",
    )
    # number of query-reduced patch tokens kept per organ expert (token reduction)
    selected_patch = 9
    # where the per-sample generated VQA report is dumped during validation/test
    vqa_report_dump_path = _env("CTRG_VQA_REPORT_DUMP", "training_generated_report_vqa.txt")

    # ── Dataset knobs ──
    # enable the optional lesion-mask / text-query branch (needs the
    # ReXGroundingCT masks under `lesion_mask_path`)
    with_lesion = False
    train_limit = 15000000
    val_limit = 1500
    only_use_1 = False
    min_finding_freq = 20
    remove_abnormality_location = False
    remove_abnormality_attributes = False
    group_imp_by_finding = False
    filter_imp_by_min_finding_freq = False
    order_imp_by_anatomy = False
    imp_chain_mode = False
    imp_order_dropout = 0.0
    imp_finding_dropout = 0.0


@ex.named_config
def task_pretrain_m3ae():
    exp_name = "task_pretrain_m3ae"
    datasets = ["medicat", "roco"]
    loss_names = _loss_names({"itm": 1, "mlm": 1, "mim": 1})
    batch_size = 256
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 64
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    learning_rate = 1e-5
    val_check_interval = 1.0
    lr_multiplier_head = 5
    lr_multiplier_multi_modal = 5
    num_top_layer = 6
    hidden_size = 768
    num_heads = 12

    precision = 16
    mim_layer = 3


@ex.named_config
def task_pretrain_m3ae_3D():
    exp_name = "task_pretrain_m3ae"
    datasets = ["medicat", "roco"]
    loss_names = _loss_names({"itm": 0, "mlm": 1, "mim": 0})
    batch_size = 4
    max_epoch = 10
    max_steps = 100000
    warmup_steps = 0.1
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 64
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    learning_rate = 1e-4
    val_check_interval = 1.0
    lr_multiplier_head = 5
    lr_multiplier_multi_modal = 5
    num_top_layer = 6
    hidden_size = 768
    num_heads = 12

    precision = 16
    mim_layer = 3


@ex.named_config
def task_finetune_vqa():
    """M3D-QAdapter VQA fine-tuning (Stage-2): frozen Stage-1 query encoder
    -> query-driven masked-FPS token reduction (model CTRG_3D_lmae_vqa_v18)
    -> LLM decoder (LoRA).

    v18 consumes the Stage-1 ``image_feature`` features (4096x768) plus the
    matching 16x16x16 organ masks (10 organs x 4096 tokens) -- i.e. the SAME
    feature grid produced by the Stage-1 pretrain model, NOT the Qwen3-VL
    spatial features used by the (alternative) v20_qwen trainer.
    """
    exp_name = "task_pretrain_m3ae_vqa"
    datasets = ["medicat", "roco"]
    loss_names = _loss_names({"itm": 0, "mlm": 0, "mim": 0, "rg": 1})
    batch_size = 1
    per_gpu_batchsize = 1
    max_epoch = 20
    max_steps = None
    warmup_steps = 0.025
    whole_word_masking = True

    vocab_size = 30522
    max_text_len = 64
    image_size = 224
    tokenizer = "bert-base-uncased"
    train_transform_keys = ["clip"]
    val_transform_keys = ["clip"]
    learning_rate = 3e-5
    val_check_interval = 1.0
    lr_multiplier_head = 5
    lr_multiplier_multi_modal = 5
    num_top_layer = 6
    hidden_size = 768
    num_heads = 12
    # Stage-1 image_feature embedding dim fed to the query encoder.
    input_image_embed_size = 768

    precision = 16
    mim_layer = 3

    use_feature = True
    selected_patch = 9

    # LLM decoder. Default Llama-3.2-3B works with transformers>=4.30; switch to
    # Qwen3-4B (llm_dim=2560, transformers>=4.50) to match the released ckpt.
    decoder_path = _env("CTRG_DECODER_PATH", "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-3.2-3B")
    llm_dim = 3072
    # the VQA datamodule tokenizes question/answer with the LLM tokenizer,
    # NOT the CXR-BERT one used by pretraining -> point it at the decoder.
    text_tokenlizer_path = _env("CTRG_DECODER_PATH", "/apdcephfs_cq10/share_1290796/lh/dataset/Llama-3.2-3B")
    # BERT used to encode the VQA question.
    text_encoder_path = _env("CTRG_TEXT_ENCODER_PATH", "/apdcephfs_cq10/share_1290796/lh/dataset/BiomedVLP_cxr_bert")
    # v18 needs the Stage-1 extracted features: per volume
    #   <feat>/<key>.npzimage_feature.npy  -> (1, 4096, 768)   (batch["all_image"])
    #   <feat>/<key>.npzselected_patch.npy -> (1, N, 768)      (batch["image"])
    imgfea_path_train = _env("CTRG_IMGFEA_TRAIN", "/jizhicfs/datalh/M3AE/img_feature/version_75_e8_train")
    imgfea_path_test = _env("CTRG_IMGFEA_TEST", "/jizhicfs/datalh/M3AE/img_feature/version_75_e8_train")
    # v18 uses the 16x16x16 organ masks (10 organs x 4096 tokens), matching the
    # image_feature grid -- the standard pretrain `*_fast` masks.
    mask_fast_path_train = _env(
        "CTRG_MASK_FAST_TRAIN",
        "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/train_fast",
    )
    mask_fast_path_test = _env(
        "CTRG_MASK_FAST_TEST",
        "/apdcephfs_cq10/share_1290796/lh/M3AE-master/CTRATE/data_volumes/mask_processed/valid_fast",
    )
