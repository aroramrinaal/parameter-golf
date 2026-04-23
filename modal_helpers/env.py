from __future__ import annotations

import os


FORWARDED_ENV_VARS = (
    "EXPERIMENT_NAME",
    "DATA_VARIANT",
    "DATASET_TRAIN_SHARDS",
    "DATASET_WITH_DOCS",
    "DATASET_SKIP_MANIFEST",
    "RUN_ID",
    "SEED",
    "VAL_BATCH_SIZE",
    "VAL_LOSS_EVERY",
    "TRAIN_LOG_EVERY",
    "EVAL_MODE",
    "EVAL_SLIDING_STRIDE",
    "EVAL_CONTEXT_LEN",
    "FINAL_TTT_EVAL",
    "TTT_EPOCHS",
    "TTT_LR",
    "TTT_SGD_MOMENTUM",
    "TTT_GRAD_CLIP_NORM",
    "TTT_FREEZE_BLOCKS",
    "TTT_CHUNK_TOKENS",
    "ITERATIONS",
    "WARMDOWN_ITERS",
    "WARMUP_STEPS",
    "TRAIN_BATCH_TOKENS",
    "TRAIN_SEQ_LEN",
    "MAX_WALLCLOCK_SECONDS",
    "QK_GAIN_INIT",
    "VOCAB_SIZE",
    "NUM_LAYERS",
    "PHYSICAL_LAYERS",
    "LOOP_LAYERS",
    "PARALLEL_START_LAYER",
    "NUM_KV_HEADS",
    "MODEL_DIM",
    "NUM_HEADS",
    "MLP_MULT",
    "RANDOM_MLP_PROJ",
    "RANDOM_MLP_PROJ_RANK",
    "RANDOM_MLP_PROJ_GAIN",
    "RANDOM_MLP_PROJ_SEED",
    "TIE_EMBEDDINGS",
    "DEPTH_EMBEDDING",
    "VIRTUAL_LAYER_SCALES",
    "ROPE_BASE",
    "LOGIT_SOFTCAP",
    "EMBED_LR",
    "HEAD_LR",
    "TIED_EMBED_LR",
    "TIED_EMBED_INIT_STD",
    "MATRIX_LR",
    "SCALAR_LR",
    "MUON_MOMENTUM",
    "MUON_BACKEND_STEPS",
    "MUON_MOMENTUM_WARMUP_START",
    "MUON_MOMENTUM_WARMUP_STEPS",
    "BETA1",
    "BETA2",
    "ADAM_EPS",
    "GRAD_CLIP_NORM",
    "CONTROL_TENSOR_NAME_PATTERNS",
    "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
    "QUANT_INT6_NAME_PATTERNS",
    "QUANT_BYTE_SHUFFLE",
    "MATCHED_FINEWEB_REPO_ID",
    "MATCHED_FINEWEB_REMOTE_ROOT_PREFIX",
)


def remote_env_defaults(dataset_dir: str, tokenizer_path: str) -> dict[str, str]:
    return {
        "DATA_PATH": dataset_dir,
        "TOKENIZER_PATH": tokenizer_path,
        "PYTHONUNBUFFERED": "1",
    }


def collect_forwarded_env(dataset_dir: str, tokenizer_path: str) -> dict[str, str]:
    env = remote_env_defaults(dataset_dir, tokenizer_path)
    for key in FORWARDED_ENV_VARS:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value

    extra_env = os.environ.get("FORWARDED_ENV_VARS", "")
    for key in [item.strip() for item in extra_env.split(",") if item.strip()]:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value

    return env
