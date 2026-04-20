"""Modal launcher for the repository root train_gpt.py baseline.

This file mirrors the documented remote workflow more closely than the original
local-upload starter:

1. Start a Modal GPU container.
2. Download the cached challenge dataset inside the container with
   `data/cached_challenge_fineweb.py`.
3. Run `torchrun ... train_gpt.py` against those downloaded shards.

Example baseline run:

    RUN_ID=baseline_sp1024 \
    VOCAB_SIZE=1024 \
    MODAL_GPU=H100 \
    modal run modal_train_gpt.py

If you want a smaller smoke subset while iterating, add:

    DATASET_TRAIN_SHARDS=1

For a multi-GPU single-node run, set both the Modal GPU request and torchrun
process count before `modal run`, for example:

    MODAL_GPU=H100:8 \
    NPROC_PER_NODE=8 \
    modal run modal_train_gpt.py

Optional fallback:
    Set `DOWNLOAD_DATASET_IN_CONTAINER=0` to skip the remote download step and
    instead upload a local dataset/tokenizer via `LOCAL_DATA_PATH` and
    `LOCAL_TOKENIZER_PATH`.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


APP_NAME = "parameter-golf-train-gpt"
PYTHON_VERSION = "3.11"
REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "train_gpt.py"
DOWNLOAD_SCRIPT = REPO_ROOT / "data" / "cached_challenge_fineweb.py"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"

REMOTE_PROJECT_DIR = Path("/root/parameter-golf")
REMOTE_DATA_DIR = REMOTE_PROJECT_DIR / "data"

GPU_CONFIG = os.environ.get("MODAL_GPU", "H100")
TIMEOUT_SECONDS = int(os.environ.get("MODAL_TIMEOUT_SECONDS", str(24 * 60 * 60)))
TORCHRUN_EXTRA_ARGS = shlex.split(os.environ.get("TORCHRUN_EXTRA_ARGS", ""))
DOWNLOAD_DATASET_IN_CONTAINER = (
    os.environ.get("DOWNLOAD_DATASET_IN_CONTAINER", "1") != "0"
)


def _require_file(path: Path, *, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{description} was not found at {path}")
    return path


def _infer_nproc_per_node(gpu: str) -> int:
    if ":" not in gpu:
        return 1
    _, count = gpu.split(":", maxsplit=1)
    return int(count)


def _infer_data_variant() -> str:
    explicit = os.environ.get("DATA_VARIANT")
    if explicit:
        return explicit

    vocab_size = os.environ.get("VOCAB_SIZE", "1024")
    if vocab_size == "260":
        return "byte260"
    return f"sp{vocab_size}"


def _dataset_dir_name(variant: str) -> str:
    if variant == "byte260":
        return "fineweb10B_byte260"
    if variant.startswith("sp") and variant[2:].isdigit():
        return f"fineweb10B_{variant}"
    raise ValueError(f"Unsupported DATA_VARIANT={variant!r}")


def _tokenizer_file_name(variant: str) -> str:
    if variant == "byte260":
        return "fineweb_byte260.tok"
    if variant.startswith("sp") and variant[2:].isdigit():
        return f"fineweb_{variant[2:]}_bpe.model"
    raise ValueError(f"Unsupported DATA_VARIANT={variant!r}")


def _local_upload_paths() -> tuple[Path, Path]:
    data_path = (
        Path(os.environ.get("LOCAL_DATA_PATH", "./data/datasets/fineweb10B_sp1024"))
        .expanduser()
        .resolve()
    )
    tokenizer_path = (
        Path(
            os.environ.get(
                "LOCAL_TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"
            )
        )
        .expanduser()
        .resolve()
    )
    return data_path, tokenizer_path


def _remote_dataset_paths(variant: str) -> tuple[Path, Path]:
    dataset_dir = REMOTE_DATA_DIR / "datasets" / _dataset_dir_name(variant)
    tokenizer_path = REMOTE_DATA_DIR / "tokenizers" / _tokenizer_file_name(variant)
    return dataset_dir, tokenizer_path


def _download_command(variant: str, env: dict[str, str] | None = None) -> list[str]:
    env = env or os.environ
    train_shards = env.get("DATASET_TRAIN_SHARDS", "80")
    cmd = [
        "python3",
        "data/cached_challenge_fineweb.py",
        "--variant",
        variant,
        "--train-shards",
        train_shards,
    ]
    if env.get("DATASET_WITH_DOCS", "0") == "1":
        cmd.append("--with-docs")
    if env.get("DATASET_SKIP_MANIFEST", "0") == "1":
        cmd.append("--skip-manifest")
    return cmd


def _remote_env_defaults(variant: str) -> dict[str, str]:
    dataset_dir, tokenizer_path = _remote_dataset_paths(variant)
    return {
        "DATA_PATH": str(dataset_dir),
        "TOKENIZER_PATH": str(tokenizer_path),
        "PYTHONUNBUFFERED": "1",
    }


FORWARDED_ENV_VARS = (
    "DATA_VARIANT",
    "DATASET_TRAIN_SHARDS",
    "DATASET_WITH_DOCS",
    "DATASET_SKIP_MANIFEST",
    "RUN_ID",
    "SEED",
    "VAL_BATCH_SIZE",
    "VAL_LOSS_EVERY",
    "TRAIN_LOG_EVERY",
    "ITERATIONS",
    "WARMDOWN_ITERS",
    "WARMUP_STEPS",
    "TRAIN_BATCH_TOKENS",
    "TRAIN_SEQ_LEN",
    "MAX_WALLCLOCK_SECONDS",
    "QK_GAIN_INIT",
    "VOCAB_SIZE",
    "NUM_LAYERS",
    "NUM_KV_HEADS",
    "MODEL_DIM",
    "NUM_HEADS",
    "MLP_MULT",
    "TIE_EMBEDDINGS",
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
    "MATCHED_FINEWEB_REPO_ID",
    "MATCHED_FINEWEB_REMOTE_ROOT_PREFIX",
)


def _collect_forwarded_env(variant: str) -> dict[str, str]:
    env = _remote_env_defaults(variant)
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


app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install_from_requirements(str(REQUIREMENTS_FILE))
    .workdir(str(REMOTE_PROJECT_DIR))
    .add_local_file(TRAIN_SCRIPT, remote_path=str(REMOTE_PROJECT_DIR / "train_gpt.py"))
    .add_local_file(
        DOWNLOAD_SCRIPT,
        remote_path=str(REMOTE_DATA_DIR / "cached_challenge_fineweb.py"),
    )
)

if not DOWNLOAD_DATASET_IN_CONTAINER:
    local_data_path, local_tokenizer_path = _local_upload_paths()
    if not local_data_path.is_dir():
        raise FileNotFoundError(f"LOCAL_DATA_PATH was not found at {local_data_path}")
    if not local_tokenizer_path.is_file():
        raise FileNotFoundError(
            f"LOCAL_TOKENIZER_PATH was not found at {local_tokenizer_path}"
        )
    variant_for_upload = _infer_data_variant()
    remote_dataset_dir, remote_tokenizer_path = _remote_dataset_paths(
        variant_for_upload
    )
    image = image.add_local_dir(
        str(local_data_path), remote_path=str(remote_dataset_dir)
    ).add_local_file(local_tokenizer_path, remote_path=str(remote_tokenizer_path))


@app.function(image=image, gpu=GPU_CONFIG, timeout=TIMEOUT_SECONDS)
def run_train(env: dict[str, str], nproc_per_node: int, variant: str) -> None:
    script_size = (REMOTE_PROJECT_DIR / "train_gpt.py").stat().st_size
    remote_env = os.environ.copy()
    remote_env.update(env)

    if DOWNLOAD_DATASET_IN_CONTAINER:
        download_cmd = _download_command(variant, env=remote_env)
        print(f"[modal] Downloading dataset with: {' '.join(download_cmd)}")
        completed = subprocess.run(
            download_cmd,
            cwd=str(REMOTE_PROJECT_DIR),
            env=remote_env,
            check=False,
        )
        print(f"[modal] Dataset download exit code: {completed.returncode}")
        if completed.returncode != 0:
            raise RuntimeError(
                f"Dataset download failed with exit code {completed.returncode}"
            )
    else:
        print("[modal] Using uploaded local dataset/tokenizer files.")

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc_per_node}",
        *TORCHRUN_EXTRA_ARGS,
        "train_gpt.py",
    ]

    print(f"[modal] Launching {' '.join(cmd)} on {GPU_CONFIG}...")
    print(f"[modal] Script size: {script_size} bytes")
    print(f"[modal] DATA_PATH={remote_env['DATA_PATH']}")
    print(f"[modal] TOKENIZER_PATH={remote_env['TOKENIZER_PATH']}")

    completed = subprocess.run(
        cmd,
        cwd=str(REMOTE_PROJECT_DIR),
        env=remote_env,
        check=False,
    )

    print(f"[modal] Exit code: {completed.returncode}")
    if completed.returncode != 0:
        raise RuntimeError(f"torchrun exited with code {completed.returncode}")


@app.local_entrypoint()
def main() -> None:
    _require_file(TRAIN_SCRIPT, description="train_gpt.py")
    _require_file(DOWNLOAD_SCRIPT, description="data/cached_challenge_fineweb.py")
    _require_file(REQUIREMENTS_FILE, description="requirements.txt")

    variant = _infer_data_variant()
    nproc_per_node = int(
        os.environ.get("NPROC_PER_NODE", str(_infer_nproc_per_node(GPU_CONFIG)))
    )
    env = _collect_forwarded_env(variant)
    run_train.remote(env=env, nproc_per_node=nproc_per_node, variant=variant)
