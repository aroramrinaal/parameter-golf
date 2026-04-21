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
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import modal
from modal_helpers.env import collect_forwarded_env


APP_NAME = "parameter-golf-train-gpt"
PYTHON_VERSION = "3.11"
REPO_ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = REPO_ROOT / "train_gpt.py"
DOWNLOAD_SCRIPT = REPO_ROOT / "data" / "cached_challenge_fineweb.py"
REQUIREMENTS_FILE = REPO_ROOT / "requirements.txt"
MODAL_HELPERS_DIR = REPO_ROOT / "modal_helpers"

REMOTE_PROJECT_DIR = Path("/root/parameter-golf")
REMOTE_DATA_DIR = REMOTE_PROJECT_DIR / "data"
REMOTE_HELPERS_DIR = Path("/root/modal_helpers")
REMOTE_EXPERIMENTS_DIR = Path("/mnt/experiments")

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


app = modal.App(APP_NAME)
experiments_volume = modal.Volume.from_name(
    "parameter-golf-experiments", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install_from_requirements(str(REQUIREMENTS_FILE))
    .workdir(str(REMOTE_PROJECT_DIR))
    .add_local_dir(str(MODAL_HELPERS_DIR), remote_path=str(REMOTE_HELPERS_DIR))
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


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    timeout=TIMEOUT_SECONDS,
    volumes={str(REMOTE_EXPERIMENTS_DIR): experiments_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
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

    experiment_name = env.get("EXPERIMENT_NAME", env.get("RUN_ID", "unnamed"))
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    tag = f"{experiment_name}_{timestamp}"

    log_src = REMOTE_PROJECT_DIR / "logs" / f"{env.get('RUN_ID', 'run')}.txt"
    log_dst = REMOTE_EXPERIMENTS_DIR / "logs"
    log_dst.mkdir(parents=True, exist_ok=True)
    if log_src.exists():
        shutil.copy2(log_src, log_dst / f"{tag}.txt")
        print(f"[modal] Saved log to {log_dst / f'{tag}.txt'}")

    model_src = REMOTE_PROJECT_DIR / "final_model.int8.ptz"
    model_dst = REMOTE_EXPERIMENTS_DIR / "models"
    model_dst.mkdir(parents=True, exist_ok=True)
    if model_src.exists():
        shutil.copy2(model_src, model_dst / f"{tag}.int8.ptz")
        print(f"[modal] Saved model to {model_dst / f'{tag}.int8.ptz'}")

    experiments_volume.commit()
    print("[modal] Experiment artifacts committed to volume.")


@app.local_entrypoint()
def main() -> None:
    _require_file(TRAIN_SCRIPT, description="train_gpt.py")
    _require_file(DOWNLOAD_SCRIPT, description="data/cached_challenge_fineweb.py")
    _require_file(REQUIREMENTS_FILE, description="requirements.txt")
    if not MODAL_HELPERS_DIR.is_dir():
        raise FileNotFoundError(f"modal_helpers directory was not found at {MODAL_HELPERS_DIR}")

    variant = _infer_data_variant()
    nproc_per_node = int(
        os.environ.get("NPROC_PER_NODE", str(_infer_nproc_per_node(GPU_CONFIG)))
    )
    dataset_dir, tokenizer_path = _remote_dataset_paths(variant)
    env = collect_forwarded_env(str(dataset_dir), str(tokenizer_path))
    run_train.remote(env=env, nproc_per_node=nproc_per_node, variant=variant)
