"""
The `train_gpt.py` and `train_gpt_mlx.py` scripts are intended as good launching-off points for new participants, not SOTA configs. We'll accept PRs that tune, improve, or simplify these scripts without significantly increasing complexity, but competitive submissions should stay in the `/records` folder.

Hard stop: To keep readable for newcomers, let's make sure `train_gpt.py` and `train_gpt_mlx.py` never are longer than 1500 lines.
"""

from __future__ import annotations

import copy
import brotli
import glob
import io
import lzma
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func
    _FLASH_ATTN_3_IMPORT_ERROR = None
except Exception:
    flash_attn_3_func = None
    _FLASH_ATTN_3_IMPORT_ERROR = sys.exc_info()[1]

# -----------------------------
# HYPERPARAMETERS
# -----------------------------
# Default Simple Baseline run:
# - 9 transformer blocks at width 512
# - 8 attention heads with 4 KV heads (GQA) and 2x MLP expansion
# - vocab size 1024, sequence length 1024, tied embeddings
# - 524,288 train tokens per step for 20,000 iterations with a ~10 minute cap


def _parse_optional_int_csv(raw_value: str) -> tuple[int, ...]:
    raw_value = raw_value.strip()
    if not raw_value:
        return ()
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("Expected a comma-separated list of integers with no empty entries")
        values.append(int(item))
    return tuple(values)


def _parse_optional_float_csv(raw_value: str) -> tuple[float, ...]:
    raw_value = raw_value.strip()
    if not raw_value:
        return ()
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("Expected a comma-separated list of floats with no empty entries")
        values.append(float(item))
    return tuple(values)


class Hyperparameters:
    # Data paths are shard globs produced by the existing preprocessing pipeline.
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    # Validation cadence and batch size. Validation always uses the full fineweb_val split.
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", os.environ.get("VAL_BATCH_TOKENS", "524288")))
    val_loss_every = int(os.environ.get("VAL_LOSS_EVERY", 1000))
    train_log_every = int(os.environ.get("TRAIN_LOG_EVERY", 200))
    eval_mode = os.environ.get(
        "EVAL_MODE",
        "sliding" if bool(int(os.environ.get("SLIDING_WINDOW_ENABLED", "0"))) else "standard",
    ).strip().lower()
    eval_sliding_stride = int(os.environ.get("EVAL_SLIDING_STRIDE", os.environ.get("EVAL_STRIDE", "64")))
    final_ttt_eval = bool(int(os.environ.get("FINAL_TTT_EVAL", os.environ.get("TTT_ENABLED", "0"))))

    # Training length.
    iterations = int(os.environ.get("ITERATIONS", 20000))
    warmdown_iters = int(os.environ.get("WARMDOWN_ITERS", 1200))
    warmdown_frac = (
        float(os.environ["WARMDOWN_FRAC"])
        if "WARMDOWN_FRAC" in os.environ
        else None
    )
    warmup_steps = int(os.environ.get("WARMUP_STEPS", 20))
    train_batch_tokens = int(os.environ.get("TRAIN_BATCH_TOKENS", 524_288))
    train_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", 1024))
    eval_context_len = int(os.environ.get("EVAL_CONTEXT_LEN", os.environ.get("EVAL_SEQ_LEN", str(train_seq_len))))
    ttt_chunk_tokens = int(os.environ.get("TTT_CHUNK_TOKENS", str(eval_context_len)))
    max_wallclock_seconds = float(os.environ.get("MAX_WALLCLOCK_SECONDS", 600.0))
    gptq_reserve_seconds = float(os.environ.get("GPTQ_RESERVE_SECONDS", "0.0"))
    min_lr = float(os.environ.get("MIN_LR", "0.0"))
    qk_gain_init = float(os.environ.get("QK_GAIN_INIT", 1.5))
    ttt_epochs = int(os.environ.get("TTT_EPOCHS", 3))
    ttt_lr = float(os.environ.get("TTT_LR", 0.005))
    ttt_sgd_momentum = float(os.environ.get("TTT_SGD_MOMENTUM", os.environ.get("TTT_MOMENTUM", "0.9")))
    ttt_grad_clip_norm = float(os.environ.get("TTT_GRAD_CLIP_NORM", 1.0))
    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))
    ttt_batch_seqs = int(os.environ.get("TTT_BATCH_SEQS", "0"))

    # Model shape.
    vocab_size = int(os.environ.get("VOCAB_SIZE", 1024))
    num_layers = int(os.environ.get("NUM_LAYERS", 9))
    physical_layers = int(os.environ.get("PHYSICAL_LAYERS", num_layers))
    loop_layers = _parse_optional_int_csv(os.environ.get("LOOP_LAYERS", ""))
    num_loops = int(os.environ.get("NUM_LOOPS", "0"))
    loop_start = int(os.environ.get("LOOP_START", "0"))
    loop_end = int(os.environ.get("LOOP_END", "-1"))
    enable_looping_at = float(os.environ.get("ENABLE_LOOPING_AT", "0.0"))
    parallel_start_layer = int(os.environ.get("PARALLEL_START_LAYER", str(num_layers)))
    num_kv_heads = int(os.environ.get("NUM_KV_HEADS", 4))
    model_dim = int(os.environ.get("MODEL_DIM", 512))
    num_heads = int(os.environ.get("NUM_HEADS", 8))
    mlp_mult = int(os.environ.get("MLP_MULT", 2))
    random_mlp_proj = bool(int(os.environ.get("RANDOM_MLP_PROJ", "0")))
    random_mlp_proj_rank = int(os.environ.get("RANDOM_MLP_PROJ_RANK", 32))
    random_mlp_proj_gain = bool(int(os.environ.get("RANDOM_MLP_PROJ_GAIN", "1")))
    random_mlp_proj_seed = int(os.environ.get("RANDOM_MLP_PROJ_SEED", seed))
    tie_embeddings = bool(int(os.environ.get("TIE_EMBEDDINGS", "1")))
    depth_embedding = bool(int(os.environ.get("DEPTH_EMBEDDING", "0")))
    virtual_layer_scales = bool(int(os.environ.get("VIRTUAL_LAYER_SCALES", "0")))
    rope_base = float(os.environ.get("ROPE_BASE", 10000.0))
    rope_dims = int(os.environ.get("ROPE_DIMS", "0"))
    rope_train_seq_len = int(os.environ.get("ROPE_TRAIN_SEQ_LEN", str(train_seq_len)))
    logit_softcap = float(os.environ.get("LOGIT_SOFTCAP", 30.0))
    ln_scale = bool(int(os.environ.get("LN_SCALE", "0")))
    xsa_last_n = int(os.environ.get("XSA_LAST_N", "0"))
    skip_gates_enabled = bool(int(os.environ.get("SKIP_GATES_ENABLED", "0")))
    mlp_activation = os.environ.get("MLP_ACTIVATION", "relu_squared").strip().lower()
    mlp_leaky_slope = float(os.environ.get("MLP_LEAKY_SLOPE", "0.5"))
    attn_backend = os.environ.get("ATTN_BACKEND", "auto").strip().lower()

    # Optimizer hyperparameters.
    embed_lr = float(os.environ.get("EMBED_LR", 0.6))
    head_lr = float(os.environ.get("HEAD_LR", 0.008))
    tied_embed_lr = float(os.environ.get("TIED_EMBED_LR", 0.05))
    tied_embed_init_std = float(os.environ.get("TIED_EMBED_INIT_STD", 0.005))
    matrix_lr = float(os.environ.get("MATRIX_LR", 0.04))
    scalar_lr = float(os.environ.get("SCALAR_LR", 0.04))
    muon_momentum = float(os.environ.get("MUON_MOMENTUM", 0.95))
    muon_backend_steps = int(os.environ.get("MUON_BACKEND_STEPS", 5))
    muon_momentum_warmup_start = float(os.environ.get("MUON_MOMENTUM_WARMUP_START", 0.85))
    muon_momentum_warmup_steps = int(os.environ.get("MUON_MOMENTUM_WARMUP_STEPS", 500))
    beta1 = float(os.environ.get("BETA1", 0.9))
    beta2 = float(os.environ.get("BETA2", 0.95))
    adam_eps = float(os.environ.get("ADAM_EPS", 1e-8))
    grad_clip_norm = float(os.environ.get("GRAD_CLIP_NORM", 0.0))
    embed_wd = float(os.environ.get("EMBED_WD", os.environ.get("ADAM_WD", "0.0")))
    head_wd = float(os.environ.get("HEAD_WD", os.environ.get("ADAM_WD", "0.0")))
    scalar_wd = float(os.environ.get("SCALAR_WD", os.environ.get("ADAM_WD", "0.0")))
    muon_wd = float(os.environ.get("MUON_WD", "0.0"))
    ema_decay = float(os.environ.get("EMA_DECAY", "0.0"))
    compressor = os.environ.get("COMPRESSOR", "brotli").strip().lower()
    gptq_calibration_batches = int(os.environ.get("GPTQ_CALIBRATION_BATCHES", "8"))
    embed_bits = int(os.environ.get("EMBED_BITS", "8"))
    matrix_bits = int(os.environ.get("MATRIX_BITS", "6"))

# -----------------------------
# MUON OPTIMIZER 
# -----------------------------
# 
# As borrowed from modded-nanogpt
# Background on Muon: https://kellerjordan.github.io/posts/muon/

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 10, eps: float = 1e-7) -> Tensor:
    # Orthogonalize a 2D update matrix with a fast Newton-Schulz iteration.
    # MuonEq-R row-equalizes the matrix before Newton-Schulz orthogonalization.
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    row_norm = X.float().norm(dim=1, keepdim=True).clamp_min(eps)
    X = X / row_norm.to(dtype=X.dtype)
    X /= X.norm() + eps
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float,
        momentum: float,
        backend_steps: int,
        nesterov: bool = True,
        weight_decay: float = 0.0,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                momentum=momentum,
                backend_steps=backend_steps,
                nesterov=nesterov,
                weight_decay=weight_decay,
            ),
        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0

        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            weight_decay = group["weight_decay"]

            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)

            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    # Scale correction from Muon reference implementations.
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr : curr + p.numel()] = g.reshape(-1)
                curr += p.numel()

            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            curr = 0
            for p in params:
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)
                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()

        return loss


# -----------------------------
# TOKENIZER-AGNOSTIC EVALUATION SETUP 
# -----------------------------
#
# It's common for small models have a large fraction of their parameters be embeddings, since the 2 * d_model * d_vocab vectors can be gigantic.
# Instead of locking the tokenizer, we let you bring your own and calculate our validation metrics on the average compression of the validation set.
# We calculate BPB (bits-per-byte) instead of validation loss, so we need methods to count the number of bits per token in the tokenizer.
# Note: Submissions that edit the tokenizer will be examined more carefully, since screwing this up might unjustly improve your score.

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    # The export pipeline writes the fixed first-50k-doc validation set to fineweb_val_*.
    return torch.cat([load_data_shard(file) for file in files]).contiguous()


def _accumulate_val_bpb(
    x: Tensor,
    y: Tensor,
    score_mask: Tensor,
    per_token_losses: Tensor,
    val_loss_sum: Tensor,
    val_token_count: Tensor,
    val_byte_count: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> None:
    scored_losses = per_token_losses[score_mask].to(torch.float64)
    val_loss_sum += scored_losses.sum()
    val_token_count += float(scored_losses.numel())
    prev_ids = x[score_mask]
    tgt_ids = y[score_mask]
    token_bytes = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
    token_bytes += (has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]).to(dtype=torch.int16)
    val_byte_count += token_bytes.to(torch.float64).sum()


def _finalize_val_metrics(
    model: nn.Module,
    val_loss_sum: Tensor,
    val_token_count: Tensor,
    val_byte_count: Tensor,
) -> tuple[float, float]:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_standard(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.train_seq_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sequence per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, TRAIN_SEQ_LEN={args.train_seq_len}"
        )
    local_batch_seqs = local_batch_tokens // args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // args.train_seq_len
    if total_seqs <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={args.train_seq_len}")
    seq_start = (total_seqs * rank) // world_size
    seq_end = (total_seqs * (rank + 1)) // world_size
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    model.eval()
    with torch.inference_mode():
        for batch_seq_start in range(seq_start, seq_end, local_batch_seqs):
            batch_seq_end = min(batch_seq_start + local_batch_seqs, seq_end)
            raw_start = batch_seq_start * args.train_seq_len
            raw_end = batch_seq_end * args.train_seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, args.train_seq_len)
            y = local[1:].reshape(-1, args.train_seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                per_token_losses = model(x, y, reduction="none").detach()
            _accumulate_val_bpb(
                x,
                y,
                torch.ones_like(per_token_losses, dtype=torch.bool),
                per_token_losses,
                val_loss_sum,
                val_token_count,
                val_byte_count,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )

    return _finalize_val_metrics(model, val_loss_sum, val_token_count, val_byte_count)


def _run_sliding_batch(
    model: nn.Module,
    device: torch.device,
    batch_windows: list[Tensor],
    batch_targets: list[Tensor],
    score_ranges: list[tuple[int, int]],
    desired_batch_size: int,
    context_len: int,
    val_loss_sum: Tensor,
    val_token_count: Tensor,
    val_byte_count: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> None:
    if not batch_windows:
        return
    padded_windows = list(batch_windows)
    padded_targets = list(batch_targets)
    padded_score_ranges = list(score_ranges)
    if len(padded_windows) > desired_batch_size:
        raise ValueError(
            f"Sliding eval batch overflow: got {len(padded_windows)} windows for desired_batch_size={desired_batch_size}"
        )
    pad_window = torch.zeros((context_len,), dtype=batch_windows[0].dtype)
    while len(padded_windows) < desired_batch_size:
        padded_windows.append(pad_window)
        padded_targets.append(pad_window)
        padded_score_ranges.append((0, 0))
    x = torch.stack(padded_windows).to(device=device, dtype=torch.int64, non_blocking=True)
    y = torch.stack(padded_targets).to(device=device, dtype=torch.int64, non_blocking=True)
    score_mask = torch.zeros_like(y, dtype=torch.bool)
    for i, (score_from, score_to) in enumerate(padded_score_ranges):
        score_mask[i, score_from:score_to] = True
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        per_token_losses = model(x, y, reduction="none").detach()
    _accumulate_val_bpb(
        x,
        y,
        score_mask,
        per_token_losses,
        val_loss_sum,
        val_token_count,
        val_byte_count,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )


def _pad_sliding_sequence(tokens: Tensor, target_len: int) -> Tensor:
    if tokens.numel() > target_len:
        raise ValueError(f"Sliding eval sequence length {tokens.numel()} exceeds target_len={target_len}")
    if tokens.numel() == target_len:
        return tokens
    return torch.cat((tokens, torch.zeros((target_len - tokens.numel(),), dtype=tokens.dtype)))


def eval_val_sliding(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    if args.eval_context_len <= 0:
        raise ValueError(f"EVAL_CONTEXT_LEN must be positive, got {args.eval_context_len}")
    if args.eval_sliding_stride <= 0:
        raise ValueError(f"EVAL_SLIDING_STRIDE must be positive, got {args.eval_sliding_stride}")
    total_targets = val_tokens.numel() - 1
    if total_targets <= 0:
        raise ValueError("Validation split is empty")

    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    if local_batch_tokens < args.eval_context_len:
        raise ValueError(
            "VAL_BATCH_SIZE must provide at least one sliding window per rank; "
            f"got VAL_BATCH_SIZE={args.val_batch_size}, WORLD_SIZE={world_size}, "
            f"GRAD_ACCUM_STEPS={grad_accum_steps}, EVAL_CONTEXT_LEN={args.eval_context_len}"
        )
    local_batch_windows_max = max(local_batch_tokens // args.eval_context_len, 1)
    local_target_start = (total_targets * rank) // world_size
    local_target_end = (total_targets * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)

    batch_windows: list[Tensor] = []
    batch_targets: list[Tensor] = []
    score_ranges: list[tuple[int, int]] = []
    prev_end = 0

    def flush() -> None:
        nonlocal batch_windows, batch_targets, score_ranges
        _run_sliding_batch(
            model,
            device,
            batch_windows,
            batch_targets,
            score_ranges,
            local_batch_windows_max,
            args.eval_context_len,
            val_loss_sum,
            val_token_count,
            val_byte_count,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        batch_windows = []
        batch_targets = []
        score_ranges = []

    model.eval()
    with torch.inference_mode():
        for begin in range(0, total_targets, args.eval_sliding_stride):
            end = min(begin + args.eval_context_len, total_targets)
            score_segment_start = prev_end
            score_segment_end = end
            prev_end = end

            overlap_start = max(score_segment_start, local_target_start)
            overlap_end = min(score_segment_end, local_target_end)
            if overlap_start >= overlap_end:
                if end >= total_targets:
                    break
                continue

            local = val_tokens[begin : end + 1]
            x = _pad_sliding_sequence(local[:-1], args.eval_context_len)
            y = _pad_sliding_sequence(local[1:], args.eval_context_len)
            score_from = overlap_start - begin
            score_to = overlap_end - begin

            batch_windows.append(x)
            batch_targets.append(y)
            score_ranges.append((score_from, score_to))
            if len(batch_windows) >= local_batch_windows_max:
                flush()

            if end >= total_targets:
                break
        flush()

    return _finalize_val_metrics(model, val_loss_sum, val_token_count, val_byte_count)


def eval_val(
    args: Hyperparameters,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    # Validation computes two metrics:
    # - val_loss: token cross-entropy (natural log)
    # - val_bpb: tokenizer-agnostic compression metric used by the challenge
    if args.eval_mode == "standard":
        return eval_val_standard(
            args,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
    if args.eval_mode == "sliding":
        return eval_val_sliding(
            args,
            model,
            rank,
            world_size,
            device,
            grad_accum_steps,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
    raise ValueError(f"Unsupported EVAL_MODE={args.eval_mode!r}; expected 'standard' or 'sliding'")


def _set_ttt_frozen_blocks(model: "GPT", freeze_blocks: int) -> dict[str, bool]:
    original: dict[str, bool] = {}
    for name, param in model.named_parameters():
        original[name] = bool(param.requires_grad)
        block_idx = None
        if name.startswith("blocks."):
            parts = name.split(".", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                block_idx = int(parts[1])
        param.requires_grad_(block_idx is None or block_idx >= freeze_blocks)
    return original


def _restore_requires_grad(model: "GPT", original: dict[str, bool]) -> None:
    for name, param in model.named_parameters():
        param.requires_grad_(original.get(name, True))


def _build_ttt_chunk_windows(
    tokens: Tensor,
    context_len: int,
    stride: int,
) -> list[tuple[Tensor, Tensor, tuple[int, int]]]:
    total_targets = tokens.numel() - 1
    prev_end = 0
    windows: list[tuple[Tensor, Tensor, tuple[int, int]]] = []
    for begin in range(0, total_targets, stride):
        end = min(begin + context_len, total_targets)
        score_from = prev_end - begin
        score_to = end - begin
        prev_end = end
        if score_from >= score_to:
            if end >= total_targets:
                break
            continue
        local = tokens[begin : end + 1]
        x = _pad_sliding_sequence(local[:-1], context_len)
        y = _pad_sliding_sequence(local[1:], context_len)
        windows.append((x, y, (score_from, score_to)))
        if end >= total_targets:
            break
    return windows


def _build_ttt_train_windows(tokens: Tensor, context_len: int) -> list[tuple[Tensor, Tensor, int]]:
    total_targets = tokens.numel() - 1
    windows: list[tuple[Tensor, Tensor, int]] = []
    for begin in range(0, total_targets, context_len):
        end = min(begin + context_len, total_targets)
        local = tokens[begin : end + 1]
        x = _pad_sliding_sequence(local[:-1], context_len)
        y = _pad_sliding_sequence(local[1:], context_len)
        windows.append((x, y, local.numel() - 1))
    return windows


def _score_ttt_chunk(
    model: "GPT",
    device: torch.device,
    tokens: Tensor,
    stride: int,
    context_len: int,
    desired_batch_size: int,
    val_loss_sum: Tensor,
    val_token_count: Tensor,
    val_byte_count: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> list[tuple[Tensor, Tensor, int]]:
    windows = _build_ttt_chunk_windows(tokens, context_len, stride)
    train_windows = _build_ttt_train_windows(tokens, context_len)
    batch_windows: list[Tensor] = []
    batch_targets: list[Tensor] = []
    score_ranges: list[tuple[int, int]] = []
    model.eval()
    with torch.inference_mode():
        for x, y, score_range in windows:
            batch_windows.append(x)
            batch_targets.append(y)
            score_ranges.append(score_range)
            if len(batch_windows) >= desired_batch_size:
                _run_sliding_batch(
                    model,
                    device,
                    batch_windows,
                    batch_targets,
                    score_ranges,
                    desired_batch_size,
                    context_len,
                    val_loss_sum,
                    val_token_count,
                    val_byte_count,
                    base_bytes_lut,
                    has_leading_space_lut,
                    is_boundary_token_lut,
                )
                batch_windows = []
                batch_targets = []
                score_ranges = []
        if batch_windows:
            _run_sliding_batch(
                model,
                device,
                batch_windows,
                batch_targets,
                score_ranges,
                desired_batch_size,
                context_len,
                val_loss_sum,
                val_token_count,
                val_byte_count,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
    return train_windows


def _ttt_update_chunk(
    args: Hyperparameters,
    model: "GPT",
    device: torch.device,
    train_windows: list[tuple[Tensor, Tensor, int]],
) -> None:
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params or args.ttt_epochs <= 0 or not train_windows:
        return
    optimizer = torch.optim.SGD(trainable_params, lr=args.ttt_lr, momentum=args.ttt_sgd_momentum)
    model.train()
    batch_size = args.ttt_batch_seqs if args.ttt_batch_seqs > 0 else max(args.val_batch_size // max(args.eval_context_len, 1), 1)
    total_steps = args.ttt_epochs * math.ceil(len(train_windows) / batch_size)
    global_step = 0
    for _epoch in range(args.ttt_epochs):
        for batch_start in range(0, len(train_windows), batch_size):
            cosine = 0.5 * (1.0 + math.cos(math.pi * global_step / max(total_steps, 1)))
            for group in optimizer.param_groups:
                group["lr"] = args.ttt_lr * cosine
            batch = train_windows[batch_start : batch_start + batch_size]
            x = torch.stack([window_x for window_x, _window_y, _valid_len in batch]).to(device=device, dtype=torch.int64, non_blocking=True)
            y = torch.stack([window_y for _window_x, window_y, _valid_len in batch]).to(device=device, dtype=torch.int64, non_blocking=True)
            valid_lens = [valid_len for _window_x, _window_y, valid_len in batch]
            train_mask = torch.zeros_like(y, dtype=torch.bool)
            for row_idx, valid_len in enumerate(valid_lens):
                train_mask[row_idx, :valid_len] = True
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                per_token_loss = model(x, y, reduction="none")
                loss = per_token_loss[train_mask].mean()
            loss.backward()
            if args.ttt_grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.ttt_grad_clip_norm)
            optimizer.step()
            global_step += 1
    optimizer.zero_grad(set_to_none=True)
    model.eval()


def eval_val_score_first_ttt(
    args: Hyperparameters,
    model: "GPT",
    rank: int,
    world_size: int,
    device: torch.device,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
) -> tuple[float, float]:
    if args.ttt_chunk_tokens <= 0:
        raise ValueError(f"TTT_CHUNK_TOKENS must be positive, got {args.ttt_chunk_tokens}")
    total_targets = val_tokens.numel() - 1
    if total_targets <= 0:
        raise ValueError("Validation split is empty")
    local_target_start = (total_targets * rank) // world_size
    local_target_end = (total_targets * (rank + 1)) // world_size

    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    local_batch_windows_max = max(args.val_batch_size // max(world_size * args.eval_context_len, 1), 1)

    original_requires_grad = _set_ttt_frozen_blocks(model, args.ttt_freeze_blocks)
    try:
        for chunk_start in range(local_target_start, local_target_end, args.ttt_chunk_tokens):
            chunk_end = min(chunk_start + args.ttt_chunk_tokens, local_target_end)
            chunk_tokens = val_tokens[chunk_start : chunk_end + 1]
            train_windows = _score_ttt_chunk(
                model,
                device,
                chunk_tokens,
                args.eval_sliding_stride,
                args.eval_context_len,
                local_batch_windows_max,
                val_loss_sum,
                val_token_count,
                val_byte_count,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            _ttt_update_chunk(args, model, device, train_windows)
    finally:
        _restore_requires_grad(model, original_requires_grad)

    return _finalize_val_metrics(model, val_loss_sum, val_token_count, val_byte_count)

# -----------------------------
# POST-TRAINING QUANTIZATION
# -----------------------------
#
# It's silly to export our model, which is trained in bf16 and fp32, at that same precision.
# Instead, we get approximately the same model (with a small hit) by quantizing the model to int8 & Brotli compressing.
# We can then decompress the model and run in higher precision for evaluation, after closing in under the size limit.

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        (
            "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,"
            "virtual_attn_scale,virtual_attn_scales,virtual_mlp_scale,virtual_mlp_scales,"
            "virtual_resid_mix,virtual_resid_mixes,depth_embed,q_gain,skip_weight,skip_weights,"
            "skip_gates,"
            "lane_merge"
        ),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_AFFINE_OFFSET_DTYPE = torch.float16
SDCLIP_STD_MULT = 2.5
EMBED_GPTQ_CLIP_STD_MULTS = _parse_optional_float_csv(
    os.environ.get("EMBED_GPTQ_CLIP_STD_MULTS", "8.0,12.0,16.0,20.0,24.0")
)
MATRIX_INT8_GPTQ_CLIP_STD_MULTS = _parse_optional_float_csv(
    os.environ.get("INT8_GPTQ_CLIP_STD_MULTS", "8.0,12.0,16.0,20.0,24.0")
)
MLP_INT6_GPTQ_CLIP_STD_MULTS = _parse_optional_float_csv(
    os.environ.get("INT6_GPTQ_CLIP_STD_MULTS", "4.0,6.0,8.0,10.0,12.85,16.0")
)
FULL_GPTQ_INT6 = bool(int(os.environ.get("FULL_GPTQ_INT6", "0")))
EMBED_BITS = int(os.environ.get("EMBED_BITS", "8"))
MATRIX_BITS = int(os.environ.get("MATRIX_BITS", "6"))
EMBED_QUANT_QMAX = (1 << (EMBED_BITS - 1)) - 1
MATRIX_QUANT_QMAX = (1 << (MATRIX_BITS - 1)) - 1
QUANT_INT6_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "QUANT_INT6_NAME_PATTERNS",
        "mlp.fc.weight,mlp.proj.weight",
    ).split(",")
    if pattern
)
QUANT_BYTE_SHUFFLE = bool(int(os.environ.get("QUANT_BYTE_SHUFFLE", "0")))
BYTE_SHUFFLE_GROUP_SIZE = 8

def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())

def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def _reshape_row_scale(scale: Tensor, q: Tensor) -> Tensor:
    return scale.to(dtype=torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))


def _quantize_rowwise_gptq_symmetric(
    t: Tensor,
    hdiag: Tensor | None,
    qmax: int,
    clip_std_mults: tuple[float, ...],
) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim != 2:
        raise ValueError(f"Expected a 2D tensor for rowwise GPTQ quantization, got shape={tuple(t32.shape)}")
    if hdiag is None:
        hdiag = torch.ones((t32.shape[1],), dtype=torch.float32, device=t32.device)
    else:
        hdiag = hdiag.to(dtype=torch.float32, device=t32.device)
    row_std = t32.std(dim=1, unbiased=False)
    row_absmax = t32.abs().amax(dim=1)
    best_err = torch.full((t32.shape[0],), float("inf"), dtype=torch.float32, device=t32.device)
    best_q = torch.zeros_like(t32, dtype=torch.int8)
    best_scale = torch.ones((t32.shape[0],), dtype=torch.float32, device=t32.device)
    min_scale = 1.0 / max(qmax, 1)
    for std_mult in clip_std_mults:
        clip_abs = torch.minimum(row_absmax, row_std * std_mult).clamp_min(min_scale)
        q = torch.clamp(
            torch.round(torch.clamp(t32, -clip_abs[:, None], clip_abs[:, None]) / clip_abs[:, None] * qmax),
            -qmax,
            qmax,
        ).to(torch.int8)
        scale = (clip_abs / qmax).clamp_min(min_scale)
        deq = q.float() * scale[:, None]
        err = ((t32 - deq).square() * hdiag[None, :]).sum(dim=1)
        improved = err < best_err
        if improved.any():
            best_err = torch.where(improved, err, best_err)
            best_q[improved] = q[improved]
            best_scale[improved] = scale[improved]
    return best_q.contiguous(), best_scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()


def _pack_int6_signed(q: Tensor) -> tuple[Tensor, int]:
    if q.dtype != torch.int8:
        raise ValueError(f"Expected int8 input for int6 pack, got {q.dtype}")
    flat = q.reshape(-1).to(dtype=torch.int16)
    qmin = -MATRIX_QUANT_QMAX
    qmax = MATRIX_QUANT_QMAX
    if bool(((flat < qmin) | (flat > qmax)).any().item()):
        raise ValueError(f"Int6 pack expected values in [{qmin}, {qmax}]")
    original_numel = int(flat.numel())
    pad = (-original_numel) % 4
    if pad:
        flat = torch.cat((flat, torch.zeros((pad,), dtype=torch.int16, device=flat.device)))
    u = (flat + 32).reshape(-1, 4)
    b0 = (u[:, 0] | ((u[:, 1] & 0x03) << 6)).to(dtype=torch.uint8)
    b1 = (((u[:, 1] >> 2) & 0x0F) | ((u[:, 2] & 0x0F) << 4)).to(dtype=torch.uint8)
    b2 = (((u[:, 2] >> 4) & 0x03) | (u[:, 3] << 2)).to(dtype=torch.uint8)
    return torch.stack((b0, b1, b2), dim=1).reshape(-1).contiguous(), original_numel


def _unpack_int6_signed(packed: Tensor, original_numel: int, shape: tuple[int, ...]) -> Tensor:
    flat = packed.reshape(-1).to(dtype=torch.int16)
    if flat.numel() % 3 != 0:
        raise ValueError(f"Packed int6 tensor expected a multiple of 3 bytes, got {flat.numel()}")
    triple = flat.reshape(-1, 3)
    v0 = triple[:, 0] & 0x3F
    v1 = ((triple[:, 0] >> 6) & 0x03) | ((triple[:, 1] & 0x0F) << 2)
    v2 = ((triple[:, 1] >> 4) & 0x0F) | ((triple[:, 2] & 0x03) << 4)
    v3 = (triple[:, 2] >> 2) & 0x3F
    unpacked = torch.stack((v0, v1, v2, v3), dim=1).reshape(-1)[:original_numel]
    return (unpacked - 32).to(dtype=torch.int8).reshape(shape).contiguous()


def _byte_shuffle_bytes(data: bytes, group_size: int) -> bytes:
    if group_size <= 1 or len(data) <= group_size:
        return data
    return b"".join(data[offset::group_size] for offset in range(group_size))


def _byte_unshuffle_bytes(data: bytes, group_size: int) -> bytes:
    if group_size <= 1 or len(data) <= group_size:
        return data
    base = len(data) // group_size
    rem = len(data) % group_size
    lanes = []
    cursor = 0
    for lane_idx in range(group_size):
        lane_len = base + (1 if lane_idx < rem else 0)
        lanes.append(data[cursor : cursor + lane_len])
        cursor += lane_len
    out = bytearray(len(data))
    for lane_idx, lane in enumerate(lanes):
        out[lane_idx::group_size] = lane
    return bytes(out)


def _compress_quant_bytes(data: bytes, compressor: str) -> bytes:
    if compressor == "brotli":
        return brotli.compress(data, quality=11)
    if compressor == "lzma":
        return lzma.compress(data, preset=6)
    raise ValueError(f"Unsupported COMPRESSOR={compressor!r}")


def _decompress_quant_bytes(data: bytes, compressor: str) -> bytes:
    if compressor == "brotli":
        return brotli.decompress(data)
    if compressor == "lzma":
        return lzma.decompress(data)
    raise ValueError(f"Unsupported COMPRESSOR={compressor!r}")


def quant_group_for_name(name: str) -> str:
    if name == "tok_emb.weight" or name == "lm_head.weight":
        return "embedding"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    if any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS):
        return "control"
    if "norm" in name:
        return "norm"
    return "misc"


def make_quant_group_stats() -> dict[str, dict[str, int]]:
    return {
        group: {"raw_bytes": 0, "payload_bytes": 0, "int6_bytes": 0, "num_tensors": 0}
        for group in ("embedding", "attention", "mlp", "control", "norm", "misc")
    }


def record_quant_group_stats(
    stats: dict[str, object],
    name: str,
    *,
    raw_bytes: int,
    payload_bytes: int,
    int6_bytes: int = 0,
) -> None:
    group = quant_group_for_name(name)
    group_stats = stats["group_stats"][group]
    group_stats["raw_bytes"] += raw_bytes
    group_stats["payload_bytes"] += payload_bytes
    group_stats["int6_bytes"] += int6_bytes
    group_stats["num_tensors"] += 1

def quantize_float_tensor_sdclip(t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        mean = t32.mean(dim=1)
        std = t32.std(dim=1, unbiased=False)
        clip_lo = mean - SDCLIP_STD_MULT * std
        clip_hi = mean + SDCLIP_STD_MULT * std
        degenerate = clip_hi <= clip_lo
        if degenerate.any():
            row_absmax = t32.abs().amax(dim=1)
            clip_lo = torch.where(degenerate, -row_absmax, clip_lo)
            clip_hi = torch.where(degenerate, row_absmax, clip_hi)
        offset = 0.5 * (clip_hi + clip_lo)
        scale = ((clip_hi - clip_lo) / 254.0).clamp_min(1.0 / 127.0)
        clipped = torch.maximum(torch.minimum(t32, clip_hi[:, None]), clip_lo[:, None])
        q = torch.clamp(torch.round((clipped - offset[:, None]) / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return (
            q,
            scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous(),
            offset.to(dtype=INT8_AFFINE_OFFSET_DTYPE).contiguous(),
        )

    mean = t32.mean() if t32.numel() else torch.tensor(0.0, dtype=torch.float32)
    std = t32.std(unbiased=False) if t32.numel() else torch.tensor(0.0, dtype=torch.float32)
    clip_lo = mean - SDCLIP_STD_MULT * std if t32.numel() else torch.tensor(0.0)
    clip_hi = mean + SDCLIP_STD_MULT * std if t32.numel() else torch.tensor(0.0)
    if t32.numel() and clip_hi <= clip_lo:
        clip_abs = t32.abs().amax()
        clip_lo = -clip_abs
        clip_hi = clip_abs
    offset = 0.5 * (clip_hi + clip_lo)
    scale = ((clip_hi - clip_lo) / 254.0).clamp_min(1.0 / 127.0)
    clipped = torch.clamp(t32, float(clip_lo.item()), float(clip_hi.item()))
    q = torch.clamp(torch.round((clipped - offset) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale.to(dtype=torch.float32), offset.to(dtype=torch.float32)

def quantize_embedding_tensor_gptq(t: Tensor, hdiag: Tensor, qmax: int) -> tuple[Tensor, Tensor]:
    return _quantize_rowwise_gptq_symmetric(t, hdiag, qmax, EMBED_GPTQ_CLIP_STD_MULTS)

def collect_gptq_calibration(
    model: "GPT",
    train_files: str,
    train_seq_len: int,
    device: torch.device,
    calibration_batches: int,
) -> dict[str, Tensor]:
    stream = TokenStream(train_files)
    seed = stream.take(train_seq_len).to(dtype=torch.int64, device=device).unsqueeze(0)
    hdiag = torch.zeros((model.tok_emb.weight.shape[1],), dtype=torch.float32, device=device)
    linear_hdiag: dict[str, Tensor] = {}
    linear_counts: dict[str, int] = {}
    hooks = []
    for module_name, module in model.named_modules():
        if isinstance(module, CastedLinear):
            weight_name = f"{module_name}.weight"
            linear_hdiag[weight_name] = torch.zeros((module.weight.shape[1],), dtype=torch.float32, device=device)
            linear_counts[weight_name] = 0
            def _hook(_module, inputs, _output, *, weight_name: str = weight_name) -> None:
                x = inputs[0]
                flat_x = x.reshape(-1, x.shape[-1]).float()
                linear_hdiag[weight_name] += flat_x.square().sum(dim=0)
                linear_counts[weight_name] += flat_x.shape[0]
            hooks.append(module.register_forward_hook(_hook))
    total = 0
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(calibration_batches):
                hidden = model.forward_hidden(seed)
                flat_hidden = hidden.reshape(-1, hidden.size(-1)).float()
                hdiag += flat_hidden.square().sum(dim=0)
                total += flat_hidden.size(0)
                logits = model.project_logits(hidden)
                seed = logits.argmax(dim=-1)
    finally:
        for hook in hooks:
            hook.remove()
        if was_training:
            model.train()
    calibration = {"tok_emb.weight": hdiag.detach().cpu().contiguous()}
    if total > 0:
        calibration["tok_emb.weight"] = (hdiag / total).detach().cpu().contiguous()
    for name, diag in linear_hdiag.items():
        count = linear_counts[name]
        if count > 0:
            calibration[name] = (diag / count).detach().cpu().contiguous()
    return calibration


def should_quantize_matrix_int6(name: str, t: Tensor) -> bool:
    return (
        MATRIX_BITS == 6
        and
        t.ndim == 2
        and name != "tok_emb.weight"
        and t.numel() > INT8_KEEP_FLOAT_MAX_NUMEL
        and (FULL_GPTQ_INT6 or any(pattern in name for pattern in QUANT_INT6_NAME_PATTERNS))
    )

def quantize_state_dict_int8(state_dict: dict[str, Tensor], gptq_calibration: dict[str, Tensor] | None = None):
    # Single supported clean-script export format:
    # - per-row int8 for 2D float tensors
    # - per-tensor int8 for other float tensors
    # - exact passthrough for non-floats
    # - passthrough for small float tensors, stored as fp16 to save bytes
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    offsets: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_float_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
            "int6_packed_bytes",
        ),
        0,
    )
    stats["group_stats"] = make_quant_group_stats()

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        raw_bytes = tensor_nbytes(t)
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += raw_bytes

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            payload_bytes = tensor_nbytes(t)
            stats["int8_payload_bytes"] += payload_bytes
            record_quant_group_stats(stats, name, raw_bytes=raw_bytes, payload_bytes=payload_bytes)
            continue

        # Small float tensors are cheap enough to keep directly. We still downcast
        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.
        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            payload_bytes = tensor_nbytes(kept)
            stats["int8_payload_bytes"] += payload_bytes
            record_quant_group_stats(stats, name, raw_bytes=raw_bytes, payload_bytes=payload_bytes)
            continue

        stats["num_float_tensors"] += 1
        if gptq_calibration is not None and name == "tok_emb.weight" and name in gptq_calibration:
            q, s = quantize_embedding_tensor_gptq(
                t,
                gptq_calibration[name].to(dtype=torch.float32),
                EMBED_QUANT_QMAX,
            )
            qmeta[name] = {"scheme": "gptq_diag_per_row", "axis": 0}
            quantized[name] = q
            scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            payload_bytes = tensor_nbytes(q) + tensor_nbytes(s)
            stats["int8_payload_bytes"] += payload_bytes
            record_quant_group_stats(stats, name, raw_bytes=raw_bytes, payload_bytes=payload_bytes)
            continue
        if should_quantize_matrix_int6(name, t):
            hdiag = gptq_calibration.get(name) if gptq_calibration is not None else None
            q, s = _quantize_rowwise_gptq_symmetric(t, hdiag, MATRIX_QUANT_QMAX, MLP_INT6_GPTQ_CLIP_STD_MULTS)
            packed, original_numel = _pack_int6_signed(q)
            qmeta[name] = {
                "scheme": "gptq_diag_per_row_int6_packed",
                "axis": 0,
                "shape": list(t.shape),
                "original_numel": original_numel,
            }
            quantized[name] = packed
            scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            packed_bytes = tensor_nbytes(packed)
            payload_bytes = packed_bytes + tensor_nbytes(s)
            stats["int6_packed_bytes"] += packed_bytes
            stats["int8_payload_bytes"] += payload_bytes
            record_quant_group_stats(
                stats,
                name,
                raw_bytes=raw_bytes,
                payload_bytes=payload_bytes,
                int6_bytes=packed_bytes,
            )
            continue
        if t.ndim == 2:
            hdiag = gptq_calibration.get(name) if gptq_calibration is not None else None
            qmax = 127 if MATRIX_BITS >= 8 else MATRIX_QUANT_QMAX
            q, s = _quantize_rowwise_gptq_symmetric(t, hdiag, qmax, MATRIX_INT8_GPTQ_CLIP_STD_MULTS)
            qmeta[name] = {"scheme": "gptq_diag_per_row", "axis": 0}
            quantized[name] = q
            scales[name] = s
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            payload_bytes = tensor_nbytes(q) + tensor_nbytes(s)
            stats["int8_payload_bytes"] += payload_bytes
            record_quant_group_stats(stats, name, raw_bytes=raw_bytes, payload_bytes=payload_bytes)
            continue
        else:
            q, s, offset = quantize_float_tensor_sdclip(t)
            qmeta[name] = {"scheme": "sdclip_affine", "axis": 0 if t.ndim == 2 else None}
            offsets[name] = offset
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        payload_bytes = tensor_nbytes(q) + tensor_nbytes(s)
        stats["int8_payload_bytes"] += payload_bytes
        if name in offsets:
            payload_bytes += tensor_nbytes(offsets[name])
            stats["int8_payload_bytes"] += tensor_nbytes(offsets[name])
        record_quant_group_stats(stats, name, raw_bytes=raw_bytes, payload_bytes=payload_bytes)

    obj: dict[str, object] = {
        "__quant_format__": "int6_int8_sdclip_gptq_v3",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if offsets:
        obj["offsets"] = offsets
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats

def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
    qmeta = obj.get("qmeta", {})
    offsets = obj.get("offsets", {})
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    for name, q in obj["quantized"].items():
        dtype = getattr(torch, obj["dtypes"][name])
        s = obj["scales"][name]
        offset = offsets.get(name)
        scheme = qmeta.get(name, {}).get("scheme")
        if scheme == "gptq_diag_per_row":
            out[name] = (q.float() * _reshape_row_scale(s, q)).to(dtype=dtype).contiguous()
        elif scheme == "gptq_diag_per_row_int6_packed":
            shape = tuple(qmeta[name]["shape"])
            unpacked = _unpack_int6_signed(q, int(qmeta[name]["original_numel"]), shape)
            out[name] = (unpacked.float() * _reshape_row_scale(s, unpacked)).to(dtype=dtype).contiguous()
        elif s.ndim > 0:
            scale = _reshape_row_scale(s, q)
            base = q.float() * scale
            if offset is not None:
                base = base + offset.to(dtype=torch.float32).view(q.shape[0], *([1] * (q.ndim - 1)))
            out[name] = base.to(dtype=dtype).contiguous()
        else:
            scale = float(s.item())
            base = q.float() * scale
            if offset is not None:
                base = base + float(offset.item())
            out[name] = base.to(dtype=dtype).contiguous()
    for name, t in obj["passthrough"].items():
        # Restore small tensors, undoing the temporary fp16 storage cast if needed.
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


# -----------------------------
# DATA LOADING 
# -----------------------------

def load_data_shard(file: Path) -> Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    # SHARD HEADER INTS & SHARD_MAGIC
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if file.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {file}: expected {expected_size} bytes")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    if tokens_np.size != num_tokens:
        raise ValueError(f"Short read for {file}")
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


class TokenStream:
    # Reads shards sequentially and wraps around forever. The training loop therefore
    # has deterministic, simple streaming behavior with no sampling or workers.
    def __init__(self, pattern: str):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        if not self.files:
            raise FileNotFoundError(f"No files found for pattern: {pattern}")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n: int) -> Tensor:
        chunks: list[Tensor] = []
        remaining = n
        while remaining > 0:
            avail = self.tokens.numel() - self.pos
            if avail <= 0:
                self._advance_file()
                continue
            k = min(remaining, avail)
            chunks.append(self.tokens[self.pos : self.pos + k])
            self.pos += k
            remaining -= k
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


class DistributedTokenLoader:
    # Each call consumes a contiguous chunk from the shared token stream, then slices out
    # one disjoint span per rank. The extra "+1" token lets us build (x, y) by shifting.
    def __init__(self, pattern: str, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.stream = TokenStream(pattern)

    def next_batch(self, global_tokens: int, seq_len: int, grad_accum_steps: int) -> tuple[Tensor, Tensor]:
        local_tokens = global_tokens // (self.world_size * grad_accum_steps)
        per_rank_span = local_tokens + 1
        chunk = self.stream.take(per_rank_span * self.world_size)
        start = self.rank * per_rank_span
        local = chunk[start : start + per_rank_span].to(dtype=torch.int64)
        x = local[:-1].reshape(-1, seq_len)
        y = local[1:].reshape(-1, seq_len)
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# -----------------------------
# TRANSFORMER MODULES
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps: float | None = None):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)


class CastedLinear(nn.Linear):
    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.
    def forward(self, x: Tensor) -> Tensor:
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), bias)


class RandomProjAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, seed: int, gain: bool):
        super().__init__()
        g = torch.Generator(device="cpu"); g.manual_seed(seed)
        w = torch.randn((out_dim, in_dim), generator=g, dtype=torch.float32) / math.sqrt(in_dim)
        self.register_buffer("random_weight", w, persistent=False)
        self.down = CastedLinear(in_dim, rank, bias=False)
        self.up = CastedLinear(rank, out_dim, bias=False); self.up._zero_init = True
        self.gain = nn.Parameter(torch.ones(out_dim, dtype=torch.float32)) if gain else None

    def forward(self, x: Tensor) -> Tensor:
        y = F.linear(x, self.random_weight.to(dtype=x.dtype))
        if self.gain is not None:
            y = (y.reshape(-1, y.size(-1)) * self.gain.to(dtype=x.dtype)).view_as(y)
        return y + self.up(self.down(x))


def restore_low_dim_params_to_fp32(module: nn.Module) -> None:
    # Keep small/control parameters in fp32 even when the model body runs in bf16.
    with torch.no_grad():
        for name, param in module.named_parameters():
            if (param.ndim < 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
                param.data = param.data.float()


class Rotary(nn.Module):
    # Caches cos/sin tables per sequence length on the current device.
    def __init__(self, dim: int, base: float = 10000.0, train_seq_len: int = 1024, rope_dims: int = 0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        if (
            self._cos_cached is None
            or self._sin_cached is None
            or self._seq_len_cached != seq_len
            or self._cos_cached.device != device
        ):
            if seq_len > self.train_seq_len and self.rope_dims > 2:
                scale = seq_len / self.train_seq_len
                new_base = self.base * scale ** (self.rope_dims / (self.rope_dims - 2))
                inv_freq = 1.0 / (
                    new_base
                    ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32, device=device) / self.rope_dims)
                )
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int,
        rope_train_seq_len: int,
        attn_backend: str,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        if self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        kv_dim = self.num_kv_heads * self.head_dim
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, kv_dim, bias=False)
        self.c_v = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = rope_dims if rope_dims > 0 else self.head_dim
        self.rotary = Rotary(
            self.head_dim,
            base=rope_base,
            train_seq_len=rope_train_seq_len,
            rope_dims=self.rope_dims,
        )
        self.use_xsa = False
        self.attn_backend = attn_backend

    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:
        batch, seq, heads, dim = y.shape
        kv_heads = v.size(2)
        group = heads // kv_heads
        y_grouped = y.reshape(batch, seq, kv_heads, group, dim)
        v_norm = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_grouped * v_norm).sum(dim=-1, keepdim=True) * v_norm
        return (y_grouped - proj).reshape(batch, seq, heads, dim)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]
        use_flash3 = (
            self.attn_backend == "flash3"
            or (
                self.attn_backend == "auto"
                and flash_attn_3_func is not None
                and x.is_cuda
            )
        )
        if use_flash3:
            if flash_attn_3_func is None:
                raise RuntimeError(
                    "ATTN_BACKEND=flash3 requested but flash_attn_interface import failed: "
                    f"{_FLASH_ATTN_3_IMPORT_ERROR!r}"
                )
            q_fa = q.transpose(1, 2).contiguous()
            k_fa = k.transpose(1, 2).contiguous()
            v_fa = v.transpose(1, 2).contiguous()
            y = flash_attn_3_func(q_fa, k_fa, v_fa, causal=True)
            if self.use_xsa:
                y = self._xsa_efficient(y, v_fa)
            y = y.reshape(bsz, seqlen, dim)
        else:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                is_causal=True,
                enable_gqa=(self.num_kv_heads != self.num_heads),
            )
            if self.use_xsa:
                y = self._xsa_efficient(y.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    # relu^2 MLP from the original modded-nanogpt setup
    def __init__(
        self,
        dim: int,
        mlp_mult: int,
        random_proj: bool,
        random_proj_rank: int,
        random_proj_seed: int,
        random_proj_gain: bool,
        activation: str,
        leaky_slope: float,
    ):
        super().__init__()
        hidden = mlp_mult * dim
        self.fc = CastedLinear(dim, hidden, bias=False)
        self.proj = RandomProjAdapter(hidden, dim, random_proj_rank, random_proj_seed, random_proj_gain) if random_proj else CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.activation = activation
        self.leaky_slope = leaky_slope

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        if self.activation == "leaky_relu_squared":
            x = F.leaky_relu(x, negative_slope=self.leaky_slope)
        elif self.activation == "relu_squared":
            x = torch.relu(x)
        else:
            raise ValueError(f"Unsupported MLP_ACTIVATION={self.activation!r}")
        return self.proj(x.square())


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        random_mlp_proj: bool,
        random_mlp_proj_rank: int,
        random_mlp_proj_seed: int,
        random_mlp_proj_gain: bool,
        rope_base: float,
        qk_gain_init: float,
        rope_dims: int,
        rope_train_seq_len: int,
        layer_idx: int,
        ln_scale: bool,
        mlp_activation: str,
        mlp_leaky_slope: float,
        attn_backend: str,
    ):
        super().__init__()
        self.attn_norm = RMSNorm()
        self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim,
            num_heads,
            num_kv_heads,
            rope_base,
            qk_gain_init,
            rope_dims,
            rope_train_seq_len,
            attn_backend,
        )
        self.mlp = MLP(
            dim,
            mlp_mult,
            random_mlp_proj,
            random_mlp_proj_rank,
            random_mlp_proj_seed,
            random_mlp_proj_gain,
            mlp_activation,
            mlp_leaky_slope,
        )
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(
        self,
        x: Tensor,
        x0: Tensor,
        resid_mix_override: Tensor | None = None,
        attn_scale_override: Tensor | None = None,
        mlp_scale_override: Tensor | None = None,
        parallel_residual: bool = False,
        mlp_x: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        mix_src = self.resid_mix if resid_mix_override is None else resid_mix_override
        attn_scale_src = self.attn_scale if attn_scale_override is None else attn_scale_override
        mlp_scale_src = self.mlp_scale if mlp_scale_override is None else mlp_scale_override
        mix = mix_src.to(dtype=x.dtype)
        attn_scale = attn_scale_src.to(dtype=x.dtype)[None, None, :]
        mlp_scale = mlp_scale_src.to(dtype=x.dtype)[None, None, :]
        if parallel_residual:
            mlp_x = x if mlp_x is None else mlp_x
            attn_lane = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
            mlp_lane = mlp_x + mlp_scale * self.mlp(self.mlp_norm(mlp_x) * self.ln_scale_factor)
            attn_lane = attn_lane + attn_scale * self.attn(self.attn_norm(attn_lane) * self.ln_scale_factor)
            return attn_lane, mlp_lane
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x) * self.ln_scale_factor)
        x = x + attn_scale * attn_out
        x = x + mlp_scale * self.mlp(self.mlp_norm(x) * self.ln_scale_factor)
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        physical_layers: int,
        loop_layers: tuple[int, ...],
        num_loops: int,
        loop_start: int,
        loop_end: int,
        parallel_start_layer: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        random_mlp_proj: bool,
        random_mlp_proj_rank: int,
        random_mlp_proj_seed: int,
        random_mlp_proj_gain: bool,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        depth_embedding: bool,
        virtual_layer_scales: bool,
        logit_softcap: float,
        rope_base: float,
        rope_dims: int,
        rope_train_seq_len: int,
        qk_gain_init: float,
        ln_scale: bool,
        xsa_last_n: int,
        skip_gates_enabled: bool,
        mlp_activation: str,
        mlp_leaky_slope: float,
        attn_backend: str,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        if physical_layers <= 0:
            raise ValueError(f"physical_layers must be positive, got {physical_layers}")
        if physical_layers > num_layers:
            raise ValueError(
                f"physical_layers must be <= num_layers, got physical_layers={physical_layers} num_layers={num_layers}"
            )
        if parallel_start_layer < 0 or parallel_start_layer > num_layers:
            raise ValueError(
                f"parallel_start_layer must be within [0, {num_layers}], got {parallel_start_layer}"
            )
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_layers = num_layers
        self.physical_layers = physical_layers
        self.loop_layers = loop_layers
        self.num_loops = num_loops
        self.loop_start = loop_start
        self.loop_end = loop_end
        self.parallel_start_layer = parallel_start_layer
        self.parallel_residual_enabled = parallel_start_layer < num_layers
        self.depth_embedding_enabled = depth_embedding
        self.virtual_layer_scales_enabled = virtual_layer_scales
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        if self.parallel_residual_enabled:
            self.lane_merge = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        else:
            self.register_parameter("lane_merge", None)
        if self.num_loops > 0 and loop_layers:
            raise ValueError("NUM_LOOPS/LOOP_START/LOOP_END cannot be combined with LOOP_LAYERS")
        base_encoder_block_ids, base_decoder_block_ids = self._build_block_schedule(
            num_layers,
            physical_layers,
            loop_layers,
            self.num_encoder_layers,
            self.num_decoder_layers,
        )
        if self.num_loops > 0:
            loop_encoder_block_ids, loop_decoder_block_ids = self._build_recurrent_schedule(
                physical_layers,
                self.num_loops,
                self.loop_start,
                self.loop_end,
            )
        else:
            loop_encoder_block_ids, loop_decoder_block_ids = base_encoder_block_ids, base_decoder_block_ids
        self.encoder_block_ids = tuple(base_encoder_block_ids)
        self.decoder_block_ids = tuple(base_decoder_block_ids)
        self.loop_encoder_block_ids = tuple(loop_encoder_block_ids)
        self.loop_decoder_block_ids = tuple(loop_decoder_block_ids)
        self.looping_active = False
        self.num_skip_weights = min(len(self.loop_encoder_block_ids), len(self.loop_decoder_block_ids))
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        if skip_gates_enabled:
            self.skip_gates = nn.Parameter(torch.zeros(self.num_skip_weights, model_dim, dtype=torch.float32))
        else:
            self.register_parameter("skip_gates", None)
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    random_mlp_proj,
                    random_mlp_proj_rank,
                    random_mlp_proj_seed,
                    random_mlp_proj_gain,
                    rope_base,
                    qk_gain_init,
                    rope_dims,
                    rope_train_seq_len,
                    i,
                    ln_scale,
                    mlp_activation,
                    mlp_leaky_slope,
                    attn_backend,
                )
                for i in range(physical_layers)
            ]
        )
        if xsa_last_n > 0:
            for i in range(max(0, physical_layers - xsa_last_n), physical_layers):
                self.blocks[i].attn.use_xsa = True
        if depth_embedding:
            self.depth_embed = nn.Parameter(torch.zeros(num_layers, model_dim, dtype=torch.float32))
        else:
            self.register_parameter("depth_embed", None)
        if virtual_layer_scales:
            self.virtual_attn_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
            self.virtual_mlp_scales = nn.Parameter(torch.ones(num_layers, model_dim, dtype=torch.float32))
            self.virtual_resid_mixes = nn.Parameter(
                torch.stack(
                    (
                        torch.ones((num_layers, model_dim), dtype=torch.float32),
                        torch.zeros((num_layers, model_dim), dtype=torch.float32),
                    ),
                    dim=1,
                )
            )
        else:
            self.register_parameter("virtual_attn_scales", None)
            self.register_parameter("virtual_mlp_scales", None)
            self.register_parameter("virtual_resid_mixes", None)
        self.final_norm = RMSNorm()
        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    @staticmethod
    def _build_block_schedule(
        num_layers: int,
        physical_layers: int,
        loop_layers: tuple[int, ...],
        num_encoder_layers: int,
        num_decoder_layers: int,
    ) -> tuple[list[int], list[int]]:
        if loop_layers:
            if len(set(loop_layers)) != len(loop_layers):
                raise ValueError(f"loop_layers must be unique, got {list(loop_layers)}")
            expected_loop_layers = tuple(range(loop_layers[0], loop_layers[-1] + 1))
            if loop_layers != expected_loop_layers:
                raise ValueError(
                    f"loop_layers must be a contiguous increasing range, got {list(loop_layers)}"
                )
            if loop_layers[0] < 0 or loop_layers[-1] >= physical_layers:
                raise ValueError(
                    f"loop_layers must be within [0, {physical_layers - 1}], got {list(loop_layers)}"
                )
            full_schedule: list[int] = []
            for physical_idx in range(physical_layers):
                full_schedule.append(physical_idx)
                if physical_idx == loop_layers[-1]:
                    full_schedule.extend(loop_layers)
            expected_num_layers = physical_layers + len(loop_layers)
            if num_layers != expected_num_layers:
                raise ValueError(
                    "num_layers must equal physical_layers + len(loop_layers) when LOOP_LAYERS is set, "
                    f"got num_layers={num_layers} physical_layers={physical_layers} "
                    f"loop_layers={list(loop_layers)}"
                )
            return (
                full_schedule[:num_encoder_layers],
                full_schedule[num_encoder_layers:],
            )
        if physical_layers == num_layers:
            return (
                list(range(num_encoder_layers)),
                list(range(num_encoder_layers, num_layers)),
            )
        return (
            [i % physical_layers for i in range(num_encoder_layers)],
            [physical_layers - 1 - (i % physical_layers) for i in range(num_decoder_layers)],
        )

    @staticmethod
    def _build_recurrent_schedule(
        physical_layers: int,
        num_loops: int,
        loop_start: int,
        loop_end: int,
    ) -> tuple[list[int], list[int]]:
        if loop_start < 0 or loop_end < loop_start or loop_end >= physical_layers:
            raise ValueError(
                f"loop range must satisfy 0 <= LOOP_START <= LOOP_END < {physical_layers}, "
                f"got LOOP_START={loop_start} LOOP_END={loop_end}"
            )
        if num_loops <= 0:
            raise ValueError(f"NUM_LOOPS must be positive when using recurrent schedule, got {num_loops}")
        loop_segment = list(range(loop_start, loop_end + 1))
        full_schedule = list(range(loop_start))
        for _ in range(num_loops + 1):
            full_schedule.extend(loop_segment)
        full_schedule.extend(range(loop_end + 1, physical_layers))
        num_encoder = len(full_schedule) // 2
        return full_schedule[:num_encoder], full_schedule[num_encoder:]

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _run_virtual_block(
        self,
        x: Tensor,
        x0: Tensor,
        virtual_idx: int,
        physical_idx: int,
        parallel_residual: bool = False,
        mlp_x: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        block = self.blocks[physical_idx]
        if self.depth_embed is not None:
            x = x + self.depth_embed[virtual_idx].to(dtype=x.dtype)[None, None, :]
            if mlp_x is not None:
                mlp_x = mlp_x + self.depth_embed[virtual_idx].to(dtype=mlp_x.dtype)[None, None, :]
        resid_mix = None
        if self.virtual_resid_mixes is not None:
            resid_mix = block.resid_mix * self.virtual_resid_mixes[virtual_idx]
        attn_scale = None
        if self.virtual_attn_scales is not None:
            attn_scale = block.attn_scale * self.virtual_attn_scales[virtual_idx]
        mlp_scale = None
        if self.virtual_mlp_scales is not None:
            mlp_scale = block.mlp_scale * self.virtual_mlp_scales[virtual_idx]
        return block(
            x,
            x0,
            resid_mix_override=resid_mix,
            attn_scale_override=attn_scale,
            mlp_scale_override=mlp_scale,
            parallel_residual=parallel_residual,
            mlp_x=mlp_x,
        )

    def forward_hidden(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        attn_lane: Tensor | None = None
        mlp_lane: Tensor | None = None
        encoder_block_ids = self.loop_encoder_block_ids if self.looping_active else self.encoder_block_ids
        decoder_block_ids = self.loop_decoder_block_ids if self.looping_active else self.decoder_block_ids

        # First half stores skips; second half reuses them in reverse order.
        for virtual_idx, physical_idx in enumerate(encoder_block_ids):
            if virtual_idx >= self.parallel_start_layer:
                if attn_lane is None or mlp_lane is None:
                    attn_lane = x
                    mlp_lane = x
                attn_lane, mlp_lane = self._run_virtual_block(
                    attn_lane,
                    x0,
                    virtual_idx,
                    physical_idx,
                    parallel_residual=True,
                    mlp_x=mlp_lane,
                )
                x = 0.5 * (attn_lane + mlp_lane)
            else:
                x = self._run_virtual_block(x, x0, virtual_idx, physical_idx)
            skips.append(x)
        for i, physical_idx in enumerate(decoder_block_ids):
            virtual_idx = len(encoder_block_ids) + i
            if skips:
                skip = self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
                if attn_lane is not None and mlp_lane is not None and virtual_idx >= self.parallel_start_layer:
                    if self.skip_gates is not None:
                        gate = torch.sigmoid(self.skip_gates[i].to(dtype=attn_lane.dtype))[None, None, :]
                        attn_lane = torch.lerp(skip, attn_lane, gate)
                    else:
                        attn_lane = attn_lane + skip
                    x = 0.5 * (attn_lane + mlp_lane)
                else:
                    if self.skip_gates is not None:
                        gate = torch.sigmoid(self.skip_gates[i].to(dtype=x.dtype))[None, None, :]
                        x = torch.lerp(skip, x, gate)
                    else:
                        x = x + skip
            if virtual_idx >= self.parallel_start_layer:
                if attn_lane is None or mlp_lane is None:
                    attn_lane = x
                    mlp_lane = x
                attn_lane, mlp_lane = self._run_virtual_block(
                    attn_lane,
                    x0,
                    virtual_idx,
                    physical_idx,
                    parallel_residual=True,
                    mlp_x=mlp_lane,
                )
                x = 0.5 * (attn_lane + mlp_lane)
            else:
                x = self._run_virtual_block(
                    x,
                    x0,
                    virtual_idx,
                    physical_idx,
                )

        if attn_lane is not None and mlp_lane is not None:
            lane_merge = self.lane_merge.to(dtype=x.dtype)
            x = lane_merge * attn_lane + (1.0 - lane_merge) * mlp_lane
        return self.final_norm(x)

    def project_logits(self, hidden: Tensor) -> Tensor:
        flat_hidden = hidden.reshape(-1, hidden.size(-1))
        if self.tie_embeddings:
            logits_proj = F.linear(flat_hidden, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(flat_hidden)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits.view(*hidden.shape[:-1], logits.size(-1))

    def forward(self, input_ids: Tensor, target_ids: Tensor, reduction: str = "mean") -> Tensor:
        hidden = self.forward_hidden(input_ids)
        logits = self.project_logits(hidden).reshape(-1, self.tok_emb.num_embeddings)
        targets = target_ids.reshape(-1)
        losses = F.cross_entropy(logits.float(), targets, reduction="none")
        if reduction == "mean":
            return losses.mean()
        if reduction == "none":
            return losses.view_as(target_ids)
        raise ValueError(f"Unsupported reduction={reduction!r}")


# -----------------------------
# TRAINING
# -----------------------------

def main() -> None:
    global zeropower_via_newtonschulz5

    code = Path(__file__).read_text(encoding="utf-8")
    args = Hyperparameters()
    if args.eval_mode not in {"standard", "sliding"}:
        raise ValueError(f"EVAL_MODE must be 'standard' or 'sliding', got {args.eval_mode!r}")
    if args.attn_backend not in {"auto", "sdpa", "flash3"}:
        raise ValueError(f"ATTN_BACKEND must be one of auto|sdpa|flash3, got {args.attn_backend!r}")
    if args.embed_bits < 2 or args.embed_bits > 8:
        raise ValueError(f"EMBED_BITS must be within [2, 8], got {args.embed_bits}")
    if args.matrix_bits not in {6, 8}:
        raise ValueError(f"MATRIX_BITS must currently be 6 or 8, got {args.matrix_bits}")
    zeropower_via_newtonschulz5 = torch.compile(zeropower_via_newtonschulz5)

    # -----------------------------
    # DISTRIBUTED + CUDA SETUP
    # -----------------------------

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size <= 0:
        raise ValueError(f"WORLD_SIZE must be positive, got {world_size}")
    if 8 % world_size != 0:
        raise ValueError(f"WORLD_SIZE={world_size} must divide 8 so grad_accum_steps stays integral")
    grad_accum_steps = 8 // world_size
    grad_scale = 1.0 / grad_accum_steps
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    # Fast math knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    logfile = None
    if master_process:
        os.makedirs("logs", exist_ok=True)
        logfile = f"logs/{args.run_id}.txt"
        print(logfile)

    def log0(msg: str, console: bool = True) -> None:
        if not master_process:
            return
        if console:
            print(msg)
        if logfile is not None:
            with open(logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

    log0(code, console=False)
    log0("=" * 100, console=False)
    log0(f"Running Python {sys.version}", console=False)
    log0(f"Running PyTorch {torch.__version__}", console=False)
    log0(
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False).stdout,
        console=False,
    )
    log0("=" * 100, console=False)

    # -----------------------------
    # TOKENIZER + VALIDATION METRIC SETUP
    # -----------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Script only setup for SentencePiece .model file: {args.tokenizer_path}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    dataset_dir = Path(args.data_path).resolve()
    actual_train_files = len(list(dataset_dir.glob("fineweb_train_*.bin")))
    val_tokens = load_validation_tokens(args.val_files)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(
        sp, args.vocab_size, device
    )
    log0(f"val_bpb:enabled tokenizer_kind=sentencepiece tokenizer_path={args.tokenizer_path}")
    log0(f"train_loader:dataset:{dataset_dir.name} train_shards:{actual_train_files}")
    log0(f"val_loader:shards pattern={args.val_files} tokens:{val_tokens.numel() - 1}")
    log0(
        f"eval_config:mode:{args.eval_mode} context_len:{args.eval_context_len} "
        f"stride:{args.eval_sliding_stride if args.eval_mode == 'sliding' else 0}"
    )

    # -----------------------------
    # MODEL + OPTIMIZER SETUP
    # -----------------------------

    base_model = GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        physical_layers=args.physical_layers,
        loop_layers=args.loop_layers,
        num_loops=args.num_loops,
        loop_start=args.loop_start,
        loop_end=args.loop_end,
        parallel_start_layer=args.parallel_start_layer,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        random_mlp_proj=args.random_mlp_proj,
        random_mlp_proj_rank=args.random_mlp_proj_rank,
        random_mlp_proj_seed=args.random_mlp_proj_seed,
        random_mlp_proj_gain=args.random_mlp_proj_gain,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        depth_embedding=args.depth_embedding,
        virtual_layer_scales=args.virtual_layer_scales,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        rope_dims=args.rope_dims,
        rope_train_seq_len=args.rope_train_seq_len,
        qk_gain_init=args.qk_gain_init,
        ln_scale=args.ln_scale,
        xsa_last_n=args.xsa_last_n,
        skip_gates_enabled=args.skip_gates_enabled,
        mlp_activation=args.mlp_activation,
        mlp_leaky_slope=args.mlp_leaky_slope,
        attn_backend=args.attn_backend,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    restore_low_dim_params_to_fp32(base_model)
    compiled_model = base_model if args.random_mlp_proj else torch.compile(base_model, dynamic=False, fullgraph=True)
    model: nn.Module = DDP(compiled_model, device_ids=[local_rank], broadcast_buffers=False) if distributed else compiled_model

    # Optimizer split:
    # - token embedding (Adam) uses EMBED_LR
    # - untied lm_head (Adam) uses HEAD_LR
    # - matrix params in transformer blocks use MATRIX_LR via Muon
    # - vectors/scalars use SCALAR_LR via Adam
    matrix_params = [
        p
        for name, p in base_model.named_parameters()
        if name not in {"tok_emb.weight", "lm_head.weight"}
        and p.ndim == 2
        and not any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
    ]
    scalar_params = [
        p
        for name, p in base_model.named_parameters()
        if name not in {"tok_emb.weight", "lm_head.weight"}
        and (p.ndim != 2 or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS))
    ]
    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr
    optimizer_tok = torch.optim.AdamW(
        [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.embed_wd,
        fused=True,
    )
    optimizer_muon = Muon(
        matrix_params,
        lr=args.matrix_lr,
        momentum=args.muon_momentum,
        backend_steps=args.muon_backend_steps,
        weight_decay=args.muon_wd,
    )
    for group in optimizer_muon.param_groups:
        group["base_lr"] = args.matrix_lr
    optimizer_scalar = torch.optim.AdamW(
        [{"params": scalar_params, "lr": args.scalar_lr, "base_lr": args.scalar_lr}],
        betas=(args.beta1, args.beta2),
        eps=args.adam_eps,
        weight_decay=args.scalar_wd,
        fused=True,
    )
    optimizers: list[torch.optim.Optimizer] = [optimizer_tok, optimizer_muon, optimizer_scalar]
    if base_model.lm_head is not None:
        optimizer_head = torch.optim.AdamW(
            [{"params": [base_model.lm_head.weight], "lr": args.head_lr, "base_lr": args.head_lr}],
            betas=(args.beta1, args.beta2),
            eps=args.adam_eps,
            weight_decay=args.head_wd,
            fused=True,
        )
        optimizers.insert(1, optimizer_head)

    n_params = sum(p.numel() for p in base_model.parameters())
    log0(f"model_params:{n_params}")
    log0(
        f"virtual_layers:{args.num_layers} physical_layers:{args.physical_layers} "
        f"loop_layers:{list(args.loop_layers)} num_loops:{args.num_loops} "
        f"loop_start:{args.loop_start} loop_end:{args.loop_end} "
        f"parallel_start_layer:{args.parallel_start_layer} "
        f"depth_embedding:{int(args.depth_embedding)} "
        f"virtual_layer_scales:{int(args.virtual_layer_scales)} "
        f"rope_dims:{args.rope_dims if args.rope_dims > 0 else base_model.blocks[0].attn.head_dim} "
        f"ln_scale:{int(args.ln_scale)} xsa_last_n:{args.xsa_last_n} "
        f"skip_gates:{int(args.skip_gates_enabled)} mlp_activation:{args.mlp_activation} "
        f"attn_backend:{args.attn_backend}"
    )
    log0(f"encoder_block_ids:{list(base_model.encoder_block_ids)} decoder_block_ids:{list(base_model.decoder_block_ids)}")
    log0(f"world_size:{world_size} grad_accum_steps:{grad_accum_steps}")
    log0("sdp_backends:cudnn=False flash=True mem_efficient=False math=False")
    log0(f"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}")
    log0(
        f"tie_embeddings:{args.tie_embeddings} embed_lr:{token_lr} "
        f"head_lr:{args.head_lr if base_model.lm_head is not None else 0.0} "
        f"matrix_lr:{args.matrix_lr} scalar_lr:{args.scalar_lr} "
        f"embed_wd:{args.embed_wd} head_wd:{args.head_wd if base_model.lm_head is not None else 0.0} "
        f"scalar_wd:{args.scalar_wd} muon_wd:{args.muon_wd} ema_decay:{args.ema_decay}"
    )
    log0(
        f"train_batch_tokens:{args.train_batch_tokens} train_seq_len:{args.train_seq_len} "
        f"iterations:{args.iterations} warmup_steps:{args.warmup_steps} "
        f"max_wallclock_seconds:{args.max_wallclock_seconds:.3f} "
        f"warmdown_iters:{args.warmdown_iters} warmdown_frac:{args.warmdown_frac if args.warmdown_frac is not None else -1.0:.3f} "
        f"min_lr:{args.min_lr:.3f} enable_looping_at:{args.enable_looping_at:.3f} "
        f"gptq_reserve_seconds:{args.gptq_reserve_seconds:.1f} "
        f"gptq_calibration_batches:{args.gptq_calibration_batches} compressor:{args.compressor} "
        f"embed_bits:{args.embed_bits} matrix_bits:{args.matrix_bits}"
    )
    log0(f"seed:{args.seed}")

    # -----------------------------
    # DATA LOADER & MODEL WARMUP
    # -----------------------------

    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    def zero_grad_all() -> None:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

    max_wallclock_ms = 1000.0 * args.max_wallclock_seconds if args.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None and args.gptq_reserve_seconds > 0:
        max_wallclock_ms = max(max_wallclock_ms - 1000.0 * args.gptq_reserve_seconds, 0.0)

    def training_frac(step: int, elapsed_ms: float) -> float:
        if max_wallclock_ms is None:
            return step / max(args.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-9)

    def lr_mul(step: int, elapsed_ms: float) -> float:
        if args.warmdown_frac is not None:
            if args.warmdown_frac <= 0:
                return 1.0
            frac = training_frac(step, elapsed_ms)
            if frac >= 1.0 - args.warmdown_frac:
                return max((1.0 - frac) / args.warmdown_frac, args.min_lr)
            return 1.0
        if args.warmdown_iters <= 0:
            return 1.0
        if max_wallclock_ms is None:
            warmdown_start = max(args.iterations - args.warmdown_iters, 0)
            return max((args.iterations - step) / max(args.warmdown_iters, 1), 0.0) if warmdown_start <= step < args.iterations else 1.0
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = args.warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        return max(remaining_ms / max(warmdown_ms, 1e-9), args.min_lr) if remaining_ms <= warmdown_ms else 1.0

    # Warmup primes the compiled forward/backward/optimizer paths, then we restore the
    # initial weights/optimizer state so measured training starts from the true init.
    if args.warmup_steps > 0:
        initial_model_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
        initial_optimizer_states = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for warmup_step in range(args.warmup_steps):
            zero_grad_all()
            for micro_step in range(grad_accum_steps):
                if distributed:
                    model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                    warmup_loss = model(x, y)
                (warmup_loss * grad_scale).backward()
            for opt in optimizers:
                opt.step()
            zero_grad_all()
            if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                log0(f"warmup_step:{warmup_step + 1}/{args.warmup_steps}")
        base_model.load_state_dict(initial_model_state, strict=True)
        for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
            opt.load_state_dict(state)
        zero_grad_all()
        if distributed:
            model.require_backward_grad_sync = True
        train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)
        if args.num_loops > 0:
            base_model.looping_active = True
            for warmup_step in range(args.warmup_steps):
                zero_grad_all()
                for micro_step in range(grad_accum_steps):
                    if distributed:
                        model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
                    x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                        warmup_loss = model(x, y)
                    (warmup_loss * grad_scale).backward()
                for opt in optimizers:
                    opt.step()
                zero_grad_all()
                if args.warmup_steps <= 20 or (warmup_step + 1) % 10 == 0 or warmup_step + 1 == args.warmup_steps:
                    log0(f"loop_warmup_step:{warmup_step + 1}/{args.warmup_steps}")
            base_model.load_state_dict(initial_model_state, strict=True)
            for opt, state in zip(optimizers, initial_optimizer_states, strict=True):
                opt.load_state_dict(state)
            zero_grad_all()
            base_model.looping_active = False
            if distributed:
                model.require_backward_grad_sync = True
            train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)

    # -----------------------------
    # MAIN TRAINING LOOP
    # -----------------------------

    ema_state = None
    if args.ema_decay > 0.0:
        ema_state = {
            name: tensor.detach().float().clone()
            for name, tensor in base_model.state_dict().items()
        }

    training_time_ms = 0.0
    stop_after_step: int | None = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    step = 0
    while True:
        last_step = step == args.iterations or (stop_after_step is not None and step >= stop_after_step)

        should_validate = last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)
        if should_validate:
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            log0(
                f"step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} "
                f"eval_mode:{args.eval_mode} train_time:{training_time_ms:.0f}ms "
                f"step_avg:{training_time_ms / max(step, 1):.2f}ms"
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()

        if last_step:
            if stop_after_step is not None and step < args.iterations:
                log0(
                    f"stopping_early: wallclock_cap train_time:{training_time_ms:.0f}ms "
                    f"step:{step}/{args.iterations}"
                )
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        scale = lr_mul(step, elapsed_ms)
        if args.num_loops > 0 and not base_model.looping_active and training_frac(step, elapsed_ms) >= args.enable_looping_at:
            base_model.looping_active = True
            log0(
                f"layer_loop:enabled step:{step} frac:{training_frac(step, elapsed_ms):.3f} "
                f"encoder_block_ids:{list(base_model.loop_encoder_block_ids)} "
                f"decoder_block_ids:{list(base_model.loop_decoder_block_ids)}"
            )
        zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(grad_accum_steps):
            if distributed:
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss * grad_scale).backward()
        train_loss /= grad_accum_steps

        frac = min(step / args.muon_momentum_warmup_steps, 1.0) if args.muon_momentum_warmup_steps > 0 else 1.0
        muon_momentum = (1 - frac) * args.muon_momentum_warmup_start + frac * args.muon_momentum
        for group in optimizer_muon.param_groups:
            group["momentum"] = muon_momentum

        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * scale

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)
        for opt in optimizers:
            opt.step()
        zero_grad_all()
        if ema_state is not None:
            with torch.no_grad():
                for name, tensor in base_model.state_dict().items():
                    ema_state[name].mul_(args.ema_decay).add_(tensor.detach().float(), alpha=1.0 - args.ema_decay)

        step += 1
        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        should_log_train = (
            args.train_log_every > 0
            and (step <= 10 or step % args.train_log_every == 0 or stop_after_step is not None)
        )
        if should_log_train:
            log0(
                f"step:{step}/{args.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms / step:.2f}ms"
            )

        # Needed to sync whether we've reached the wallclock cap.
        reached_cap = max_wallclock_ms is not None and approx_training_time_ms >= max_wallclock_ms
        if distributed and max_wallclock_ms is not None:
            reached_cap_tensor = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(reached_cap_tensor, op=dist.ReduceOp.MAX)
            reached_cap = bool(reached_cap_tensor.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log0(
        f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
        f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB"
    )
    if ema_state is not None:
        log0("ema:applying EMA weights")
        current_state = base_model.state_dict()
        ema_state_cast = {name: tensor.to(dtype=current_state[name].dtype) for name, tensor in ema_state.items()}
        base_model.load_state_dict(ema_state_cast, strict=True)

    # -----------------------------
    # SERIALIZATION + ROUNDTRIP VALIDATION
    # -----------------------------
    # Save the raw state (useful for debugging/loading in PyTorch directly), then always produce
    # the compressed quantized artifact and validate the round-tripped weights.

    if master_process:
        torch.save(base_model.state_dict(), "final_model.pt")
        model_bytes = os.path.getsize("final_model.pt")
        code_bytes = len(code.encode("utf-8"))
        log0(f"Serialized model: {model_bytes} bytes")
        log0(f"Code size: {code_bytes} bytes")
        log0(f"Total submission size: {model_bytes + code_bytes} bytes")

    if master_process:
        gptq_calibration = collect_gptq_calibration(
            base_model,
            args.train_files,
            args.train_seq_len,
            device,
            args.gptq_calibration_batches,
        )
        quant_obj, quant_stats = quantize_state_dict_int8(
            base_model.state_dict(),
            gptq_calibration=gptq_calibration,
        )
        quant_buf = io.BytesIO()
        torch.save(quant_obj, quant_buf)
        quant_raw = quant_buf.getvalue()
        quant_bytes_for_compressor = (
            _byte_shuffle_bytes(quant_raw, BYTE_SHUFFLE_GROUP_SIZE)
            if QUANT_BYTE_SHUFFLE
            else quant_raw
        )
        quant_blob = _compress_quant_bytes(quant_bytes_for_compressor, args.compressor)
        quant_raw_bytes = len(quant_raw)
        with open("final_model.int8.ptz", "wb") as f:
            f.write(quant_blob)
        quant_file_bytes = os.path.getsize("final_model.int8.ptz")
        code_bytes = len(code.encode("utf-8"))
        ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
        log0(
            f"Serialized model int8+{args.compressor}: {quant_file_bytes} bytes "
            f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} "
            f"payload_ratio:{ratio:.2f}x int6_bytes:{quant_stats['int6_packed_bytes']} "
            f"byte_shuffle:{int(QUANT_BYTE_SHUFFLE)})"
        )
        log0(f"Total submission size int8+{args.compressor}: {quant_file_bytes + code_bytes} bytes")
        for group_name, group_stats in quant_stats["group_stats"].items():
            if group_stats["num_tensors"] <= 0:
                continue
            group_ratio = group_stats["raw_bytes"] / max(group_stats["payload_bytes"], 1)
            log0(
                f"quant_group:{group_name} tensors:{group_stats['num_tensors']} "
                f"raw_bytes:{group_stats['raw_bytes']} payload_bytes:{group_stats['payload_bytes']} "
                f"int6_bytes:{group_stats['int6_bytes']} payload_ratio:{group_ratio:.2f}x"
            )

    if distributed:
        dist.barrier()
    with open("final_model.int8.ptz", "rb") as f:
        quant_blob_disk = f.read()
    quant_raw_disk = _decompress_quant_bytes(quant_blob_disk, args.compressor)
    if QUANT_BYTE_SHUFFLE:
        quant_raw_disk = _byte_unshuffle_bytes(quant_raw_disk, BYTE_SHUFFLE_GROUP_SIZE)
    quant_state = torch.load(io.BytesIO(quant_raw_disk), map_location="cpu")
    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    torch.cuda.synchronize()
    t_qeval = time.perf_counter()
    q_val_loss, q_val_bpb = eval_val(
        args,
        model,
        rank,
        world_size,
        device,
        grad_accum_steps,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    log0(
        f"final_int8_brotli_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} "
        f"eval_mode:{args.eval_mode} eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms"
    )
    log0(
        f"final_int8_brotli_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f} "
        f"eval_mode:{args.eval_mode}"
    )

    # Phase 7: Score-first Test-Time Training (TTT) evaluation
    if args.final_ttt_eval:
        torch.cuda.synchronize()
        t_ttt = time.perf_counter()
        ttt_val_loss, ttt_val_bpb = eval_val_score_first_ttt(
            args,
            base_model,
            rank,
            world_size,
            device,
            val_tokens,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        torch.cuda.synchronize()
        log0(
            f"final_score_first_ttt val_loss:{ttt_val_loss:.4f} val_bpb:{ttt_val_bpb:.4f} "
            f"ttt_epochs:{args.ttt_epochs} ttt_lr:{args.ttt_lr} ttt_freeze_blocks:{args.ttt_freeze_blocks} "
            f"ttt_chunk_tokens:{args.ttt_chunk_tokens} eval_time:{1000.0 * (time.perf_counter() - t_ttt):.0f}ms"
        )
        log0(
            f"final_score_first_ttt_exact val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f}"
        )

        # After TTT, run a final frozen sliding-window eval pass with adapted weights
        if args.eval_mode == "sliding":
            torch.cuda.synchronize()
            t_frozen = time.perf_counter()
            frozen_val_loss, frozen_val_bpb = eval_val_sliding(
                args,
                model,
                rank,
                world_size,
                device,
                grad_accum_steps,
                val_tokens,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
            )
            torch.cuda.synchronize()
            log0(
                f"final_frozen_sliding_after_ttt val_loss:{frozen_val_loss:.4f} val_bpb:{frozen_val_bpb:.4f} "
                f"eval_time:{1000.0 * (time.perf_counter() - t_frozen):.0f}ms"
            )
            log0(
                f"final_frozen_sliding_after_ttt_exact val_loss:{frozen_val_loss:.8f} val_bpb:{frozen_val_bpb:.8f}"
            )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
