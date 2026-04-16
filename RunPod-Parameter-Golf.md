March 25th, 2026

# OpenAI's Parameter Golf: Train the Best Language Model That Fits in 16MB on Runpod

## What is Parameter Golf?

Parameter Golf is OpenAI's Model Craft Challenge: train the best language model you can that fits inside a 16MB artifact and trains in under 10 minutes on 8×H100 GPUs on Runpod. The metric is simple: lowest bits per byte (BPB) on the FineWeb validation set wins. No tricks with tokenizer scoring; it's tokenizer-agnostic compression performance, full stop.

If you're familiar with neural scaling laws, the challenge is essentially L(N) optimization: push validation loss as low as possible given a fixed parameter budget, with no constraints on architecture, data volume, or training steps (beyond the 10-minute wall clock). It's the parameter-constrained cousin of the [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt) (which optimizes training time) and the [NanoGPT Slowrun](https://github.com/qlabs-eng/slowrun) (which constrains dataset size).

The challenge runs from **March 18 to April 30, 2026**, and OpenAI is backing it with **$1M in compute credits** to help participants get started. The leaderboard already has 18 record submissions, with the top score dropping from the naive baseline of 1.2244 BPB down to 1.1228 BPB in just five days.

## Why this matters

Parameter Golf isn't just a leaderboard game. It's pushing the community toward ideas that matter for the future of efficient AI: depth recurrence, aggressive parameter tying, quantization-aware training, novel tokenizers, test-time compute, and low-rank factorizations. The constraint forces creativity in a way that "throw more H100s at it" never does. The search for AI excellence is over breadth (more GPU compute) and depth (more efficient, targeted use of that compute.) This contest focuses on digging as deep as possible to eke out performance on a small GPU.

For OpenAI, it's also a talent pipeline. The challenge is designed in the spirit of competitive mathematics and programming olympiads, and strong participants may catch the attention of OpenAI's research team for early-career hiring.

## How scoring works

The 16MB cap is measured as **code bytes plus compressed model bytes**. All of your training code lives in a single `train_gpt.py` script, and the model is int8-quantized and zlib-compressed after training. No external downloads, network calls, or dataset access are allowed during evaluation. The artifact must be fully self-contained.

Validation performance is measured in **bits per byte (BPB)** on the FineWeb validation set, which is a fixed first-50k-document slice. Because the metric is BPB rather than cross-entropy over a fixed vocabulary, changing your tokenizer doesn't give you a free scoring advantage; the metric normalizes for it.

New leaderboard records must beat the current SOTA by at least 0.005 nats, with enough run logs to demonstrate statistical significance at p < 0.01. Typically, averaging over three runs is sufficient.

## Getting started on Runpod

OpenAI is partnering directly with Runpod to make the setup as frictionless as possible. There's even a [pre-built Runpod template](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th) with all Python dependencies pre-installed.

### Step 1: Launch a GPU pod

[Create a Runpod account](https://console.runpod.io/deploy) if you don't already have one, and set up an SSH key in Settings so you can connect to your pod (or just use Jupyter Notebook if you prefer.). Then deploy a GPU Cloud Pod using the official Parameter Golf template linked above.

Start with a **1×H100** for development and experimentation. An 8×H100 SXM box (required for final leaderboard submissions) costs roughly $20/hr, so you'll want to validate your ideas on cheaper hardware first. A single H100 or even an A100 works well for iteration.

### Step 2: Clone and download data

Once your pod is running and you've SSH'd in:

```javascript
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

Download the cached FineWeb dataset with the 1024-token BPE vocabulary:

```javascript
python3 data/cached_challenge_fineweb.py --variant sp1024
```

This pulls the full validation split plus 80 training shards (around 8B tokens). For faster iteration, use `--train-shards 1` to grab just a single shard.

### Step 3: Run the baseline

Launch a training run on your single GPU:

```javascript
RUN_ID=baseline_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

The script enforces a 10-minute wall clock by default. When the run finishes, it prints `val_loss`, `val_bpb`, and the compressed model size. The baseline config — 9 layers, 512 hidden dim, 1024 vocab, tied embeddings, 4 KV heads — should land around **1.2244 BPB** with a compressed artifact well under 16MB.

To enable periodic validation logging during training, set `VAL_LOSS_EVERY=200`. To remove the wall clock cap for unlimited experimentation, set `MAX_WALLCLOCK_SECONDS=0`.

### Step 4: Scale to 8×H100

When you're ready to submit, spin up an 8×H100 SXM pod on Runpod using the same template, and change the launch command:

```javascript
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Everything else stays the same. The training script handles distributed data loading and gradient synchronization internally.

## Running your first experiments

Here are a few practical experiments to run on a single-GPU Runpod pod as you get your bearings.

### Test 1: Baseline sanity check

Run the stock baseline to confirm your environment is working and you can reproduce the published 1.2244 score:

```javascript
RUN_ID=test_baseline \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Watch the validation loss curve — it should plateau around the expected value. If your compressed model is over 16MB, something is wrong with quantization or compression.

### Test 2: Longer context

One of the earliest wins on the leaderboard was simply increasing sequence length from the default to 2048 or 4096. Try it:

```javascript
RUN_ID=test_seq2048 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
SEQ_LEN=2048 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

This alone moved the score from 1.2244 to 1.206 in the original leaderboard submissions.

### Test 3: Layer depth vs. width

The baseline uses 9 layers with a 512 hidden dimension. Try 10 or 11 layers at the same dimension — the top submissions all use 10–11 layers, which suggests that depth is more parameter-efficient than width for this task:

```javascript
RUN_ID=test_11layers \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
NUM_LAYERS=11 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## What the top submissions are doing

The leaderboard has evolved fast. Here's a look at the key techniques that have driven scores from 1.2244 down to 1.1228 BPB in less than a week.

### Quantization-aware training (QAT)

Nearly every competitive submission uses int6 or mixed int5/int6 quantization with straight-through estimator (STE) gradients. Since the artifact size is measured post-compression, training the model to be robust to low-precision weights is one of the highest-leverage moves. The current leader uses GPTQ-lite clip search combined with QAT at 0.15 to squeeze maximum information into the 16MB budget.

### Wider MLPs

Several top runs use a 3× MLP expansion ratio instead of the standard 4×, which keeps the parameter count balanced differently between attention and feed-forward layers. Combined with int6 quantization, this allows more expressive capacity per compressed byte.

### Sliding window evaluation

A surprisingly impactful evaluation technique: instead of evaluating the model at a fixed context length, top submissions use a sliding window with a stride of 64 tokens, effectively increasing the evaluation context. This doesn't change the model itself, but it better captures the model's actual compression ability. The first submission to use this technique jumped from ~1.20 to ~1.19 BPB.

### Exponential moving average (EMA) and stochastic weight averaging (SWA)

Model averaging during training produces smoother final weights, which compress better and generalize better on the validation set. The transition from SWA (used in earlier submissions) to EMA in the March 21 submissions coincided with another significant score drop.

### Cross-sequence attention (XSA)

One of the more creative architectural changes: applying attention across sequence boundaries in the deepest layers of the model, allowing the model to borrow context from adjacent sequences during evaluation. The partial variant (applying XSA to only the last 3–4 layers) keeps the computational overhead manageable.

### Test-time training (TTT)

The LoRA TTT submission takes a different approach entirely: the model adapts its own weights at evaluation time using low-rank updates on already-evaluated tokens. This doesn't violate the rules because it only trains on tokens that have already been scored. It achieved 1.1928 BPB, competitive with much more traditional approaches.

## Key rules and constraints to know

**Artifact size**: 16,000,000 bytes (decimal, not MiB). Code bytes plus compressed model bytes.

**Training compute**: 10 minutes wall clock on 8×H100 SXM GPUs.

**Evaluation compute**: An additional 10 minutes on 8×H100s. You can evaluate at any sequence length.

**External packages**: You can import any Python library (FlashAttention, custom CUDA kernels, etc.) as long as it doesn't sneak in additional data or violate the spirit of the challenge. Package bytes don't count toward the 16MB limit.

**No validation data during training**: You cannot train on the validation set. Test-time training is allowed only on tokens you've already evaluated.

**Statistical significance**: New records must beat the current SOTA by ≥0.005 nats at p < 0.01.

## How to submit

All submissions are made as pull requests to the [parameter-golf repo](https://github.com/openai/parameter-golf). Your PR should add a new folder under `records/track_10min_16mb/` containing:

1. A `README.md` explaining your approach
2. A `submission.json` with your name, GitHub handle, `val_bpb`, and metadata
3. A training log demonstrating statistical significance (usually 3 runs)
4. A working `train_gpt.py` and any dependencies

Non-record submissions, like interesting approaches that don't beat SOTA are welcome too, especially novel or unconventional ideas.

## Resources

- **Repository**: [github.com/openai/parameter-golf](https://github.com/openai/parameter-golf)
- **Runpod template**: [Launch on Runpod](https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th)
- **Runpod × OpenAI landing page**: [modelcraft.runpod.io](https://modelcraft.runpod.io/)
- **Compute credits**: [Request a grant from OpenAI](https://openai.com/index/parameter-golf/#credit-form)
- **Discord**: Join the OpenAI Discord server, channels `#parameter-golf-discussions` and `#parameter-golf-announcements`
- **Participant form**: [Sign up with OpenAI](https://jobs.ashbyhq.com/openai/form/open-ai-challenge-parameter-golf) (optional, but helps with attribution)

The challenge runs through April 30. The baseline is right there waiting to be beaten, the Runpod template makes setup a five-minute task, and OpenAI is handing out compute credits. Fore!
