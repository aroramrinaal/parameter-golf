# parameter golf experiment tracker

**setup:** 1xH100 SXM (Modal) | 1 training shard | 10-min wallclock cap
**baseline comparison:** runpod 1xH100 baseline val_bpb = 1.36785

## phases

- `env-only baseline`: experiments 0-5 below, all launched through `modal_train_gpt.py` with forwarded env only and the stock baseline evaluation path.
- `sliding-eval code`: reserved for runs after the `train_gpt.py` code change that adds `EVAL_MODE`, `EVAL_SLIDING_STRIDE`, and `EVAL_CONTEXT_LEN`.
- `universal-transformer code`: reserved for runs after the `train_gpt.py` code change that adds `PHYSICAL_LAYERS`, `DEPTH_EMBEDDING`, and `VIRTUAL_LAYER_SCALES` for shared-block virtual depth.
- `random-mlp-proj code`: reserved for runs after the `train_gpt.py` code change that adds frozen-random `mlp.proj` plus a learned low-rank adapter and output gain.

## results

| # | phase | experiment | code version | tokenizer / data | eval mode | comparable to | changes | val_bpb | delta vs baseline | steps | compressed size | valid (<16MB) |
|---|-------|------------|--------------|------------------|-----------|---------------|---------|---------|-------------------|-------|----------------|---------------|
| 0 | env-only baseline | baseline (runpod) | env-only baseline | `sp1024` / `sp1024` | standard | baseline | stock config: 9L, 512d, seq 1024 | 1.36785 | — | 1161 | 13.1 MB | yes |
| 1 | env-only baseline | seq_len 2048 | env-only baseline | `sp1024` / `sp1024` | standard | 0, 5 | `TRAIN_SEQ_LEN=2048` | 1.33526 | -0.0326 | ~1500 | 14.05 MB | yes |
| 2 | env-only baseline | 11 layers | env-only baseline | `sp1024` / `sp1024` | standard | 0 | `NUM_LAYERS=11` | 1.34528 | -0.0226 | 1432 | 16.77 MB | no |
| 3 | env-only baseline | 11 layers + seq 2048 | env-only baseline | `sp1024` / `sp1024` | standard | 2 | `NUM_LAYERS=11` + `TRAIN_SEQ_LEN=2048` | 1.33722 | -0.0306 | 1245 | 16.07 MB | borderline |
| 4 | env-only baseline | 11L + MLP3 + seq 2048 | env-only baseline | `sp1024` / `sp1024` | standard | 3 | `NUM_LAYERS=11` + `MLP_MULT=3` + `TRAIN_SEQ_LEN=2048` | 1.33321 | -0.0346 | 1104 | 19.49 MB | no |
| 5 | env-only baseline | 10L + seq 2048 | env-only baseline | `sp1024` / `sp1024` | standard | 0, 1 | `NUM_LAYERS=10` + `TRAIN_SEQ_LEN=2048` | 1.34019 | -0.0277 | 1322 | 14.93 MB | yes |
| 6 | sliding-eval code | seq2048 standard | sliding-eval code | `sp1024` / `sp1024` | standard | 1 | `TRAIN_SEQ_LEN=2048` | 1.36353 | -0.0043 | 1022 | 12.50 MB | yes |
| 7 | sliding-eval code | seq2048 sliding64 rerun | sliding-eval code | `sp1024` / `sp1024` | sliding (stride=64) | 6 | `TRAIN_SEQ_LEN=2048` + `EVAL_MODE=sliding` + `EVAL_SLIDING_STRIDE=64` + `EVAL_CONTEXT_LEN=2048` | 1.31375 | -0.0541 | 1528 | 14.07 MB | yes |
| 8 | sliding-eval code | sp4096 seq2048 sliding64 repooverride | sliding-eval code | `sp4096` / `sp4096` | sliding (stride=64) | 7 only for eval-style, not tokenizer-family | `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf` + `VOCAB_SIZE=4096` + `DATA_VARIANT=sp4096` + `TRAIN_SEQ_LEN=2048` + `EVAL_MODE=sliding` + `EVAL_SLIDING_STRIDE=64` + `EVAL_CONTEXT_LEN=2048` | 1.28724 | -0.0806 | 1456 | 15.37 MB | yes |
| 9 | universal-transformer code | sp4096 universal-transformer main | universal-transformer code | `sp4096` / `sp4096` | sliding (stride=64) | 8 | `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf` + `VOCAB_SIZE=4096` + `DATA_VARIANT=sp4096` + `TRAIN_SEQ_LEN=2048` + `NUM_LAYERS=12` + `PHYSICAL_LAYERS=3` + `MODEL_DIM=640` + `NUM_HEADS=10` + `NUM_KV_HEADS=5` + `MLP_MULT=3` + `DEPTH_EMBEDDING=1` + `VIRTUAL_LAYER_SCALES=1` + `EVAL_MODE=sliding` + `EVAL_SLIDING_STRIDE=64` + `EVAL_CONTEXT_LEN=2048` | 1.35815 | -0.0097 | 811 | 9.64 MB | yes |
| 10 | universal-transformer code | sp4096 UT half-untied main | universal-transformer code | `sp4096` / `sp4096` | sliding (stride=64) | 9, 8 | `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf` + `VOCAB_SIZE=4096` + `DATA_VARIANT=sp4096` + `TRAIN_SEQ_LEN=2048` + `NUM_LAYERS=10` + `PHYSICAL_LAYERS=5` + `MODEL_DIM=640` + `NUM_HEADS=10` + `NUM_KV_HEADS=5` + `MLP_MULT=3` + `DEPTH_EMBEDDING=1` + `VIRTUAL_LAYER_SCALES=1` + `EVAL_MODE=sliding` + `EVAL_SLIDING_STRIDE=64` + `EVAL_CONTEXT_LEN=2048` | 1.31372 | -0.0541 | 954 | 15.03 MB | yes |
| 11 | universal-transformer code | sp4096 UT half-untied no depth emb | universal-transformer code | `sp4096` / `sp4096` | sliding (stride=64) | 10 | `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf` + `VOCAB_SIZE=4096` + `DATA_VARIANT=sp4096` + `TRAIN_SEQ_LEN=2048` + `NUM_LAYERS=10` + `PHYSICAL_LAYERS=5` + `MODEL_DIM=640` + `NUM_HEADS=10` + `NUM_KV_HEADS=5` + `MLP_MULT=3` + `DEPTH_EMBEDDING=0` + `VIRTUAL_LAYER_SCALES=1` + `EVAL_MODE=sliding` + `EVAL_SLIDING_STRIDE=64` + `EVAL_CONTEXT_LEN=2048` | 1.31040 | -0.0575 | 963 | 15.03 MB | yes |
| 12 | random-mlp-proj code | sp4096 random mlp proj adapter | random-mlp-proj code | `sp4096` / `sp4096` | sliding (stride=64) | 8 | `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf` + `VOCAB_SIZE=4096` + `DATA_VARIANT=sp4096` + `TRAIN_SEQ_LEN=2048` + `NUM_LAYERS=10` + `MODEL_DIM=640` + `NUM_HEADS=10` + `NUM_KV_HEADS=5` + `MLP_MULT=3` + `RANDOM_MLP_PROJ=1` + `RANDOM_MLP_PROJ_RANK=32` + `RANDOM_MLP_PROJ_GAIN=1` + `RANDOM_MLP_PROJ_SEED=1337` + `EVAL_MODE=sliding` + `EVAL_SLIDING_STRIDE=64` + `EVAL_CONTEXT_LEN=2048` | 1.55121 | +0.1834 | 356 | 16.75 MB | no |
| 13 | sliding-eval code | sp4096 seq2048 sliding64 qkgain3 | sliding-eval code | `sp4096` / `sp4096` | sliding (stride=64) | 8 | `MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf` + `VOCAB_SIZE=4096` + `DATA_VARIANT=sp4096` + `TRAIN_SEQ_LEN=2048` + `EVAL_MODE=sliding` + `EVAL_SLIDING_STRIDE=64` + `EVAL_CONTEXT_LEN=2048` + `QK_GAIN_INIT=3.0` | 1.28450 | -0.0834 | 1461 | 15.47 MB | yes |
