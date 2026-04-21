# parameter golf experiment tracker

**setup:** 1xH100 SXM (Modal) | 1 training shard | 10-min wallclock cap
**baseline comparison:** runpod 1xH100 baseline val_bpb = 1.36785

## phases

- `env-only baseline`: experiments 0-5 below, all launched through `modal_train_gpt.py` with forwarded env only and the stock baseline evaluation path.
- `sliding-eval code`: reserved for runs after the `train_gpt.py` code change that adds `EVAL_MODE`, `EVAL_SLIDING_STRIDE`, and `EVAL_CONTEXT_LEN`.

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
