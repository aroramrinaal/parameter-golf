# parameter golf experiment tracker

**setup:** 1xH100 SXM (Modal) | 1 training shard | 10-min wallclock cap | sp1024 tokenizer
**baseline comparison:** runpod 1xH100 baseline val_bpb = 1.36785

## results

| # | experiment | changes | val_bpb | delta vs baseline | steps | compressed size | valid (<16MB) |
|---|-----------|---------|---------|-------------------|-------|----------------|---------------|
| 0 | baseline (runpod) | stock config: 9L, 512d, seq 1024 | 1.36785 | — | 1161 | 13.1 MB | yes |
| 1 | seq_len 2048 | `TRAIN_SEQ_LEN=2048` | 1.33526 | -0.0326 | ~1500 | 14.05 MB | yes |
| 2 | 11 layers | `NUM_LAYERS=11` | 1.34528 | -0.0226 | 1432 | 16.77 MB | no |
| 3 | 11 layers + seq 2048 | `NUM_LAYERS=11` + `TRAIN_SEQ_LEN=2048` | 1.33722 | -0.0306 | 1245 | 16.07 MB | borderline |
| 4 | 11L + MLP3 + seq 2048 | `NUM_LAYERS=11` + `MLP_MULT=3` + `TRAIN_SEQ_LEN=2048` | 1.33321 | -0.0346 | 1104 | 19.49 MB | no |
| 5 | 10L + seq 2048 | `NUM_LAYERS=10` + `TRAIN_SEQ_LEN=2048` | 1.34019 | -0.0277 | 1322 | 14.93 MB | yes |
