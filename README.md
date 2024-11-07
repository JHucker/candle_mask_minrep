# candle_mask_minrep
Masked fill ops timings for tch vs candle

## Results

```bash
name, compute_cap, driver_version
NVIDIA GeForce RTX 4090, 8.9, 560.35.03
```

```
| batch_size | tch_μs_average | cdl_μs_average |
|------------|----------------|----------------|
|          1 |             14 |             10 |
|          2 |             13 |             10 |
|          4 |             13 |             10 |
|          8 |             13 |             13 |
|         16 |             16 |             20 |
|         32 |             18 |             30 |
|         64 |             21 |             60 |
|        128 |             29 |             91 |
|        256 |             51 |            447 |
|        512 |            201 |            911 |
|       1024 |            515 |           1278 |
|       2048 |           1027 |           2007 |
```
