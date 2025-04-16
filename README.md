# Background Removal

This is background removal.

## Instructions

HPCC Instructions:
```
module load Miniforge3
```

Create Conda environment:
```
conda create -n rembg_env python=3.11
```

1. Conda environment
```
conda activate rembg_env
```

Timings:
- Time taken for 50 files with rembg: 267.58 seconds
- Time taken for 100 files with u2_net: CUDA (10.016s)
- Time taken for 100 files with u2_net (CUDA):  (168.21s)
