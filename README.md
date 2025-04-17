# Background Removal

This is background removal.

## Instructions

HPCC Instructions:

```bash
module load Miniforge3
```

Create Conda environment:

```bash
conda create -n rembg_env python=3.11
```

1. Conda environment

```bash
conda activate rembg_env
```

Timings:

- Time taken for 50 files with rembg: 267.58s
- Time taken for 50 files with u2_net: 10.016s
- Time taken for 50 files with u2_net: (CUDA): 16.21s
