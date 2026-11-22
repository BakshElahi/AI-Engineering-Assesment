# Hardware used for the full 50-file run

All 50 Parquet files (`data/train-00000` to `train-00049`) were processed on **Kaggle notebooks**
(free tier, "GPU T4 x2" accelerator), not on my own machine.

| Item | Value |
|---|---|
| GPU | 2 x NVIDIA Tesla T4, 14.56 GB each (both used, one process per GPU) |
| CPU | Intel Xeon @ 2.00 GHz, 4 vCPUs (2 physical cores, 4 threads) |
| RAM | 31 GB |
| Disk | Kaggle-provided volumes: 20 GB `/kaggle/working` (ext4 loop device) and a large `/kaggle/temp` overlay where the data was downloaded. Kaggle does not report the disk type (SSD or hard disk). |
| OS | Linux 6.18 (x86_64, glibc 2.39) |
| Python / PyTorch | Python 3.13, PyTorch 2.11.0+cu130, Ultralytics 8.4.216, PyAV 19.0.2 |

## How the run was done

The run was split into two notebooks, each using both T4 GPUs, so that a failure would not lose
everything and to stay inside Kaggle's 12-hour session limit. The two notebooks ran one after the
other, not at the same time.

| Notebook | Files | Wall time of `run.py` (log line `TOTAL wall time`) |
|---|---|---|
| A | train-00000 .. train-00024 | 10357.1 s (2 h 52 min) |
| B | train-00025 .. train-00049 | 10143 s (2 h 49 min) |

Total: **FILL IN (A + B)**. These times exclude about 1.5 minutes per notebook for installing packages and
downloading the data.

Settings: `--stride 3` (every 3rd frame within each second), FP16, chunks of 32 frames, one decoder
thread per GPU process. Per-file speed was about 102 to 115 sampled frames per second per GPU.

The results in `results.jsonl.gz` are the concatenation of the output of the two notebooks
(1,650 videos, 461,757 seconds, 0 error lines).
