# Video Indexing at Speed

`run.py` reads FineVideo Parquet files, runs YOLO11l on the video frames, and writes one JSON line
per second of video: which of `person, car, dog, laptop, cell phone` are present, and how many
people are visible.

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

## Install

Target: Ubuntu 22.04, Python 3.11, NVIDIA driver with CUDA 12 support. Docker is not needed.

System packages (needed by OpenCV, which Ultralytics installs):

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

Python packages:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Preparation command (needs internet, run once before the timed run)

```bash
python prepare.py
```

This downloads the official `yolo11l.pt` weights, then tries to build a TensorRT FP16 engine
(`yolo11l.engine`) for the GPU of this machine. For that it installs the packages listed in
`requirements-tensorrt.txt`, with `requirements.txt` as a constraints file so the pinned torch
version is never changed. It needs about 10 GB of free disk space and took roughly 15 minutes on a
Tesla T4 (it builds a 4.3 GB TensorRT libraries wheel and the engine build itself took about 14 minutes).

The TensorRT step is optional. If it fails for any reason (no GPU, no internet, not enough disk,
build error), the script prints a message and exits normally, and `run.py` uses `yolo11l.pt` instead.
The timed run itself needs no internet. Run both commands from the repository root.

## Run

```bash
python run.py --input /path/to/parquet_folder --output results.jsonl
```

Optional flags (the defaults are what I used for the results file):

| Flag | Default | Meaning |
|---|---|---|
| `--stride` | 3 | Run the model on every 3rd frame within each second. `--stride 1` uses every frame. |
| `--chunk` | 32 | Frames per model call |
| `--procs` | one per GPU | Worker processes, each pinned to one GPU |
| `--workers` | 1 | Decoder threads per process |
| `--weights` | auto | Default: `yolo11l.engine` if it exists, else `yolo11l.pt` |
| `--no-trt` | off | Ignore the engine and use PyTorch |
| `--fp32` | off | Use FP32 instead of FP16 |
| `--max-videos` | 0 (all) | Development only: limit videos per file |

## What the script does

1. Finds `*.parquet` files in the input folder. Each GPU process takes the next unprocessed file.
2. Reads one video (`mp4` bytes) at a time from the Parquet file.
3. Decodes it with PyAV on a background thread. Frame time is relative to the first decoded frame, so
   second `s` holds frames with time in `[s, s+1)`. Frames are converted to BGR and rotated if the stream has a `rotate` tag.
4. Keeps every 3rd frame within each second (see "Frame sampling" below) and sends chunks of 32 frames to
   YOLO11l (`imgsz=640`, `conf=0.4`, `iou=0.7`, Ultralytics letterbox and class-aware NMS, FP16, classes limited to the five).
5. For each second: an object is present if it is found in at least half of the sampled frames; the people
   count is the median of the per-frame person counts, rounded down at halves.
6. Writes a line for every second from 0 to the second that holds the last frame, empty seconds included.
7. If a video cannot be read or decoded, it writes an error line and goes on. If it fails partway, the lines
   already produced are kept and an error line is added. Per-file outputs are merged into the single output file at the end.

## Frame sampling (important, read this)

The reference runs the model on every frame. To be faster, this script runs the model on every 3rd frame within
each second, and decides presence and the people count from those frames. For videos with a low frame rate the stride is
reduced so that each second still has about 8 sampled frames.
This is an approximation of the reference, so I measured it. Against my own reference (the same pipeline with FP32 and every frame),
on 6 videos from two files (2,215 seconds):

| Setting | Objects F1 | People exact | People within 1 |
|---|---|---|---|
| Pass thresholds | 0.95 | 0.85 | 0.95 |
| stride 3, FP16 | 0.9912 | 0.9783 | 0.9991 |

On 4 videos from the first file: FP16 with every frame gave F1 1.0000 / exact 0.9993 / within-1 1.0000; stride 2 gave
0.9952 / 0.9799 / 1.0000; stride 3 gave 0.9889 / 0.9625 / 0.9987.
These numbers compare against my own FP32 every-frame run, not the official reference, and the sample is small.
`--stride 1` removes the approximation at about 3 times the cost.

## TensorRT (optional speed-up)

If `yolo11l.engine` exists next to `run.py`, it is used; otherwise `yolo11l.pt`. If the engine fails to
load, or fails on a chunk of frames, the script falls back to PyTorch (`--no-trt` forces PyTorch).
The engine is exported in FP16 with a dynamic input size, so Ultralytics keeps its default rectangular letterbox.
A TensorRT engine only works on the GPU it was built on, which is why it is built in the preparation step and not
stored in the repository.

Measured on a Tesla T4 (not on an RTX 4090, which I could not test):

- 256 frames, 32 per call: PyTorch FP16 111 FPS, TensorRT FP16 191 FPS (1.72x); model inference time per frame 7.1 ms to 3.5 ms.
  The two gave the same detections on those frames (159 each, identical class lists and person counts).
- Full pipeline on 3 videos: about 103 to 110 sampled frames per second with PyTorch, about 171 with TensorRT.
- Compared with `results.jsonl.gz` on the same 3 videos (1,082 seconds): 1,079 to 1,081 seconds identical (99.7% to 99.9%)
  with TensorRT, 1,080 (99.8%) with PyTorch. Two different engine builds gave slightly different counts.

Installing the TensorRT export packages upgrades `sympy` to 1.14, which makes pip print a version warning for torch 2.6.0.
I did not see any effect from it.

## How `results.jsonl.gz` was made

On Kaggle (2 x Tesla T4, see `HARDWARE.md`) with the default settings (`--stride 3`, FP16), in two notebooks:
files 0 to 24 and files 25 to 49, then concatenated. It has 1,650 videos, 461,757 seconds and no error lines.
It was made with PyTorch FP16 (no TensorRT engine) before the TensorRT step was added.

That environment used Python 3.13, PyTorch 2.11.0+cu130, PyAV 19.0.2 and OpenCV 4.14.
`requirements.txt` pins versions that work on Python 3.11 with a CUDA 12 driver (PyTorch 2.6.0 with CUDA 12.4 wheels,
PyAV 18.1.1, OpenCV 5.1.0.95). I installed this set in a fresh Python 3.11 environment on a Kaggle T4 and reran 3 videos
from the first file: 1,080 of 1,082 seconds (99.8%) were identical to `results.jsonl.gz`.
I did not rerun the full set with the pinned versions, and I could not test on a CUDA 12-only driver.

## Known limitations

- Rotation is handled only when the stream has a `rotate` tag, not for display-matrix side data.
- The frame sampling above is an approximation.
- Damaged videos get an error line; I only tested this on a small number of cases.
- I could not test on an RTX 4090. The TensorRT engine is built on the grading machine by `prepare.py`; if that fails, PyTorch is used.
