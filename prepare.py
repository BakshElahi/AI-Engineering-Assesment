"""Preparation step (needs internet, run once before the timed run).

1. Downloads the official yolo11l.pt weights.
2. Tries to build a TensorRT FP16 engine (yolo11l.engine) for the GPU of this machine.

Step 2 is optional: if anything fails, the script prints a message, removes partial files and
exits with status 0. run.py then simply uses yolo11l.pt.
The TensorRT packages are installed with requirements.txt as a constraints file, so they can
never change the pinned torch/numpy/etc. versions.

Usage:  python prepare.py            (installs TensorRT packages with pip, then exports)
        python prepare.py --skip-install   (packages already installed)
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    from ultralytics import YOLO
    YOLO("yolo11l.pt")  # downloads the weights if missing
    print("yolo11l.pt ready", flush=True)

    engine = os.path.join(HERE, "yolo11l.engine")
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU visible; skipping TensorRT")
        if not a.skip_install:
            subprocess.check_call([sys.executable, "-m", "pip", "install",
                                   "-r", os.path.join(HERE, "requirements-tensorrt.txt"),
                                   "-c", os.path.join(HERE, "requirements.txt")])
        t0 = time.time()
        # dynamic=True keeps Ultralytics' rectangular letterbox (e.g. 640x384) and allows batch 1..N
        YOLO("yolo11l.pt").export(format="engine", half=True, dynamic=True, batch=a.batch,
                                  imgsz=640, device=0, verbose=False)
        print(f"engine exported in {time.time() - t0:.0f}s", flush=True)
        # check that it loads and runs
        import numpy as np
        m = YOLO(engine, task="detect")
        r = m.predict([np.zeros((360, 640, 3), np.uint8)] * 4, imgsz=640, conf=0.4, iou=0.7,
                      device=0, verbose=False)
        assert len(r) == 4
        print("TensorRT engine OK:", engine, flush=True)
    except BaseException as e:  # never fail the preparation step because of TensorRT
        print(f"TensorRT export skipped/failed ({type(e).__name__}: {e}). run.py will use PyTorch.", flush=True)
        for p in (engine, os.path.join(HERE, "yolo11l.onnx")):
            if os.path.exists(p):
                os.remove(p)


if __name__ == "__main__":
    main()
