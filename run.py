import argparse, glob, io, json, logging, math, multiprocessing as mp, os, queue, shutil, threading, time, warnings
import numpy as np

CLASSES = [0, 2, 16, 63, 67]
NAMES = {0: "person", 2: "car", 16: "dog", 63: "laptop", 67: "cell phone"}
MIN_SAMPLES_PER_SEC = 8
HERE = os.path.dirname(os.path.abspath(__file__))


def find_file(name):
    for d in (os.getcwd(), HERE):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def get_rotation(stream, cv2):
    # Tag-based only (TODO: display-matrix side data).
    try:
        r = int(float(stream.metadata.get("rotate", 0))) % 360
    except Exception:
        return None
    return {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(r)


def video_iter(path, max_videos):
    import pyarrow.parquet as pq
    name = os.path.basename(path)
    try:
        pf = pq.ParquetFile(path)
        for i, b in enumerate(pf.iter_batches(batch_size=1, columns=["mp4"])):
            if max_videos and i >= max_videos:
                break
            yield f"{name}:{i}", b.column(0)[0].as_py()
    except Exception as e:
        yield f"{name}:?", e


def process_file(path, predict, a, out_path):
    import av, cv2
    it, lock, q = video_iter(path, a.max_videos), threading.Lock(), queue.Queue(maxsize=8)

    def decode_one(vid, data):
        last, err, cs, cf = -1, None, [], []
        try:
            if data is None or isinstance(data, Exception):
                raise ValueError("could not read video bytes")
            with av.open(io.BytesIO(data)) as c:
                s = c.streams.video[0]
                s.thread_type = "AUTO"
                rot = get_rotation(s, cv2)
                try:
                    fps = float(s.average_rate) if s.average_rate else 0.0
                except Exception:
                    fps = 0.0
                stride = a.stride if fps <= 0 else max(1, min(a.stride, int(round(fps / MIN_SAMPLES_PER_SEC))))
                t0, cur, k = None, None, 0
                for f in c.decode(s):
                    if f.pts is None:
                        continue
                    t = f.pts * f.time_base
                    if t0 is None:
                        t0 = t
                    sec = max(0, math.floor(t - t0))
                    last = max(last, sec)
                    if sec != cur:
                        cur, k = sec, 0
                    take = (k % stride == 0)
                    k += 1
                    if not take:
                        continue
                    img = f.to_ndarray(format="bgr24")
                    if rot is not None:
                        img = cv2.rotate(img, rot)
                    cs.append(sec); cf.append(img)
                    if len(cf) >= a.chunk:
                        q.put(("frames", vid, cs, cf)); cs, cf = [], []
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]
        if cf:
            q.put(("frames", vid, cs, cf))
        if last < 0 and err is None:
            err = "could not decode video"
        q.put(("done", vid, last, err))

    def worker():
        while True:
            with lock:
                item = next(it, None)
            if item is None:
                break
            decode_one(*item)
        q.put(("exit",))

    for _ in range(a.workers):
        threading.Thread(target=worker, daemon=True).start()

    states, bad, alive = {}, {}, a.workers
    n_frames, t_wait, t_inf = 0, 0.0, 0.0
    with open(out_path + ".tmp", "w") as out:
        while alive:
            t = time.perf_counter(); m = q.get(); t_wait += time.perf_counter() - t
            if m[0] == "exit":
                alive -= 1
            elif m[0] == "frames":
                _, vid, secs, frames = m
                if vid in bad:
                    continue
                t = time.perf_counter()
                try:
                    res = predict(frames)
                    st = states.setdefault(vid, {})
                    for sec, r in zip(secs, res):
                        cls = r.boxes.cls.int().tolist()
                        d = st.setdefault(sec, [0, {}, []])
                        d[0] += 1
                        for c in set(cls):
                            d[1][c] = d[1].get(c, 0) + 1
                        d[2].append(cls.count(0))
                    n_frames += len(frames)
                except Exception as e:
                    bad[vid] = f"inference error: {type(e).__name__}: {e}"[:200]
                t_inf += time.perf_counter() - t
            else:
                _, vid, last, err = m
                err = bad.pop(vid, None) or err
                st = states.pop(vid, {})
                for s in range(last + 1):
                    d = st.get(s)
                    if not d:
                        objs, people = [], 0
                    else:
                        objs = [NAMES[c] for c in CLASSES if d[1].get(c, 0) * 2 >= d[0]]
                        pc = sorted(d[2]); n = len(pc)
                        people = pc[n // 2] if n % 2 else (pc[n // 2 - 1] + pc[n // 2]) // 2
                    out.write(json.dumps({"video": vid, "second": s, "objects": objs, "people": people}) + "\n")
                if err:
                    out.write(json.dumps({"video": vid, "error": err}) + "\n")
                out.flush()
    os.replace(out_path + ".tmp", out_path)
    return n_frames, t_wait, t_inf


def gpu_worker(dev, fq, a, parts_dir):
    warnings.filterwarnings("ignore")
    import cv2, torch
    from ultralytics import YOLO
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    cv2.setNumThreads(1); torch.set_num_threads(2)
    kw = dict(imgsz=640, conf=0.4, iou=0.7, classes=CLASSES, device=dev, half=not a.fp32, verbose=False)
    warm = [np.zeros((360, 640, 3), np.uint8)] * 4
    is_engine = a.weights.endswith(".engine")
    try:
        model = YOLO(a.weights, task="detect")
        model.predict(warm, **kw)  # warm-up
    except Exception as e:
        if not is_engine:
            raise
        print(f"[gpu{dev}] TensorRT engine failed ({type(e).__name__}: {str(e)[:150]}); falling back to PyTorch", flush=True)
        is_engine = False
        model = YOLO(a.pt_weights, task="detect")
        model.predict(warm, **kw)
    fb = {"model": None}

    def predict(frames):
        try:
            return model.predict(frames, **kw)
        except Exception:
            if not is_engine:
                raise
            # engine failed on this chunk: retry with the PyTorch model instead of losing the video
            if fb["model"] is None:
                fb["model"] = YOLO(a.pt_weights, task="detect")
            return fb["model"].predict(frames, **kw)

    print(f"[gpu{dev}] backend={'tensorrt' if is_engine else 'pytorch'}", flush=True)
    while True:
        path = fq.get()
        if path is None:
            break
        out = os.path.join(parts_dir, os.path.basename(path) + ".jsonl")
        if os.path.exists(out):
            continue  # resume
        t0 = time.time()
        n, tw, ti = process_file(path, predict, a, out)
        dt = max(1e-9, time.time() - t0)
        print(f"[gpu{dev}] {os.path.basename(path)} sampled_frames={n} total={dt:.0f}s "
              f"infer={ti:.0f}s wait_decode={tw:.0f}s -> {n/dt:.1f} sampled fps", flush=True)


def main():
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--weights", default=None, help="default: yolo11l.engine if present, else yolo11l.pt")
    ap.add_argument("--no-trt", action="store_true", help="ignore yolo11l.engine and use PyTorch")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=32)
    ap.add_argument("--workers", type=int, default=1, help="decoder threads per process")
    ap.add_argument("--procs", type=int, default=0, help="processes (default: one per GPU)")
    ap.add_argument("--max-videos", type=int, default=0, help="dev only: videos per file")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--keep-parts", action="store_true")
    a = ap.parse_args()

    a.pt_weights = find_file("yolo11l.pt") or "yolo11l.pt"
    if a.weights is None:
        eng = None if a.no_trt else find_file("yolo11l.engine")
        a.weights = eng or a.pt_weights
    ngpu = torch.cuda.device_count()
    a.procs = a.procs or max(1, ngpu)
    files = sorted(glob.glob(os.path.join(a.input, "*.parquet")))
    parts_dir = a.output + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    print(f"files={len(files)} gpus={ngpu} procs={a.procs} stride={a.stride} fp16={not a.fp32} weights={a.weights}", flush=True)

    ctx = mp.get_context("spawn")
    fq = ctx.Queue()
    for f in files:
        fq.put(f)
    for _ in range(a.procs):
        fq.put(None)
    t0 = time.time()
    ps = [ctx.Process(target=gpu_worker, args=(i % max(1, ngpu), fq, a, parts_dir)) for i in range(a.procs)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()

    with open(a.output, "wb") as out:
        for f in files:
            name = os.path.basename(f)
            part = os.path.join(parts_dir, name + ".jsonl")
            if os.path.exists(part):
                with open(part, "rb") as src:
                    shutil.copyfileobj(src, out)
            else:
                out.write((json.dumps({"video": f"{name}:?", "error": "file not processed"}) + "\n").encode())
    if not a.keep_parts and all(os.path.exists(os.path.join(parts_dir, os.path.basename(f) + ".jsonl")) for f in files):
        shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"TOTAL wall time: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
