"""Đánh giá độ chính xác (mAP) và tốc độ CPU của các mô hình .pt/.onnx, xuất bảng kết quả.

Ví dụ:
  python training/evaluate.py --data ../datasets/tsr5_vn/data.yaml --split test \
      --models models/tsr_320.onnx models/tsr_416.onnx --out docs/results.csv
"""
import argparse
import csv
import re
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from ultralytics import YOLO


def cpu_latency(onnx_path, threads, runs=50):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]
    x = np.random.rand(*inp.shape).astype(np.float32)
    for _ in range(5):
        s.run(None, {inp.name: x})
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        s.run(None, {inp.name: x})
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=4, help="số luồng CPU khi đo tốc độ (Pi 4/5 có 4 nhân)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = []
    for m in args.models:
        size = int(re.search(r"_(\d+)", Path(m).stem).group(1)) if m.endswith(".onnx") else 640
        r = YOLO(m, task="detect").val(data=args.data, split=args.split, imgsz=size, batch=1 if m.endswith(".onnx") else 16,
                                       device="cpu" if m.endswith(".onnx") else 0, plots=False, verbose=False)
        row = {
            "model": Path(m).name, "imgsz": size,
            "precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
            "mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4),
        }
        for i, name in r.names.items():
            if i in r.box.ap_class_index:
                row[f"AP50_{name}"] = round(float(r.box.ap50[list(r.box.ap_class_index).index(i)]), 4)
        if m.endswith(".onnx"):
            row["cpu_ms"] = round(cpu_latency(m, args.threads), 2)
            row["size_MB"] = round(Path(m).stat().st_size / 1e6, 2)
        rows.append(row)
        print(row)

    if args.out:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
