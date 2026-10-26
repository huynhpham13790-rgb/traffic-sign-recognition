"""Phiên bản Python (ONNX Runtime + OpenCV) dùng để đối chiếu tốc độ với bản C++.

  python deploy/python/detect.py --model models/tsr_416.onnx --labels models/classes.csv --source 0
  python deploy/python/detect.py --model models/tsr_416.onnx --labels models/classes.csv --source img.jpg --bench 100
"""
import argparse
import csv
import sys
import time

import cv2
import numpy as np
import onnxruntime as ort


class Detector:
    def __init__(self, path, conf=0.35, iou=0.45, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0]
        self.size = self.inp.shape[2]
        self.conf, self.iou = conf, iou

    def detect(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.size / h, self.size / w)
        nh, nw = round(h * r), round(w * r)
        top, left = (self.size - nh) // 2, (self.size - nw) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(bgr, (nw, nh))
        x = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.inp.name: x})[0][0]

        if out.shape[-1] == 6:  # YOLO26 end-to-end: x1,y1,x2,y2,score,cls
            keep = out[out[:, 4] >= self.conf]
            boxes = keep[:, :4].copy()
            scores, cls = keep[:, 4], keep[:, 5].astype(int)
        else:  # [4 + nc, N]
            out = out.T
            scores_all = out[:, 4:]
            cls = scores_all.argmax(1)
            scores = scores_all[np.arange(len(cls)), cls]
            m = scores >= self.conf
            b, scores, cls = out[m, :4], scores[m], cls[m]
            if len(b) and b.max() <= 1.0:
                b = b * self.size
            boxes = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)
            xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], 1)
            idx = cv2.dnn.NMSBoxesBatched(xywh.tolist(), scores.tolist(), cls.tolist(), self.conf, self.iou) if len(boxes) else []
            idx = np.array(idx, dtype=int).reshape(-1)
            boxes, scores, cls = boxes[idx], scores[idx], cls[idx]
        boxes = (boxes - [left, top, left, top]) / r
        return [(int(c), float(s), b.astype(int).tolist()) for b, s, c in zip(boxes, scores, cls)]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--source", default="0")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--bench", type=int, default=0)
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    with open(args.labels, encoding="utf-8") as f:
        names = {int(r["id"]): r for r in csv.DictReader(f)}
    det = Detector(args.model, args.conf, threads=args.threads)

    if args.source.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(args.source)
        if args.bench:
            for _ in range(5):
                det.detect(img)
            t = []
            for _ in range(args.bench):
                t0 = time.perf_counter()
                det.detect(img)
                t.append((time.perf_counter() - t0) * 1000)
            print(f"input={det.size} n={args.bench} mean={np.mean(t):.2f}ms p50={np.median(t):.2f}ms fps={1000 / np.mean(t):.2f}")
            return
        for c, s, b in det.detect(img):
            print(c, names[c]["vi"], round(s, 3), b)
        return

    cap = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    fps, last = 0.0, time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = det.detect(frame)
        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 / (now - last) if fps else 1 / (now - last)
        last = now
        if not args.headless:
            for c, s, (x1, y1, x2, y2) in dets:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{names[c]['en']} {s:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("TSR (Python)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    print(f"FPS: {fps:.1f}")


if __name__ == "__main__":
    main()
