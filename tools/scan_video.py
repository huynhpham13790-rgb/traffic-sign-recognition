"""Quét một video dài, tìm các đoạn có biển báo để cắt làm clip demo.

  python tools/scan_video.py video.mp4 --weights models/best.pt --out scan.csv
  python tools/scan_video.py video.mp4 --weights models/best.pt --cut demo_videos/clips --pad 4

Mỗi "sự kiện" = các lần phát hiện cùng lớp cách nhau không quá --gap giây.
"""
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--stride", type=int, default=5, help="xử lý 1 trong mỗi N khung hình")
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--min-hits", type=int, default=3, help="số lần phát hiện tối thiểu để tính là một sự kiện")
    ap.add_argument("--out", type=Path, default=None, help="ghi danh sách sự kiện ra CSV")
    ap.add_argument("--cut", type=Path, default=None, help="cắt từng sự kiện thành clip mp4 vào thư mục này")
    ap.add_argument("--pad", type=float, default=4.0, help="số giây thêm trước/sau mỗi clip")
    args = ap.parse_args()

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    hits = []  # (time, class, conf)
    idx = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % args.stride == 0:
            _, frame = cap.retrieve()
            r = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            for c, s in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                hits.append((idx / fps, int(c), s))
            if idx % (args.stride * 600) == 0:
                print(f"{idx}/{total} khung, {len(hits)} lần phát hiện")
        idx += 1

    events = []
    for c in sorted({h[1] for h in hits}):
        times = [(t, s) for t, cc, s in hits if cc == c]
        cur = [times[0]]
        for t, s in times[1:]:
            if t - cur[-1][0] <= args.gap:
                cur.append((t, s))
            else:
                events.append((c, cur))
                cur = [(t, s)]
        events.append((c, cur))
    events = [(c, e) for c, e in events if len(e) >= args.min_hits]
    events.sort(key=lambda x: x[1][0][0])

    rows = [{"class_id": c, "name": model.names[c], "start_s": round(e[0][0], 1), "end_s": round(e[-1][0], 1),
             "hits": len(e), "max_conf": round(max(s for _, s in e), 3)} for c, e in events]
    for r in rows:
        m, s = divmod(r["start_s"], 60)
        print(f"{int(m):02d}:{s:04.1f}  {r['name']:16s} {r['hits']:3d} lần  conf max {r['max_conf']}")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["class_id"])
            w.writeheader()
            w.writerows(rows)

    if args.cut:
        args.cut.mkdir(parents=True, exist_ok=True)
        for i, r in enumerate(rows):
            start = max(0.0, r["start_s"] - args.pad)
            dur = r["end_s"] - start + args.pad
            out = args.cut / f"{args.video.stem}_{i:02d}_{r['name']}_{int(r['start_s'])}s.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.1f}", "-i", str(args.video), "-t", f"{dur:.1f}",
                            "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-an", str(out)], check=True)
            print("Đã cắt:", out)


if __name__ == "__main__":
    main()
