"""Dựng video demo từ bản ghi `tsr --record`: đúng tốc độ thật và có tiếng cảnh báo.

  python tools/mux_alerts.py demo.avi --sounds deploy/cpp/sounds --out demo.mp4

Đọc demo.avi.frames.csv (thời điểm từng khung, ms) để giữ mỗi khung đúng khoảng thời gian nó hiện trên màn hình,
và demo.avi.alerts.csv (ms,class_id) để chèn sounds/<class_id>.wav đúng lúc cảnh báo.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

FPS = 30


def retime(video, frames_csv, out):
    """Lặp mỗi khung cho đến khi tới thời điểm của khung tiếp theo (tốc độ khung hình cố định FPS)."""
    times = [int(x) for x in Path(frames_csv).read_text().split()]
    cap = cv2.VideoCapture(str(video))
    w = None
    n = 0
    for i, t in enumerate(times):
        ok, f = cap.read()
        if not ok:
            break
        if w is None:
            w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (f.shape[1], f.shape[0]))
        end = times[i + 1] if i + 1 < len(times) else t + 1000 // FPS
        while n * 1000 / FPS < end:
            w.write(f)
            n += 1
    w.release()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--sounds", type=Path, default=Path("deploy/cpp/sounds"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    alerts = []
    for line in Path(str(args.video) + ".alerts.csv").read_text().split():
        ms, cls = line.split(",")
        alerts.append((int(ms), int(cls)))

    tmp = Path(tempfile.mkdtemp()) / "retimed.avi"
    retime(args.video, str(args.video) + ".frames.csv", tmp)

    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(tmp)]
    for _, cls in alerts:
        cmd += ["-i", str(args.sounds / f"{cls}.wav")]
    if alerts:
        parts = [f"[{i + 1}:a]adelay={ms}|{ms}[a{i}]" for i, (ms, _) in enumerate(alerts)]
        mix = "".join(f"[a{i}]" for i in range(len(alerts)))
        flt = ";".join(parts) + f";{mix}amix=inputs={len(alerts)}:normalize=0,apad[aout]"
        cmd += ["-filter_complex", flt, "-map", "0:v", "-map", "[aout]", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", str(args.out)]
    subprocess.run(cmd, check=True)
    print(f"{args.out}: {len(alerts)} cảnh báo")


if __name__ == "__main__":
    main()
