"""Làm sạch bộ dữ liệu Vietnam Traffic Sign Detection và sinh data.yaml.

- Loại ảnh trùng tuyệt đối (MD5) trong cùng tập.
- Loại ảnh ở val/test trùng hoặc gần trùng (dHash) với ảnh trong train -> tránh rò rỉ dữ liệu.
- Bỏ các dòng nhãn sai định dạng.
"""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SPLITS = ("train", "val", "test")


def dhash(path, size=8):
    img = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    return np.packbits(a[:, 1:] > a[:, :-1]).tobytes()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path, help="thư mục dataset gốc (chứa train/ val/ test/)")
    ap.add_argument("dst", type=Path, help="thư mục xuất bộ dữ liệu đã làm sạch")
    ap.add_argument("--classes", type=Path, required=True, help="classid.csv")
    args = ap.parse_args()

    with open(args.classes, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    names = [r["classes-en"].strip() for r in rows]

    seen_md5, train_dh = set(), set()
    stats = {}
    for split in SPLITS:
        out_img, out_lab = args.dst / split / "images", args.dst / split / "labels"
        out_img.mkdir(parents=True, exist_ok=True)
        out_lab.mkdir(parents=True, exist_ok=True)
        kept = dup = leak = bad_lines = 0
        for p in sorted((args.src / split / "images").glob("*.jpg")):
            md5 = hashlib.md5(p.read_bytes()).hexdigest()
            if md5 in seen_md5:
                dup += 1
                continue
            h = dhash(p)
            if split != "train" and h in train_dh:
                leak += 1
                continue
            seen_md5.add(md5)
            if split == "train":
                train_dh.add(h)
            lab = args.src / split / "labels" / (p.stem + ".txt")
            good = []
            for line in (lab.read_text().splitlines() if lab.exists() else []):
                parts = line.split()
                if not parts:
                    continue
                if len(parts) != 5 or not 0 <= int(parts[0]) < len(names):
                    bad_lines += 1
                    continue
                good.append(" ".join(parts))
            shutil.copy2(p, out_img / p.name)
            (out_lab / (p.stem + ".txt")).write_text("\n".join(good) + ("\n" if good else ""))
            kept += 1
        stats[split] = {"kept": kept, "exact_dup_removed": dup, "leak_removed": leak, "bad_label_lines": bad_lines}
        print(f"{split:5s}: giữ {kept}, bỏ trùng {dup}, bỏ rò rỉ {leak}, dòng nhãn lỗi {bad_lines}")

    yaml = [f"path: {args.dst.resolve().as_posix()}", "train: train/images", "val: val/images", "test: test/images", "names:"]
    yaml += [f"  {i}: {json.dumps(n)}" for i, n in enumerate(names)]
    (args.dst / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    (args.dst / "clean_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
