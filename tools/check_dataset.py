"""Kiểm tra chất lượng bộ dữ liệu YOLO: nhãn hợp lệ, phân bố lớp, kích thước box, ảnh trùng giữa các tập."""
import argparse
import collections
import hashlib
import json
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
    ap.add_argument("root", type=Path, help="thư mục chứa train/ val/ test/")
    ap.add_argument("--nc", type=int, default=82)
    ap.add_argument("--out", type=Path, default=None, help="ghi báo cáo JSON")
    args = ap.parse_args()

    report = {"splits": {}, "problems": []}
    md5_owner, dh_owner = {}, {}
    exact_dup, near_dup = collections.Counter(), collections.Counter()
    sizes = collections.Counter()
    rel_areas = []

    for split in SPLITS:
        imgs = sorted((args.root / split / "images").glob("*.*"))
        cls_count = collections.Counter()
        n_boxes = n_empty = 0
        for p in imgs:
            lab = args.root / split / "labels" / (p.stem + ".txt")
            md5 = hashlib.md5(p.read_bytes()).hexdigest()
            if md5 in md5_owner:
                exact_dup[(md5_owner[md5], split)] += 1
            else:
                md5_owner[md5] = split
            with Image.open(p) as im:
                sizes[im.size] += 1
            h = dhash(p)
            if h in dh_owner and dh_owner[h] != split:
                near_dup[(dh_owner[h], split)] += 1
            dh_owner.setdefault(h, split)
            if not lab.exists():
                report["problems"].append(f"thiếu nhãn: {split}/{p.name}")
                continue
            lines = [l.split() for l in lab.read_text().splitlines() if l.strip()]
            if not lines:
                n_empty += 1
            for parts in lines:
                if len(parts) != 5:
                    report["problems"].append(f"dòng sai định dạng: {split}/{lab.name}")
                    continue
                c, x, y, w, hh = int(parts[0]), *map(float, parts[1:])
                if not 0 <= c < args.nc or not all(0 <= v <= 1 for v in (x, y, w, hh)):
                    report["problems"].append(f"giá trị ngoài miền: {split}/{lab.name}")
                cls_count[c] += 1
                rel_areas.append(w * hh)
                n_boxes += 1
        report["splits"][split] = {
            "images": len(imgs), "boxes": n_boxes, "background": n_empty,
            "classes_present": len(cls_count),
            "class_counts": dict(sorted(cls_count.items())),
        }

    rel_areas.sort()
    q = lambda f: rel_areas[int(f * (len(rel_areas) - 1))]
    report["box_rel_area_quantiles"] = {"p10": q(0.1), "p50": q(0.5), "p90": q(0.9)}
    report["image_sizes_top"] = [[f"{w}x{h}", n] for (w, h), n in sizes.most_common(5)]
    report["exact_duplicates"] = {f"{a}->{b}": n for (a, b), n in exact_dup.items()}
    report["near_duplicates_cross_split"] = {f"{a}->{b}": n for (a, b), n in near_dup.items()}

    for s, v in report["splits"].items():
        print(f"{s:5s}: {v['images']} ảnh, {v['boxes']} box, {v['background']} ảnh nền, {v['classes_present']} lớp")
    print("Kích thước ảnh phổ biến:", report["image_sizes_top"])
    print("Diện tích box tương đối (p10/p50/p90):", {k: round(v, 5) for k, v in report["box_rel_area_quantiles"].items()})
    print("Trùng tuyệt đối:", report["exact_duplicates"])
    print("Gần trùng giữa các tập:", report["near_duplicates_cross_split"])
    print("Số lỗi nhãn:", len(report["problems"]), report["problems"][:5])
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
