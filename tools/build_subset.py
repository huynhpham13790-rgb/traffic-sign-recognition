"""Tạo bộ dữ liệu theo nhóm lớp đích từ bộ VN (đã làm sạch) và GTSRB.

Nhóm lớp (--set):
  tsr7 (mặc định, theo mục 11.2 thuyết minh): GioiHan40, GioiHan50, GioiHan60, CamDiNguocChieu,
       CamReTrai, CamRePhai, VongXuyen
  tsr5 (phiên bản cũ): GioiHan60, VongXuyen, DuongCam, CamDiNguocChieu, Stop

- Ảnh VN: chỉ giữ nhãn của các lớp đích, các biển khác bị bỏ nhãn -> ảnh vẫn được dùng làm mẫu âm.
- GTSRB (tuỳ chọn --gtsrb): ảnh crop biển báo được dán ngẫu nhiên lên ảnh nền VN (copy-paste)
  để tạo mẫu phát hiện. Chỉ thêm vào tập train; val/test giữ 100% ảnh thực tế VN.
- --gtsrb-test: tạo thêm tập test_gtsrb từ GTSRB Test (dán lên ảnh nền val) để đánh giá cả 5 lớp.
"""
import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# tên lớp, tên tiếng Việt, ánh xạ id VN -> id đích, ánh xạ id GTSRB -> id đích
CLASS_SETS = {
    "tsr7": (
        ["GioiHan40", "GioiHan50", "GioiHan60", "CamDiNguocChieu", "CamReTrai", "CamRePhai", "VongXuyen"],
        ["Giới hạn tốc độ 40km/h", "Giới hạn tốc độ 50km/h", "Giới hạn tốc độ 60km/h", "Cấm đi ngược chiều",
         "Cấm rẽ trái", "Cấm rẽ phải", "Vòng xuyến"],
        {2: 0, 39: 1, 12: 2, 0: 3, 17: 4, 18: 5, 56: 6},
        {2: 1, 3: 2, 17: 3, 40: 6},  # GTSRB không có biển 40 km/h và biển cấm rẽ
    ),
    "tsr5": (
        ["GioiHan60", "VongXuyen", "DuongCam", "CamDiNguocChieu", "Stop"],
        ["Giới hạn tốc độ 60km/h", "Vòng xuyến", "Đường cấm", "Cấm đi ngược chiều", "Dừng lại (Stop)"],
        {12: 0, 56: 1, 0: 3, 64: 4},
        {3: 0, 40: 1, 15: 2, 17: 3, 14: 4},
    ),
}
NAMES, NAMES_VI, VN_MAP, GTSRB_MAP = CLASS_SETS["tsr7"]


def read_vn(src, split, out_img, out_lab):
    """Sao chép một tập VN, ánh xạ nhãn. Trả về danh sách ảnh không chứa lớp đích (dùng làm nền)."""
    backgrounds = []
    for p in sorted((src / split / "images").glob("*.jpg")):
        lines = []
        for line in (src / split / "labels" / (p.stem + ".txt")).read_text().splitlines():
            c, *box = line.split()
            if int(c) in VN_MAP:
                lines.append(" ".join([str(VN_MAP[int(c)]), *box]))
        if not lines:
            backgrounds.append(p)
        if out_img is not None:
            shutil.copy2(p, out_img / p.name)
            (out_lab / (p.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    return backgrounds


def load_gtsrb(root, csv_name, per_class, step, rng, tracks=None):
    """Đọc crop GTSRB theo ROI. Lấy cách `step` khung hình trong mỗi track để giảm ảnh gần trùng.

    tracks: None = mọi track; "train"/"val" = tách theo track (track id chia hết cho 10 -> val),
    để ảnh val không cùng đoạn video với ảnh train.
    """
    by_cls = {c: [] for c in GTSRB_MAP}
    with open(root / csv_name, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            c = int(r["ClassId"])
            if c not in GTSRB_MAP:
                continue
            frame = int(Path(r["Path"]).stem.split("_")[-1]) if "_" in Path(r["Path"]).stem else 0
            if frame % step:
                continue
            if tracks is not None:
                is_val = int(Path(r["Path"]).stem.split("_")[1]) % 10 == 0
                if is_val != (tracks == "val"):
                    continue
            by_cls[c].append((root / r["Path"], int(r["Roi.X1"]), int(r["Roi.Y1"]), int(r["Roi.X2"]), int(r["Roi.Y2"])))
    for c in by_cls:
        rng.shuffle(by_cls[c])
        by_cls[c] = by_cls[c][:per_class]
    return by_cls


def paste(bg, crop, rng, taken):
    """Dán crop (đã cắt theo ROI, có viền mềm) vào bg ở vị trí không chồng lên box đã có."""
    # Ảnh VN có thể có viền đen (letterbox) -> chỉ dán trong vùng nội dung thật.
    gray = bg.max(axis=2)
    rows, cols = np.where(gray.max(axis=1) > 10)[0], np.where(gray.max(axis=0) > 10)[0]
    top, bottom, left, right = rows[0], rows[-1], cols[0], cols[-1]
    ch = bottom - top
    for _ in range(20):
        side = int(rng.uniform(16, 96))  # phân bố gần với kích thước biển thực tế trong ảnh 640
        h = side
        w = max(8, int(side * crop.shape[1] / crop.shape[0]))
        # nửa trên-giữa/ hai bên ảnh, nơi biển báo thường xuất hiện
        if right - left < w + 2 or 0.65 * ch < h + 2:
            continue
        x = rng.randint(left, right - w - 1)
        y = rng.randint(top + int(0.05 * ch), top + int(0.65 * ch) - h)
        box = (x, y, x + w, y + h)
        if any(not (box[2] < t[0] or box[0] > t[2] or box[3] < t[1] or box[1] > t[3]) for t in taken):
            continue
        c = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR if w > crop.shape[1] else cv2.INTER_AREA)
        # khớp độ sáng với vùng nền để bớt "lạc tông"
        region = bg[y:y + h, x:x + w].astype(np.float32)
        c = c.astype(np.float32) * (0.6 + 0.4 * region.mean() / max(c.mean(), 1.0))
        mask = np.zeros((h, w), np.float32)
        cv2.rectangle(mask, (1, 1), (w - 2, h - 2), 1.0, -1)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)[..., None]
        bg[y:y + h, x:x + w] = np.clip(c * mask + region * (1 - mask), 0, 255).astype(np.uint8)
        return box
    return None


def synthesize(by_cls, backgrounds, out_img, out_lab, prefix, rng, max_per_img=3):
    items = [(GTSRB_MAP[c], it) for c, lst in by_cls.items() for it in lst]
    rng.shuffle(items)
    n = 0
    i = 0
    while i < len(items):
        bg = cv2.imread(str(rng.choice(backgrounds)))
        H, W = bg.shape[:2]
        k = rng.randint(1, max_per_img)
        taken, lines = [], []
        for cls, (path, x1, y1, x2, y2) in items[i:i + k]:
            img = cv2.imread(str(path))
            crop = img[y1:y2 + 1, x1:x2 + 1]
            if crop.size == 0:
                continue
            box = paste(bg, crop, rng, taken)
            if box is None:
                continue
            taken.append(box)
            bx1, by1, bx2, by2 = box
            lines.append(f"{cls} {(bx1 + bx2) / 2 / W:.6f} {(by1 + by2) / 2 / H:.6f} {(bx2 - bx1) / W:.6f} {(by2 - by1) / H:.6f}")
        i += k
        name = f"{prefix}_{n:05d}"
        cv2.imwrite(str(out_img / f"{name}.jpg"), bg, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (out_lab / f"{name}.txt").write_text("\n".join(lines) + "\n")
        n += 1
    return n


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("vn", type=Path, help="bộ VN đã làm sạch (prepare_dataset.py)")
    ap.add_argument("dst", type=Path)
    ap.add_argument("--gtsrb", type=Path, default=None, help="thư mục GTSRB (Train.csv, Train/)")
    ap.add_argument("--per-class", type=int, default=400, help="số crop GTSRB tối đa mỗi lớp")
    ap.add_argument("--gtsrb-test", action="store_true")
    ap.add_argument("--gtsrb-only", action="store_true",
                    help="train/val chỉ gồm ảnh GTSRB dán lên nền VN (không dùng nhãn VN); test vẫn là ảnh VN thật")
    ap.add_argument("--set", choices=CLASS_SETS, default="tsr7", help="nhóm lớp đích")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    global NAMES, NAMES_VI, VN_MAP, GTSRB_MAP
    NAMES, NAMES_VI, VN_MAP, GTSRB_MAP = CLASS_SETS[args.set]
    rng = random.Random(args.seed)

    splits = ["train", "val", "test"]
    bgs = {}
    for s in splits:
        oi, ol = args.dst / s / "images", args.dst / s / "labels"
        oi.mkdir(parents=True, exist_ok=True)
        ol.mkdir(parents=True, exist_ok=True)
        bgs[s] = read_vn(args.vn, s, oi, ol)
        print(f"{s}: VN, {len(bgs[s])} ảnh không chứa lớp đích")

    if args.gtsrb_only:
        for s in ("train", "val"):
            shutil.rmtree(args.dst / s)
            (args.dst / s / "images").mkdir(parents=True)
            (args.dst / s / "labels").mkdir(parents=True)
            by_cls = load_gtsrb(args.gtsrb, "Train.csv", args.per_class, step=2, rng=rng, tracks=s)
            print(f"GTSRB {s} crop:", {NAMES[GTSRB_MAP[c]]: len(v) for c, v in by_cls.items()})
            n = synthesize(by_cls, bgs[s], args.dst / s / "images", args.dst / s / "labels", f"gtsrb_{s}", rng)
            print(f"Đã tạo {n} ảnh tổng hợp cho {s}")
    elif args.gtsrb:
        by_cls = load_gtsrb(args.gtsrb, "Train.csv", args.per_class, step=3, rng=rng)
        print("GTSRB train crop:", {NAMES[GTSRB_MAP[c]]: len(v) for c, v in by_cls.items()})
        n = synthesize(by_cls, bgs["train"], args.dst / "train" / "images", args.dst / "train" / "labels", "gtsrb", rng)
        print(f"Đã tạo {n} ảnh tổng hợp cho train")
        if args.gtsrb_test:
            ti, tl = args.dst / "test_gtsrb" / "images", args.dst / "test_gtsrb" / "labels"
            ti.mkdir(parents=True, exist_ok=True)
            tl.mkdir(parents=True, exist_ok=True)
            by_cls = load_gtsrb(args.gtsrb, "Test.csv", 150, step=1, rng=rng)
            n = synthesize(by_cls, bgs["test"], ti, tl, "gtest", rng)
            print(f"Đã tạo {n} ảnh test_gtsrb (nền lấy từ ảnh test VN)")

    yaml = [f"path: {args.dst.resolve().as_posix()}", "train: train/images", "val: val/images", "test: test/images", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (args.dst / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    with open(args.dst / "classes.csv", "w", encoding="utf-8", newline="") as f:
        f.write("id,en,vi\n")
        for i, (n, v) in enumerate(zip(NAMES, NAMES_VI)):
            f.write(f"{i},{n},{v}\n")


if __name__ == "__main__":
    main()
