"""Xuất mô hình sang ONNX (FP32) ở nhiều kích thước đầu vào và lượng tử hoá INT8 (tuỳ chọn).

Ví dụ:
  python training/export.py --weights runs/v8n/weights/best.pt --imgsz 320 416 640 \
      --int8 --calib ../datasets/tsr5_vn/val/images --out models/
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def letterbox(img, size):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out


def quantize_int8(fp32_path, int8_path, calib_dir, size, n_calib=200):
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            files = sorted(Path(calib_dir).glob("*.jpg"))
            random.Random(0).shuffle(files)
            self.files = iter(files[:n_calib])

        def get_next(self):
            f = next(self.files, None)
            if f is None:
                return None
            img = letterbox(cv2.imread(str(f)), size)[:, :, ::-1].transpose(2, 0, 1)
            return {"images": (img[None].astype(np.float32) / 255.0)}

    pre = fp32_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    # Giữ đầu ra phát hiện (Detect head cuối) ở FP32: lượng tử hoá phần hồi quy toạ độ làm giảm mAP mạnh.
    quantize_static(
        str(pre), str(int8_path), Reader(),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=[n for n in _detect_head_nodes(pre)],
    )
    pre.unlink()


def _detect_head_nodes(path):
    import onnx
    m = onnx.load(str(path))
    # các node phía sau DFL (giải mã box) và concat cuối cùng
    return [n.name for n in m.graph.node if "/model.22/" in n.name and any(k in n.name for k in ("dfl", "Concat", "Split", "Sigmoid", "Mul", "Add", "Sub", "Div"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[320, 416, 640])
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib", type=Path, help="thư mục ảnh hiệu chỉnh cho INT8")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for size in args.imgsz:
        f = Path(YOLO(args.weights).export(format="onnx", imgsz=size, opset=17, simplify=True, dynamic=False))
        dst = args.out / f"tsr_{size}.onnx"
        shutil.move(f, dst)
        print("FP32:", dst)
        if args.int8:
            q = args.out / f"tsr_{size}_int8.onnx"
            quantize_int8(dst, q, args.calib, size)
            print("INT8:", q)


if __name__ == "__main__":
    main()
