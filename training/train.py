"""Huấn luyện mô hình YOLO phát hiện biển báo giao thông Việt Nam."""
import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="đường dẫn data.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="trọng số khởi tạo (yolov8n.pt, yolo11n.pt, ...)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--project", default="runs")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=0,
        deterministic=True,
        patience=30,
        cos_lr=True,
        close_mosaic=10,
        # Biển báo có hướng (rẽ trái/phải, cua trái/phải) -> không được lật ngang.
        fliplr=0.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
