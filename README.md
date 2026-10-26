# Nhận diện biển báo giao thông thời gian thực trên Raspberry Pi

Đề tài NCKH sinh viên 2026, Trường ĐH CNTT & TT – ĐH Thái Nguyên:
**"Xây dựng mô hình nhận diện biển báo giao thông thời gian thực trên phương tiện giao thông"**.

Hệ thống phát hiện 7 loại biển báo (theo phạm vi mục 11.2 thuyết minh) bằng YOLOv8n, chạy độc lập trên Raspberry Pi bằng C++ và ONNX Runtime,
phát cảnh báo âm thanh/hình ảnh cho người lái và ghi log.

| ID | Lớp | Biển báo |
|---|---|---|
| 0 | GioiHan40 | Giới hạn tốc độ 40 km/h |
| 1 | GioiHan50 | Giới hạn tốc độ 50 km/h |
| 2 | GioiHan60 | Giới hạn tốc độ 60 km/h |
| 3 | CamDiNguocChieu | Cấm đi ngược chiều |
| 4 | CamReTrai | Cấm rẽ trái |
| 5 | CamRePhai | Cấm rẽ phải |
| 6 | VongXuyen | Vòng xuyến |

## Kết quả chính

Mô hình YOLOv8n 7 lớp, train trên ảnh đường phố VN + biển GTSRB dán lên nền VN.
Đánh giá trên 713 ảnh đường phố Việt Nam thật (chi tiết: [docs/results.md](docs/results.md)):

| Mô hình | Dung lượng | mAP50 | CPU PC (ms) |
|---|---|---|---|
| `best.pt` (640) | 6,5 MB | 0,961 | – |
| `tsr_640.onnx` FP32 | 12,8 MB | 0,963 | 24,3 |
| **`tsr_416_int8.onnx`** (khuyến nghị cho Pi) | 6,1 MB | **0,906** | **9,5** |
| `tsr_320_int8.onnx` | 6,1 MB | 0,809 | 6,1 |

Thí nghiệm nguồn dữ liệu (bộ 5 lớp, xem docs/results.md §3): mô hình chỉ học GTSRB đạt mAP50 0,06 trên ảnh VN thật,
so với 0,99 khi có dữ liệu VN → bắt buộc dùng dữ liệu Việt Nam; GTSRB chỉ dùng làm dữ liệu bổ sung.

## Cấu trúc

```
tools/        check_dataset.py   kiểm tra nhãn, phân bố, ảnh trùng/rò rỉ giữa các tập
              prepare_dataset.py làm sạch bộ VN
              build_subset.py    tạo bộ 7 lớp (--set tsr7) / 5 lớp, ghép GTSRB bằng copy-paste
              scan_video.py      quét video đường phố, tìm và cắt đoạn có biển để demo
training/     train.py  export.py (ONNX FP32/INT8)  evaluate.py (mAP + tốc độ)
deploy/cpp/   chương trình C++ chạy trên Pi (ONNX Runtime + OpenCV)
deploy/python bản Python dùng để so sánh tốc độ
models/       best.pt, tsr_{320,416,640}[_int8].onnx, classes.csv, baseline_v1_128.onnx (bản cũ)
docs/         dataset.md  results.md  deploy_raspberry_pi.md
```

## Bắt đầu nhanh

**Trên Raspberry Pi (OS 64-bit):**

```bash
git clone https://github.com/huynhpham13790-rgb/traffic-sign-recognition
cd traffic-sign-recognition/deploy/cpp
bash scripts/setup_pi.sh
./build/tsr --model ../../models/tsr_416_int8.onnx --labels ../../models/classes.csv --source 0
```

**Huấn luyện lại trên PC:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
# chuẩn bị dữ liệu: docs/dataset.md §5
python training/train.py --data datasets/tsr7/data.yaml --model yolov8n.pt --name v8n_tsr7
```

## Tài liệu

- [Dữ liệu](docs/dataset.md): nguồn, làm sạch (loại 549 ảnh rò rỉ), copy-paste GTSRB, thống kê
- [Kết quả](docs/results.md): so sánh nguồn dữ liệu, đánh đổi kích thước/INT8, biểu đồ
- [Triển khai Raspberry Pi](docs/deploy_raspberry_pi.md): cài đặt, chạy, systemd, đo FPS

## Demo bằng video

Không cần giơ biển trước webcam: chạy trực tiếp trên video quay đường phố.

```bash
./build/tsr --model ../../models/tsr_416_int8.onnx --labels ../../models/classes.csv --source duong_pho.mp4
python tools/scan_video.py duong_pho.mp4 --weights models/best.pt --cut demo_clips   # cắt các đoạn có biển
```

Video thử nghiệm: kênh YouTube "Vietnam Silent Routes" (CC BY), không đưa vào repo.

## Dữ liệu & giấy phép

- Vietnam Traffic Sign Detection: <https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam> (CC BY 4.0)
- GTSRB: Stallkamp et al., *The German Traffic Sign Recognition Benchmark*, IJCNN 2011
- Ultralytics YOLO (AGPL-3.0)
