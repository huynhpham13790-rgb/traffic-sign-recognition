# Kết quả thực nghiệm

Cấu hình chung: YOLOv8n (3,0 M tham số), khởi tạo từ trọng số COCO, 80 epoch, ảnh 640, batch 32,
SGD + cosine LR, `fliplr=0` (không lật ngang vì biển có hướng), `seed=0`, GPU RTX 5060 8 GB.
Số liệu chi tiết: `docs/results/*.csv`.

## 1. So sánh nguồn dữ liệu huấn luyện

**Tập test VN thật** (713 ảnh, 135 biển thuộc 4 lớp; không có "Đường cấm"):

| Mô hình | Dữ liệu train | P | R | mAP50 | mAP50-95 | AP50 GH60 | AP50 Vòng xuyến | AP50 Cấm ngược chiều | AP50 Stop |
|---|---|---|---|---|---|---|---|---|---|
| Phiên bản cũ (input 128) | GTSRB | 0,245 | 0,229 | 0,229 | 0,198 | 0,000 | 0,003 | 0,000 | 0,915 |
| C | chỉ GTSRB (tổng hợp) | 0,110 | 0,158 | 0,059 | 0,038 | 0,002 | 0,049 | 0,102 | 0,085 |
| A | chỉ VN | 0,960 | 0,974 | **0,992** | **0,783** | 0,994 | 0,995 | 0,984 | 0,995 |
| **B** | **VN + GTSRB** | **0,977** | 0,970 | 0,990 | 0,774 | 0,984 | 0,995 | 0,987 | 0,995 |

**Tập `test_gtsrb`** (354 ảnh: crop GTSRB *Test* dán lên ảnh test VN, đủ 5 lớp):

| Mô hình | mAP50 | mAP50-95 | AP50 Đường cấm |
|---|---|---|---|
| Phiên bản cũ (input 128) | 0,000 | 0,000 | 0,000 |
| C – chỉ GTSRB | **0,994** | **0,964** | 0,995 |
| A – chỉ VN | 0,275 | 0,200 | 0,000 |
| **B – VN + GTSRB** | 0,987 | 0,951 | 0,984 |

**Nhận xét**

1. **Chênh lệch miền dữ liệu (domain gap) rất lớn.** Mô hình chỉ học GTSRB (C) gần như hoàn hảo trên biển
   kiểu Đức nhưng chỉ đạt mAP50 = 0,06 trên ảnh đường phố VN. Ngược lại, mô hình chỉ học VN (A) chỉ đạt 0,28
   trên biển GTSRB. Điều này giải thích vì sao hệ thống cũ hoạt động khi giơ biển trước webcam nhưng
   không phù hợp với đường thật: dùng GTSRB "làm nền tảng" là chưa đủ, bắt buộc phải có dữ liệu Việt Nam.
2. **Kết hợp hai nguồn (B) giữ được cả hai**: trên ảnh VN tương đương A (0,990 so với 0,992 — chênh lệch nằm
   trong sai số do tập test nhỏ), đồng thời nhận được lớp "Đường cấm" (không có trong dữ liệu VN) và biển
   kiểu GTSRB. → **Chọn B làm mô hình triển khai.**
3. Mô hình cũ với input 128 không phát hiện được biển nhỏ trong cảnh rộng (biển 34 px chỉ còn ~7 px);
   nó chỉ nhận tốt Stop vì các ảnh Stop trong tập test phần lớn là biển cỡ lớn, chụp gần.

**Lưu ý về độ tin cậy:** mAP50 ≈ 0,99 trên tập test VN cao hơn nhiều công bố khác. Nguyên nhân có thể là tập test
có ít mẫu (Vòng xuyến: 9, Stop: 12) và ảnh trong bộ dữ liệu có thể được trích từ cùng đoạn video với ảnh train
(dù đã loại các ảnh gần trùng ở mức dHash). Cần kiểm chứng thêm bằng video tự quay trên đường ở Thái Nguyên.

## 2. Đánh đổi tốc độ – độ chính xác khi triển khai (mô hình B)

Tập test VN thật. Thời gian suy luận đo bằng ONNX Runtime 1.31, 4 luồng CPU **trên PC** (không phải Pi), chỉ để so sánh tương đối giữa các cấu hình.

| Mô hình | Input | Kiểu số | Dung lượng | mAP50 | mAP50-95 | CPU PC (ms) | Tăng tốc so với 640 FP32 |
|---|---|---|---|---|---|---|---|
| `tsr_640.onnx` | 640 | FP32 | 12,8 MB | **0,989** | **0,773** | 35,4 | 1,0× |
| `tsr_640_int8.onnx` | 640 | INT8 | 3,6 MB | 0,913 | 0,649 | 20,6 | 1,7× |
| `tsr_416.onnx` | 416 | FP32 | 12,7 MB | 0,932 | 0,705 | 15,4 | 2,3× |
| **`tsr_416_int8.onnx`** | 416 | INT8 | 3,5 MB | **0,936** | 0,686 | **9,5** | **3,7×** |
| `tsr_320.onnx` | 320 | FP32 | 12,7 MB | 0,811 | 0,623 | 9,0 | 3,9× |
| `tsr_320_int8.onnx` | 320 | INT8 | 3,5 MB | 0,794 | 0,612 | 6,1 | 5,8× |

**Nhận xét**

- Kích thước input ảnh hưởng mạnh đến độ chính xác do biển trong ảnh nhỏ: 640 → 320 làm mAP50 giảm từ 0,99 xuống 0,81,
  chủ yếu ở biển giới hạn 60 và cấm ngược chiều nhìn từ xa.
- Lượng tử hoá INT8 (tĩnh, QDQ, hiệu chỉnh trên 200 ảnh train) giảm dung lượng 3,6× và tăng tốc 1,5–1,7×.
  Ở 416, mAP50 gần như giữ nguyên (0,932 → 0,936), nhưng recall giảm (0,88 → 0,79): biển rất nhỏ có điểm tin cậy
  thấp hơn. Ví dụ một biển 60 cỡ 13 px được bản FP32 nhận (0,65) nhưng bản INT8 bỏ sót ở ngưỡng 0,35.
- **Khuyến nghị cho Raspberry Pi:** `tsr_416_int8.onnx` (cân bằng tốt nhất). Khi cần độ nhạy tối đa và Pi đủ mạnh (Pi 5): `tsr_416.onnx`.

## 3. Đường cong huấn luyện và ma trận nhầm lẫn

| A – chỉ VN | B – VN + GTSRB | C – chỉ GTSRB |
|---|---|---|
| ![](images/curves_tsr5_vn.png) | ![](images/curves_tsr5_vn_gtsrb.png) | ![](images/curves_tsr5_gtsrb.png) |

Ma trận nhầm lẫn (chuẩn hoá) của mô hình B trên tập val: ![](images/confusion_tsr5_vn_gtsrb.png)

Ví dụ dự đoán trên tập val: ![](images/val_pred_example.jpg)

## 4. Đo trên Raspberry Pi (cần bổ sung)

Điền sau khi chạy trên thiết bị thật (lệnh: `docs/deploy_raspberry_pi.md` §6):

| Thiết bị | Mô hình | Python (ms / FPS) | C++ (ms / FPS) | FPS toàn hệ thống (camera + hiển thị) | Nhiệt độ CPU |
|---|---|---|---|---|---|
| Pi 4 / Pi 5 | `tsr_416_int8.onnx` | | | | |
| Pi 4 / Pi 5 | `tsr_416.onnx` | | | | |
| Pi 4 / Pi 5 | `baseline_v1_128.onnx` (cũ) | | | | |

## 5. Tái tạo

```bash
python training/train.py --data datasets/tsr5_vn_gtsrb/data.yaml --model yolov8n.pt --name v8n_tsr5_vn_gtsrb
python training/export.py --weights runs/v8n_tsr5_vn_gtsrb/weights/best.pt --imgsz 320 416 640 --int8 \
    --calib datasets/tsr5_vn_gtsrb/train/images --out models
python training/evaluate.py --data datasets/tsr5_vn_gtsrb/data.yaml --split test \
    --models models/tsr_416.onnx models/tsr_416_int8.onnx --out docs/results/onnx_vn_test.csv
```
