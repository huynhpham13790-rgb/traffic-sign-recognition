# Dữ liệu

## 1. Lớp biển báo nghiên cứu

Theo phạm vi đề tài (mục 11.2 thuyết minh), hệ thống tập trung vào 5 loại biển phổ biến, rủi ro cao:

| ID | Tên lớp | Biển báo | Mã QCVN 41:2019 |
|---|---|---|---|
| 0 | `GioiHan60` | Giới hạn tốc độ tối đa 60 km/h | P.127 |
| 1 | `VongXuyen` | Nơi giao nhau chạy theo vòng xuyến | R.303 |
| 2 | `DuongCam` | Đường cấm | P.101 |
| 3 | `CamDiNguocChieu` | Cấm đi ngược chiều | P.102 |
| 4 | `Stop` | Dừng lại | R.122 |

Thứ tự lớp giữ nguyên như hệ thống phiên bản trước để tương thích.

## 2. Nguồn dữ liệu

### 2.1. Vietnam Traffic Sign Detection (nguồn chính)

- Nguồn: <https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam> (CC BY 4.0)
- 10.157 ảnh 640×640 chụp đường phố Việt Nam, 82 lớp, nhãn YOLO.
- Ánh xạ sang 5 lớp: `12 → GioiHan60`, `56 → VongXuyen`, `0 → CamDiNguocChieu`, `64 → Stop`.
  Bộ này **không có** lớp "Đường cấm".
- Ảnh chỉ chứa các biển khác (cấm đỗ, giới hạn 50, ...) vẫn được giữ, bỏ nhãn → đóng vai trò mẫu âm,
  giúp mô hình không báo nhầm biển tương tự (ví dụ 50 ↔ 60).

### 2.2. GTSRB (dữ liệu cũ của nhóm)

- German Traffic Sign Recognition Benchmark [Stallkamp et al., 2011], file `archive.zip` của phiên bản trước.
- Là ảnh **crop** từng biển (15–250 px), dùng cho bài toán phân loại, không phải ảnh cảnh đường phố.
- Lớp dùng: `3 → GioiHan60`, `40 → VongXuyen`, `15 → DuongCam`, `17 → CamDiNguocChieu`, `14 → Stop`
  (trùng với 5 lớp nhãn `labels_my-project-name_2025-06-14...zip` nhóm đã gán trước đây).

Để dùng được cho bài toán phát hiện, crop GTSRB được **dán (copy-paste augmentation)** lên ảnh nền VN
không chứa lớp đích (`tools/build_subset.py`):

- kích thước biển ngẫu nhiên 16–96 px (gần với phân bố biển thật trong ảnh 640),
- vị trí ở 5–65% chiều cao vùng ảnh thật (bỏ qua viền đen letterbox), không chồng nhau,
- chỉnh độ sáng theo vùng nền, làm mềm viền,
- 1–3 biển/ảnh; lấy cách 3 khung hình trong mỗi track GTSRB để giảm ảnh gần trùng.

Ảnh tổng hợp **chỉ đưa vào tập train**. Tập val/test giữ 100% ảnh thật VN để đánh giá công bằng.

## 3. Làm sạch dữ liệu (`tools/check_dataset.py`, `tools/prepare_dataset.py`)

Kiểm tra bộ VN gốc cho thấy:

| Vấn đề | Số lượng | Xử lý |
|---|---|---|
| Ảnh trùng tuyệt đối (MD5) | 89 | Bỏ bản sau |
| Ảnh val/test gần trùng ảnh train (dHash 64-bit giống hệt) | 258 (val) + 291 (test) | Bỏ khỏi val/test → tránh rò rỉ dữ liệu |
| Dòng nhãn sai định dạng | 1 (`train/8380.txt`) | Bỏ dòng |

Nếu không loại các ảnh rò rỉ, ~30% ảnh val/test thực chất đã xuất hiện trong train và mAP sẽ bị đánh giá cao hơn thực tế.

Đặc điểm kích thước: diện tích box tương đối p10 / p50 / p90 = 0,026% / 0,27% / 4,5% ảnh,
tức cạnh biển trung vị ≈ 34 px và 10% biển chỉ ≈ 10 px trên ảnh 640. Đây là lý do mô hình cũ với input
128×128 gần như không phát hiện được biển ở xa (biển 34 px chỉ còn ≈ 7 px).

## 4. Thống kê bộ dữ liệu cuối

| Tập | Thí nghiệm A (`tsr5_vn`) | Thí nghiệm B (`tsr5_vn_gtsrb`) |
|---|---|---|
| train | 8.067 ảnh | 8.067 + 684 ảnh tổng hợp |
| val | 739 ảnh thật | 739 ảnh thật |
| test | 713 ảnh thật | 713 ảnh thật |
| test_gtsrb | – | 354 ảnh tổng hợp từ GTSRB **Test** (nền từ ảnh test VN) |

Số đối tượng theo lớp (train A → B / val / test):

| Lớp | train A | train B | val | test |
|---|---|---|---|---|
| GioiHan60 | 579 | 979 | 83 | 57 |
| VongXuyen | 233 | 353 | 13 | 9 |
| DuongCam | 0 | 210 | 0 | 0 |
| CamDiNguocChieu | 441 | 811 | 55 | 57 |
| Stop | 105 | 365 | 9 | 12 |

### Thí nghiệm C (`tsr5_gtsrb`) — chỉ dùng GTSRB (cách làm của phiên bản cũ)

- train: 955 ảnh tổng hợp từ GTSRB Train (lấy cách 2 khung/track), val: 129 ảnh tổng hợp từ các track
  GTSRB khác hẳn train (track id chia hết cho 10), nền là ảnh VN không chứa lớp đích, không dùng nhãn VN.
- test: cùng 713 ảnh thật VN như A, B (+ đánh giá thêm trên `test_gtsrb`).
- Mục đích: định lượng mức chênh lệch miền (domain gap) giữa biển báo Đức và biển báo/bối cảnh Việt Nam.

```bash
python tools/build_subset.py datasets/vn_tsd/clean datasets/tsr5_gtsrb --gtsrb datasets/gtsrb --gtsrb-only --per-class 1000
```

**Hạn chế:** tập test thật có ít mẫu `VongXuyen`, `Stop` và không có `DuongCam` → kết quả từng lớp này có
độ bất định lớn. Cần bổ sung ảnh tự chụp tại Việt Nam (đặc biệt biển "Đường cấm") theo mục 11.1 thuyết minh.

## 5. Tái tạo dữ liệu

```bash
# 1. tải bộ VN
mkdir -p datasets/vn_tsd && cd datasets/vn_tsd
wget https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam/resolve/main/traffic-sign-detection-vietnam.zip -O data.zip
wget https://huggingface.co/datasets/star092304/Traffic-sign-detection-VietNam/resolve/main/classid.csv
unzip -q data.zip -d raw && cd ../..

# 2. GTSRB: giải nén archive.zip (Kaggle "GTSRB - German Traffic Sign") vào datasets/gtsrb

# 3. làm sạch + tạo bộ 5 lớp
python tools/check_dataset.py datasets/vn_tsd/raw/dataset
python tools/prepare_dataset.py datasets/vn_tsd/raw/dataset datasets/vn_tsd/clean --classes datasets/vn_tsd/classid.csv
python tools/build_subset.py datasets/vn_tsd/clean datasets/tsr5_vn
python tools/build_subset.py datasets/vn_tsd/clean datasets/tsr5_vn_gtsrb --gtsrb datasets/gtsrb --gtsrb-test
```

Mọi bước dùng `seed=0` nên kết quả tái tạo được.
