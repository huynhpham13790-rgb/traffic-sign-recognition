# Triển khai trên Raspberry Pi

## 1. Vì sao phiên bản cũ bị giật (FPS thấp)

| Nguyên nhân | Phiên bản cũ | Phiên bản mới |
|---|---|---|
| Ngôn ngữ vòng lặp xử lý | Python thuần (tiền/hậu xử lý bằng vòng lặp Python/NumPy) | C++17, bộ đệm cấp phát một lần, NMS viết tay |
| Đọc camera | Cách làm phổ biến là đọc cùng luồng với suy luận → khung hình dồn trong bộ đệm, hình ảnh bị trễ | Luồng camera riêng, chỉ giữ khung **mới nhất**, `CAP_PROP_BUFFERSIZE=1` |
| Runtime | (tuỳ phiên bản) | ONNX Runtime, tối ưu đồ thị `ORT_ENABLE_ALL`, 4 luồng |
| Cảnh báo | Mỗi khung hình đều có thể kích hoạt | Lọc theo thời gian K/N khung + thời gian chờ → không báo dồn dập, ít báo nhầm |

Lưu ý: mô hình cũ dùng input 128×128 nên rất nhanh nhưng không thấy biển nhỏ/ở xa (xem `dataset.md` §3).
Mô hình mới (7 lớp) dùng 320–416, chọn kích thước theo bảng đánh đổi tốc độ/độ chính xác trong `results.md`.

## 2. Phần cứng & hệ điều hành

- Raspberry Pi 4 (≥ 4 GB) hoặc Pi 5, **Raspberry Pi OS 64-bit (Bookworm)** — bắt buộc 64-bit vì ONNX Runtime chỉ phát hành bản `aarch64`.
- Camera USB (UVC) hoặc Camera Module (CSI), loa qua jack 3.5 mm/USB, màn hình HDMI/DSI (tuỳ chọn).
- Tản nhiệt + quạt: Pi 4 bị hạ xung khi > 80 °C, FPS sẽ tụt sau vài phút nếu không tản nhiệt.

## 3. Cài đặt và biên dịch

```bash
git clone https://github.com/huynhpham13790-rgb/traffic-sign-recognition
cd traffic-sign-recognition/deploy/cpp
bash scripts/setup_pi.sh          # cài OpenCV, tải ONNX Runtime 1.20.1 aarch64, build
./build/tsr --help
```

Đã kiểm tra biên dịch và chạy trong container `debian:bookworm` arm64 (cùng nền với Raspberry Pi OS).

## 4. Chạy

```bash
cd deploy/cpp
# camera USB, hiển thị lên màn hình
./build/tsr --model ../../models/tsr_416_int8.onnx --labels ../../models/classes.csv --source 0

# chạy nền không màn hình, có âm thanh và ghi log
./build/tsr --model ../../models/tsr_416_int8.onnx --labels ../../models/classes.csv \
    --source 0 --headless --sound-dir sounds --log alerts.csv

# Camera Module (CSI) qua libcamera + GStreamer
./build/tsr --model ../../models/tsr_416_int8.onnx --labels ../../models/classes.csv --source \
  "libcamerasrc ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"

# khuyến nghị cho Pi 4: chỉ nhận diện dải ảnh chứa biển (xem results.md §6.1)
./build/tsr --model ../../models/tsr_160x512_int8.onnx --labels ../../models/classes.csv     --source 0 --roi 0,0.15,1,0.55 --sound-dir sounds

# "zoom": xử lý xen kẽ nửa trái/phải của dải ROI, biển lớn hơn ~1,4 lần, cảnh báo sớm hơn ~0,8 s (results.md §6.2)
./build/tsr --model ../../models/tsr_224x416_int8.onnx --labels ../../models/classes.csv \n    --source 0 --roi 0,0.15,1,0.55 --tiles --sound-dir sounds

# trình diễn bằng video trên màn hình nhỏ: đúng tốc độ thật, toàn màn hình, ghi lại để dựng video có tiếng
./build/tsr --model ../../models/tsr_160x512_int8.onnx --labels ../../models/classes.csv --source clip.mp4     --roi 0,0.15,1,0.55 --realtime --fullscreen --sound-dir sounds --record demo.avi
# trên máy tính: python tools/mux_alerts.py demo.avi --sounds deploy/cpp/sounds --out demo.mp4

# đo tốc độ suy luận thuần
./build/tsr --model ../../models/tsr_416_int8.onnx --labels ../../models/classes.csv --source test.jpg --bench 100
```

Âm thanh: `deploy/cpp/sounds/<id>.wav` có sẵn giọng đọc tiếng Việt cho 7 lớp (ví dụ `2.wav` = "Chú ý, tốc độ tối đa
sáu mươi cây số một giờ"); chương trình gọi `aplay` không chặn. Pi 4 xuất tiếng qua jack 3.5 mm
(`~/.asoundrc`: `defaults.pcm.card 2`); loa trần 3 W 8 Ω cần mạch khuếch đại (vd. PAM8403), tai nghe/loa có
nguồn thì cắm thẳng.

Tham số cảnh báo: `--window 5 --hits 2` (mặc định) nghĩa là biển phải xuất hiện ở ≥ 2 trong 5 khung gần nhất;
`--cooldown 4000` nghĩa là cùng một biển chỉ cảnh báo lại sau 4 giây.

Log CSV (`--log`): `time,class_id,name_en,name_vi,score` — phục vụ nội dung "ghi log, phân tích dữ liệu" của đề tài.

## 5. Tự khởi động cùng Pi (systemd)

```ini
# /etc/systemd/system/tsr.service
[Unit]
Description=Traffic sign recognition
After=sound.target

[Service]
WorkingDirectory=/home/pi/traffic-sign-recognition/deploy/cpp
ExecStart=/home/pi/traffic-sign-recognition/deploy/cpp/build/tsr --model ../../models/tsr_416_int8.onnx --labels ../../models/classes.csv --source 0 --headless --sound-dir sounds --log alerts.csv
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now tsr
```

## 6. So sánh Python và C++

`deploy/python/detect.py` dùng cùng mô hình ONNX, cùng tiền/hậu xử lý, để đo đối chứng:

```bash
python3 deploy/python/detect.py --model models/tsr_416_int8.onnx --labels models/classes.csv --source test.jpg --bench 100
./deploy/cpp/build/tsr          --model models/tsr_416_int8.onnx --labels models/classes.csv --source test.jpg --bench 100
```

Kết quả đo trên Pi 4: `results.md` §6.

## 7. Gợi ý tăng tốc thêm

- Giảm `--size` camera (640x480) — giảm chi phí đọc/resize.
- Đã mặc định dùng INT8; nếu bỏ sót biển nhỏ/xa, thử `tsr_416.onnx` (FP32) hoặc hạ `--conf` xuống 0.25.
- Pi 5 nhanh hơn Pi 4 khoảng 2–3 lần với cùng mô hình.
