# Tổng quan các nghiên cứu liên quan và khoảng trống đề tài giải quyết

> Số liệu của các công trình dưới đây lấy từ tóm tắt/trang công bố của bài báo; khi trích dẫn chính thức
> trong báo cáo cần mở bản đầy đủ để kiểm tra lại.

## 1. Các hướng nghiên cứu chính

| Hướng | Nội dung | Ví dụ |
|---|---|---|
| Phân loại biển đã cắt sẵn | Ảnh chỉ chứa biển báo, bài toán chỉ còn phân loại | GTSRB (43 lớp, Đức) |
| Phát hiện biển trong cảnh đường phố | Tìm và phân loại biển nhỏ trong ảnh toàn cảnh | TT100K (Trung Quốc), Zalo AI 2020 / dữ liệu VN trên Kaggle |
| Mô hình nhẹ cho thiết bị nhúng | Rút gọn mạng (GhostNet, YOLO-tiny), FP16/INT8 | Jetson Nano/TX2, Raspberry Pi |
| Hệ thống hỗ trợ lái (ADAS) hoàn chỉnh | Camera → mô hình → cảnh báo người lái | Các đồ án trên Raspberry Pi |

## 2. Các công trình tương tự

| Công trình | Phần cứng | Kết quả công bố | Hạn chế còn lại |
|---|---|---|---|
| YOLOv5 cải tiến cho biển báo VN, FP32→FP16 trên Jetson (ETASR, 2023) [1] | Jetson | Đạt thời gian thực | Cần GPU Jetson (đắt hơn Pi nhiều lần); FP16 không dùng được trên CPU Pi |
| YOLOv5 cho biển nguy hiểm/cảnh báo VN, 11.138 ảnh (JESTEC, 2024) [2] | PC | Độ chính xác cao trên tập test | Chỉ nhóm biển cảnh báo; không triển khai nhúng, không thử video thực |
| Benchmark trên Jetson TX2, ảnh 1280×720 (RICE 2022) [3] | Jetson TX2 | Tỷ lệ phát hiện 79% (camera tĩnh), 89% (camera động) | Bỏ sót ~1/5 biển; phần cứng GPU |
| YOLO-tiny cho biển VN, 5.000 ảnh (ĐH Cần Thơ, 2019) [4] | CPU laptop | F1 > 92%, ~0,17 s/ảnh (~6 FPS) | Chưa đạt thời gian thực, chưa có cảnh báo |
| YOLOv8 + dữ liệu Kaggle VN trên Raspberry Pi, nối Google Maps [5] | Raspberry Pi | — | Chạy Python thuần; không công bố đánh giá trên video đường phố dài |
| YOLOv8 + GhostNet nhẹ trên Raspberry Pi (Luo và cs.) [6] | Raspberry Pi | mAP 93,5% (thời tiết xấu), 82,9% (tập gốc) | Dữ liệu Trung Quốc; độ trễ trên Pi còn lớn |
| ADAS trên Raspberry Pi 4 (Computación y Sistemas) [7] | Pi 4 | 0,4 FPS, mAP 53% | Quá chậm cho xe đang chạy |
| YOLOv8 phát hiện biển báo trên Raspberry Pi (Pekommas) [8] | Pi 4 | — | Tác giả nhận định tốc độ xử lý là nút thắt lớn nhất |
| So sánh Faster R-CNN / YOLOv4 (Heliyon, 2022) [9] | GPU | Faster R-CNN 6 FPS; YOLOv4 35 FPS, mAP 59,9% | Mô hình hai giai đoạn không đạt thời gian thực |

## 3. Những vấn đề chưa được giải quyết triệt để

1. **Tốc độ trên phần cứng rẻ.** Bài nào cũng đạt thời gian thực thì đều dùng Jetson (GPU). Các bài dùng Pi thì
   chỉ đạt 0,4–vài FPS hoặc không công bố FPS. Hầu hết chạy bằng Python và đọc camera chung luồng với suy
   luận, nên hình bị trễ dần. Phiên bản cũ của chính đề tài này cũng gặp lỗi đó.
2. **Khoảng cách giữa dữ liệu chuẩn và đường thật.** GTSRB là bài toán phân loại: biển chiếm gần hết ảnh [10].
   Mô hình đạt rất cao trên GTSRB nhưng không chứng minh được gì khi chạy trên ảnh toàn cảnh. Biển báo VN có
   khác biệt so với biển Đức (phông chữ, biển phụ, cách treo).
3. **Báo nhầm từ vật thể giống biển.** Các tổng quan ghi nhận lỗi phổ biến nhất là nhầm biển quảng cáo và hình
   dạng tương tự [10, 11]. Phần lớn bài chỉ báo cáo mAP trên ảnh tĩnh, không đếm số lần báo nhầm khi chạy
   video dài, trong khi với người lái thì báo nhầm liên tục còn tệ hơn bỏ sót.
4. **Rò rỉ dữ liệu.** Các bộ dữ liệu VN công khai thường được tăng cường (lật, làm mờ) trước khi chia tập,
   nên cùng một biển nằm cả trong train và test, làm kết quả bị thổi phồng. Ít bài nói rõ đã kiểm tra trùng lặp.
5. **Lật ngang ảnh khi tăng cường.** Lật ngang biến biển "cấm rẽ trái" thành "cấm rẽ phải", nên nhãn bị sai.
   Đây là lỗi tăng cường mặc định của YOLO (`fliplr=0.5`) mà nhiều đồ án không để ý.
6. **Thiếu phần hệ thống.** Ít công trình có đủ chuỗi: lọc cảnh báo theo thời gian, âm thanh tiếng Việt,
   ghi log, tự khởi động, và có thể tái lập (mã nguồn và mô hình công khai).

## 4. Đề tài này đóng góp thêm gì

| Vấn đề (§3) | Cách đề tài xử lý | Minh chứng |
|---|---|---|
| 1. Tốc độ trên Pi | ONNX Runtime + C++17, luồng camera riêng chỉ giữ khung mới nhất, INT8 tĩnh giữ đầu Detect ở FP32 (mAP50 0,906 so với 0,734 nếu lượng tử hoá toàn bộ), có bảng đánh đổi 320/416/640 | `results.md` §2; FPS trên Pi: §6 (đang đo) |
| 2. Khoảng cách dữ liệu | So sánh có kiểm soát: chỉ GTSRB, chỉ VN, VN + GTSRB dán lên nền VN; luôn đánh giá trên ảnh đường phố VN | `results.md` §3 |
| 3. Báo nhầm | Đánh giá trên ~115 phút video đường phố Hà Nội, đếm từng sự kiện bằng mắt (92/133 đúng). Chỉ ra cụ thể nguồn nhầm (biển hạn chế chiều cao, logo Hà Nội). Bộ lọc K/N khung + thời gian chờ | `results.md` §5 |
| 4. Rò rỉ dữ liệu | Loại trùng bằng MD5 + dHash trước khi chia tập | `dataset.md` |
| 5. Lật ảnh | Tắt `fliplr`, tách riêng lớp cấm rẽ trái/phải; kết quả không nhầm giữa hai lớp | ma trận nhầm lẫn |
| 6. Hệ thống | Cảnh báo âm thanh, CSV log, systemd, mã nguồn và mô hình công khai trên GitHub | `deploy_raspberry_pi.md` |

**Giới hạn đề tài thừa nhận:** chỉ 7 loại biển; chưa thử ban đêm và mưa; số liệu FPS trên Pi chưa đo; báo nhầm
vòng xuyến và biển tốc độ trên video còn cao. Hướng tiếp theo: thêm ảnh âm khó (hard negative), ngưỡng riêng
cho từng lớp, mở rộng số lớp từ GTSRB.

## 5. Các ngách có thể phát triển tiếp

- **Ảnh âm khó cho bối cảnh VN:** biển hạn chế chiều cao/tải trọng, logo, biển quảng cáo tròn đỏ.
- **Theo dõi đối tượng (tracking)** giữa các khung để mỗi biển chỉ cảnh báo đúng một lần và đọc ổn định hơn.
- **Đọc biển phụ** ("trừ xe buýt", khung giờ). Đây là đặc thù VN mà dữ liệu nước ngoài không có.
- **Ghép GPS + bản đồ biển báo:** tạo bản đồ biển từ log nhiều chuyến đi, phát hiện biển bị che hoặc mất.
- **Đánh giá trong điều kiện khó:** đêm, mưa, ngược sáng, camera hành trình rẻ tiền.

## Tài liệu tham khảo

1. ETASR — Vietnamese traffic sign recognition with YOLOv5 on Jetson. https://etasr.com/index.php/ETASR/article/download/6658/3428/24748
2. JESTEC Vol.19 Iss.1 (2024) — Vietnamese danger/warning sign detection. https://jestec.taylors.edu.my/Vol%2019%20Issue%201%20February%20%202024/19_1_10.pdf
3. RICE 2022, ACSIS Vol.33. https://annals-csis.org/proceedings/rice2022/drp/05.html
4. ĐH Cần Thơ (2019). https://qldiem.ctu.edu.vn/ql/docgia/tapchiquocte-2019/baibao-72396.html
5. i-us.ru — YOLOv8 Vietnamese traffic signs, Raspberry Pi. https://i-us.ru/index.php/ius/article/view/16350/15394
6. Luo và cs. — Lightweight real-time traffic sign detection based on YOLOv8. https://scholar.hit.edu.cn/en/publications/a-novel-lightweight-real-time-traffic-sign-detection-method-based/
7. Computación y Sistemas — ADAS traffic elements. https://biblat.unam.mx/en/revista/computacion-y-sistemas/articulo/automatic-detection-of-vehicular-traffic-elements-based-on-deep-learning-for-advanced-driving-assistance-systems
8. Pekommas — Traffic Signs Detection System Based on YOLOv8 using Raspberry Pi. https://journal.mmtc.ac.id/index.php/pekommas/article/view/5832
9. Heliyon (Dec 2022). https://doaj.org/article/6c7de714a9de436e958016bbe2975783
10. Traffic Sign Recognition in Autonomous Driving: Dataset, Benchmark, and Field Experiment (arXiv 2603.23034). https://arxiv.org/pdf/2603.23034
11. An overview of traffic sign detection and classification methods (Saadna & Behloul). https://researchgate.net/profile/Yassmina-Saadna/publication/317776221
