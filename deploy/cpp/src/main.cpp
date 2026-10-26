// Hệ thống nhận diện và cảnh báo biển báo giao thông thời gian thực trên Raspberry Pi.
//
// Luồng xử lý:
//   [luồng camera] đọc liên tục, chỉ giữ khung hình mới nhất (không dồn bộ đệm -> không trễ)
//   [luồng chính]  letterbox -> ONNX Runtime -> hậu xử lý -> lọc thời gian -> cảnh báo/hiển thị/ghi log
#include "alert.hpp"
#include "detector.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Args {
    std::string model = "models/best.onnx";
    std::string labels = "models/classes.csv";
    std::string source = "0";
    std::string sound_dir;
    std::string log_path;
    float conf = 0.35f;
    int threads = 4;
    int width = 1280, height = 720;
    int window = 5, min_hits = 3, cooldown_ms = 4000;
    int bench = 0;
    bool headless = false;
};

struct Label {
    std::string en, vi;
};

static void usage() {
    std::cout << "Cách dùng: tsr [tuỳ chọn]\n"
                 "  --model PATH        mô hình ONNX (mặc định models/best.onnx)\n"
                 "  --labels PATH       file classes.csv (id,en,vi)\n"
                 "  --source SRC        0 = camera, đường dẫn video/ảnh, hoặc pipeline GStreamer\n"
                 "  --size WxH          độ phân giải camera (mặc định 1280x720)\n"
                 "  --conf F            ngưỡng tin cậy (0.35)\n"
                 "  --threads N         số luồng ONNX Runtime (4)\n"
                 "  --window N --hits K cảnh báo khi thấy >= K lần trong N khung (5, 3)\n"
                 "  --cooldown MS       thời gian chờ trước khi cảnh báo lại cùng biển (4000)\n"
                 "  --sound-dir DIR     phát DIR/<id>.wav (hoặc DIR/default.wav) bằng aplay\n"
                 "  --log PATH          ghi log CSV các cảnh báo\n"
                 "  --bench N           đo tốc độ suy luận N lần trên khung đầu tiên rồi thoát\n"
                 "  --headless          không mở cửa sổ hiển thị\n";
}

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("Thiếu giá trị cho " + k);
            return argv[++i];
        };
        if (k == "--model") a.model = next();
        else if (k == "--labels") a.labels = next();
        else if (k == "--source") a.source = next();
        else if (k == "--size") {
            std::string v = next();
            a.width = std::stoi(v.substr(0, v.find('x')));
            a.height = std::stoi(v.substr(v.find('x') + 1));
        } else if (k == "--conf") a.conf = std::stof(next());
        else if (k == "--threads") a.threads = std::stoi(next());
        else if (k == "--window") a.window = std::stoi(next());
        else if (k == "--hits") a.min_hits = std::stoi(next());
        else if (k == "--cooldown") a.cooldown_ms = std::stoi(next());
        else if (k == "--sound-dir") a.sound_dir = next();
        else if (k == "--log") a.log_path = next();
        else if (k == "--bench") a.bench = std::stoi(next());
        else if (k == "--headless") a.headless = true;
        else if (k == "-h" || k == "--help") {
            usage();
            std::exit(0);
        } else throw std::runtime_error("Tuỳ chọn không hợp lệ: " + k);
    }
    return a;
}

static std::unordered_map<int, Label> load_labels(const std::string& path) {
    std::unordered_map<int, Label> labels;
    std::ifstream f(path);
    std::string line;
    std::getline(f, line);  // bỏ dòng tiêu đề
    while (std::getline(f, line)) {
        std::stringstream ss(line);
        std::string id, en, vi;
        if (std::getline(ss, id, ',') && std::getline(ss, en, ',') && std::getline(ss, vi))
            labels[std::stoi(id)] = {en, vi};
    }
    return labels;
}

static bool is_image(const std::string& s) {
    auto ext = fs::path(s).extension().string();
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp";
}

// Đọc camera ở luồng riêng, luôn ghi đè khung hình mới nhất.
class FrameGrabber {
public:
    FrameGrabber(cv::VideoCapture& cap, bool live) : cap_(cap), live_(live) {
        if (live_) thread_ = std::thread([this] { loop(); });
    }
    ~FrameGrabber() {
        stop_ = true;
        if (thread_.joinable()) thread_.join();
    }

    // Lấy khung hình mới (chờ nếu chưa có). Trả về false khi hết nguồn.
    bool read(cv::Mat& out) {
        if (!live_) return cap_.read(out);
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return fresh_ || stop_; });
        if (!fresh_) return false;
        std::swap(out, frame_);
        fresh_ = false;
        return true;
    }

private:
    void loop() {
        cv::Mat f;
        while (!stop_) {
            if (!cap_.read(f) || f.empty()) {
                stop_ = true;
                break;
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                f.copyTo(frame_);
                fresh_ = true;
            }
            cv_.notify_one();
        }
        cv_.notify_all();
    }

    cv::VideoCapture& cap_;
    bool live_;
    std::thread thread_;
    std::mutex mu_;
    std::condition_variable cv_;
    cv::Mat frame_;
    bool fresh_ = false;
    std::atomic<bool> stop_{false};
};

static void play_sound(const std::string& dir, int cls) {
    if (dir.empty()) return;
    fs::path p = fs::path(dir) / (std::to_string(cls) + ".wav");
    if (!fs::exists(p)) p = fs::path(dir) / "default.wav";
    if (!fs::exists(p)) return;
    std::string cmd = "aplay -q \"" + p.string() + "\" &";
    if (std::system(cmd.c_str()) != 0) std::cerr << "Không phát được âm thanh\n";
}

static std::string timestamp() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    localtime_r(&t, &tm);
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return os.str();
}

static int run_bench(Detector& det, const cv::Mat& frame, int n) {
    for (int i = 0; i < 5; ++i) det.detect(frame);  // khởi động
    std::vector<double> ms;
    for (int i = 0; i < n; ++i) {
        auto t0 = Clock::now();
        det.detect(frame);
        ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    double mean = std::accumulate(ms.begin(), ms.end(), 0.0) / n;
    std::cout << std::fixed << std::setprecision(2) << "input=" << det.input_size() << " n=" << n
              << " mean=" << mean << "ms p50=" << ms[n / 2] << "ms p95=" << ms[n * 95 / 100]
              << "ms fps=" << 1000.0 / mean << "\n";
    return 0;
}

int main(int argc, char** argv) try {
    Args args = parse_args(argc, argv);
    auto labels = load_labels(args.labels);
    Detector det({args.model, args.conf, 0.45f, args.threads});
    std::cout << "Đã nạp mô hình " << args.model << " (input " << det.input_size() << "), " << labels.size() << " lớp\n";

    cv::Mat frame;
    if (is_image(args.source)) {
        frame = cv::imread(args.source);
        if (frame.empty()) throw std::runtime_error("Không đọc được ảnh " + args.source);
        if (args.bench > 0) return run_bench(det, frame, args.bench);
        for (const auto& d : det.detect(frame))
            std::cout << d.class_id << " " << labels[d.class_id].vi << " " << d.score << " " << d.box << "\n";
        return 0;
    }

    cv::VideoCapture cap;
    const bool is_cam = args.source.size() <= 2 && std::isdigit(static_cast<unsigned char>(args.source[0]));
    const bool is_gst = args.source.find('!') != std::string::npos;
    if (is_cam) {
        cap.open(std::stoi(args.source), cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FRAME_WIDTH, args.width);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, args.height);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    } else if (is_gst) {
        cap.open(args.source, cv::CAP_GSTREAMER);
    } else {
        cap.open(args.source);
    }
    if (!cap.isOpened()) throw std::runtime_error("Không mở được nguồn " + args.source);

    if (args.bench > 0) {
        if (!cap.read(frame)) throw std::runtime_error("Không đọc được khung hình");
        return run_bench(det, frame, args.bench);
    }

    std::ofstream log;
    if (!args.log_path.empty()) {
        const bool exists = fs::exists(args.log_path);
        log.open(args.log_path, std::ios::app);
        if (!exists) log << "time,class_id,name_en,name_vi,score\n";
    }

    AlertFilter filter(args.window, args.min_hits, std::chrono::milliseconds(args.cooldown_ms));
    FrameGrabber grabber(cap, is_cam || is_gst);
    double fps = 0;
    auto last = Clock::now();

    while (grabber.read(frame)) {
        auto dets = det.detect(frame);

        for (int cls : filter.update(dets)) {
            float score = 0;
            for (const auto& d : dets)
                if (d.class_id == cls) score = std::max(score, d.score);
            const Label& l = labels[cls];
            std::cout << "[CẢNH BÁO] " << l.vi << " (" << std::setprecision(2) << score << ")\n";
            play_sound(args.sound_dir, cls);
            if (log.is_open()) log << timestamp() << "," << cls << "," << l.en << "," << l.vi << "," << score << "\n" << std::flush;
        }

        auto now = Clock::now();
        const double dt = std::chrono::duration<double>(now - last).count();
        last = now;
        fps = fps == 0 ? 1.0 / dt : 0.9 * fps + 0.1 / dt;

        if (!args.headless) {
            for (const auto& d : dets) {
                cv::rectangle(frame, d.box, {0, 255, 0}, 2);
                // putText của OpenCV không hỗ trợ dấu tiếng Việt -> hiển thị tên tiếng Anh.
                std::ostringstream txt;
                txt << labels[d.class_id].en << " " << std::setprecision(2) << d.score;
                cv::putText(frame, txt.str(), d.box.tl() + cv::Point(0, -5), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
            }
            std::ostringstream f;
            f << "FPS " << std::fixed << std::setprecision(1) << fps;
            cv::putText(frame, f.str(), {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 0, 255}, 2);
            cv::imshow("Traffic Sign Recognition", frame);
            if ((cv::waitKey(1) & 0xFF) == 'q') break;
        }
    }
    std::cout << "FPS trung bình gần nhất: " << fps << "\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Lỗi: " << e.what() << "\n";
    return 1;
}
