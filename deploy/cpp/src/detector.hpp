#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct Detection {
    int class_id;
    float score;
    cv::Rect box;  // toạ độ trên ảnh gốc
};

struct DetectorConfig {
    std::string model_path;
    float conf_threshold = 0.35f;
    float nms_threshold = 0.45f;
    int num_threads = 4;
};

// Bộ phát hiện YOLO (đầu ra dạng Ultralytics) chạy bằng ONNX Runtime trên CPU.
// Hỗ trợ cả hai kiểu đầu ra:
//   - YOLOv8/YOLO11: [1, 4 + nc, N]  (cần NMS)
//   - YOLO26 end-to-end: [1, N, 6] = x1, y1, x2, y2, score, class (không cần NMS)
class Detector {
public:
    explicit Detector(const DetectorConfig& cfg);

    std::vector<Detection> detect(const cv::Mat& bgr);

    int input_size() const { return input_w_; }

private:
    void preprocess(const cv::Mat& bgr);
    std::vector<Detection> postprocess_raw(const float* out, int64_t channels, int64_t anchors, const cv::Size& orig);
    std::vector<Detection> postprocess_e2e(const float* out, int64_t rows, int64_t cols, const cv::Size& orig);

    DetectorConfig cfg_;
    Ort::Env env_;
    Ort::Session session_{nullptr};
    Ort::MemoryInfo mem_info_;
    std::string input_name_, output_name_;
    int input_w_ = 640, input_h_ = 640;

    // Bộ đệm dùng lại giữa các khung hình để tránh cấp phát lặp.
    std::vector<float> blob_;
    cv::Mat letterbox_;
    float scale_ = 1.f;
    int pad_x_ = 0, pad_y_ = 0;
};
