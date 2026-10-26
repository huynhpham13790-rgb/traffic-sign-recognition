#include "detector.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace {

float iou(const cv::Rect& a, const cv::Rect& b) {
    const float inter = static_cast<float>((a & b).area());
    const float uni = static_cast<float>(a.area() + b.area()) - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

// NMS theo từng lớp: chỉ loại các box trùng nhau cùng lớp.
std::vector<Detection> nms(std::vector<Detection> dets, float thr) {
    std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.score > b.score; });
    std::vector<Detection> keep;
    std::vector<bool> removed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        keep.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (!removed[j] && dets[j].class_id == dets[i].class_id && iou(dets[i].box, dets[j].box) > thr)
                removed[j] = true;
        }
    }
    return keep;
}

}  // namespace

Detector::Detector(const DetectorConfig& cfg)
    : cfg_(cfg),
      env_(ORT_LOGGING_LEVEL_WARNING, "tsr"),
      mem_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(cfg_.num_threads);
    opts.SetInterOpNumThreads(1);
    opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(env_, cfg_.model_path.c_str(), opts);

    Ort::AllocatorWithDefaultOptions alloc;
    input_name_ = session_.GetInputNameAllocated(0, alloc).get();
    output_name_ = session_.GetOutputNameAllocated(0, alloc).get();

    auto shape = session_.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0)
        throw std::runtime_error("Mô hình cần input tĩnh dạng [1,3,H,W]");
    input_h_ = static_cast<int>(shape[2]);
    input_w_ = static_cast<int>(shape[3]);
    blob_.resize(static_cast<size_t>(3) * input_h_ * input_w_);
    letterbox_ = cv::Mat(input_h_, input_w_, CV_8UC3);
}

void Detector::preprocess(const cv::Mat& bgr) {
    scale_ = std::min(static_cast<float>(input_w_) / bgr.cols, static_cast<float>(input_h_) / bgr.rows);
    const int new_w = static_cast<int>(std::round(bgr.cols * scale_));
    const int new_h = static_cast<int>(std::round(bgr.rows * scale_));
    pad_x_ = (input_w_ - new_w) / 2;
    pad_y_ = (input_h_ - new_h) / 2;

    letterbox_.setTo(cv::Scalar(114, 114, 114));
    cv::resize(bgr, letterbox_(cv::Rect(pad_x_, pad_y_, new_w, new_h)), cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // HWC BGR uint8 -> CHW RGB float32 [0,1]
    const size_t plane = static_cast<size_t>(input_h_) * input_w_;
    float* r = blob_.data();
    float* g = r + plane;
    float* b = g + plane;
    constexpr float k = 1.f / 255.f;
    for (int y = 0; y < input_h_; ++y) {
        const uchar* p = letterbox_.ptr<uchar>(y);
        const size_t row = static_cast<size_t>(y) * input_w_;
        for (int x = 0; x < input_w_; ++x, p += 3) {
            b[row + x] = p[0] * k;
            g[row + x] = p[1] * k;
            r[row + x] = p[2] * k;
        }
    }
}

std::vector<Detection> Detector::detect(const cv::Mat& bgr) {
    preprocess(bgr);
    const std::array<int64_t, 4> in_shape{1, 3, input_h_, input_w_};
    Ort::Value input = Ort::Value::CreateTensor<float>(mem_info_, blob_.data(), blob_.size(), in_shape.data(), in_shape.size());

    const char* in_names[] = {input_name_.c_str()};
    const char* out_names[] = {output_name_.c_str()};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);

    const auto out_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    const float* out = outputs[0].GetTensorData<float>();
    if (out_shape.size() == 3 && out_shape[2] == 6)
        return postprocess_e2e(out, out_shape[1], out_shape[2], bgr.size());
    return postprocess_raw(out, out_shape[1], out_shape[2], bgr.size());
}

std::vector<Detection> Detector::postprocess_raw(const float* out, int64_t channels, int64_t anchors, const cv::Size& orig) {
    const int64_t nc = channels - 4;
    std::vector<Detection> dets;
    for (int64_t a = 0; a < anchors; ++a) {
        int best = 0;
        float best_score = out[4 * anchors + a];
        for (int64_t c = 1; c < nc; ++c) {
            const float s = out[(4 + c) * anchors + a];
            if (s > best_score) {
                best_score = s;
                best = static_cast<int>(c);
            }
        }
        if (best_score < cfg_.conf_threshold) continue;

        float cx = out[a], cy = out[anchors + a], w = out[2 * anchors + a], h = out[3 * anchors + a];
        // Một số bản export (export normalize) trả toạ độ trong [0,1] -> đổi về pixel của input.
        if (cx <= 1.f && cy <= 1.f && w <= 1.f && h <= 1.f) {
            cx *= input_w_;
            w *= input_w_;
            cy *= input_h_;
            h *= input_h_;
        }
        const float x1 = (cx - w / 2 - pad_x_) / scale_;
        const float y1 = (cy - h / 2 - pad_y_) / scale_;
        cv::Rect box(cv::Point(static_cast<int>(x1), static_cast<int>(y1)), cv::Size(static_cast<int>(w / scale_), static_cast<int>(h / scale_)));
        dets.push_back({best, best_score, box & cv::Rect(0, 0, orig.width, orig.height)});
    }
    return nms(std::move(dets), cfg_.nms_threshold);
}

std::vector<Detection> Detector::postprocess_e2e(const float* out, int64_t rows, int64_t cols, const cv::Size& orig) {
    std::vector<Detection> dets;
    for (int64_t i = 0; i < rows; ++i) {
        const float* d = out + i * cols;
        if (d[4] < cfg_.conf_threshold) continue;
        const float x1 = (d[0] - pad_x_) / scale_, y1 = (d[1] - pad_y_) / scale_;
        const float x2 = (d[2] - pad_x_) / scale_, y2 = (d[3] - pad_y_) / scale_;
        cv::Rect box(cv::Point(static_cast<int>(x1), static_cast<int>(y1)), cv::Point(static_cast<int>(x2), static_cast<int>(y2)));
        dets.push_back({static_cast<int>(d[5]), d[4], box & cv::Rect(0, 0, orig.width, orig.height)});
    }
    return dets;
}
