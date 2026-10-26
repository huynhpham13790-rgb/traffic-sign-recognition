#pragma once

#include "detector.hpp"

#include <algorithm>
#include <chrono>
#include <deque>
#include <string>
#include <unordered_map>
#include <vector>

// Lọc theo thời gian: một biển báo chỉ được xác nhận khi xuất hiện ở ít nhất
// `min_hits` trong `window` khung hình gần nhất, sau đó chờ `cooldown` trước khi
// cảnh báo lại cùng loại. Giảm cảnh báo giả do nhận nhầm ở một khung hình lẻ.
class AlertFilter {
public:
    AlertFilter(int window, int min_hits, std::chrono::milliseconds cooldown)
        : window_(window), min_hits_(min_hits), cooldown_(cooldown) {}

    // Trả về danh sách class_id cần phát cảnh báo ở khung hình hiện tại.
    std::vector<int> update(const std::vector<Detection>& dets) {
        std::vector<int> present;
        for (const auto& d : dets) present.push_back(d.class_id);
        history_.push_back(present);
        if (static_cast<int>(history_.size()) > window_) history_.pop_front();

        std::unordered_map<int, int> hits;
        for (const auto& frame : history_) {
            std::vector<int> uniq = frame;
            std::sort(uniq.begin(), uniq.end());
            uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
            for (int c : uniq) ++hits[c];
        }

        const auto now = std::chrono::steady_clock::now();
        std::vector<int> fire;
        for (const auto& [cls, n] : hits) {
            if (n < min_hits_) continue;
            auto it = last_alert_.find(cls);
            if (it != last_alert_.end() && now - it->second < cooldown_) continue;
            last_alert_[cls] = now;
            fire.push_back(cls);
        }
        return fire;
    }

private:
    int window_, min_hits_;
    std::chrono::milliseconds cooldown_;
    std::deque<std::vector<int>> history_;
    std::unordered_map<int, std::chrono::steady_clock::time_point> last_alert_;
};
