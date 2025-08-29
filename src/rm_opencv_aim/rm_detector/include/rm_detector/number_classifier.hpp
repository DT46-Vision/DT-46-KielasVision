#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

class NumberClassifier {
public:
    struct Result {
        int   class_id{-1};
        float confidence{0.f};   // softmax 概率
    };

    explicit NumberClassifier(const std::string& onnx_path,
                              cv::Size input_size = {20, 28},
                              bool invert_binary = false);

    bool isLoaded() const { return !net_.empty(); }

    Result classify(const cv::Mat& armor_img);
    cv::Size inputSize() const { return input_size_; }

private:
    cv::dnn::Net net_;
    cv::Size     input_size_;
    bool         invert_binary_;

    // --- 内存复用 buffer ---
    cv::Mat gray_;
    cv::Mat bin_;
    cv::Mat bin_f_;
};
