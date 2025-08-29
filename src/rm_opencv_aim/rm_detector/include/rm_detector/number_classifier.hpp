#pragma once
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <utility>
#include <vector>

class NumberClassifier {
public:
    struct Result {
        int   class_id{-1};
        float confidence{0.f};   // softmax 概率
    };

    // onnx_path: 模型文件路径
    // input_size: 模型输入分辨率（默认 20x28：宽x高）
    // invert_binary: 是否在 Otsu 后取反（数据集前景/背景相反时可打开）
    explicit NumberClassifier(const std::string& onnx_path,
                              cv::Size input_size = {20, 28},
                              bool invert_binary = false);

    bool isLoaded() const { return !net_.empty(); }

    // 传入一张 ROI（BGR 或灰度均可）
    // 返回 (class_id, confidence)
    Result classify(const cv::Mat& armor_img);

    // 把预测结果画到 ROI 上（可选）
    void annotate(cv::Mat& roi, const Result& r,
                  const std::vector<std::string>& labels = {},
                  const cv::Point& org = {6, 22}) const;

    cv::Size inputSize() const { return input_size_; }

private:
    cv::dnn::Net net_;
    cv::Size     input_size_;
    bool         invert_binary_;
};

