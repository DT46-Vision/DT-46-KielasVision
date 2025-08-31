#include "rm_detector/number_classifier.hpp"

namespace DT46_VISION {

    NumberClassifier::NumberClassifier(const std::string& onnx_path,
                                       cv::Size input_size,
                                       bool invert_binary)
        : input_size_(input_size), invert_binary_(invert_binary)
    {
        net_ = cv::dnn::readNetFromONNX(onnx_path);
        if (net_.empty()) {
            throw std::runtime_error("NumberClassifier: failed to load ONNX model: " + onnx_path);
        }

        // ---- DNN 加速配置 ----
        try {
            net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        } catch (...) {
            try {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_OPENCL);
            } catch (...) {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
        }

        // 提前分配 buffer，避免 classify() 时频繁 malloc/free
        gray_.create(input_size_, CV_8UC1);
        bin_.create(input_size_, CV_8UC1);
    }

    NumberClassifier::Result NumberClassifier::classify(const cv::Mat& armor_img)
    {
        Result out;
        if (armor_img.empty()) return out;

        // --- 灰度 ---
        if (armor_img.channels() == 3) {
            cv::cvtColor(armor_img, gray_, cv::COLOR_BGR2GRAY);
        } else if (armor_img.size() != input_size_) {
            cv::resize(armor_img, gray_, input_size_, 0, 0, cv::INTER_AREA);
        } else {
            gray_ = armor_img;
        }

        // --- Otsu 二值化 ---
        cv::threshold(gray_, bin_, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (invert_binary_) cv::bitwise_not(bin_, bin_);

        // --- resize (保证输入大小和模型一致) ---
        if (bin_.size() != input_size_) {
            cv::resize(bin_, bin_, input_size_, 0, 0, cv::INTER_AREA);
        }

        // --- 直接转 blob (归一化) ---
        cv::Mat blob = cv::dnn::blobFromImage(bin_, 1.0 / 255.0, input_size_, cv::Scalar(), false, false, CV_32F);

        // --- 前向推理 ---
        net_.setInput(blob);
        cv::Mat logits = net_.forward(); // shape: 1xC

        // --- 取最大值/类别 ---
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(logits, nullptr, &confidence, nullptr, &classIdPoint);

        out.class_id = classIdPoint.x;
        out.confidence = static_cast<float>(confidence);  // 注意：这是未归一化概率, 不是 softmax ！！

        return out;
    }

} // namespace DT46_VISION
