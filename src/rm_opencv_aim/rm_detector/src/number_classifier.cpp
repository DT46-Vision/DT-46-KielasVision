#include "rm_detector/number_classifier.hpp"

namespace DT46_VISION {

    NumberClassifier::NumberClassifier(const std::string& onnx_path,
                                       cv::Size input_size)
        : input_size_(input_size)
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

        // 如果输入是三通道，则把它转成灰度并作为回退（优先支持单通道二值图）
        if (armor_img.channels() == 3) {
            cv::cvtColor(armor_img, gray_, cv::COLOR_BGR2GRAY);
            // 作为回退：如果你确实需要，也可以在这里做 Otsu 或 adaptive threshold，
            // 但既然你希望 detector 做预处理，这里不再做 Canny/OTSU（保持轻量）。
            cv::threshold(gray_, bin_, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        } else {
            // 假定 armor_img 已经是二值/边缘图（单通道）
            bin_ = armor_img;
        }

        // 确保大小和模型输入一致（如果不一致就 resize）
        if (bin_.size() != input_size_) {
            cv::resize(bin_, bin_, input_size_, 0, 0, cv::INTER_AREA);
        }

        // 转 blob 并归一化（单通道也可以直接 blobFromImage）
        cv::Mat blob = cv::dnn::blobFromImage(bin_, 1.0 / 255.0, input_size_, cv::Scalar(), false, false, CV_32F);

        // 前向推理
        net_.setInput(blob);
        cv::Mat logits = net_.forward(); // shape: 1xC

        // 取最大值/类别
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(logits, nullptr, &confidence, nullptr, &classIdPoint);

        out.class_id = classIdPoint.x;
        out.confidence = static_cast<float>(confidence);  // 仍然是 logit / 未 softmax 的值

        return out;
    }

} // namespace DT46_VISION
