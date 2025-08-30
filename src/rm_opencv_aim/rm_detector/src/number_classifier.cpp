#include "rm_detector/number_classifier.hpp"

namespace DT46_VISION {  // 添加命名空间

    NumberClassifier::NumberClassifier(const std::string& onnx_path,
                                    cv::Size input_size,
                                    bool invert_binary)
        : input_size_(input_size), invert_binary_(invert_binary)
    {
        net_ = cv::dnn::readNetFromONNX(onnx_path);
        if (net_.empty()) {
            throw std::runtime_error("NumberClassifier: failed to load ONNX model: " + onnx_path);
        }

        // 提前分配 buffer，避免 classify() 时频繁 malloc/free
        gray_.create(input_size_, CV_8UC1);
        bin_.create(input_size_, CV_8UC1);
        bin_f_.create(input_size_, CV_32F);
    }

    NumberClassifier::Result NumberClassifier::classify(const cv::Mat& armor_img)
    {
        Result out;
        if (armor_img.empty()) return out;

        // --- 灰度 ---
        if (armor_img.channels() == 3) {
            cv::cvtColor(armor_img, gray_, cv::COLOR_BGR2GRAY);
        } else {
            cv::resize(armor_img, gray_, input_size_);  // 保证大小一致
        }

        // --- Otsu 二值化 ---
        cv::threshold(gray_, bin_, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        if (invert_binary_) cv::bitwise_not(bin_, bin_);

        // --- resize (如果输入大小和模型不一致) ---
        if (bin_.size() != input_size_) {
            cv::resize(bin_, bin_, input_size_, 0, 0, cv::INTER_AREA);
        }

        // --- float 归一化 ---
        bin_.convertTo(bin_f_, CV_32F, 1.0 / 255.0);

        // --- blob ---
        cv::Mat blob = cv::dnn::blobFromImage(bin_f_, 1.0, input_size_, cv::Scalar(), false, false, CV_32F);

        // --- 前向推理 ---
        net_.setInput(blob);
        cv::Mat logits = net_.forward(); // 1xC

        // --- softmax ---
        cv::Mat row = logits.reshape(1, 1);
        const int C = row.cols;
        const float* p = row.ptr<float>(0);

        float max_logit = *std::max_element(p, p + C);
        std::vector<float> exps(C);
        float denom = 0.f;
        for (int i = 0; i < C; ++i) {
            exps[i] = std::exp(p[i] - max_logit);
            denom += exps[i];
        }

        int best_id = 0;
        float best_p = (denom > 0.f) ? (exps[0] / denom) : 0.f;
        for (int i = 1; i < C; ++i) {
            float prob = (denom > 0.f) ? (exps[i] / denom) : 0.f;
            if (prob > best_p) {
                best_p = prob;
                best_id = i;
            }
        }

        out.class_id = best_id;
        out.confidence = best_p;
        return out;
    }
    
}