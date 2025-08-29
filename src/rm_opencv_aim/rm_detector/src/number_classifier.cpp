#include "rm_detector/number_classifier.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <stdexcept>
#include <cmath>
#include <sstream>

NumberClassifier::NumberClassifier(const std::string& onnx_path,
                                   cv::Size input_size,
                                   bool invert_binary)
    : input_size_(input_size), invert_binary_(invert_binary)
{
    net_ = cv::dnn::readNetFromONNX(onnx_path);
    if (net_.empty()) {
        throw std::runtime_error("NumberClassifier: failed to load ONNX model: " + onnx_path);
    }
    // 可按需设置后端/设备：
    // net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    // net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

NumberClassifier::Result NumberClassifier::classify(const cv::Mat& armor_img)
{
    Result out;

    if (armor_img.empty()) {
        return out; // 返回 class_id=-1, confidence=0
    }

    // 1) 灰度
    cv::Mat gray;
    if (armor_img.channels() == 3) {
        cv::cvtColor(armor_img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = armor_img;
    }

    // 2) Otsu 二值化（必要时可反相）
    cv::Mat bin;
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (invert_binary_) cv::bitwise_not(bin, bin);

    // 3) 尺寸匹配到模型输入（默认 20x28，宽x高）
    if (bin.size() != input_size_) {
        cv::resize(bin, bin, input_size_, 0, 0, cv::INTER_AREA);
    }

    // 4) 归一化到 [0,1]，并构造 NCHW blob (1x1xHxW)
    cv::Mat bin_f;
    bin.convertTo(bin_f, CV_32F, 1.0 / 255.0);
    cv::Mat blob = cv::dnn::blobFromImage(
        bin_f,            // 输入图
        1.0,              // 已经做过 /255，这里不再缩放
        input_size_,      // (W,H)
        cv::Scalar(),     // 均值
        false,            // 不做通道翻转
        false,            // 不做裁剪
        CV_32F
    );

    // 5) 前向
    net_.setInput(blob);
    cv::Mat logits = net_.forward(); // 形如 (1 x C)

    // 6) softmax
    cv::Mat row = logits.reshape(1, 1); // 1xC
    const int C = row.cols;
    const float* p = row.ptr<float>(0);

    float max_logit = p[0];
    for (int i = 1; i < C; ++i) max_logit = std::max(max_logit, p[i]);

    std::vector<float> exps(C);
    float denom = 0.f;
    for (int i = 0; i < C; ++i) {
        exps[i] = std::exp(p[i] - max_logit);
        denom  += exps[i];
    }

    int   best_id = 0;
    float best_p  = (denom > 0.f) ? (exps[0] / denom) : 0.f;
    for (int i = 1; i < C; ++i) {
        float prob = (denom > 0.f) ? (exps[i] / denom) : 0.f;
        if (prob > best_p) {
            best_p = prob;
            best_id = i;
        }
    }

    out.class_id   = best_id;
    out.confidence = best_p;
    return out;
}

void NumberClassifier::annotate(cv::Mat& roi, const Result& r,
                                const std::vector<std::string>& labels,
                                const cv::Point& org) const
{
    std::ostringstream ss;
    if (!labels.empty() && r.class_id >= 0 && r.class_id < (int)labels.size()) {
        ss << labels[r.class_id] << " ";
    } else {
        ss << "#" << r.class_id << " ";
    }
    ss.setf(std::ios::fixed);
    ss.precision(1);
    ss << (r.confidence * 100.0f) << "%";

    cv::putText(roi, ss.str(), org,
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}
