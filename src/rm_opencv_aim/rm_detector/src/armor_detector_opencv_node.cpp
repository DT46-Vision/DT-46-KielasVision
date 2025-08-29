#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <std_msgs/msg/header.hpp>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <optional>
#include <string>
#include "rm_detector/armor_detector_opencv.hpp"
#include "rm_detector/pnp.hpp"
#include "rm_detector/number_classifier.hpp"
#include "rm_interfaces/msg/armor_cpp_info.hpp"
#include "rm_interfaces/msg/armors_cpp_msg.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

using namespace std::chrono;
using namespace cv;
using namespace std;

static std::string default_model_path() {
    try {
        const std::string pkg_share = ament_index_cpp::get_package_share_directory("rm_detector");
        return pkg_share + "/model/mlp.onnx";
    } catch (...) { return std::string(); }
}

class ArmorDetectorNode : public rclcpp::Node {
public:
    ArmorDetectorNode() : Node("rm_detector") {
        // ---------------- 参数声明（给出可用的默认模型路径，避免默认被禁用） ----------------
        this->declare_parameter<std::string>("cls_model_path", default_model_path());
        this->declare_parameter<bool>("cls_invert_binary", false);

        this->declare_parameter<int>("light_area_min", 5);
        this->declare_parameter<double>("light_h_w_ratio", 3.f);
        this->declare_parameter<int>("light_angle_min", -35);
        this->declare_parameter<int>("light_angle_max", 35);
        this->declare_parameter<double>("light_red_ratio", 2.0f);
        this->declare_parameter<double>("light_blue_ratio", 2.0f);

        this->declare_parameter<double>("height_rate_tol", 1.3f);
        this->declare_parameter<double>("height_multiplier_min", 1.8f);
        this->declare_parameter<double>("height_multiplier_max", 3.0f);

        this->declare_parameter<int>("binary_val", 120);
        this->declare_parameter<int>("detect_color", 2);
        this->declare_parameter<int>("display_mode", 0);
        this->declare_parameter<bool>("use_geometric_center", true);
        this->declare_parameter<int>("max_processing_fps", 0);

        // ---------------- Detector 初始化 ----------------
        Light_params light_params = {
            static_cast<int>(this->get_parameter("light_area_min").as_int()),
            this->get_parameter("light_h_w_ratio").as_double(),
            static_cast<int>(this->get_parameter("light_angle_min").as_int()),
            static_cast<int>(this->get_parameter("light_angle_max").as_int()),
            this->get_parameter("light_red_ratio").as_double(),
            this->get_parameter("light_blue_ratio").as_double(),

            this->get_parameter("height_rate_tol").as_double(),
            this->get_parameter("height_multiplier_min").as_double(),
            this->get_parameter("height_multiplier_max").as_double()
        };

        const int detect_color = static_cast<int>(this->get_parameter("detect_color").as_int());
        const int display_mode = static_cast<int>(this->get_parameter("display_mode").as_int());
        const int binary_val   = static_cast<int>(this->get_parameter("binary_val").as_int());

        detector_ = std::make_shared<ArmorDetector>(detect_color, display_mode, binary_val, light_params);
        pnp_      = std::make_shared<PNP>(this->get_logger());

        // 先按当前参数加载一次分类器
        reload_classifier_from_params_();

        // 动态参数回调
        callback_handle_ = this->add_on_set_parameters_callback(
            std::bind(&ArmorDetectorNode::parameters_callback, this, std::placeholders::_1));

        // ---------------- 订阅/发布 ----------------
        auto sensor_qos = rclcpp::SensorDataQoS();
        sub_image_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", sensor_qos, std::bind(&ArmorDetectorNode::image_callback, this, std::placeholders::_1));

        sub_camera_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", 10, std::bind(&ArmorDetectorNode::camera_info_callback, this, std::placeholders::_1));

        publisher_armors_     = this->create_publisher<rm_interfaces::msg::ArmorsCppMsg>("/detector/armors_info", 10);
        publisher_result_img_ = this->create_publisher<sensor_msgs::msg::Image>("/detector/result", 10);
        publisher_bin_img_    = this->create_publisher<sensor_msgs::msg::Image>("/detector/bin_img", 10);
        publisher_armor_img_  = this->create_publisher<sensor_msgs::msg::Image>("/detector/armor_img", 10);

        // 工作线程
        running_.store(true);
        worker_ = std::thread(&ArmorDetectorNode::processing_loop, this);

        RCLCPP_INFO(this->get_logger(), "Armor Detector Node has been started.");
    }

    ~ArmorDetectorNode() override {
        running_.store(false);
        if (worker_.joinable()) worker_.join();
    }

private:
    // ---------------- 分类器加载 ----------------
    void reload_classifier_from_params_() {
        const auto onnx_path = this->get_parameter("cls_model_path").as_string();
        const bool invert    = this->get_parameter("cls_invert_binary").as_bool();
        reload_classifier_impl_(onnx_path, invert);
    }

    void reload_classifier_impl_(const std::string& onnx_path, bool invert) {
        if (onnx_path.empty()) {
            classifier_.reset();
            if (detector_) detector_->set_classifier(nullptr);
            RCLCPP_WARN(this->get_logger(), "[Classifier] Disabled (cls_model_path is empty).");
            return;
        }
        try {
            // 训练里 ROI=20x28；构造时指定输入尺寸，内部会做 Otsu + 可选反相 + 归一化 + blob
            classifier_ = std::make_shared<NumberClassifier>(onnx_path, cv::Size(20, 28), invert);
            if (detector_) detector_->set_classifier(classifier_);
            RCLCPP_INFO(this->get_logger(), "[Classifier] Loaded ONNX: %s | invert=%s",
                        onnx_path.c_str(), invert ? "true" : "false");
        } catch (const std::exception& e) {
            classifier_.reset();
            if (detector_) detector_->set_classifier(nullptr);
            RCLCPP_ERROR(this->get_logger(), "[Classifier] Load failed: %s", e.what());
        }
    }

    // ---------------- 回调/线程 ----------------
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv_bridge::CvImageConstPtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                  "cv_bridge toCvShare failed: %s", e.what());
            return;
        }
        std::lock_guard<std::mutex> lock(frame_mtx_);
        latest_frame_ = cv_ptr;
    }

    void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        std::lock_guard<std::mutex> lock(caminfo_mtx_);
        latest_caminfo_ = msg;
    }

    void processing_loop() {
        using clock = std::chrono::steady_clock;
        auto window_start = clock::now();
        size_t processed_in_window = 0;
        bool saw_frame_in_window = false;
        bool detected_in_window  = false;

        const std::string GREEN = "\033[32m";
        const std::string CYAN  = "\033[96m";
        const std::string RESET = "\033[0m";

        int max_fps = this->get_parameter("max_processing_fps").as_int();
        std::chrono::microseconds min_period(0);
        if (max_fps > 0) {
            min_period = std::chrono::microseconds(1'000'000 / max_fps);
            RCLCPP_INFO(this->get_logger(), "Max processing FPS = %d", max_fps);
        }

        while (rclcpp::ok() && running_.load()) {
            auto loop_start = clock::now();

            cv_bridge::CvImageConstPtr frame_ptr;
            {
                std::lock_guard<std::mutex> lock(frame_mtx_);
                frame_ptr = latest_frame_;
            }
            if (!frame_ptr) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }

            sensor_msgs::msg::CameraInfo::SharedPtr caminfo_ptr;
            {
                std::lock_guard<std::mutex> lock(caminfo_mtx_);
                caminfo_ptr = latest_caminfo_;
            }

            cv::Mat frame = frame_ptr->image;

            cv::Mat bin, result, armor_img;
            std::vector<Armor> armors;
            bool detection_error = false;
            try {
                armors = detector_->detect_armors(frame);
                std::tie(bin, result, armor_img) = detector_->display();
            } catch (const std::exception& e) {
                RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
                detection_error = true;
            }

            rm_interfaces::msg::ArmorsCppMsg armors_msg;
            armors_msg.header.stamp = this->get_clock()->now();
            armors_msg.header.frame_id = "camera_frame";

            if (!detection_error && !armors.empty()) {
                detected_in_window = true;
                bool use_geometric_center = this->get_parameter("use_geometric_center").as_bool();
                for (const auto& armor : armors) {
                    rm_interfaces::msg::ArmorCppInfo armor_info;
                    armor_info.armor_id = armor.armor_id;

                    auto [dx, dy, dz] = pnp_->processArmorCorners(
                        caminfo_ptr, use_geometric_center, frame, armor, armor.armor_id);

                    armor_info.dx = dx; armor_info.dy = dy; armor_info.dz = dz;

                    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                        "发布 %sarmor_id:%s %s%d%s | %sdx:%s %.2f | %sdy:%s %.2f | %sdz:%s %.2f",
                        CYAN.c_str(), RESET.c_str(),
                        GREEN.c_str(), armor_info.armor_id, RESET.c_str(),
                        CYAN.c_str(), RESET.c_str(), armor_info.dx,
                        CYAN.c_str(), RESET.c_str(), armor_info.dy,
                        CYAN.c_str(), RESET.c_str(), armor_info.dz);
                    armors_msg.armors.push_back(armor_info);
                }
            }

            publisher_armors_->publish(armors_msg);
            if (!bin.empty()) {
                sensor_msgs::msg::Image bin_img_msg =
                *cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", bin).toImageMsg();
                publisher_bin_img_->publish(bin_img_msg);
            }
            if (!result.empty()) {
                sensor_msgs::msg::Image result_img_msg =
                *cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", result).toImageMsg();
                publisher_result_img_->publish(result_img_msg);
            }
            if (!armor_img.empty()) {
                sensor_msgs::msg::Image armor_img_msg =
                *cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", armor_img).toImageMsg();
                publisher_armor_img_->publish(armor_img_msg);
            }

            processed_in_window++; saw_frame_in_window = true;

            auto now = clock::now();
            if (now - window_start >= 1s) {
                if (saw_frame_in_window && !detected_in_window) {
                    RCLCPP_INFO(this->get_logger(), "No armors detected.");
                }
                if (processed_in_window > 0) {
                    double secs = duration<double>(now - window_start).count();
                    double fps  = processed_in_window / secs;
                    RCLCPP_WARN(this->get_logger(), "[FPS_detect] %.1f (processed)", fps);
                }
                window_start = now;
                processed_in_window = 0;
                saw_frame_in_window = false;
                detected_in_window  = false;
            }

            if (min_period.count() > 0) {
                auto loop_cost = clock::now() - loop_start;
                if (loop_cost < min_period) std::this_thread::sleep_for(min_period - loop_cost);
            }
        }
    }

    // ---------------- 动态参数回调 ----------------
    rcl_interfaces::msg::SetParametersResult
    parameters_callback(const std::vector<rclcpp::Parameter>& parameters) {
        rcl_interfaces::msg::SetParametersResult result; result.successful = true; result.reason = "success";
        std::optional<std::string> new_model_path;
        std::optional<bool>        new_invert;

        for (const auto& param : parameters) {
            const auto& name = param.get_name();
            if (name == "cls_model_path") {
                new_model_path = param.as_string();
            } else if (name == "cls_invert_binary") {
                new_invert = param.as_bool();
            } else if (name == "light_area_min") {
                detector_->update_light_area_min(param.as_int());
            } else if (name == "light_h_w_ratio") {
                detector_->update_light_h_w_ratio(param.as_double());
            } else if (name == "light_angle_min") {
                detector_->update_light_angle_min(param.as_int());
            } else if (name == "light_angle_max") {
                detector_->update_light_angle_max(param.as_int());
            } else if (name == "light_red_ratio") {
                detector_->update_light_red_ratio(param.as_double());
            } else if (name == "light_blue_ratio") {
                detector_->update_light_blue_ratio(param.as_double());
            } else if (name == "height_rate_tol") {
                detector_->update_height_rate_tol(param.as_double());
            } else if (name == "height_multiplier_min") {
                detector_->update_height_multiplier_min(param.as_double());
            } else if (name == "height_multiplier_max") {
                detector_->update_height_multiplier_max(param.as_double());
            } else if (name == "binary_val") {
                detector_->update_binary_val(param.as_int());
            } else if (name == "detect_color") {
                detector_->update_detect_color(param.as_int());
            } else if (name == "display_mode") {
                detector_->update_display_mode(param.as_int());
            } else if (name == "use_geometric_center" || name == "max_processing_fps") {
                // 读取时在循环里生效
            }
        }

        if (new_model_path || new_invert) {
            const std::string path = new_model_path ? *new_model_path
                                                    : this->get_parameter("cls_model_path").as_string();
            const bool inv = new_invert ? *new_invert
                                        : this->get_parameter("cls_invert_binary").as_bool();
            reload_classifier_impl_(path, inv);
        }
        return result;
    }

private:
    // ---- ROS 通道
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr        sub_image_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr   sub_camera_info_;
    rclcpp::Publisher<rm_interfaces::msg::ArmorsCppMsg>::SharedPtr  publisher_armors_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_result_img_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_armor_img_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_bin_img_;
    rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr callback_handle_;

    // ---- 模块实例
    std::shared_ptr<ArmorDetector> detector_;
    std::shared_ptr<PNP>           pnp_;
    std::shared_ptr<NumberClassifier> classifier_;

    // ---- 缓存与线程
    std::mutex frame_mtx_;
    cv_bridge::CvImageConstPtr latest_frame_;
    std::mutex caminfo_mtx_;
    sensor_msgs::msg::CameraInfo::SharedPtr latest_caminfo_;
    std::thread worker_;
    std::atomic<bool> running_{false};
};

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}