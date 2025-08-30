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
#include <filesystem>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "rm_detector/armor_detector_opencv.hpp"
#include "rm_detector/pnp.hpp"
#include "rm_detector/number_classifier.hpp"
#include "rm_interfaces/msg/armor_cpp_info.hpp"
#include "rm_interfaces/msg/armors_cpp_msg.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

using namespace std::chrono;
using namespace cv;
using namespace std;

namespace DT46_VISION {

    class ArmorDetectorNode : public rclcpp::Node {
    public:
        ArmorDetectorNode() : Node("rm_detector") {
            // ---------------- 参数声明 ----------------
            // 先声明所有参数（从 detector_params.yaml 复制默认值）
            this->declare_parameter<std::string>("cls_model_file", "mlp.onnx");
            this->declare_parameter<bool>("cls_invert_binary", false);

            // 灯条过滤参数
            this->declare_parameter<int>("light_area_min", 5);
            this->declare_parameter<double>("light_h_w_ratio", 10.0);
            this->declare_parameter<int>("light_angle_min", -35);
            this->declare_parameter<int>("light_angle_max", 35);
            this->declare_parameter<double>("light_red_ratio", 2.0);
            this->declare_parameter<double>("light_blue_ratio", 2.0);

            // 装甲板匹配参数
            this->declare_parameter<double>("height_rate_tol", 1.3);
            this->declare_parameter<double>("height_multiplier_min", 1.8);
            this->declare_parameter<double>("height_multiplier_max", 3.0);

            // 图像处理参数
            this->declare_parameter<int>("binary_val", 120);
            this->declare_parameter<int>("detect_color", 2);
            this->declare_parameter<int>("display_mode", 0);

            // 其他参数
            this->declare_parameter<bool>("use_geometric_center", true);
            this->declare_parameter<int>("print_period_ms", 1000); // ms

            // ---------------- 参数读取 ----------------
            std::string cls_model_file   = get_required_param<std::string>("cls_model_file");
            cls_invert_binary_           = get_required_param<bool>("cls_invert_binary");

            // 拼接模型路径
            std::string pkg_share = ament_index_cpp::get_package_share_directory("rm_detector");
            cls_model_path_ = (std::filesystem::path(pkg_share) / "model" / cls_model_file).string();

            RCLCPP_INFO(this->get_logger(), "Classifier model file: %s", cls_model_file.c_str());
            RCLCPP_INFO(this->get_logger(), "Classifier model path: %s", cls_model_path_.c_str());

            Params params = {
                get_required_param<int>("light_area_min"),
                get_required_param<double>("light_h_w_ratio"),
                get_required_param<int>("light_angle_min"),
                get_required_param<int>("light_angle_max"),
                get_required_param<double>("light_red_ratio"),
                get_required_param<double>("light_blue_ratio"),
                get_required_param<double>("height_rate_tol"),
                get_required_param<double>("height_multiplier_min"),
                get_required_param<double>("height_multiplier_max")
            };

            detect_color_        = get_required_param<int>("detect_color");
            display_mode_        = get_required_param<int>("display_mode");
            binary_val_          = get_required_param<int>("binary_val");
            use_geometric_center_= get_required_param<bool>("use_geometric_center");
            print_period_ms_.store(get_required_param<int>("print_period_ms")); // 多线程访问

            // ---------------- Detector 初始化 ----------------
            detector_ = std::make_shared<ArmorDetector>(detect_color_, display_mode_, binary_val_, params);
            pnp_      = std::make_shared<PNP>(this->get_logger());

            reload_classifier_impl_(cls_model_path_, cls_invert_binary_);

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
            publisher_img_armor_  = this->create_publisher<sensor_msgs::msg::Image>("/detector/img_armor", 10);

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
        // ---------------- 工具函数 ----------------
        template<typename T>
        T get_required_param(const std::string& name) {
            T value;
            if (!this->get_parameter(name, value)) {
                RCLCPP_ERROR(this->get_logger(), "Required parameter '%s' not found in YAML!", name.c_str());
                throw std::runtime_error("Missing required parameter: " + name);
            }
            return value;
        }

        // ---------------- 分类器加载 ----------------
        void reload_classifier_impl_(const std::string& onnx_path, bool invert) {
            if (onnx_path.empty()) {
                classifier_.reset();
                if (detector_) detector_->set_classifier(nullptr);
                RCLCPP_WARN(this->get_logger(), "[Classifier] Disabled (cls_model_file is empty).");
                return;
            }
            try {
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

        // ---------------- 图像回调 ----------------
        void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
            cv_bridge::CvImageConstPtr cv_ptr;
            try { cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8); }
            catch (const std::exception& e) {
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                    "cv_bridge toCvShare failed: %s", e.what());
                return;
            }
            std::lock_guard<std::mutex> lock(frame_mtx_);
            latest_frame_ = cv_ptr;
        }

        // ---------------- 相机内参回调 ----------------
        void camera_info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
            std::lock_guard<std::mutex> lock(caminfo_mtx_);
            latest_caminfo_ = msg;
        }

        // ---------------- 主处理线程 ----------------
        void processing_loop() {
            using clock = std::chrono::steady_clock;

            const std::string GREEN = "\033[32m";
            const std::string CYAN  = "\033[96m";
            const std::string PINK  = "\033[38;5;218m";
            const std::string RESET = "\033[0m";

            auto last_print = clock::now();

            // 周期内状态
            bool had_detection_in_period = false;
            std::vector<rm_interfaces::msg::ArmorCppInfo> last_detected_armors;

            // FPS 统计
            int frame_count = 0;
            auto last_fps_time = clock::now();
            double current_fps = 0.0;

            while (rclcpp::ok() && running_.load()) {
                frame_count++;  // 每帧计数

                cv_bridge::CvImageConstPtr frame_ptr;
                { std::lock_guard<std::mutex> lock(frame_mtx_); frame_ptr = latest_frame_; }
                if (!frame_ptr) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }

                sensor_msgs::msg::CameraInfo::SharedPtr caminfo_ptr;
                { std::lock_guard<std::mutex> lock(caminfo_mtx_); caminfo_ptr = latest_caminfo_; }

                cv::Mat frame = frame_ptr->image;

                cv::Mat bin, result, img_armor;
                std::vector<Armor> armors;
                bool detection_error = false;
                try {
                    armors = detector_->detect_armors(frame);
                    std::tie(bin, result, img_armor) = detector_->display();
                } catch (const std::exception& e) {
                    RCLCPP_ERROR(this->get_logger(), "Detection error: %s", e.what());
                    detection_error = true;
                }

                rm_interfaces::msg::ArmorsCppMsg armors_msg;
                armors_msg.header.stamp = this->get_clock()->now();
                armors_msg.header.frame_id = "camera_frame";

                if (!detection_error && !armors.empty()) {
                    had_detection_in_period = true;  // 标记周期内有检测
                    last_detected_armors.clear();

                    for (const auto& armor : armors) {
                        rm_interfaces::msg::ArmorCppInfo armor_info;
                        armor_info.armor_id = armor.armor_id;
                        auto [dx, dy, dz] = pnp_->processArmorCorners(
                            caminfo_ptr, use_geometric_center_, frame, armor, armor.armor_id);
                        armor_info.dx = dx; armor_info.dy = dy; armor_info.dz = dz;

                        last_detected_armors.push_back(armor_info);
                        armors_msg.armors.push_back(armor_info);
                    }
                }

                publisher_armors_->publish(armors_msg);
                if (!bin.empty())
                    publisher_bin_img_->publish(*cv_bridge::CvImage(std_msgs::msg::Header(), "mono8", bin).toImageMsg());
                if (!result.empty())
                    publisher_result_img_->publish(*cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", result).toImageMsg());
                if (!img_armor.empty())
                    publisher_img_armor_->publish(*cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", img_armor).toImageMsg());

                // -------- 打印节流逻辑 --------
                int pp_ms = print_period_ms_.load();
                auto now = clock::now();
                if (pp_ms <= 0 ||
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count() >= pp_ms) {

                    // 计算 FPS
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time).count();
                    if (elapsed > 0) {
                        current_fps = frame_count * 1000.0 / elapsed;
                    }
                    frame_count = 0;
                    last_fps_time = now;

                    if (had_detection_in_period) {
                        for (const auto& armor_info : last_detected_armors) {
                            RCLCPP_INFO(this->get_logger(),
                                "发布 %sarmor_id:%s %s%d%s | %sdx:%s %.2f | %sdy:%s %.2f | %sdz:%s %.2f",
                                CYAN.c_str(), RESET.c_str(),
                                GREEN.c_str(), armor_info.armor_id, RESET.c_str(),
                                CYAN.c_str(), RESET.c_str(), armor_info.dx,
                                CYAN.c_str(), RESET.c_str(), armor_info.dy,
                                CYAN.c_str(), RESET.c_str(), armor_info.dz);
                        }
                    } else {
                        RCLCPP_INFO(this->get_logger(), "%sNo armors detected%s", PINK.c_str(), RESET.c_str());
                    }

                    // 打印 FPS (警告级别)
                    RCLCPP_WARN(this->get_logger(), "[FPS_detector] %.1f", current_fps);

                    // 重置周期状态
                    had_detection_in_period = false;
                    last_detected_armors.clear();
                    last_print = now;
                }
            }
        }



        // ---------------- 动态参数回调 ----------------
        rcl_interfaces::msg::SetParametersResult
        parameters_callback(const std::vector<rclcpp::Parameter>& parameters) {
            rcl_interfaces::msg::SetParametersResult result;
            result.successful = true; result.reason = "success";

            std::optional<std::string> new_model_file;
            std::optional<bool>        new_invert;

            for (const auto& param : parameters) {
                const auto& name = param.get_name();
                if (name == "cls_model_file") {
                    new_model_file = param.as_string();
                } else if (name == "cls_invert_binary") {
                    new_invert = param.as_bool();
                } else if (name == "light_area_min") { detector_->update_light_area_min(param.as_int());
                } else if (name == "light_h_w_ratio") { detector_->update_light_h_w_ratio(param.as_double());
                } else if (name == "light_angle_min") { detector_->update_light_angle_min(param.as_int());
                } else if (name == "light_angle_max") { detector_->update_light_angle_max(param.as_int());
                } else if (name == "light_red_ratio") { detector_->update_light_red_ratio(param.as_double());
                } else if (name == "light_blue_ratio") { detector_->update_light_blue_ratio(param.as_double());
                } else if (name == "height_rate_tol") { detector_->update_height_rate_tol(param.as_double());
                } else if (name == "height_multiplier_min") { detector_->update_height_multiplier_min(param.as_double());
                } else if (name == "height_multiplier_max") { detector_->update_height_multiplier_max(param.as_double());
                } else if (name == "binary_val") { detector_->update_binary_val(param.as_int());
                } else if (name == "detect_color") { detector_->update_detect_color(param.as_int());
                } else if (name == "display_mode") { detector_->update_display_mode(param.as_int());
                } else if (name == "print_period_ms") { print_period_ms_.store(param.as_int());
                }
            }

            if (new_model_file || new_invert) {
                std::string pkg_share = ament_index_cpp::get_package_share_directory("rm_detector");
                std::string new_path = cls_model_path_;
                if (new_model_file) {
                    new_path = (std::filesystem::path(pkg_share) / "model" / *new_model_file).string();
                }
                reload_classifier_impl_(new_path, new_invert.value_or(cls_invert_binary_));
            }
            return result;
        }

    private:
        // ---- ROS 通道
        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr        sub_image_;
        rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr   sub_camera_info_;
        rclcpp::Publisher<rm_interfaces::msg::ArmorsCppMsg>::SharedPtr  publisher_armors_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_result_img_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_img_armor_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr           publisher_bin_img_;
        rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr callback_handle_;

        // ---- 模块实例
        std::shared_ptr<ArmorDetector> detector_;
        std::shared_ptr<PNP>           pnp_;
        std::shared_ptr<NumberClassifier> classifier_;

        // ---- 参数缓存
        std::string cls_model_path_;
        bool cls_invert_binary_;
        int detect_color_;
        int display_mode_;
        int binary_val_;
        bool use_geometric_center_;
        std::atomic<int> print_period_ms_{1000};

        // ---- 缓存与线程
        std::mutex frame_mtx_;
        cv_bridge::CvImageConstPtr latest_frame_;
        std::mutex caminfo_mtx_;
        sensor_msgs::msg::CameraInfo::SharedPtr latest_caminfo_;
        std::thread worker_;
        std::atomic<bool> running_{false};
    };

}

int main(int argc, char *argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DT46_VISION::ArmorDetectorNode>());
    rclcpp::shutdown();
    return 0;
}