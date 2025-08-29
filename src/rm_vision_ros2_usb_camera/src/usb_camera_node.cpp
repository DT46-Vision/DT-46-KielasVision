#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <camera_info_manager/camera_info_manager.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <chrono>
#include <mutex>

// Linux V4L2 控制：用于在 OpenCV 不生效时兜底设置相机控制项
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

using namespace std::chrono_literals;

class UsbCameraNode : public rclcpp::Node {
public:
  UsbCameraNode() : Node("usb_camera_node") {
    // ========= 1) 声明参数（有默认值） =========
    camera_index_ = declare_parameter<int>("camera_index", 0);
    device_path_  = declare_parameter<std::string>("device_path", ""); // 若非空优先使用

    frame_id_     = declare_parameter<std::string>("frame_id", "camera_optical_frame");
    camera_name_  = declare_parameter<std::string>("camera_name", "usb_camera");

    width_  = declare_parameter<int>("frame_width", 640);
    height_ = declare_parameter<int>("frame_height", 480);
    fps_    = declare_parameter<double>("fps", 30.0);

    pixel_format_ = declare_parameter<std::string>("pixel_format", "MJPG"); // MJPG/YUYV/H264/GREY 等

    use_sensor_data_qos_ = declare_parameter<bool>("use_sensor_data_qos", false);

    // 图像几何处理
    flip_horizontal_ = declare_parameter<bool>("flip_horizontal", false);
    flip_vertical_   = declare_parameter<bool>("flip_vertical", false);
    rotate_degree_   = declare_parameter<int>("rotate_degree", 0); // 0/90/180/270

    // 画质/曝光/白平衡/对焦等（取决于设备支持）
    exposure_auto_      = declare_parameter<bool>("exposure_auto", true);
    exposure_time_      = declare_parameter<int>("exposure_time", 800); // 仅手动曝光时生效
    gain_               = declare_parameter<int>("gain", 16);
    brightness_         = declare_parameter<int>("brightness", -1);
    contrast_           = declare_parameter<int>("contrast", -1);
    saturation_         = declare_parameter<int>("saturation", -1);
    sharpness_          = declare_parameter<int>("sharpness", -1);
    gamma_              = declare_parameter<int>("gamma", -1);
    white_balance_auto_ = declare_parameter<bool>("white_balance_auto", true);
    white_balance_temp_ = declare_parameter<int>("white_balance_temperature", -1);
    focus_auto_         = declare_parameter<bool>("focus_auto", true);
    focus_absolute_     = declare_parameter<int>("focus_absolute", -1);
    power_line_freq_    = declare_parameter<int>("power_line_frequency", 1); // 0:Disabled,1:50Hz,2:60Hz

    camera_info_url_    = declare_parameter<std::string>("camera_info_url", "");

    // ========= 2) CameraInfoManager（加载标定） =========
    cinfo_mgr_ = std::make_shared<camera_info_manager::CameraInfoManager>(this, camera_name_, camera_info_url_);
    if (!camera_info_url_.empty()) {
      if (cinfo_mgr_->loadCameraInfo(camera_info_url_)) {
        RCLCPP_INFO(get_logger(), "Loaded camera_info from: %s", camera_info_url_.c_str());
      } else {
        RCLCPP_WARN(get_logger(), "Failed to load camera_info from: %s", camera_info_url_.c_str());
      }
    }

    // ========= 3) 发布器 =========
    if (use_sensor_data_qos_) {
      image_pub_ = image_transport::create_publisher(this, "image_raw",
                  rclcpp::SensorDataQoS().get_rmw_qos_profile());
      cinfo_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info",
                  rclcpp::SensorDataQoS());
    } else {
      image_pub_ = image_transport::create_publisher(this, "image_raw");
      cinfo_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera_info", 10);
    }

    // ========= 4) 打开相机并应用参数 =========
    openCamera();

    // ========= 5) 定时器（按 fps 节拍发布） =========
    updateTimerFromFps();

    // ========= 6) 动态参数回调 =========
    param_cb_handle_ = this->add_on_set_parameters_callback(
      std::bind(&UsbCameraNode::onParamSet, this, std::placeholders::_1)
    );

    RCLCPP_INFO(get_logger(), "rm_vision_ros2_usb_camera_node started.");
  }

  ~UsbCameraNode() { closeCamera(); }

private:
  // 将字符串 FOURCC 转换成 int（例如 "MJPG"）
  static int fourccFromString(const std::string &fmt) {
    if (fmt.size() != 4) return 0;
    return cv::VideoWriter::fourcc(fmt[0], fmt[1], fmt[2], fmt[3]);
  }

  std::string devicePathFromIndex(int index) {
    if (!device_path_.empty()) return device_path_;
    return "/dev/video" + std::to_string(index);
  }

  // ---------- 打开/关闭相机 ----------
  void openCamera() {
    std::scoped_lock lk(mtx_);

    // 关闭旧的
    closeCameraUnlocked();

    const std::string dev = devicePathFromIndex(camera_index_);
    RCLCPP_INFO(get_logger(), "Opening camera: %s", dev.c_str());

    // 1) 用 OpenCV（V4L2 后端）打开
    cap_.open(dev, cv::CAP_V4L2);
    if (!cap_.isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open camera via OpenCV: %s", dev.c_str());
      return;
    }

    // 2) 先设置像素格式（要尽早）
    const int fourcc = fourccFromString(pixel_format_);
    if (fourcc != 0) {
      bool ok = cap_.set(cv::CAP_PROP_FOURCC, fourcc);
      RCLCPP_INFO(get_logger(), "Set FOURCC %s -> %s", pixel_format_.c_str(), ok ? "OK" : "FAIL");
    } else {
      RCLCPP_WARN(get_logger(), "Invalid pixel_format '%s' (skip setting FOURCC)", pixel_format_.c_str());
    }

    // 3) 分辨率 + 帧率（注意最终是否成功取决于硬件支持）
    cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width_);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height_);
    cap_.set(cv::CAP_PROP_FPS,          fps_);

    // 4) 打开 V4L2 fd，用于控制项兜底
    v4l2_fd_ = ::open(dev.c_str(), O_RDWR);
    if (v4l2_fd_ < 0) {
      RCLCPP_WARN(get_logger(), "Open V4L2 fd failed. Some controls may not work.");
    }

    // 5) 应用控制项（曝光/白平衡/对焦等）
    applyControls();

    // 6) 记录实际设置
    double rw = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
    double rh = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
    double rfps = cap_.get(cv::CAP_PROP_FPS);
    RCLCPP_INFO(get_logger(), "Camera ready: %.0fx%.0f @ %.2f FPS", rw, rh, rfps);
  }

  void closeCamera() {
    std::scoped_lock lk(mtx_);
    closeCameraUnlocked();
  }

  void closeCameraUnlocked() {
    if (cap_.isOpened()) cap_.release();
    if (v4l2_fd_ >= 0) { ::close(v4l2_fd_); v4l2_fd_ = -1; }
  }

  // ---------- V4L2 控件封装 ----------
  bool v4l2SetCtrl(unsigned int id, int32_t value) {
    if (v4l2_fd_ < 0) return false;
    struct v4l2_control ctrl{};
    ctrl.id = id;
    ctrl.value = value;
    int ret = ioctl(v4l2_fd_, VIDIOC_S_CTRL, &ctrl);
    return ret == 0;
  }

  bool setExposureAuto(bool enable) {
    // 常见 UVC：V4L2_CID_EXPOSURE_AUTO = 1(Manual)/3(Auto)
    const int V4L2_EXPOSURE_MANUAL = 1;
    const int V4L2_EXPOSURE_AUTO   = 3;
    bool ok = v4l2SetCtrl(V4L2_CID_EXPOSURE_AUTO, enable ? V4L2_EXPOSURE_AUTO : V4L2_EXPOSURE_MANUAL);
    if (!ok) {
      // OpenCV 兜底（注意不同驱动实现不同，这里采用常见的 0.25/0.75 写法）
      cap_.set(cv::CAP_PROP_AUTO_EXPOSURE, enable ? 0.75 : 0.25);
    }
    return true;
  }

  void applyControls() {
    // 曝光模式
    setExposureAuto(exposure_auto_);
    // 手动曝光时间
    if (!exposure_auto_ && exposure_time_ >= 0) {
      v4l2SetCtrl(V4L2_CID_EXPOSURE_ABSOLUTE, exposure_time_);
    }

    if (gain_        >= 0) v4l2SetCtrl(V4L2_CID_GAIN, gain_);
    if (brightness_  >= 0) v4l2SetCtrl(V4L2_CID_BRIGHTNESS, brightness_);
    if (contrast_    >= 0) v4l2SetCtrl(V4L2_CID_CONTRAST, contrast_);
    if (saturation_  >= 0) v4l2SetCtrl(V4L2_CID_SATURATION, saturation_);
    if (sharpness_   >= 0) v4l2SetCtrl(V4L2_CID_SHARPNESS, sharpness_);
    if (gamma_       >= 0) v4l2SetCtrl(V4L2_CID_GAMMA, gamma_);

    // 白平衡
    v4l2SetCtrl(V4L2_CID_AUTO_WHITE_BALANCE, white_balance_auto_ ? 1 : 0);
    if (!white_balance_auto_ && white_balance_temp_ > 0) {
      v4l2SetCtrl(V4L2_CID_WHITE_BALANCE_TEMPERATURE, white_balance_temp_);
    }

    // 对焦
    v4l2SetCtrl(V4L2_CID_FOCUS_AUTO, focus_auto_ ? 1 : 0);
    if (!focus_auto_ && focus_absolute_ >= 0) {
      v4l2SetCtrl(V4L2_CID_FOCUS_ABSOLUTE, focus_absolute_);
    }

    // 电源频率（抗闪烁）
    if (power_line_freq_ >= 0) {
      v4l2SetCtrl(V4L2_CID_POWER_LINE_FREQUENCY, power_line_freq_);
    }
  }

  // ---------- 定时器 ----------
  void updateTimerFromFps() {
    double fps = (fps_ > 1e-6) ? fps_ : 30.0;
    auto period = std::chrono::duration<double>(1.0 / fps);
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(period),
      std::bind(&UsbCameraNode::onTimer, this)
    );
  }

  void onTimer() {
    std::lock_guard<std::mutex> lk(mtx_);
    if (!cap_.isOpened()) return;

    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Failed to read frame");
      return;
    }

    // 翻转
    if (flip_horizontal_) cv::flip(frame, frame, 1);
    if (flip_vertical_)   cv::flip(frame, frame, 0);

    // 旋转
    if (rotate_degree_ == 90)       cv::rotate(frame, frame, cv::ROTATE_90_CLOCKWISE);
    else if (rotate_degree_ == 180) cv::rotate(frame, frame, cv::ROTATE_180);
    else if (rotate_degree_ == 270) cv::rotate(frame, frame, cv::ROTATE_90_COUNTERCLOCKWISE);

    // 时间戳 & header
    rclcpp::Time stamp = this->now();
    std_msgs::msg::Header hdr;
    hdr.stamp = stamp;
    hdr.frame_id = frame_id_;

    // 根据通道选择编码：单通道 -> mono8；三通道 -> bgr8
    const std::string encoding = (frame.channels() == 1) ? "mono8" : "bgr8";
    auto img_msg = cv_bridge::CvImage(hdr, encoding, frame).toImageMsg();

    // CameraInfo：从 manager 获取并更新 stamp/尺寸/frame_id
    auto ci = cinfo_mgr_->getCameraInfo();
    ci.header.stamp = stamp;
    ci.header.frame_id = frame_id_;

    image_pub_.publish(*img_msg);
    cinfo_pub_->publish(ci);
  }

  // ---------- 动态参数回调 ----------
  rcl_interfaces::msg::SetParametersResult
  onParamSet(const std::vector<rclcpp::Parameter> &params) {
    bool need_reopen = false;
    bool fps_changed = false;

    for (const auto &p : params) {
      const auto &n = p.get_name();

      if (n == "camera_index") { camera_index_ = p.as_int(); need_reopen = true; }
      else if (n == "device_path") { device_path_ = p.as_string(); need_reopen = true; }
      else if (n == "frame_id") { frame_id_ = p.as_string(); }
      else if (n == "camera_name") {
        camera_name_ = p.as_string();
        cinfo_mgr_.reset(new camera_info_manager::CameraInfoManager(this, camera_name_, camera_info_url_));
      }
      else if (n == "frame_width") { width_ = p.as_int(); need_reopen = true; }
      else if (n == "frame_height") { height_ = p.as_int(); need_reopen = true; }
      else if (n == "fps") { fps_ = p.as_double(); fps_changed = true; cap_.set(cv::CAP_PROP_FPS, fps_); }
      else if (n == "pixel_format") { pixel_format_ = p.as_string(); need_reopen = true; }
      else if (n == "flip_horizontal") { flip_horizontal_ = p.as_bool(); }
      else if (n == "flip_vertical") { flip_vertical_ = p.as_bool(); }
      else if (n == "rotate_degree") { rotate_degree_ = p.as_int(); }
      else if (n == "exposure_auto") { exposure_auto_ = p.as_bool(); setExposureAuto(exposure_auto_); }
      else if (n == "exposure_time") { exposure_time_ = p.as_int(); if (!exposure_auto_) v4l2SetCtrl(V4L2_CID_EXPOSURE_ABSOLUTE, exposure_time_); }
      else if (n == "gain") { gain_ = p.as_int(); if (gain_>=0) v4l2SetCtrl(V4L2_CID_GAIN, gain_); }
      else if (n == "brightness") { brightness_ = p.as_int(); if (brightness_>=0) v4l2SetCtrl(V4L2_CID_BRIGHTNESS, brightness_); }
      else if (n == "contrast") { contrast_ = p.as_int(); if (contrast_>=0) v4l2SetCtrl(V4L2_CID_CONTRAST, contrast_); }
      else if (n == "saturation") { saturation_ = p.as_int(); if (saturation_>=0) v4l2SetCtrl(V4L2_CID_SATURATION, saturation_); }
      else if (n == "sharpness") { sharpness_ = p.as_int(); if (sharpness_>=0) v4l2SetCtrl(V4L2_CID_SHARPNESS, sharpness_); }
      else if (n == "gamma") { gamma_ = p.as_int(); if (gamma_>=0) v4l2SetCtrl(V4L2_CID_GAMMA, gamma_); }
      else if (n == "white_balance_auto") { white_balance_auto_ = p.as_bool(); v4l2SetCtrl(V4L2_CID_AUTO_WHITE_BALANCE, white_balance_auto_?1:0); }
      else if (n == "white_balance_temperature") { white_balance_temp_ = p.as_int(); if (!white_balance_auto_ && white_balance_temp_>0) v4l2SetCtrl(V4L2_CID_WHITE_BALANCE_TEMPERATURE, white_balance_temp_); }
      else if (n == "focus_auto") { focus_auto_ = p.as_bool(); v4l2SetCtrl(V4L2_CID_FOCUS_AUTO, focus_auto_?1:0); }
      else if (n == "focus_absolute") { focus_absolute_ = p.as_int(); if (!focus_auto_ && focus_absolute_>=0) v4l2SetCtrl(V4L2_CID_FOCUS_ABSOLUTE, focus_absolute_); }
      else if (n == "power_line_frequency") { power_line_freq_ = p.as_int(); if (power_line_freq_>=0) v4l2SetCtrl(V4L2_CID_POWER_LINE_FREQUENCY, power_line_freq_); }
      else if (n == "camera_info_url") {
        camera_info_url_ = p.as_string();
        auto new_mgr = std::make_shared<camera_info_manager::CameraInfoManager>(this, camera_name_, camera_info_url_);
        if (!camera_info_url_.empty()) new_mgr->loadCameraInfo(camera_info_url_);
        cinfo_mgr_ = new_mgr;
      }
      else if (n == "use_sensor_data_qos") { use_sensor_data_qos_ = p.as_bool(); /* 发布器 QoS 不动态更换，通常重启生效 */ }
    }

    if (need_reopen) { openCamera(); }
    if (fps_changed) { updateTimerFromFps(); }

    rcl_interfaces::msg::SetParametersResult ret;
    ret.successful = true;
    return ret;
  }

private:
  // 参数
  int camera_index_{};
  std::string device_path_{};
  std::string frame_id_{};
  std::string camera_name_{};

  int width_{};
  int height_{};
  double fps_{};
  std::string pixel_format_{};

  bool use_sensor_data_qos_{};

  bool flip_horizontal_{};
  bool flip_vertical_{};
  int  rotate_degree_{};

  bool exposure_auto_{};
  int  exposure_time_{};
  int  gain_{};
  int  brightness_{};
  int  contrast_{};
  int  saturation_{};
  int  sharpness_{};
  int  gamma_{};
  bool white_balance_auto_{};
  int  white_balance_temp_{};
  bool focus_auto_{};
  int  focus_absolute_{};
  int  power_line_freq_{};
  std::string camera_info_url_{};

  // 组件
  image_transport::Publisher image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr cinfo_pub_;
  std::shared_ptr<camera_info_manager::CameraInfoManager> cinfo_mgr_;
  rclcpp::TimerBase::SharedPtr timer_;

  // 采集
  cv::VideoCapture cap_;
  int v4l2_fd_ = -1;
  std::mutex mtx_;

  // 参数回调
  OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<UsbCameraNode>());
  rclcpp::shutdown();
  return 0;
}
