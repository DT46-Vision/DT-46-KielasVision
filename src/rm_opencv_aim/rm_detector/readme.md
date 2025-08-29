⸻


# rm_detector — Armor Detector with OpenCV + PnP

## 📌 简介
本包实现了基于 **OpenCV** 和 **PnP 解算** 的装甲板检测与姿态估计，支持高帧率运行（>700 FPS）。  
采用 **ROS 2 (Humble)**，通过独立线程实现图像处理与 ROS 通信解耦。  

---

## ⚡ 功能特性
- **高帧率检测**：独立线程解耦订阅与处理，支持 >700 FPS。
- **零拷贝优化**：使用 `cv_bridge::toCvShare` 避免不必要的数据复制。
- **光心校正**：可选择使用标定光心或图像几何中心。
- **参数可调**：支持动态参数调整（`rqt_reconfigure` 或 `ros2 param set`）。
- **可视化**：发布二值化图像 `/detector/bin_img` 与标注结果 `/detector/armors_img`。
- **日志优化**：关键日志保留，冗余日志节流。

---

## 🛠 参数说明

| 参数名                  | 类型   | 默认值 | 说明 |
|--------------------------|--------|--------|------|
| `light_area_min`         | int    | 5      | 灯条最小面积 |
| `light_angle_min`        | int    | -35    | 灯条最小角度 |
| `light_angle_max`        | int    | 35     | 灯条最大角度 |
| `light_red_ratio`        | float  | 2.0    | 红色通道比例阈值 |
| `light_blue_ratio`       | float  | 2.0    | 蓝色通道比例阈值 |
| `cy_tol`                 | int    | 5      | y 坐标容差 |
| `height_rate_tol`        | float  | 1.3    | 高度比例容差 |
| `vertical_discretization`| float  | 1.5    | 垂直离散化阈值 |
| `height_multiplier_min`  | float  | 1.8    | 高度缩放最小值 |
| `height_multiplier_max`  | float  | 3.0    | 高度缩放最大值 |
| `binary_val`             | int    | 20     | 二值化阈值 |
| `detect_color`           | int    | 2      | 检测颜色（1=红，2=蓝） |
| `display_mode`           | int    | 0      | 显示模式（0=不显示，1=部分显示，2=调试显示） |
| `camera_id`              | int    | 1      | 相机编号 |
| `use_geometric_center`   | bool   | true   | 是否用图像几何中心代替标定光心 |
| `max_processing_fps`     | int    | 0      | 最大处理帧率限制（0=不限速） |

---

## 🚀 运行

### 启动节点
```bash
ros2 run rm_detector armor_detector_opencv_node

常用调参

ros2 param set /armor_detector_opencv_node detect_color 1
ros2 param set /armor_detector_opencv_node display_mode 2
ros2 param set /armor_detector_opencv_node use_geometric_center true


⸻

📡 话题接口

订阅
	•	/image_raw (sensor_msgs/msg/Image)
输入相机图像
	•	/camera_info (sensor_msgs/msg/CameraInfo)
相机标定参数

发布
	•	/detector/armors_info (rm_interfaces/msg/ArmorsCppMsg)
装甲板位姿信息（yaw, pitch, deep）
	•	/detector/armors_img (sensor_msgs/msg/Image)
绘制装甲板的调试图像
	•	/detector/bin_img (sensor_msgs/msg/Image)
二值化图像

⸻

🔧 Debug 经验总结

1. 光心偏差
	•	问题：标定光心 (cx, cy) 与图像几何中心不一致，yaw/pitch 偏移。
	•	解决：加入 use_geometric_center 参数，允许用几何中心替代光心。

2. 日志过多
	•	问题：RCLCPP_INFO 高频输出拖慢性能。
	•	解决：改用 RCLCPP_INFO_THROTTLE，保留关键信息。

3. 帧率低（不开 GUI）
	•	问题：不开 rqt 显示时，FPS 掉到 ~80。
	•	原因：DDS 缓冲积压，生产者降速。
	•	解决：使用 SensorDataQoS + 独立线程缓存最新帧，避免延迟。

4. OpenCL 加速
	•	问题：尝试 cv::ocl 报错（未启用 OpenCL）。
	•	解决：去掉 OpenCL，加速意义不大。

5. FPS 统计
	•	优化：仅统计处理帧率（processed FPS），每秒 WARN 打印一次。

⸻

📈 性能结果
	•	旧版本：不开 GUI 时 ~80 FPS，开 GUI 时 ~150 FPS。
	•	改进后：稳定 700~900 FPS，CPU 利用率合理。

⸻

✨ TODO
	•	支持 TensorRT 加速检测
	•	多相机同步处理
	•	深度信息融合（激光测距 / 深度相机）

⸻


---
