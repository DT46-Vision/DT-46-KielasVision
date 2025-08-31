# 基础镜像：ROS2 Humble Desktop
FROM osrf/ros:humble-desktop

# 设置时区，避免 tzdata 交互
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata \
    && rm -rf /var/lib/apt/lists/*

# 安装 ROS2 构建常用工具
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 初始化 rosdep（加 || true 防止重复 init 报错）
RUN rosdep init || true && rosdep update

# 设置工作空间
WORKDIR /ros_ws

# 拷贝源码到容器中
COPY ./ros_ws/ ./

# 构建 ROS2 项目
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --symlink-install"

# 设置容器启动时进入 bash
CMD ["/bin/bash"]
