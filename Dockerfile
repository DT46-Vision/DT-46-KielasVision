# 基础镜像：ROS2 Humble Desktop
FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8

# 时区 + 本地化
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata locales \
    && locale-gen en_US en_US.UTF-8 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# 安装 curl、pip，并执行 chsrc 换源（只换 Ubuntu + pip，不动 ROS）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl python3-pip \
    && curl -sL https://chsrc.run/posix -o /tmp/chsrc.sh \
    && bash /tmp/chsrc.sh || echo "chsrc init failed, skipping" \
    && command -v chsrc >/dev/null 2>&1 || { echo "chsrc not found, skipping"; exit 0; } \
    && chsrc set ubuntu || echo "ubuntu mirror switch failed" \
    && chsrc set pip || echo "pip mirror switch failed" \
    && rm -rf /var/lib/apt/lists/*

# 基础构建工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    git build-essential \
    ros-humble-ament-cmake ros-humble-ament-package ros-humble-camera-info-manager \
    ros-humble-rqt* ros-humble-foxglove-bridge ros-humble-rviz2 \
    && rm -rf /var/lib/apt/lists/*

# 确保 pip 可用（系统上通常已有 pip3）
RUN if [ ! -e /usr/bin/pip ]; then ln -s /usr/bin/pip3 /usr/bin/pip; fi

# Python 依赖
RUN pip install --no-cache-dir \
    numpy==1.23.5 \
    torch==2.0.1 \
    torchvision==0.15.2 \
    onnx \
    matplotlib \
    catkin_pkg \
    empy==3.3.4 \
    lark-parser

# （可选）安装 MindVision/Hik SDK
COPY ./MindVisionSDK /tmp/MindVisionSDK
RUN if [ -d /tmp/MindVisionSDK ]; then \
      cd /tmp/MindVisionSDK && bash install.sh && cp lib/x64/libMVSDK.so /usr/lib/ && echo "/usr/lib" > /etc/ld.so.conf.d/mvskd.conf && ldconfig && rm -rf /tmp/MindVisionSDK; \
    else echo "No MindVisionSDK provided, skipping"; fi

COPY ./MVS.deb /tmp/MVS.deb
RUN if [ -f /tmp/MVS.deb ]; then dpkg -i /tmp/MVS.deb || apt-get -f install -y && echo "/usr/lib/hik" > /etc/ld.so.conf.d/hik.conf && ldconfig && rm /tmp/MVS.deb; else echo "No MVS.deb, skipping"; fi

# 初始化 rosdep
RUN if [ ! -d /etc/ros/rosdep/sources.list.d ]; then rosdep init || true; else echo "rosdep already initialized"; fi \
    && rosdep fix-permissions || true

# 工作目录 & 拷贝源码
WORKDIR /ros_ws
COPY ./ros_ws ./

# 安装工作空间依赖
RUN apt-get update && \
    rosdep install --from-paths src --ignore-src -r -y || echo "rosdep install failed, continuing" && \
    rm -rf /var/lib/apt/lists/*

# 构建 ROS2 工作空间
RUN . /opt/ros/humble/setup.sh && \
    colcon build --symlink-install

# 入口
CMD ["/bin/bash"]
