# 项目镜像和容器名
IMAGE_NAME = kielas_rmvision:latest
CONTAINER_NAME = rmvision-core

.DEFAULT_GOAL := help

help: ## 显示可用命令
	@echo "可用命令："
	@echo "  make build     - 编译镜像 ($(IMAGE_NAME))"
	@echo "  make up        - 启动容器 ($(CONTAINER_NAME)) 并运行 ros2 launch"
	@echo "  make exec      - 进入容器交互式终端 (环境已加载)"
	@echo "  make rqt       - 在容器内运行 rqt"
	@echo "  make rviz      - 在容器内运行 rviz2"
	@echo "  make foxglove  - 在容器内运行 foxglove_bridge"
	@echo "  make down      - 停止容器"
	@echo "  make clean     - 停止并删除容器+镜像+卷"

build: ## 编译镜像
	docker compose build

up: ## 启动核心容器 (运行 ros2 launch)
	docker compose up

exec: ## 进入容器交互式终端
	docker exec -it $(CONTAINER_NAME) bash -c "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash || true && bash"

rqt: ## 在容器内运行 rqt
	docker exec -it $(CONTAINER_NAME) bash -c "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash || true && rqt"

rviz: ## 在容器内运行 rviz2
	docker exec -it $(CONTAINER_NAME) bash -c "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash || true && rviz2"

foxglove: ## 在容器内运行 foxglove_bridge
	docker exec -it $(CONTAINER_NAME) bash -c "source /opt/ros/humble/setup.bash && source /ros_ws/install/setup.bash || true && ros2 run foxglove_bridge foxglove_bridge"

down: ## 停止容器
	docker compose down

clean: ## 删除容器+镜像+卷
	docker compose down --rmi all --volumes --remove-orphans
