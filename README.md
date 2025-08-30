# DT-46-KielasVison
新的梓喵系统，添加了稳定的 PnP 2D 转 3D 坐标功能和装甲板信息识别，帧率提升，接管了电控的发弹管理并且添加了弹道解算。
# DT-46-Classifier_training
装甲板图案分类器训练相关代码

<img src="DT46-vision.svg" alt="DT46_vision" width="200" height="200">

## 使用 CIFAR-100 作为负样本

下载地址：https://www.cs.toronto.edu/~kriz/cifar.html

下载解压后，使用 [process_cifra100.py](process_cifra100.py) 对其进行处理

## 装甲板图案数据采集

1. 启动相机节点与识别器
2. 将装甲板置于相机视野中，检查识别器的 img_armor 话题图像是否准确
3. 改变装甲板姿态，若此时角点依然准确，录制该类别的 rosbag

    ```
    ros2 bag record /detector/img_armor -o <output_path>
    ```
- 总共 3 类 装甲板图案
1. 1   -- armor_bag_1
2. 3   -- armor_bag_2
3. 哨兵 -- armor_bag_3

4. 从 bag 中提取出图片作为数据集

    ```
    python3 extract_bag_bin.py armor_bag_1 datasets/1/
    ```

5. 按照下列结构放置图片作为数据集

    ```
    datasets
    ├─1 -1 == 1
    ├─2 -1 == 3
    ├─3 -1 == 哨兵
    ├─4negative -1 == 啥也不是
    ```

## 训练

运行 [mlp_training.py](/training_scripts/mpl_training.py)
