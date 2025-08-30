# DT-46-KielasVison
梓喵-铭

<br><img src="DT46-vision.svg" alt="DT46_vision" width="200" height="200">

## **[技术文档](https://www.notion.so/DT46-RM-vision-25ba064aa1788083afacddc497af92c8)**
## **[技术文档-本地](梓喵-KielasVision/梓喵-DT46-RM-vision.md)**

**一键看包**

```bash
python3 deps_finder.py src/
```

### DT-46-KielasVison-Armor_type_definition
| 编号 | 含义             | 序号 |
|------|------------------|------|
| B1   | 蓝方1号 装甲板   | 0    |
| B2   | 蓝方2号 装甲板   | 1    |
| B3   | 蓝方3号 装甲板   | 2    |
| B4   | 蓝方4号 装甲板   | 3    |
| B5   | 蓝方5号 装甲板   | 4    |
| B7   | 蓝方哨兵 装甲板   | 5    |
| R1   | 红方1号 装甲板   | 6    |
| R2   | 红方2号 装甲板   | 7    |
| R3   | 红方3号 装甲板   | 8    |
| R4   | 红方4号 装甲板   | 9    |
| R5   | 红方5号 装甲板   | 10   |
| R7   | 红方哨兵 装甲板   | 11   |

---

## DT-46-Classifier_training
装甲板图案分类器训练相关代码

## 创建 datasets 文件夹

```bash
mkdir datasets/1/
mkdir datasets/2/
mkdir datasets/3/
mkdir datasets/4negative/
```

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
