# Copyright (c) 2023 ChenJun

import rosbag2_py
import cv2
import sys
import os

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.serialization import deserialize_message


def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)

    return storage_options, converter_options


def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_images.py <rosbag_path> <save_path>")
        return

    bag_path = sys.argv[1]
    save_path = sys.argv[2]

    # 自动创建保存目录
    os.makedirs(save_path, exist_ok=True)

    storage_options, converter_options = get_rosbag_options(bag_path)

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    storage_filter = rosbag2_py.StorageFilter(
        topics=['/detector/img_armor_processed'])
    reader.set_filter(storage_filter)

    bridge = CvBridge()

    count = 0
    print("Saving binarized images to %s" % save_path)
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        msg = deserialize_message(data, Image)

        # 转换为 OpenCV 图像
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

        # --- 灰度 ---
        # gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        # 只保存二值化图
        out_file = os.path.join(save_path, "%06i.png" % count)
        cv2.imwrite(out_file, cv_img)

        print("Writing binarized image %i" % count)
        count += 1

    print("Finished! Total saved:", count)


if __name__ == '__main__':
    main()
