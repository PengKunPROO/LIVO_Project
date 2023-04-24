#!/usr/bin/python3
import rosbag
import rospy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


class ImageCreator():
    def __init__(self, bag_path):
        # 这个装所有数据的缓存，即 data_cache[0]代表一个 (x,y) ——x,y都是numpy数组
        self.data_cache = []
        self.iteration_count = 0
        self.img_len = 0
        self.lidar_len = 0
        with rosbag.Bag(bag_path, 'r') as bag:  # 要读取的bag文件；
            img_topic = []
            lidar_topic = []
            img_count = bag.get_message_count('/camera/image')
            lidar_count = bag.get_message_count('/livox/lidar')
            if (lidar_count >= img_count):
                self.iteration_count = lidar_count
            else:
                self.iteration_count = img_count
            for topic, msg, t in bag.read_messages():
                if (img_topic.__len__() > self.iteration_count) and (lidar_topic.__len__() > self.iteration_count):
                    break
                if topic == "/camera/image":  # 图像的topic；
                    try:
                        img_topic.append(msg.header.stamp.to_sec())
                    except Exception as e:
                        print(e)
                elif topic == "/livox/lidar":  # 图像的topic；
                    try:
                        lidar_topic.append(msg.header.stamp.to_sec())
                    except Exception as e:
                        print(e)
        self.img_len = img_topic.__len__()
        self.lidar_len = lidar_topic.__len__()
        self.data_cache.append(img_topic)
        self.data_cache.append(lidar_topic)


def getNearsetLidarFrame(img_frame):
    min = np.abs(img_frame - lidar_c[0])
    min_frame = img_frame
    for sec in lidar_c:
        if np.abs(sec - img_frame) < min:
            min = np.abs(sec - img_frame)
            min_frame = sec
    return min_frame


if __name__ == '__main__':
    bag_path = '/home/sulab/.ros/2023-03-19-21-55-24.bag'
    # rospy.init_node(PKG)
    try:
        image_creator = ImageCreator(bag_path)
        img_c = np.array(image_creator.data_cache[0])
        lidar_c = np.array(image_creator.data_cache[1])
        print(img_c[0])
        print(lidar_c[85])
        print(img_c[0] - lidar_c[83])
        pivot = img_c[0]
        img_c = img_c - pivot
        lidar_c = (lidar_c - pivot)
        ratio = (float)(image_creator.img_len / image_creator.lidar_len)
        # 已经知道雷达总时长多一点，所以计算出总的时长
        max_len = lidar_c[image_creator.lidar_len-1] - lidar_c[0]
        const1 = np.full((image_creator.img_len), 1)
        const2 = np.full((image_creator.lidar_len), 1.1)

        if (ratio >= 1):
            ratio = 1 / ratio
        print(lidar_c[0])
        print(img_c[0])
        ax = plt.axes()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
        plt.title('Synchronization Show')
        plt.xticks(rotation=90)
        plt.plot(img_c, const1,
                 'o', color='green', label='img sampling frequency')
        plt.plot(lidar_c, const2,
                 '*', color='red', label='lidar sampling frequency')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('time stamp')
        plt.show()

        img_c = np.array(image_creator.data_cache[0])
        lidar_c = np.array(image_creator.data_cache[1])
        array_lidar = []
        for sec in img_c:
            # 寻找最近的雷达帧，用于计算偏移
            array_lidar.append(getNearsetLidarFrame(sec))
        len_delay = 0
        if (image_creator.img_len > image_creator.lidar_len-83):
            len_delay = image_creator.lidar_len-83
        else:
            len_delay = image_creator.img_len
        array_delay = np.array(np.array(array_lidar) - img_c)

        plt.title('Delay Show')
        plt.xticks(rotation=90)
        plt.plot(np.arange(image_creator.img_len), array_delay,
                 color='red', label='lidar sampling frequency')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('time diff')
        plt.show()
    except rospy.ROSInterruptException:
        pass
