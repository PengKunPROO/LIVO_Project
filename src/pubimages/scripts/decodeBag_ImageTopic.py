#!/usr/bin/python3
# coding:utf-8

# Extract images from a bag file.
import rosbag
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError
     
# Reading bag filename from command line or roslaunch parameter.
#import os
#import sys
     
rgb_path = '/home/sulab/image_cache/test_front_train_station/'   #已经建立好的存储rgb彩色图文件的目录
     
class ImageCreator():
    def __init__(self,bag_path):
        self.bridge = CvBridge()
        with rosbag.Bag(bag_path, 'r') as bag:  #要读取的bag文件；
            for topic,msg,t in bag.read_messages():
                if topic == "/camera/image": #图像的topic；
                        try:
                            cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
                        except CvBridgeError as e:
                            print(e)
                        timestr = str(msg.header.stamp.to_sec())
                        #%.6f表示小数点后带有6位，可根据精确度需要修改；
                        image_name = timestr+ ".jpg" #图像命名：时间戳.png
                        cv2.imwrite(rgb_path + image_name, cv_image)  #保存；
                # elif topic == "camera/depth_registered/image_raw": #图像的topic；
                #         try:
                #             cv_image = self.bridge.imgmsg_to_cv2(msg,"16UC1")
                #         except CvBridgeError as e:
                #             print (e)
                #         timestr = "%.6f" %  msg.header.stamp.to_sec()
                #         #%.6f表示小数点后带有6位，可根据精确度需要修改；
                #         image_name = timestr+ ".png" #图像命名：时间戳.png
                #         cv2.imwrite(depth_path + image_name, cv_image)  #保存；
     
if __name__ == '__main__':
    bag_path = '/opt/test_front_train_station.bag'
    #rospy.init_node(PKG)
    try:
        image_creator = ImageCreator(bag_path)
    except rospy.ROSInterruptException:
        pass