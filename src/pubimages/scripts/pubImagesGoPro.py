#!/usr/bin/python3
# single_webcam.py/Open GoPro, Version 2.0 (C) Copyright 2021 GoPro, Inc. (http://gopro.com/OpenGoPro).
# This copyright was auto-generated on Fri Nov 11 20:03:39 UTC 2022

import logging
import argparse
import sys
import rospy
import signal
import time

from multi_webcam import GoProWebcamPlayer
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError

logging.basicConfig(level=logging.DEBUG)




def kill_cbk(signum, frame):
    global is_looping
    is_looping = False
    rospy.loginfo('exit the process!')


if __name__ == "__main__":

    global is_looping
    is_looping = True
    signal.signal(signal.SIGINT, kill_cbk)
    rospy.init_node('gopro_nodes', anonymous=True)
    img_pub = rospy.Publisher('/camera/image', Image, queue_size=1000)
    text_pub = rospy.Publisher('/time_end', String, queue_size=10)
    port = rospy.get_param('common/port')
    fov = rospy.get_param('common/fov', default=None)
    resolution = rospy.get_param('common/resolution', default=None)
    serials = rospy.get_param('common/serials')
    paras = {'port': port, 'fov': fov,
             'resolution': resolution, 'serials': serials}
    webcam1 = GoProWebcamPlayer(paras['serials'], paras['port'])
    webcam1.open()
    webcam1.play(paras['resolution'], paras['fov'])
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        #webcam1.player._cache_lock.acquire()
        # print(webcam1.player._global_img_cache.__len__())
        data = webcam1.player.queue.get()
        msg = CvBridge().cv2_to_imgmsg(data.img, 'bgr8')
        # stamp是真实的采集时间
        msg.header.stamp = data.stamp
        # frameid是消息发送的时间
        msg.header.frame_id = str(rospy.Time.now().to_sec())
        img_pub.publish(msg)
        #webcam1.player._cache_lock.release()

        rate.sleep()
        # if webcam1.player.is_change:
        # print("outside loop")
        # image_delivery = webcam1.player.image
        # msg = CvBridge().cv2_to_imgmsg(image_delivery, "bgr8")
        # msg.header.stamp = rospy.Time.now()
        # msg.header.frame_id = "camera_frame"
        # img_pub.publish(msg)
        # webcam1.player.is_changed = True
        # webcam1.player.is_change = False

    if (is_looping is False):
        print("end up with signal 1")
        webcam1.close()
