#!/usr/bin/env python2.7

import time
import numpy as np
import cv2
import cv_bridge
from PIL import ImageDraw, ImageFont
from PIL import Image as PIL_Image
from cv_bridge.boost.cv_bridge_boost import getCvType

# ros package
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import PoseStamped

# custom package
from yolo4_tiny import YoloBody
from utils import (DecodeBox, letterbox_image, non_max_suppression, yolo_correct_boxes)
from yolo import YOLO


class object_detect:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        self.yolo = YOLO()
        rospy.init_node('object_detection')
        
        # Subscribe color and depth image
        rospy.Subscriber("/d435/color/image_raw",Image,self.color_callback)
        rospy.Subscriber("/d435/depth/image_raw",Image,self.depth_callback)

        # Subscribe camera info
        rospy.Subscriber("/d435/depth/camera_info",CameraInfo,self.depth_camera_info_callback)
        rospy.Subscriber("/d435/color/camera_info",CameraInfo,self.color_camera_info_callback)

        self.box_pub = rospy.Publisher("/desired/input/box", BoundingBox, queue_size=1)
        self.boxarray_pub = rospy.Publisher("/desired/input/box_array", BoundingBoxArray, queue_size=1)

        self.boxes = BoundingBoxArray()
        self.oak_box = BoundingBox()
        self.pine_box = BoundingBox()
        self.tree1_box = BoundingBox()
        self.tree2_box = BoundingBox()
        print ("init finish!")

    def depth_callback(self,data):
        # Depth image callback
        self.depth_image = self.bridge.imgmsg_to_cv2(data)
        self.depth_image = cv2.resize(self.depth_image,(640,480))   # 640 width, 480 height
        self.depth_image = np.array(self.depth_image, dtype=np.float32)
        self.depth_array = self.depth_image/1000.0
        #Display
        #cv2.imshow('depth_image',self.depth_image)
        #cv2.waitKey(1)

    def color_callback(self,data):
        # RGB image callback
        self.rgb_image = self.bridge.imgmsg_to_cv2(data)
        self.img = self.rgb_image
        return self.img
        #self.rgb_image = cv2.cvtColor(self.rgb_image,cv2.COLOR_RGB2BGR)
        # Display
        #cv2.imshow('rgb_image',self.rgb_image)
        #cv2.waitKey(1)    

    # this is depth camera info callback
    def depth_camera_info_callback(self, data):
        self.depth_height = data.height
        self.depth_width = data.width

        # Pixels coordinates of the principal point (center of projection)
        self.depth_u = data.P[2]
        self.depth_v = data.P[6]

        # Focal Length of image (multiple of pixel width and height)
        self.depth_fx = data.P[0]
        self.depth_fy = data.P[5]

    # this is color camera info callback
    def color_camera_info_callback(self, data):
        self.rgb_height = data.height
        self.rgb_width = data.width

        # Pixels coordinates of the principal point (center of projection)
        self.rgb_u = data.P[2]
        self.rgb_v = data.P[6]

        # Focal Length of image (multiple of pixel width and height)
        self.rgb_fx = data.P[0]
        self.rgb_fy = data.P[5]

    # Oak Algorithm Core
    def oak_core(self,rgb_image,depth_image, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):

        self.oak_expected_3d_top_left_x = ((top_left_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.oak_expected_3d_top_left_y = ((top_left_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        self.oak_expected_3d_top_right_x = ((bottom_right_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.oak_expected_3d_top_right_y = ((top_left_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        self.oak_expected_3d_bottom_left_x = ((top_left_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.oak_expected_3d_bottom_left_y = ((bottom_right_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        self.oak_expected_3d_bottom_right_x = ((bottom_right_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.oak_expected_3d_bottom_right_y = ((bottom_right_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        self.oak_expected_3d_center_x = ((center_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.oak_expected_3d_center_y = ((center_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        print("self.oak_expected_3d_top_left_x : {}".format(self.oak_expected_3d_top_left_x))
        print("self.oak_expected_3d_top_right_x : {}".format(self.oak_expected_3d_top_right_x))
        print("self.oak_expected_3d_top_left_y : {}".format(self.oak_expected_3d_top_left_y))
        print("self.oak_expected_3d_bottom_left_y : {}".format(self.oak_expected_3d_bottom_right_y))
        print("self.depth_array[center_y,center_x] : {}".format(self.depth_array[center_y,center_x]))
        print("self.top_left_x : {}".format(top_left_x))
        print("self.rgb_u : {}".format(self.rgb_u))
        print("self.rgb_fx : {}".format(self.rgb_fx))

        if (abs(self.oak_expected_3d_top_left_x)>100)&(abs(self.oak_expected_3d_top_left_y)>100)&(abs(self.oak_expected_3d_top_right_x)>100)&(abs(self.oak_expected_3d_top_right_y)>100)&(abs(self.oak_expected_3d_bottom_left_x)>100)&(abs(self.oak_expected_3d_bottom_left_y)>100)&(abs(self.oak_expected_3d_bottom_right_y)>100)&(abs(self.oak_expected_3d_bottom_right_y)>100):
            self.oak_expected_3d_top_left_x = self.oak_expected_3d_top_left_x/1000.0
            self.oak_expected_3d_top_left_y = self.oak_expected_3d_top_left_y/1000.0
            self.oak_expected_3d_top_right_x = self.oak_expected_3d_top_right_x/1000.0
            self.oak_expected_3d_top_right_y = self.oak_expected_3d_top_right_y/1000.0
            self.oak_expected_3d_bottom_left_x = self.oak_expected_3d_bottom_left_x/1000.0
            self.oak_expected_3d_bottom_left_y = self.oak_expected_3d_bottom_left_y/1000.0
            self.oak_expected_3d_bottom_right_x = self.oak_expected_3d_bottom_right_x/1000.0
            self.oak_expected_3d_bottom_right_y = self.oak_expected_3d_bottom_right_y/1000.0
            self.oak_expected_3d_center_x = self.oak_expected_3d_center_x/1000.0
            self.oak_expected_3d_center_y = self.oak_expected_3d_center_y/1000.0
            print ("Successfully divide 1000")

        self.oak_box.header.stamp = rospy.Time.now()
        self.oak_box.header.frame_id = "d435_depth_optical_frame"
        self.oak_box.pose.orientation.w = 1
        self.oak_box.pose.position.x = self.oak_expected_3d_center_x # increase in program  = move to right in rviz 
        self.oak_box.pose.position.y = self.oak_expected_3d_center_y # increase in program  = downward in rviz
        self.oak_box.pose.position.z = self.depth_image[center_y,center_x]/1000.0 # increase in program  = move away in rviz (directly use the depth distance)
        self.oak_box.dimensions.x = abs(self.oak_expected_3d_top_left_x)-abs(self.oak_expected_3d_top_right_x)
        if (self.oak_expected_3d_top_left_x < 0)&(self.oak_expected_3d_top_right_x > 0):
            self.oak_box.dimensions.x = abs(self.oak_expected_3d_top_left_x)+abs(self.oak_expected_3d_top_right_x)
        self.oak_box.dimensions.x = abs(self.oak_box.dimensions.x)
        self.oak_box.dimensions.y = abs(self.oak_expected_3d_top_left_y)-abs(self.oak_expected_3d_bottom_right_y)
        if (self.oak_expected_3d_top_left_y < 0)&(self.oak_expected_3d_bottom_right_y > 0):
            self.oak_box.dimensions.y = abs(self.oak_expected_3d_top_left_y)+abs(self.oak_expected_3d_bottom_right_y)
        self.oak_box.dimensions.y = abs(self.oak_box.dimensions.y)

        print("self.oak_box.dimensions.x : {}".format(self.oak_box.dimensions.x))
        print("self.oak_box.dimensions.y : {}".format(self.oak_box.dimensions.y))
        self.oak_box.dimensions.z = 1
        #self.box_pub.publish(self.oak_box)
        #print("box publish finish")
        #return self.oak_expected_3d_top_left_x, self.oak_expected_3d_top_left_y, self.oak_expected_3d_top_right_x, self.oak_expected_3d_top_right_y, self.oak_expected_3d_center_x, self.oak_expected_3d_center_y, self.oak_expected_3d_bottom_left_x, self.oak_expected_3d_bottom_left_y, self.oak_expected_3d_bottom_right_x, self.oak_expected_3d_bottom_right_y

    # Pine Algorithm Core
    def pine_core(self,rgb_image,depth_image, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        self.pine_expected_3d_top_left_x = ((top_left_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.pine_expected_3d_top_left_y = ((top_left_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        self.pine_expected_3d_top_right_x = ((bottom_right_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.pine_expected_3d_top_right_y = ((top_left_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        self.pine_expected_3d_bottom_left_x = ((top_left_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.pine_expected_3d_bottom_left_y = ((bottom_right_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        self.pine_expected_3d_bottom_right_x = ((bottom_right_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.pine_expected_3d_bottom_right_y = ((bottom_right_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        self.pine_expected_3d_center_x = ((center_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.pine_expected_3d_center_y = ((center_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        
        print("self.pine_expected_3d_top_left_x : {}".format(self.pine_expected_3d_top_left_x))
        print("self.pine_expected_3d_top_right_x : {}".format(self.pine_expected_3d_top_right_x))
        print("self.pine_expected_3d_top_left_y : {}".format(self.pine_expected_3d_top_left_y))
        print("self.pine_expected_3d_bottom_left_y : {}".format(self.pine_expected_3d_bottom_left_y))
        print("self.depth_array[center_y,center_x] : {}".format(self.depth_array[center_y,center_x]))
        print("self.top_left_x : {}".format(top_left_x))
        print("self.rgb_u : {}".format(self.rgb_u))
        print("self.rgb_fx : {}".format(self.rgb_fx))

        if (abs(self.pine_expected_3d_top_left_x)>100)&(abs(self.pine_expected_3d_top_left_y)>100)&(abs(self.pine_expected_3d_top_right_x)>100)&(abs(self.pine_expected_3d_top_right_y)>100)&(abs(self.pine_expected_3d_bottom_left_x)>100)&(abs(self.pine_expected_3d_bottom_left_y)>100)&(abs(self.pine_expected_3d_bottom_right_y)>100)&(abs(self.pine_expected_3d_bottom_right_y)>100):
            self.pine_expected_3d_top_left_x = self.pine_expected_3d_top_left_x/1000.0
            self.pine_expected_3d_top_left_y = self.pine_expected_3d_top_left_y/1000.0
            self.pine_expected_3d_top_right_x = self.pine_expected_3d_top_right_x/1000.0
            self.pine_expected_3d_top_right_y = self.pine_expected_3d_top_right_y/1000.0
            self.pine_expected_3d_bottom_left_x = self.pine_expected_3d_bottom_left_x/1000.0
            self.pine_expected_3d_bottom_left_y = self.pine_expected_3d_bottom_left_y/1000.0
            self.pine_expected_3d_bottom_right_x = self.pine_expected_3d_bottom_right_x/1000.0
            self.pine_expected_3d_bottom_right_y = self.pine_expected_3d_bottom_right_y/1000.0
            self.pine_expected_3d_center_x = self.pine_expected_3d_center_x/1000.0
            self.pine_expected_3d_center_y = self.pine_expected_3d_center_y/1000.0
            print ("Successfully divide 1000")

        self.pine_box.header.stamp = rospy.Time.now()
        self.pine_box.header.frame_id = "d435_depth_optical_frame"
        self.pine_box.pose.orientation.w = 1
        self.pine_box.pose.position.x = self.pine_expected_3d_center_x # increase in program  = move to right in rviz 
        self.pine_box.pose.position.y = self.pine_expected_3d_center_y # increase in program  = downward in rviz
        self.pine_box.pose.position.z = self.depth_image[center_y,center_x]/1000.0 # increase in program  = move away in rviz (directly use the depth distance)
        self.pine_box.dimensions.x = abs(self.pine_expected_3d_top_left_x)-abs(self.pine_expected_3d_top_right_x)
        if (self.pine_expected_3d_top_left_x < 0)&(self.pine_expected_3d_top_right_x > 0):
            self.pine_box.dimensions.x = abs(self.pine_expected_3d_top_left_x)+abs(self.pine_expected_3d_top_right_x)
        self.pine_box.dimensions.x = abs(self.pine_box.dimensions.x)
        self.pine_box.dimensions.y = abs(self.pine_expected_3d_top_left_y)-abs(self.pine_expected_3d_bottom_right_y)
        if (self.pine_expected_3d_top_left_y < 0)&(self.pine_expected_3d_bottom_right_y > 0):
            self.pine_box.dimensions.y = abs(self.pine_expected_3d_top_left_y)+abs(self.pine_expected_3d_bottom_right_y)
        self.pine_box.dimensions.y = abs(self.pine_box.dimensions.y)

        print("self.pine_box.dimensions.x : {}".format(self.pine_box.dimensions.x))
        print("self.pine_box.dimensions.y : {}".format(self.pine_box.dimensions.y))
        self.pine_box.dimensions.z = 1
        #self.box_pub.publish(self.pine_box)
        #print("pine box publish finish")
        #return self.pine_expected_3d_top_left_x, self.pine_expected_3d_top_left_y, self.pine_expected_3d_top_right_x, self.pine_expected_3d_top_right_y, self.pine_expected_3d_center_x, self.pine_expected_3d_center_y, self.pine_expected_3d_bottom_left_x, self.pine_expected_3d_bottom_left_y, self.pine_expected_3d_bottom_right_x, self.pine_expected_3d_bottom_right_y

    # Tree1 Algorithm Core
    def tree1_core(self,rgb_image,depth_image, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        self.tree1_expected_3d_top_left_x = ((top_left_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree1_expected_3d_top_left_y = ((top_left_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        self.tree1_expected_3d_top_right_x = ((bottom_right_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree1_expected_3d_top_right_y = ((top_left_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        self.tree1_expected_3d_bottom_left_x = ((top_left_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree1_expected_3d_bottom_left_y = ((bottom_right_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        self.tree1_expected_3d_bottom_right_x = ((bottom_right_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree1_expected_3d_bottom_right_y = ((bottom_right_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        self.tree1_expected_3d_center_x = ((center_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree1_expected_3d_center_y = ((center_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        if (abs(self.tree1_expected_3d_top_left_x)>100)&(abs(self.tree1_expected_3d_top_left_y)>100)&(abs(self.tree1_expected_3d_top_right_x)>100)&(abs(self.tree1_expected_3d_top_right_y)>100)&(abs(self.tree1_expected_3d_bottom_left_x)>100)&(abs(self.tree1_expected_3d_bottom_left_y)>100)&(abs(self.tree1_expected_3d_bottom_right_y)>100)&(abs(self.tree1_expected_3d_bottom_right_y)>100):
            self.tree1_expected_3d_top_left_x = self.tree1_expected_3d_top_left_x/1000.0
            self.tree1_expected_3d_top_left_y = self.tree1_expected_3d_top_left_y/1000.0
            self.tree1_expected_3d_top_right_x = self.tree1_expected_3d_top_right_x/1000.0
            self.tree1_expected_3d_top_right_y = self.tree1_expected_3d_top_right_y/1000.0
            self.tree1_expected_3d_bottom_left_x = self.tree1_expected_3d_bottom_left_x/1000.0
            self.tree1_expected_3d_bottom_left_y = self.tree1_expected_3d_bottom_left_y/1000.0
            self.tree1_expected_3d_bottom_right_x = self.tree1_expected_3d_bottom_right_x/1000.0
            self.tree1_expected_3d_bottom_right_y = self.tree1_expected_3d_bottom_right_y/1000.0
            self.tree1_expected_3d_center_x = self.tree1_expected_3d_center_x/1000.0
            self.tree1_expected_3d_center_y = self.tree1_expected_3d_center_y/1000.0
            print ("Successfully divide 1000")

        self.tree1_box.header.stamp = rospy.Time.now()
        self.tree1_box.header.frame_id = "d435_depth_optical_frame"
        self.tree1_box.pose.orientation.w = 1
        self.tree1_box.pose.position.x = self.tree1_expected_3d_center_x # increase in program  = move to right in rviz 
        self.tree1_box.pose.position.y = self.tree1_expected_3d_center_y # increase in program  = downward in rviz
        self.tree1_box.pose.position.z = self.depth_image[center_y,center_x]/1000.0 # increase in program  = move away in rviz (directly use the depth distance)
        self.tree1_box.dimensions.x = abs(self.tree1_expected_3d_top_left_x)-abs(self.tree1_expected_3d_top_right_x)
        if (self.tree1_expected_3d_top_left_x < 0)&(self.tree1_expected_3d_top_right_x > 0):
            self.tree1_box.dimensions.x = abs(self.tree1_expected_3d_top_left_x)+abs(self.tree1_expected_3d_top_right_x)
        self.tree1_box.dimensions.x = abs(self.tree1_box.dimensions.x)
        self.tree1_box.dimensions.y = abs(self.tree1_expected_3d_top_left_y)-abs(self.tree1_expected_3d_bottom_right_y)
        if (self.tree1_expected_3d_top_left_y < 0)&(self.tree1_expected_3d_bottom_right_y > 0):
            self.tree1_box.dimensions.y = abs(self.tree1_expected_3d_top_left_y)+abs(self.tree1_expected_3d_bottom_right_y)
        self.tree1_box.dimensions.y = abs(self.tree1_box.dimensions.y)

        print("self.tree1_box.dimensions.x : {}".format(self.tree1_box.dimensions.x))
        print("self.tree1_box.dimensions.y : {}".format(self.tree1_box.dimensions.y))
        self.tree1_box.dimensions.z = 1
        #self.box_pub.publish(self.tree1_box)
        #print("tree1 box publish finish")
        #return self.tree1_expected_3d_top_left_x, self.tree1_expected_3d_top_left_y, self.tree1_expected_3d_top_right_x, self.tree1_expected_3d_top_right_y, self.tree1_expected_3d_center_x, self.tree1_expected_3d_center_y, self.tree1_expected_3d_bottom_left_x, self.tree1_expected_3d_bottom_left_y, self.tree1_expected_3d_bottom_right_x, self.tree1_expected_3d_bottom_right_y

    # Tree2 Algorithm Core
    def tree2_core(self,rgb_image,depth_image, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        self.tree2_expected_3d_top_left_x = ((top_left_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree2_expected_3d_top_left_y = ((top_left_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        self.tree2_expected_3d_top_right_x = ((bottom_right_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree2_expected_3d_top_right_y = ((top_left_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        self.tree2_expected_3d_bottom_left_x = ((top_left_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree2_expected_3d_bottom_left_y = ((bottom_right_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy
        self.tree2_expected_3d_bottom_right_x = ((bottom_right_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree2_expected_3d_bottom_right_y = ((bottom_right_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        self.tree2_expected_3d_center_x = ((center_x - self.rgb_u)*self.depth_array[center_y,center_x])/self.rgb_fx
        self.tree2_expected_3d_center_y = ((center_y - self.rgb_v)*self.depth_array[center_y,center_x])/self.rgb_fy

        if (abs(self.tree2_expected_3d_top_left_x)>100)&(abs(self.tree2_expected_3d_top_left_y)>100)&(abs(self.tree2_expected_3d_top_right_x)>100)&(abs(self.tree2_expected_3d_top_right_y)>100)&(abs(self.tree2_expected_3d_bottom_left_x)>100)&(abs(self.tree2_expected_3d_bottom_left_y)>100)&(abs(self.tree2_expected_3d_bottom_right_y)>100)&(abs(self.tree2_expected_3d_bottom_right_y)>100):
            self.tree2_expected_3d_top_left_x = self.tree2_expected_3d_top_left_x/1000.0
            self.tree2_expected_3d_top_left_y = self.tree2_expected_3d_top_left_y/1000.0
            self.tree2_expected_3d_top_right_x = self.tree2_expected_3d_top_right_x/1000.0
            self.tree2_expected_3d_top_right_y = self.tree2_expected_3d_top_right_y/1000.0
            self.tree2_expected_3d_bottom_left_x = self.tree2_expected_3d_bottom_left_x/1000.0
            self.tree2_expected_3d_bottom_left_y = self.tree2_expected_3d_bottom_left_y/1000.0
            self.tree2_expected_3d_bottom_right_x = self.tree2_expected_3d_bottom_right_x/1000.0
            self.tree2_expected_3d_bottom_right_y = self.tree2_expected_3d_bottom_right_y/1000.0
            self.tree2_expected_3d_center_x = self.tree2_expected_3d_center_x/1000.0
            self.tree2_expected_3d_center_y = self.tree2_expected_3d_center_y/1000.0
            print ("Successfully divide 1000")

        self.tree2_box.header.stamp = rospy.Time.now()
        self.tree2_box.header.frame_id = "d435_depth_optical_frame"
        self.tree2_box.pose.orientation.w = 1
        self.tree2_box.pose.position.x = self.tree2_expected_3d_center_x # increase in program  = move to right in rviz 
        self.tree2_box.pose.position.y = self.tree2_expected_3d_center_y # increase in program  = downward in rviz
        self.tree2_box.pose.position.z = self.depth_image[center_y,center_x]/1000.0 # increase in program  = move away in rviz (directly use the depth distance)
        self.tree2_box.dimensions.x = abs(self.tree2_expected_3d_top_left_x)-abs(self.tree2_expected_3d_top_right_x)
        if (self.tree2_expected_3d_top_left_x < 0)&(self.tree2_expected_3d_top_right_x > 0):
            self.tree2_box.dimensions.x = abs(self.tree2_expected_3d_top_left_x)+abs(self.tree2_expected_3d_top_right_x)
        self.tree2_box.dimensions.x = abs(self.tree2_box.dimensions.x)
        self.tree2_box.dimensions.y = abs(self.tree2_expected_3d_top_left_y)-abs(self.tree2_expected_3d_bottom_right_y)
        if (self.tree2_expected_3d_top_left_y < 0)&(self.tree2_expected_3d_bottom_right_y > 0):
            self.tree2_box.dimensions.y = abs(self.tree2_expected_3d_top_left_y)+abs(self.tree2_expected_3d_bottom_right_y)
        self.tree2_box.dimensions.y = abs(self.tree2_box.dimensions.y)

        print("self.tree2_box.dimensions.x : {}".format(self.tree2_box.dimensions.x))
        print("self.tree2_box.dimensions.y : {}".format(self.tree2_box.dimensions.y))
        self.tree2_box.dimensions.z = 1
        #self.box_pub.publish(self.tree2_box)
        #print("tree2 box publish finish")
        #return self.tree2_expected_3d_top_left_x, self.tree2_expected_3d_top_left_y, self.tree2_expected_3d_top_right_x, self.tree2_expected_3d_top_right_y, self.tree2_expected_3d_center_x, self.tree2_expected_3d_center_y, self.tree2_expected_3d_bottom_left_x, self.tree2_expected_3d_bottom_left_y, self.tree2_expected_3d_bottom_right_x, self.tree2_expected_3d_bottom_right_y


    def image_detect(self):

        self.fps = 0.0
        while(True):
            rospy.wait_for_message("/d435/color/image_raw",Image)
            rospy.wait_for_message("/d435/depth/image_raw",Image)
            rospy.wait_for_message("/d435/depth/camera_info",CameraInfo)
            self.t1 = time.time()
            # Read a frame
            # frame is ndarray
            # convert into Image
            self.frame = PIL_Image.fromarray(np.uint8(self.img))
            # perform detection
            self.frame = np.array(self.yolo.detect_image(self.frame))
            # get the detection result
            self.oak_list, self.pine_list, self.tree1_list, self.tree2_list= self.yolo.detection_result()
            self.oak_box_center_x = float((self.oak_list[5]+self.oak_list[3])/2)
            self.oak_box_center_y = float((self.oak_list[4]+self.oak_list[2])/2)
            self.pine_box_center_x = float((self.pine_list[5]+self.pine_list[3])/2)
            self.pine_box_center_y = float((self.pine_list[4]+self.pine_list[2])/2)
            self.tree1_box_center_x = float((self.tree1_list[5]+self.tree1_list[3])/2)
            self.tree1_box_center_y = float((self.tree1_list[4]+self.tree1_list[2])/2)
            self.tree2_box_center_x = float((self.tree2_list[5]+self.tree2_list[3])/2)
            self.tree2_box_center_y = float((self.tree2_list[4]+self.tree2_list[2])/2)

            """
            print ("oak_name : {}".format(self.oak_list[0]))
            print ("oak_score : {}".format(self.oak_list[1]))
            print ("oak_top_left_y : {}".format(self.oak_list[2]))
            print ("oak_top_left_x : {}".format(self.oak_list[3]))
            print ("oak_bottom_right_y : {}".format(self.oak_list[4]))
            print ("oak_bottom_right_x : {}".format(self.oak_list[5]))
            print ("oak_box_center_x : {}".format(self.oak_box_center_y))
            print ("oak_box_center_x : {}".format(self.oak_box_center_x))

            print ("pine_name : {}".format(self.pine_list[0]))
            print ("pine_score : {}".format(self.pine_list[1]))
            print ("pine_top_left_y : {}".format(self.pine_list[2]))
            print ("pine_top_left_x : {}".format(self.pine_list[3]))
            print ("pine_bottom_right_y : {}".format(self.pine_list[4]))
            print ("pine_bottom_right_x : {}".format(self.pine_list[5]))
            print ("pine_box_center_x : {}".format(self.pine_box_center_x))
            print ("pine_box_center_y : {}".format(self.pine_box_center_y))

            print ("tree1_name : {}".format(self.tree1_list[0]))
            print ("tree1_score : {}".format(self.tree1_list[1]))
            print ("tree1_top_left_y : {}".format(self.tree1_list[2]))
            print ("tree1_top_left_x : {}".format(self.tree1_list[3]))
            print ("tree1_bottom_right_y : {}".format(self.tree1_list[4]))
            print ("tree1_bottom_right_x : {}".format(self.tree1_list[5]))
            print ("tree1_box_center_x : {}".format(self.tree1_box_center_x))
            print ("tree1_box_center_y : {}".format(self.tree1_box_center_y))

            print ("tree2_name : {}".format(self.tree2_list[0]))
            print ("tree2_score : {}".format(self.tree2_list[1]))
            print ("tree2_top_left_y : {}".format(self.tree2_list[2]))
            print ("tree2_top_left_x : {}".format(self.tree2_list[3]))
            print ("tree2_bottom_right_y : {}".format(self.tree2_list[4]))
            print ("tree2_bottom_right_x : {}".format(self.tree2_list[5]))
            print ("tree2_box_center_x : {}".format(self.tree2_box_center_x))
            print ("tree2_box_center_y : {}".format(self.tree2_box_center_y))
            """

            # convert from RGB to BGR to fulfil the opencv format
            self.frame = cv2.cvtColor(self.frame,cv2.COLOR_RGB2BGR)
            self.fps  = ( self.fps + (1./(time.time()-self.t1)) ) / 2
            #print("fps= %.2f"%(self.fps))
            self.frame = cv2.putText(self.frame, "fps= %.2f"%(self.fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #buffer time
            self.oak_core(self.rgb_image,self.depth_array,self.oak_list[3],self.oak_list[2],self.oak_list[5],self.oak_list[4],int(self.oak_box_center_x),int(self.oak_box_center_y))
            self.pine_core(self.rgb_image,self.depth_array,self.pine_list[3],self.pine_list[2],self.pine_list[5],self.pine_list[4],int(self.pine_box_center_x),int(self.pine_box_center_y))
            self.tree1_core(self.rgb_image,self.depth_array,self.tree1_list[3],self.tree1_list[2],self.tree1_list[5],self.tree1_list[4],int(self.tree1_box_center_x),int(self.tree1_box_center_y))
            self.tree2_core(self.rgb_image,self.depth_array,self.tree2_list[3],self.tree2_list[2],self.tree2_list[5],self.tree2_list[4],int(self.tree2_box_center_x),int(self.tree2_box_center_y))
            
            self.boxes.boxes = [self.oak_box, self.pine_box, self.tree1_box, self.tree2_box]
            self.boxes.header.frame_id = "d435_depth_optical_frame"
            self.boxes.header.stamp = rospy.Time.now()
            self.boxarray_pub.publish(self.boxes)
            
            cv2.imshow("d435 rgb",self.frame)
            print("open camera")
            
            self.c= cv2.waitKey(1) & 0xff 

            if self.c==27:
                cv2.capture.release()
                break
        
        cv2.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect = object_detect()
    while not rospy.is_shutdown():
        #try:
        detect.image_detect()
        #except:
        #    print("Reloading")

