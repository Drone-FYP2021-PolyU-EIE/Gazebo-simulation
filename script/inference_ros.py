import colorsys
import os
import time

import pyrealsense2 as rs2
import numpy as np
import cv2
import cv_bridge
import torch
import torch.nn as nn
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

        # Subscribe camera info [depth_rgb aligned]
        rospy.Subscriber("/d435/depth/camera_info",CameraInfo,self.camera_info_callback)

        self.oak_box_pub = rospy.Publisher("/desired/input/box", BoundingBox, queue_size=1)
        self.oak_box = BoundingBox()
        print ("init finish!")

    def depth_callback(self,data):
        # Depth image callback
        self.depth_image = self.bridge.imgmsg_to_cv2(data)
        self.depth_image = cv2.resize(self.depth_image,(640,480))   # 640 width, 480 height
        self.depth_image = np.array(self.depth_image, dtype=np.float32)
        #Display
        #cv2.imshow('depth_image',self.depth_image)
        #cv2.waitKey(1)

    def color_callback(self,data):
        # RGB image callback
        self.rgb_image = self.bridge.imgmsg_to_cv2(data)
        self.rgb_image = cv2.resize(self.rgb_image,(640,480))   # 640 width, 480 height
        self.img = self.rgb_image
        return self.img
        #self.rgb_image = cv2.cvtColor(self.rgb_image,cv2.COLOR_RGB2BGR)
        # Display
        #cv2.imshow('rgb_image',self.rgb_image)
        #cv2.waitKey(1)    

    # this is depth camera info callback
    def camera_info_callback(self, data):
        self.intrinsic = rs2.intrinsics()
        self.intrinsic.height = data.height
        self.intrinsic.width = data.width

        # Pixels coordinates of the principal point (center of projection)
        self.intrinsic.ppx = data.K[2]
        self.intrinsic.ppy = data.K[5]

        # Focal Length of image (multiple of pixel width and height)
        self.intrinsic.fx = data.K[0]
        self.intrinsic.fy = data.K[4]

        self.model = rs2.distortion.none
        self.coeffs = [i for i in data.D]

    # Finding camera link in the world
    def get_tf_transform(self,source_link,target_link):
        self.trans = [0,0,0]
        self.rot = [0,0,0]
        self.r,self.p,self.y = 0,0,0

        self.tf_listener.waitForTransform(source_link,target_link,rospy.Time(),rospy.Duration(1.0))
        
        self.now = rospy.Time.now()
        self.tf_listener.waitForTransform(source_link,target_link,self.now,rospy.Duration(1.0))
        (self.trans,self.rot) = self.tf_listener.lookupTransform(source_link,target_link,self.now)
        #trans[0]->X, trans[1]->Y, trans[2]->Z
        #rot[0]->X,rot[1]->Y,rot[2]->Z,rot[3]->W

        self.r,self.p,self.y = self.euler_from_quaternion(self.rot)

        self.rot_radian = []
        self.rot_radian.append(self.r)
        self.rot_radian.append(self.p)
        self.rot_radian.append(self.y)

        return self.trans,self.rot_radian

    # Find Z distances for bounding box markers (default depth unit is mm)
    def box_point_z_distance(self,center_x,center_y):
        """
        Return list[center point distance]
        """
        self.output = []
        self.center = self.depth_image[center_y, center_x]/1000
        #print("center point distance: {}".format(self.e))
        self.output.append(self.e)
        return self.output

    # Algorithm Core
    def core(self,rgb_image,depth_image,intrinsic, center_x, center_y):
        # TF Calculate from world to camera_link (realsense)
        self.realsense_trans,self.realsense_rot = self.get_tf_transform('base_link','d435_depth_optical_frame')
        # Image to world transformation for the bounding box center
        # Find the bounding box point z distance (in m)
        self.center_z = self.box_point_z_distance(self.depth_image,center_x,center_y)
        # Center Point
        self.center_world = rs2.rs2_deproject_pixel_to_point(intrinsic,[center_x,center_y],self.center_z[0])
        print("center_world : {}".format(self.center_world))

        # Center Point
        if(self.center_z[0]!=0):

            self.center_world_x = self.center_world[1]/100 + self.realsense_trans[0]
            self.center_world_y = self.center_world[0]/100 + self.realsense_trans[1]
            self.center_world_z = self.center_world[2]

            print("center_world[0] (y-axis) : {}".format(self.center_world[0]))
            print("center_world[1] (x-axis) : {}".format(self.center_world[1]))
        
            print("realsense_trans[0] : {}".format(self.realsense_trans[0]))
            print("realsense_trans[1] : {}".format(self.realsense_trans[1]))
            print("realsense_trans[2] : {}".format(self.realsense_trans[2]))
            print("center_world_x : {}".format(self.center_world_x))
            print("center_world_y : {}".format(self.center_world_y))

        self.oak_box.header.stamp = rospy.Time.now()
        self.oak_box.header.frame_id = "d435_depth_optical_frame"
        self.oak_box.pose.orientation.w = 1

        self.oak_box.pose.position.x = self.center_world_x # increase in program  = move to right in rviz (directly use center_world_x)
        self.oak_box.pose.position.y = self.center_world_y # increase in program  = downward in rviz (directly use center_world_y)
        self.oak_box.pose.position.z = self.center_world_z # increase in program  = move away in rviz (directly use the depth distance)
        self.oak_box.dimensions.x = self.oak_bottom_right_x-self.oak_top_left_x # dimenson same as pose.position.x dimenson (pixel width for the detection box)
        self.oak_box.dimensions.y = self.oak_bottom_right_y-self.oak_top_left_y # dimenson same as pose.position.y dimenson (pixel height for the detection box)
        self.oak_box.dimensions.z = 0.5 # dimenson same as pose.position.z dimenson (should be the minimum height of the tree trunk)

        self.oak_box_pub.publish(self.oak_box)
        print ("core finish")

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
            self.oak_name, self.oak_score, self.oak_top_left_y, self.oak_top_left_x, self.oak_bottom_right_y, self.oak_bottom_right_x, self.pine_name, self.pine_score, self.pine_top_left_y, self.pine_top_left_x, self.pine_bottom_right_y, self.pine_bottom_right_x, self.tree1_name, self.tree1_score, self.tree1_top_left_y, self.tree1_top_left_x, self.tree1_bottom_right_y, self.tree1_bottom_right_x, self.tree2_name, self.tree2_score, self.tree2_top_left_y, self.tree2_top_left_x, self.tree2_bottom_right_y, self.tree2_bottom_right_x= self.yolo.detection_result()
            print ("oak_name : {}".format(self.oak_name))
            print ("oak_score : {}".format(self.oak_score))
            print ("oak_top_left_y : {}".format(self.oak_top_left_y))
            print ("oak_top_left_x : {}".format(self.oak_top_left_x))
            print ("oak_bottom_right_y : {}".format(self.oak_bottom_right_y))
            print ("oak_bottom_right_x : {}".format(self.oak_bottom_right_x))

            print ("pine_name : {}".format(self.pine_name))
            print ("pine_score : {}".format(self.pine_score))
            print ("pine_top_left_y : {}".format(self.pine_top_left_y))
            print ("pine_top_left_x : {}".format(self.pine_top_left_x))
            print ("pine_bottom_right_y : {}".format(self.pine_bottom_right_y))
            print ("pine_bottom_right_x : {}".format(self.pine_bottom_right_x))

            print ("tree1_name : {}".format(self.tree1_name))
            print ("tree1_score : {}".format(self.tree1_score))
            print ("tree1_top_left_y : {}".format(self.tree1_top_left_y))
            print ("tree1_top_left_x : {}".format(self.tree1_top_left_x))
            print ("tree1_bottom_right_y : {}".format(self.tree1_bottom_right_y))
            print ("tree1_bottom_right_x : {}".format(self.tree1_bottom_right_x))

            print ("tree2_name : {}".format(self.tree2_name))
            print ("tree2_score : {}".format(self.tree2_score))
            print ("tree2_top_left_y : {}".format(self.tree2_top_left_y))
            print ("tree2_top_left_x : {}".format(self.tree2_top_left_x))
            print ("tree2_bottom_right_y : {}".format(self.tree2_bottom_right_y))
            print ("tree2_bottom_right_x : {}".format(self.tree2_bottom_right_x))
            # convert from RGB to BGR to fulfil the opencv format
            self.frame = cv2.cvtColor(self.frame,cv2.COLOR_RGB2BGR)
            self.fps  = ( self.fps + (1./(time.time()-self.t1)) ) / 2
            #print("fps= %.2f"%(self.fps))
            self.frame = cv2.putText(self.frame, "fps= %.2f"%(self.fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.core(self.rgb_image,self.depth_image,self.intrinsic,(self.oak_top_left_x+self.oak_bottom_right_x)/2,(self.oak_top_left_y+self.oak_bottom_right_y)/2)
            cv2.imshow("d435 rgb",self.frame)
            print("open camera")
            
            self.c= cv2.waitKey(1) & 0xff 

            if self.c==27:
                capture.release()
                break
        
        capture.release()
        out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect = object_detect()
    while not rospy.is_shutdown():
        try:
            detect.image_detect()
        except:
            print ("Reloading")

