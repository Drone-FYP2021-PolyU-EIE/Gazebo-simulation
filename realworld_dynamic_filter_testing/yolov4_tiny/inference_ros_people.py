#!/usr/bin/env python2.7

import time
import tf
import numpy as np
import cv2
import cv_bridge
from PIL import ImageDraw, ImageFont
from PIL import Image as PIL_Image
from cv_bridge.boost.cv_bridge_boost import getCvType

# ros package
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import PoseStamped

# custom package
from yolo4_tiny import YoloBody
from utils import (DecodeBox, letterbox_image, non_max_suppression, yolo_correct_boxes)
from yolo import YOLO


class object_detect:
    def __init__(self):
        self.fine_tuning = 0.0
        self.bridge = cv_bridge.CvBridge()
        self.yolo = YOLO()
        rospy.init_node('object_detection')
        
        # Subscribe color and depth image
        rospy.Subscriber("/camera/color/image_raw",Image,self.color_callback)
        rospy.Subscriber("/camera/depth/image_rect_raw",Image,self.depth_callback)
        #rospy.Subscriber("/camera/depth/color/points",PointCloud2,self.pointcloud2_callback)

        # Subscribe camera info
        rospy.Subscriber("/camera/depth/camera_info",CameraInfo,self.depth_camera_info_callback)
        rospy.Subscriber("/camera/color/camera_info",CameraInfo,self.color_camera_info_callback)

        #self.box_pub = rospy.Publisher("/desired/input/box", BoundingBox, queue_size=1)
        self.box_array_pub = rospy.Publisher("/desired/input/box_array", BoundingBoxArray, queue_size=1)


        self.box_array = BoundingBoxArray()

        self.obstacle = BoundingBox()
        self.obstacle2 = BoundingBox()
        self.obstacle3 = BoundingBox()
        self.obstacle4 = BoundingBox()
        self.obstacle5 = BoundingBox()

        self.human = BoundingBox()
        self.human2 = BoundingBox()
        self.human3 = BoundingBox()
        self.human4 = BoundingBox()
        self.human5 = BoundingBox()

        print ("init finish!")

    def depth_callback(self,data):
        # Depth image callback
        self.depth_image = self.bridge.imgmsg_to_cv2(data)
        #self.depth_image = cv2.resize(self.depth_image,(640,480))   # 640 width, 480 height
        self.depth_image = np.array(self.depth_image, dtype=np.float32)
        self.depth_array = self.depth_image/1000.0
        #print(type(self.depth_array))
        #print(self.depth_array[320,240])
        #Display
        #cv2.imshow('depth_image',self.depth_array)
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

    def transformation(self,top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        
        if (top_left_x==0)&(top_left_y==0)&(bottom_right_x==0)&(bottom_right_y==0):
            expected_3d_center_distance=0.0
            expected_3d_center_x=0.0
            expected_3d_center_y=0.0
            expected_3d_top_left_x=0.0
            expected_3d_top_left_y=0.0
            expected_3d_bottom_right_x=0.0
            expected_3d_bottom_right_y=0.0

        distance_z = self.depth_array[center_y,center_x]
        expected_3d_center_distance = distance_z
        expected_3d_top_left_x = ((top_left_x - self.rgb_u)*distance_z)/self.rgb_fx
        expected_3d_top_left_y = ((top_left_y - self.rgb_v)*distance_z)/self.rgb_fy

        expected_3d_bottom_right_x = ((bottom_right_x - self.rgb_u)*distance_z)/self.rgb_fx
        expected_3d_bottom_right_y = ((bottom_right_y - self.rgb_v)*distance_z)/self.rgb_fy

        expected_3d_center_x = ((center_x - self.rgb_u)*distance_z)/self.rgb_fx
        expected_3d_center_y = ((center_y - self.rgb_v)*distance_z)/self.rgb_fy
        """
        print("self.expected_3d_top_left_x : {}".format(expected_3d_top_left_x))
        print("self.top_left_x : {}".format(top_left_x))
        print("self.rgb_u : {}".format(self.rgb_u))
        print("distance_z : {}".format(distance_z))
        print("self.rgb_fx : {}".format(self.rgb_fx))
        print("center_y : {}".format(center_y))
        print("center_x : {}".format(center_x))
        
        print("self.human_expected_3d_top_left_x : {}".format(expected_3d_top_left_x))
        print("self.human_expected_3d_top_left_y : {}".format(expected_3d_top_left_y))
        print("self.human_expected_3d_bottom__right_x : {}".format(expected_3d_bottom_right_x))
        print("self.human_expected_3d_bottom_right_y : {}".format(expected_3d_bottom_right_y))
        print("distance_z : {}".format(distance_z))
        print("self.top_left_x : {}".format(top_left_x))
        print("self.rgb_u : {}".format(self.rgb_u))
        print("self.rgb_fx : {}".format(self.rgb_fx))


        print("_____________________________________")
        print("top_left_x : {}".format(top_left_x))
        print("top_left_y : {}".format(top_left_y))
        print("bottom_right_x : {}".format(bottom_right_x))
        print("bottom_right_y : {}".format(bottom_right_y))
        print("center_x : {}".format(center_x))
        print("center_y : {}".format(center_y))
        print("rgb_u : {}".format(self.rgb_u))
        print("rgb_v : {}".format(self.rgb_v))
        print("expected_3d_center_distance : {}".format(expected_3d_center_distance))
        print("expected_3d_top_left_x : {}".format(expected_3d_top_left_x))
        print("expected_3d_top_left_y : {}".format(expected_3d_top_left_y))
        print("expected_3d_bottom_right_x : {}".format(expected_3d_bottom_right_x))
        print("expected_3d_bottom_right_y : {}".format(expected_3d_bottom_right_y))
        print("expected_3d_center_x : {}".format(expected_3d_center_x))
        print("expected_3d_center_y : {}".format(expected_3d_center_y))
        """
        return expected_3d_center_distance,expected_3d_center_x,expected_3d_center_y, expected_3d_top_left_x, expected_3d_top_left_y, expected_3d_bottom_right_x, expected_3d_bottom_right_y


    def box_calculation(self,center_x,center_y,top_left_x,top_left_y,bottom_right_x,bottom_right_y):
        box_dimensions_x = abs(top_left_x)-abs(bottom_right_x)
        if (top_left_x < 0)&(bottom_right_x > 0):
            box_dimensions_x = abs(top_left_x)+abs(bottom_right_x)
        box_dimensions_x = abs(box_dimensions_x)

        box_dimensions_y = abs(top_left_y)-abs(bottom_right_y)
        if (top_left_y < 0)&(bottom_right_y > 0):
            box_dimensions_y = abs(top_left_y)+abs(bottom_right_y)
        box_dimensions_y = abs(box_dimensions_y)

        box_position_x = center_x
        box_position_y = center_y
        box_dimension_z = 1

        return box_dimensions_x,box_dimensions_y,box_dimension_z,box_position_x,box_position_y

    # obstacle Algorithm Core
    def obstacle_core(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.obstacle.header.stamp = rospy.Time.now()
        self.obstacle.header.frame_id = "camera_depth_optical_frame"

        self.obstacle.pose.orientation.w =  1
        self.obstacle.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.obstacle.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.obstacle.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.obstacle.dimensions.x = box_size_x
        self.obstacle.dimensions.y = box_size_y
        self.obstacle.dimensions.z = box_size_z
    
    # obstacle2 Algorithm Core
    def obstacle_core2(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.obstacle2.header.stamp = rospy.Time.now()
        self.obstacle2.header.frame_id = "camera_depth_optical_frame"

        self.obstacle2.pose.orientation.w =  1
        self.obstacle2.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.obstacle2.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.obstacle2.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.obstacle2.dimensions.x = box_size_x
        self.obstacle2.dimensions.y = box_size_y
        self.obstacle2.dimensions.z = box_size_z
    
    # obstacle3 Algorithm Core
    def obstacle_core3(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.obstacle3.header.stamp = rospy.Time.now()
        self.obstacle3.header.frame_id = "camera_depth_optical_frame"

        self.obstacle3.pose.orientation.w =  1
        self.obstacle3.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.obstacle3.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.obstacle3.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.obstacle3.dimensions.x = box_size_x
        self.obstacle3.dimensions.y = box_size_y
        self.obstacle3.dimensions.z = box_size_z

    # obstacle4 Algorithm Core
    def obstacle_core4(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.obstacle4.header.stamp = rospy.Time.now()
        self.obstacle4.header.frame_id = "camera_depth_optical_frame"

        self.obstacle4.pose.orientation.w =  1
        self.obstacle4.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.obstacle4.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.obstacle4.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.obstacle4.dimensions.x = box_size_x
        self.obstacle4.dimensions.y = box_size_y
        self.obstacle4.dimensions.z = box_size_z

    # obstacle5 Algorithm Core
    def obstacle_core5(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.obstacle5.header.stamp = rospy.Time.now()
        self.obstacle5.header.frame_id = "camera_depth_optical_frame"

        self.obstacle5.pose.orientation.w =  1
        self.obstacle5.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.obstacle5.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.obstacle5.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.obstacle5.dimensions.x = box_size_x
        self.obstacle5.dimensions.y = box_size_y
        self.obstacle5.dimensions.z = box_size_z

    # human Algorithm Core
    def human_core(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.human.header.stamp = rospy.Time.now()
        self.human.header.frame_id = "camera_depth_optical_frame"

        self.human.pose.orientation.w = 1
        self.human.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.human.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.human.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.human.dimensions.x = box_size_x
        self.human.dimensions.y = box_size_y
        self.human.dimensions.z = box_size_z

    # human2 Algorithm Core
    def human_core2(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.human2.header.stamp = rospy.Time.now()
        self.human2.header.frame_id = "camera_depth_optical_frame"

        self.human2.pose.orientation.w =  1
        self.human2.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.human2.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.human2.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.human2.dimensions.x = box_size_x
        self.human2.dimensions.y = box_size_y
        self.human2.dimensions.z = box_size_z

    # human3 Algorithm Core
    def human_core3(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.human3.header.stamp = rospy.Time.now()
        self.human3.header.frame_id = "camera_depth_optical_frame"

        self.human3.pose.orientation.w =  1
        self.human3.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.human3.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.human3.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.human3.dimensions.x = box_size_x
        self.human3.dimensions.y = box_size_y
        self.human3.dimensions.z = box_size_z

    # human4 Algorithm Core
    def human_core4(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.human4.header.stamp = rospy.Time.now()
        self.human4.header.frame_id = "camera_depth_optical_frame"

        self.human4.pose.orientation.w =  1
        self.human4.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.human4.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.human4.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.human4.dimensions.x = box_size_x
        self.human4.dimensions.y = box_size_y
        self.human4.dimensions.z = box_size_z

    # human5 Algorithm Core
    def human_core5(self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        box_position_z,cx,cy,x1,y1,x2,y2 = self.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
        box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = self.box_calculation(cx,cy,x1,y1,x2,y2)

        self.human5.header.stamp = rospy.Time.now()
        self.human5.header.frame_id = "camera_depth_optical_frame"

        self.human5.pose.orientation.w =  1
        self.human5.pose.position.x = box_position_x # increase in program  = move to right in rviz 
        self.human5.pose.position.y = box_position_y # increase in program  = downward in rviz
        self.human5.pose.position.z = box_position_z-self.fine_tuning # increase in program  = move away in rviz (directly use the depth distance)
        self.human5.dimensions.x = box_size_x
        self.human5.dimensions.y = box_size_y
        self.human5.dimensions.z = box_size_z

    def image_detect(self):
        self.fps = 0.0
        self.number_of_obstacle=0
        self.number_of_people=0

        while(True):
            rospy.wait_for_message("/camera/color/image_raw",Image)
            #rospy.wait_for_message("/camera/depth/image_rect_raw",Image)
            #rospy.wait_for_message("/camera/color/camera_info",CameraInfo)
            self.t1 = time.time()
            # Read a frame
            # frame is ndarray
            # convert into Image
            self.frame = PIL_Image.fromarray(np.uint8(self.img))
            # perform detection
            self.frame = np.array(self.yolo.detect_image(self.frame))
            # get the detection result
            self.number_of_obstacle,self.number_of_people = self.yolo.detected_number_of_object()
            self.obstacle_list, self.obstacle_list2, self.obstacle_list3, self.obstacle_list4, self.obstacle_list5, self.human_list, self.human_list2, self.human_list3, self.human_list4, self.human_list5 = self.yolo.detection_result()
            
            #print("number_of_obstacle: {}".format(self.number_of_obstacle))
            #print("number_of_people: {}".format(self.number_of_people))

            self.obstacle_center_x = float((self.obstacle_list[5]+self.obstacle_list[3])/2)
            self.obstacle_center_y = float((self.obstacle_list[4]+self.obstacle_list[2])/2)
            self.obstacle_center_x2 = float((self.obstacle_list2[5]+self.obstacle_list2[3])/2)
            self.obstacle_center_y2 = float((self.obstacle_list2[4]+self.obstacle_list2[2])/2)
            self.obstacle_center_x3 = float((self.obstacle_list3[5]+self.obstacle_list3[3])/2)
            self.obstacle_center_y3 = float((self.obstacle_list3[4]+self.obstacle_list3[2])/2)
            self.obstacle_center_x4 = float((self.obstacle_list4[5]+self.obstacle_list4[3])/2)
            self.obstacle_center_y4 = float((self.obstacle_list4[4]+self.obstacle_list4[2])/2)
            self.obstacle_center_x5 = float((self.obstacle_list5[5]+self.obstacle_list5[3])/2)
            self.obstacle_center_y5 = float((self.obstacle_list5[4]+self.obstacle_list5[2])/2)

            self.human_center_x = float((self.human_list[5]+self.human_list[3])/2)
            self.human_center_y = float((self.human_list[4]+self.human_list[2])/2)
            self.human_center_x2 = float((self.human_list2[5]+self.human_list2[3])/2)
            self.human_center_y2 = float((self.human_list2[4]+self.human_list2[2])/2)
            self.human_center_x3 = float((self.human_list3[5]+self.human_list3[3])/2)
            self.human_center_y3 = float((self.human_list3[4]+self.human_list3[2])/2)
            self.human_center_x4 = float((self.human_list4[5]+self.human_list4[3])/2)
            self.human_center_y4 = float((self.human_list4[4]+self.human_list4[2])/2)
            self.human_center_x5 = float((self.human_list5[5]+self.human_list5[3])/2)
            self.human_center_y5 = float((self.human_list5[4]+self.human_list5[2])/2)

            if self.number_of_people==1:
                self.human_core(self.human_list[3],self.human_list[2],self.human_list[5],self.human_list[4],int(self.human_center_x),int(self.human_center_y))
                self.box_array.boxes = [self.human]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_people==2:
                self.human_core(self.human_list[3],self.human_list[2],self.human_list[5],self.human_list[4],int(self.human_center_x),int(self.human_center_y))
                self.human_core2(self.human_list2[3],self.human_list2[2],self.human_list2[5],self.human_list2[4],int(self.human_center_x2),int(self.human_center_y2))
                self.box_array.boxes = [self.human,self.human2]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_people==3:
                self.human_core(self.human_list[3],self.human_list[2],self.human_list[5],self.human_list[4],int(self.human_center_x),int(self.human_center_y))
                self.human_core2(self.human_list2[3],self.human_list2[2],self.human_list2[5],self.human_list2[4],int(self.human_center_x2),int(self.human_center_y2))
                self.human_core3(self.human_list3[3],self.human_list3[2],self.human_list3[5],self.human_list3[4],int(self.human_center_x3),int(self.human_center_y3))
                self.box_array.boxes = [self.human,self.human2,self.human3]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_people==4:
                self.human_core(self.human_list[3],self.human_list[2],self.human_list[5],self.human_list[4],int(self.human_center_x),int(self.human_center_y))
                self.human_core2(self.human_list2[3],self.human_list2[2],self.human_list2[5],self.human_list2[4],int(self.human_center_x2),int(self.human_center_y2))
                self.human_core3(self.human_list3[3],self.human_list3[2],self.human_list3[5],self.human_list3[4],int(self.human_center_x3),int(self.human_center_y3))
                self.human_core4(self.human_list4[3],self.human_list4[2],self.human_list4[5],self.human_list4[4],int(self.human_center_x4),int(self.human_center_y4))
                self.box_array.boxes = [self.human,self.human2,self.human3,self.human4]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_people==5:
                self.human_core(self.human_list[3],self.human_list[2],self.human_list[5],self.human_list[4],int(self.human_center_x),int(self.human_center_y))
                self.human_core2(self.human_list2[3],self.human_list2[2],self.human_list2[5],self.human_list2[4],int(self.human_center_x2),int(self.human_center_y2))
                self.human_core3(self.human_list3[3],self.human_list3[2],self.human_list3[5],self.human_list3[4],int(self.human_center_x3),int(self.human_center_y3))
                self.human_core4(self.human_list4[3],self.human_list4[2],self.human_list4[5],self.human_list4[4],int(self.human_center_x4),int(self.human_center_y4))
                self.human_core5(self.human_list5[3],self.human_list5[2],self.human_list5[5],self.human_list5[4],int(self.human_center_x5),int(self.human_center_y5))
                self.box_array.boxes = [self.human,self.human2,self.human3,self.human4,self.human5]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_obstacle==1:
                self.obstacle_core(self.obstacle_list[3],self.obstacle_list[2],self.obstacle_list[5],self.obstacle_list[4],int(self.obstacle_center_x),int(self.obstacle_center_y))
                self.box_array.boxes = [self.obstacle]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_obstacle==2:
                self.obstacle_core(self.obstacle_list[3],self.obstacle_list[2],self.obstacle_list[5],self.obstacle_list[4],int(self.obstacle_center_x),int(self.obstacle_center_y))
                self.obstacle_core2(self.obstacle_list2[3],self.obstacle_list2[2],self.obstacle_list2[5],self.obstacle_list2[4],int(self.obstacle_center_x2),int(self.obstacle_center_y2))
                self.box_array.boxes = [self.obstacle,self.obstacle2]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_obstacle==3:
                self.obstacle_core(self.obstacle_list[3],self.obstacle_list[2],self.obstacle_list[5],self.obstacle_list[4],int(self.obstacle_center_x),int(self.obstacle_center_y))
                self.obstacle_core2(self.obstacle_list2[3],self.obstacle_list2[2],self.obstacle_list2[5],self.obstacle_list2[4],int(self.obstacle_center_x2),int(self.obstacle_center_y2))
                self.obstacle_core3(self.obstacle_list3[3],self.obstacle_list3[2],self.obstacle_list3[5],self.obstacle_list3[4],int(self.obstacle_center_x3),int(self.obstacle_center_y3))
                self.box_array.boxes = [self.obstacle,self.obstacle2,self.obstacle3]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_obstacle==4:
                self.obstacle_core(self.obstacle_list[3],self.obstacle_list[2],self.obstacle_list[5],self.obstacle_list[4],int(self.obstacle_center_x),int(self.obstacle_center_y))
                self.obstacle_core2(self.obstacle_list2[3],self.obstacle_list2[2],self.obstacle_list2[5],self.obstacle_list2[4],int(self.obstacle_center_x2),int(self.obstacle_center_y2))
                self.obstacle_core3(self.obstacle_list3[3],self.obstacle_list3[2],self.obstacle_list3[5],self.obstacle_list3[4],int(self.obstacle_center_x3),int(self.obstacle_center_y3))
                self.obstacle_core4(self.obstacle_list4[3],self.obstacle_list4[2],self.obstacle_list4[5],self.obstacle_list4[4],int(self.obstacle_center_x4),int(self.obstacle_center_y4))
                self.box_array.boxes = [self.obstacle,self.obstacle2,self.obstacle3,self.obstacle4]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_obstacle==5:
                self.obstacle_core(self.obstacle_list[3],self.obstacle_list[2],self.obstacle_list[5],self.obstacle_list[4],int(self.obstacle_center_x),int(self.obstacle_center_y))
                self.obstacle_core2(self.obstacle_list2[3],self.obstacle_list2[2],self.obstacle_list2[5],self.obstacle_list2[4],int(self.obstacle_center_x2),int(self.obstacle_center_y2))
                self.obstacle_core3(self.obstacle_list3[3],self.obstacle_list3[2],self.obstacle_list3[5],self.obstacle_list3[4],int(self.obstacle_center_x3),int(self.obstacle_center_y3))
                self.obstacle_core4(self.obstacle_list4[3],self.obstacle_list4[2],self.obstacle_list4[5],self.obstacle_list4[4],int(self.obstacle_center_x4),int(self.obstacle_center_y4))
                self.obstacle_core5(self.obstacle_list5[3],self.obstacle_list5[2],self.obstacle_list5[5],self.obstacle_list5[4],int(self.obstacle_center_x5),int(self.obstacle_center_y5))
                self.box_array.boxes = [self.obstacle,self.obstacle2,self.obstacle3,self.obstacle4,self.obstacle5]
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_people==0:
                self.box_array.boxes = []
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)
            elif self.number_of_obstacle==0:
                self.box_array.boxes = []
                self.box_array.header.frame_id = "camera_depth_optical_frame"
                self.box_array.header.stamp = rospy.Time.now()
                self.box_array_pub.publish(self.box_array)

            self.c= cv2.waitKey(1) & 0xff 

            # convert from RGB to BGR to fulfil the opencv format
            self.frame = cv2.cvtColor(self.frame,cv2.COLOR_RGB2BGR)
            self.fps  = ( self.fps + (1./(time.time()-self.t1)) ) / 2
            self.frame = cv2.putText(self.frame, "fps= %.2f"%(self.fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.frame = cv2.putText(self.frame, "obstacle= {}".format(self.number_of_obstacle), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.frame = cv2.putText(self.frame, "people= {}".format(self.number_of_people), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #buffer time
            cv2.imshow("detection",self.frame)
            #print("open camera")


            if self.c==27:
                cv2.capture.release()
                break
        
        cv2.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect = object_detect()
    #br = tf.TransformBroadcaster()
    #rate = rospy.Rate(20.0) #20hz
    while not rospy.is_shutdown():
        #try:
        detect.image_detect()
        #br.sendTransform((0.0, 0.0, 0.0),
        #                 (0.0, 0.0, 0.0, 1.0),
        #                 rospy.Time.now(),
        #                 "camera_link",
        #                 "world")
        #rate.sleep()
        #except:
        #    print("Reloading")

