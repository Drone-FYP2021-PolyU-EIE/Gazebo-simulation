#!/usr/bin/env python2.7
import time
import rospy
import math
import sys
import numpy as np

#import cv2 package
import cv2
import cv_bridge
from cv_bridge.boost.cv_bridge_boost import getCvType

#import ros package
import geometry_msgs
from nav_msgs.msg import Odometry
from geometry_msgs import msg
from geometry_msgs.msg import Point, Twist
from std_msgs.msg import Float64, Int64, Header, String
from gazebo_msgs.msg import ModelState
from std_msgs.msg import String, Duration, Header
from sensor_msgs.msg import Image, PointCloud2
from hector_uav_msgs.srv import EnableMotors

class motion():
    def __init__(self):
        rospy.init_node('motion')
        self.target_height = 2.0
        self.bridge = cv_bridge.CvBridge()                  #prepare the bridge
        self.cmd = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        rospy.Subscriber("/ground_truth/state", Odometry, self.pose_cb)   #subscribe the rostopic "/ground_truth/state"
        rospy.Subscriber("/d435/color/image_raw", Image, self.rgb_cb)   #subscribe the rostopic "/d435/color/image_raw"
        rospy.Subscriber("/d435/depth/image_raw", Image, self.depth_cb)   #subscribe the rostopic "/d435/depth/image_raw"
        rospy.Subscriber("/d435/infra1/image_raw", Image, self.infra1_cb)   #subscribe the rostopic "/d435/infra1/image_raw"        
        rospy.Subscriber("/d435/infra2/image_raw", Image, self.infra2_cb)   #subscribe the rostopic "/d435/infra2/image_raw"
        #rospy.Subscriber("/d435/depth/color/points", PointCloud2, self.pointcloud_cb)   #subscribe the rostopic "/d435/depth/color/points"
        rospy.wait_for_service('/enable_motors')
        val = rospy.ServiceProxy('/enable_motors', EnableMotors)
        resp1 = val(True)
        print (resp1.success)
        print ("Init finish")

    def rgb_cb(self, msg):
        # BEGIN BRIDGE
        self.rgb_image = self.bridge.imgmsg_to_cv2(msg)
        # Resize the image
        self.rgb_image = cv2.resize(self.rgb_image,(720,480))
        print ("rgb callback finish")

    def depth_cb(self, msg):
        # BEGIN BRIDGE
        self.depth_image = self.bridge.imgmsg_to_cv2(msg)
        # Resize the image
        self.depth_image = cv2.resize(self.depth_image,(720,480))   # 720 columns, 480 rows
        # convert to np array with float32 format
        self.depth_array = np.array(self.depth_image, dtype=np.float32)
        # prepare the empty list
        self.distance = []
        self.distance_row = []
        # convert the depth into visualize range
        self.depth_image = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.065), cv2.COLORMAP_JET)
        # convert the unit to meter for every element in depth array
        self.depth_array = self.depth_array/1000
        print ("depth callback finish")

    def infra1_cb(self, msg):
        # BEGIN BRIDGE
        self.infra1_image = self.bridge.imgmsg_to_cv2(msg)
        # Resize the image
        self.infra1_image = cv2.resize(self.infra1_image,(720,480))
        print ("infra1 callback finish")

    def infra2_cb(self, msg):
        # BEGIN BRIDGE
        self.infra2_image = self.bridge.imgmsg_to_cv2(msg)
        # Resize the image
        self.infra2_image = cv2.resize(self.infra2_image,(720,480))
        print ("infra2 callback finish")

    def pose_cb(self, data):
        self.current_x_position = float(data.pose.pose.position.x)
        self.current_y_position = float(data.pose.pose.position.y)
        self.current_z_position = float(data.pose.pose.position.z)
        self.current_x_orientation = float(data.pose.pose.orientation.x)
        self.current_y_orientation = float(data.pose.pose.orientation.y)
        self.current_z_orientation = float(data.pose.pose.orientation.z)
        self.current_w_orientation = float(data.pose.pose.orientation.z)
        print ("pose callback finish")

    def stable_in_air(self):
        self.velocity_upward = Twist()
        self.velocity_downward = Twist()
        self.velocity_upward.linear.z = 0.5
        self.velocity_downward.linear.z = -0.5
        if self.current_z_position > self.target_height:
            self.cmd.publish(self.velocity_downward)
            print ("height: {}".format(self.current_z_position))
            print ("downward")
        elif self.current_z_position < self.target_height:
            self.cmd.publish(self.velocity_upward)
            print ("height: {}".format(self.current_z_position))
            print ("upward")

    def show_image(self):
        cv2.imshow("depth camera",self.depth_array)
        cv2.imshow("rgb camera",self.rgb_image)
        #cv2.imshow("infra1 camera",self.infra1_image)
        #cv2.imshow("infra2 camera",self.infra2_image)
        cv2.waitKey(1)


    #def get_distance(self):
        




if __name__ == "__main__":
        try:
            action = motion()
            rospy.sleep(rospy.Duration(0.2)) # buffer time
            while not rospy.is_shutdown():
                action.stable_in_air()
                action.show_image()
        except rospy.ROSInterruptException:
            pass
