#!/usr/bin/env python2.7
import time
import rospy
import math
import sys
import numpy as np
import PIL
from PIL import Image

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
from sensor_msgs.msg import Image, PointCloud2, PointField
from sensor_msgs import point_cloud2
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
        rospy.Subscriber("/d435/depth/color/points", PointCloud2, self.callback_pointcloud)   #subscribe the rostopic "/d435/depth/color/points"
        #/d435/depth/color/points is XYZRGB pointcloud
  
        # saved_point_cloud_XYZ to csv file
        self.f = open("saved_point_cloud_XYZ.csv",'wb')
        self.csv_writer = csv.writer(self.f)
        
        self.rgb_pub = rospy.Publisher("/d435/color/region_of_interest", Image, queue_size=1)   #publishe the rostopic "/d435/color/region_of_interest"
        self.depth_pub = rospy.Publisher("/d435/depth/region_of_interest", Image, queue_size=1)   #publishe the rostopic "/d435/depth/region_of_interest"
        self.pointcloud_pub = rospy.Publisher("/d435/dynamic_filter/pointcloud2", PointCloud2, queue_size=1)   #publishe the rostopic "/d435/depth/region_of_interest"
        rospy.wait_for_service('/enable_motors')
        val = rospy.ServiceProxy('/enable_motors', EnableMotors)
        resp1 = val(True)
        print (resp1.success)
        print ("Init finish")

    def rgb_cb(self, msg):
        try:
            # BEGIN BRIDGE
            #print ("rgb message: {}".format(msg))
            self.rgb_image = self.bridge.imgmsg_to_cv2(msg)
            # Resize the image
            self.rgb_image = cv2.resize(self.rgb_image,(640,480))   # 640 width, 360 height
            self.cropped_rgb_image = self.rgb_image[0:360, 0:240]
            self.rgb_pub.publish(self.bridge.cv2_to_imgmsg(self.cropped_rgb_image, 'bgr8'))
            print ("rgb callback finish")
        except:
            print ("rgb wait")

    def depth_cb(self, msg):
        try:
            # BEGIN BRIDGE
            self.depth_image = self.bridge.imgmsg_to_cv2(msg)
            # Resize the image
            self.depth_image = cv2.resize(self.depth_image,(640,480))   # 640 width, 360 height
            # convert to np array with float32 format
            self.depth_image = np.array(self.depth_image, dtype=np.float32)
            # convert the depth into visualize range
            self.depth_image_with_color = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.065), cv2.COLORMAP_JET)
            # convert the unit to meter for every element in depth array
            self.depth_image = self.depth_image/1000
            self.cropped_depth_image = self.depth_image[0:360, 0:240]
            self.depth_pub.publish(self.bridge.cv2_to_imgmsg(self.cropped_depth_image))
            print ("depth callback finish")
        except:
            print ("depth wait")

    def pose_cb(self, data):
        self.current_x_position = float(data.pose.pose.position.x)
        self.current_y_position = float(data.pose.pose.position.y)
        self.current_z_position = float(data.pose.pose.position.z)
        self.current_x_orientation = float(data.pose.pose.orientation.x)
        self.current_y_orientation = float(data.pose.pose.orientation.y)
        self.current_z_orientation = float(data.pose.pose.orientation.z)
        self.current_w_orientation = float(data.pose.pose.orientation.z)
        print ("pose callback finish")

    def callback_pointcloud(self, data):
        assert isinstance(data, PointCloud2)
        self.gen = point_cloud2.read_points(data,field_names=("x","y","z"), skip_nans=True)
        time.sleep(1)
        #for p in gen:
        #    self.csv_writer.writerow(["{}".format(p[0]),"{}".format(p[1]),"{}".format(p[2])])
        #    print (" x : %.3f  y : %.3f  z : %.3f"%(p[0],p[1],p[2]))
        print ("point cloud header: {}".format(data.header))
        print ("point cloud height: {}".format(data.height))
        print ("point cloud width: {}".format(data.width))
        print ("point cloud fields: {}".format(data.fields))
        print ("is_bigendian: {}".format(data.is_bigendian))
        print ("point_step: {}".format(data.point_step))
        print ("row_step: {}".format(data.row_step))
        #print ("actual point data: {}".format(data.data))
        print ("is_dense: {}".format(data.is_dense))
        print ("point cloud callback finish")

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
        cv2.imshow("depth camera",self.depth_image)
        cv2.imshow("rgb camera",self.rgb_image)
        #cv2.imshow("cropped depth camera", self.cropped_depth_image)
        #cv2.imshow("cropped rgb camera", self.cropped_rgb_image)
        cv2.waitKey(1)

    #def get_distance(self):
        
    def xyzrgb_array_to_pointcloud2(self, points, colors, stamp=None, frame_id="d435_depth_optical_frame", seq=None):
        '''
        Create a sensor_msgs.PointCloud2 from an array of points.
        '''
        self.msg = PointCloud2()
        try:
            assert(points.shape == colors.shape)
        except:
            print(points.shape)
            print(colors.shape)
            
        if stamp:
            self.msg.header.stamp = stamp
        if frame_id:
            self.msg.header.frame_id = frame_id
        if seq:
            self.msg.header.seq = seq
        if len(points.shape) == 3 & len(colors.shape) == 3:
            self.msg.height = points.shape[0]
            self.msg.width = points.shape[1]
            self.xyzrgb = np.array(np.hstack([points, colors]), dtype=np.float32)
            self.msg.fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 16, PointField.FLOAT32, 1),
            ]
            self.msg.is_bigendian = False
            self.msg.point_step = 24
            self.msg.row_step = self.msg.point_step * self.msg.width
            self.msg.is_dense = False;
            self.msg.data = self.xyzrgb.tostring()

        self.pointcloud_pub.publish(self.msg)


if __name__ == "__main__":
        try:
            action = motion()
            rospy.sleep(rospy.Duration(0.2)) # buffer time
            while not rospy.is_shutdown():
                action.stable_in_air()
                action.show_image()
                rgb_image = action.rgb_image
                depth_image = action.depth_image_with_color
                action.xyzrgb_array_to_pointcloud2(depth_image,rgb_image, stamp=rospy.Time.now())
        except rospy.ROSInterruptException:
            pass
