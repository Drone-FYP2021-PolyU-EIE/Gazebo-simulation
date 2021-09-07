#!/usr/bin/env python

#System Packages
import numpy as np

#import cv2 package
import cv2
from cv_bridge import CvBridge, CvBridgeError

#ROS Packages
import rospy
from sensor_msgs.msg import Image, PointCloud2, CameraInfo, RegionOfInterest
from sensor_msgs import point_cloud2
from std_msgs.msg import Float64, Int64, Header, String

class camera_image_callback():
    """
    Provide the topic callback offer by realsense depth camera D435i
    """
    def depth_callback(self, data):

        """
        Provide depth callback and return depth_array(in meter), header, encoding, height, width, center x, center y
        """

        self.bridge = CvBridge()
        self.image = self.bridge.imgmsg_to_cv2(data)

        #resize for increase the fps
        #self.depth_image = cv2.resize(self.image,(640,480))   # 640 width, 360 height
        #self.depth_array = np.array(self.depth_image, dtype=np.float32)

        self.depth_array = np.array(self.image, dtype=np.float32)
        self.depth_array = self.depth_array/1000                # convert to meter
        self.depth_header = data.header
        self.depth_encoding = data.encoding
        self.depth_height = self.depth_array.shape[0]
        self.depth_width = self.depth_array.shape[1]
        self.depth_image_center_x = self.depth_width/2
        self.depth_image_center_y = self.depth_height/2

        #print("depth callback finish")
        return self.depth_array,self.depth_header,self.depth_encoding,self.depth_height,self.depth_width,self.depth_image_center_x,self.depth_image_center_y
        

    def rgb_callback(self, data):
        
        """
        Provide color callback and return rgb_image(ndarray), header, encoding, height, width, center x, center y
        """

        self.bridge = CvBridge()
        self.rgb_image = self.bridge.imgmsg_to_cv2(data)
        self.rgb_header = data.header
        self.rgb_encoding = data.encoding
        self.rgb_height = self.rgb_image.shape[0]
        self.rgb_width = self.rgb_image.shape[1]
        self.rgb_image_center_x = self.rgb_width/2
        self.rgb_image_center_y = self.rgb_height/2

        #resize for increase the fps
        #self.rgb_image = cv2.resize(image,(640,480))   # 640 width, 360 height
        #print("rgb callback finish")
        return self.rgb_image,self.rgb_header,self.rgb_encoding,self.rgb_height,self.rgb_width,self.rgb_image_center_x,self.rgb_image_center_y

    def pointcloud2_callback(self, data):

        """
        Provide pointcloud2 callback and return pointcloud_XYZ, header, height, width, fields, is_bigendian, point_step, row_step, is_dense
        """

        assert isinstance(data, PointCloud2)
        self.pointcloud_XYZ = point_cloud2.read_points(data,field_names=("x","y","z"), skip_nans=True)
        rospy.sleep(rospy.Duration(0.2)) # buffer time
        #for p in pointcloud_XYZ:
        #    print (" x : %.3f  y : %.3f  z : %.3f"%(p[0],p[1],p[2]))
        self.point_cloud_header = data.header
        self.point_cloud_height = data.height
        self.point_cloud_width = data.width
        self.point_cloud_fields = data.fields
        self.point_cloud_is_bigendian = data.is_bigendian
        self.point_cloud_point_step = data.point_step
        self.point_cloud_row_step = data.row_step
        self.point_cloud_is_dense = data.is_dense

        #print("pointcloud2 callback finish")
        return self.pointcloud_XYZ,self.point_cloud_header,self.point_cloud_height,self.point_cloud_width,self.point_cloud_fields,self.point_cloud_is_bigendian,self.point_cloud_point_step,self.point_cloud_row_step,self.point_cloud_is_dense
        

    def camera_info_callback(self, data):

        """
        Provide camera info callback and return header, height , width, distortion_model, D, K, R, P, binning_x, binning_y, roi
        """

        self.camera_info_header = data.header
        self.camera_info_height = data.height
        self.camera_info_width = data.width
        self.distortion_model = data.distortion_model
        self.D = data.D
        self.K = data.K
        self.R = data.R
        self.P = data.P
        self.binning_x = data.binning_x
        self.binning_y = data.binning_y
        self.roi = data.roi

        #print("camera info callback finish")
        return self.camera_info_header,self.camera_info_height,self.camera_info_width,self.distortion_model,self.D,self.K,self.R,self.P,self.binning_x,self.binning_y,self.roi
        

