#!/usr/bin/env python

#System Packages
from __future__ import division
import numpy as np
import pyrealsense2 as rs2
import math
import time

#import cv2 package
import cv2
from cv_bridge import CvBridge, CvBridgeError

#ROS Packages
import rospy
import tf
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PoseStamped
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
# RVIZ Coordinates
# Red   - X
# Grren - Y
# Blue  - Z

class rgbd_to_pointcloud_xy_coordinate_transform():
    def __init__(self):

        self.fps = 0.0

        self.markers_z = [0,0,0,0,0]
        self.left_top_marker_world = [0,0,0]
        self.left_down_marker_world = [0,0,0]
        self.right_top_marker_world = [0,0,0]
        self.right_down_marker_world = [0,0,0]
        self.center_world = [0,0,0]

        self.left_top_marker_world_x,self.left_top_marker_world_y,self.left_top_marker_world_z = 0,0,0
        self.left_down_marker_world_x,self.left_down_marker_world_y,self.left_down_marker_world_z = 0,0,0
        self.right_top_marker_world_x,self.right_top_marker_world_y,self.right_top_marker_world_z = 0,0,0
        self.right_down_marker_world_x,self.right_down_marker_world_y,self.right_down_marker_world_z = 0,0,0
        self.center_world_x,self.center_world_y,self.center_world_z = 0,0,0


        self.left_top_message = "" 
        self.left_down_message = "" 
        self.right_top_message = "" 
        self.right_down_message = "" 
        self.center_message = ""

        #self.left_top_message = "left top" 
        #self.left_down_message = "left down" 
        #self.right_top_message = "right top" 
        #self.right_down_message = "right down" 
        #self.center_message = "center"
        self.bgr = (255,0,0)
        self.marker_size = 5
        self.tf_listener = None
        self.marker_transform_ok = False
        self.marker_vis = None
        self.image = None

        rospy.init_node("image_pixel_XY_to_real_world_XY")
        self.tf_listener = tf.TransformListener()
        self.bridge = CvBridge()
        self.boxes = BoundingBoxArray()
        self.box = BoundingBox()
        self.marker_vis = rospy.Publisher("/d435/realsense_image_marker",Marker,queue_size=10)

        # Image Publisher for color and depth images
        #self.color_with_marker_pub = rospy.Publisher("/d435/color_with_marker",Image,queue_size=10)
        #self.depth_with_marker_pub = rospy.Publisher("/d435/depth_with_marker" ,Image,queue_size=10)

        # Subscribe color and depth image
        rospy.Subscriber("/d435/color/image_raw",Image,self.color_callback)
        rospy.Subscriber("/d435/depth/image_raw",Image,self.depth_callback)

        # Subscribe camera info [depth_rgb aligned]
        rospy.Subscriber("/d435/depth/camera_info",CameraInfo,self.camera_info_callback)

        
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

    # Find Z distances for bounding box markers (default depth unit is mm)
    def box_point_z_distance(self, depth_image, left_top_x, left_down_x, right_top_x, right_down_x, center_x, left_top_y, left_down_y, right_top_y, right_down_y, center_y):
        """
        Return list[left top point distance, left down point distance, right top point distance, right down point distance, center point distance]
        """
        self.output = []
        self.a = self.depth_image[left_top_y, left_top_x]/1000
        #print("left top point distance: {}".format(self.a))
        self.output.append(self.a)
        self.b = self.depth_image[left_down_y, left_down_x]/1000
        #print("left down point distance: {}".format(self.b))
        self.output.append(self.b)
        self.c = self.depth_image[right_top_y, right_top_x]/1000
        #print("right top point distance: {}".format(self.c))
        self.output.append(self.c)
        self.d = self.depth_image[right_down_y, right_down_x]/1000
        #print("right down point distance: {}".format(self.d))
        self.output.append(self.d)
        self.e = self.depth_image[center_y, center_x]/1000
        #print("center point distance: {}".format(self.e))
        self.output.append(self.e)
        return self.output

    # Drawing the marker on an rgb_image
    def draw_box_point_rgb(self, rgb_image,x,y,message,bgr,marker_size):
        cv2.putText(rgb_image, message,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,bgr,1,cv2.LINE_AA)
        cv2.circle(rgb_image,(x,y),marker_size,bgr,-1)
        return self.rgb_image

    # Drawing the marker z distance on an rgb_image
    def draw_box_point_z_distance_rgb(self, rgb_image,z,x,y,bgr):
        self.z = str(round(z,3)) + 'm'
        cv2.putText(rgb_image, self.z,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,bgr,1,cv2.LINE_AA)
        return self.rgb_image

    # Drawing the marker on an depth_image
    def draw_box_point_depth(self, depth_image,x,y,message,bgr,marker_size):
        cv2.putText(depth_image, message,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,bgr,1,cv2.LINE_AA)
        cv2.circle(depth_image,(x,y),marker_size,bgr,-1)
        return self.depth_image

    # Drawing the marker z distance on an depth_image
    def draw_box_point_z_distance_depth(self, depth_image,z,x,y,bgr):
        self.z = str(round(z,3)) + 'm'
        cv2.putText(depth_image, self.z,(x,y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,bgr,1,cv2.LINE_AA)
        return self.depth_image

    # Draw all markers on rgb image and depth image
    def draw_box_point_into_image(self, rgb_image,depth_image,z_distances,left_top_x, left_down_x, right_top_x, right_down_x, center_x, left_top_y, left_down_y, right_top_y, right_down_y, center_y):
        # RGB Image - Points
        self.rgb_image = self.draw_box_point_rgb(self.rgb_image,left_top_x,left_top_y,self.left_top_message,self.bgr,self.marker_size)
        self.rgb_image = self.draw_box_point_rgb(self.rgb_image,left_down_x,left_down_y,self.left_down_message,self.bgr,self.marker_size)
        self.rgb_image = self.draw_box_point_rgb(self.rgb_image,right_top_x,right_top_y,self.right_top_message,self.bgr,self.marker_size)
        self.rgb_image = self.draw_box_point_rgb(self.rgb_image,right_down_x,right_down_y,self.right_down_message,self.bgr,self.marker_size)
        self.rgb_image = self.draw_box_point_rgb(self.rgb_image,center_x,center_y,self.center_message,self.bgr,self.marker_size)

        # RGB Image - Z distances
        self.rgb_image = self.draw_box_point_z_distance_rgb(self.rgb_image,z_distances[0],left_top_x,left_top_y,self.bgr)
        self.rgb_image = self.draw_box_point_z_distance_rgb(self.rgb_image,z_distances[1],left_down_x,left_down_y,self.bgr)
        self.rgb_image = self.draw_box_point_z_distance_rgb(self.rgb_image,z_distances[2],right_top_x,right_top_y,self.bgr)
        self.rgb_image = self.draw_box_point_z_distance_rgb(self.rgb_image,z_distances[3],right_down_x,right_down_y,self.bgr)
        self.rgb_image = self.draw_box_point_z_distance_rgb(self.rgb_image,z_distances[4],center_x,center_y,self.bgr)

        # Depth Image - Points
        self.depth_image = self.draw_box_point_depth(self.depth_image,left_top_x,left_top_y,self.left_top_message,self.bgr,self.marker_size)
        self.depth_image = self.draw_box_point_depth(self.depth_image,left_down_x,left_down_y,self.left_down_message,self.bgr,self.marker_size)
        self.depth_image = self.draw_box_point_depth(self.depth_image,right_top_x,right_top_y,self.right_top_message,self.bgr,self.marker_size)
        self.depth_image = self.draw_box_point_depth(self.depth_image,right_down_x,right_down_y,self.right_down_message,self.bgr,self.marker_size)
        self.depth_image = self.draw_box_point_depth(self.depth_image,center_x,center_y,self.center_message,self.bgr,self.marker_size)

        # Depth Image - Z distances
        self.depth_image = self.draw_box_point_z_distance_depth(self.depth_image,z_distances[0],left_top_x,left_top_y,self.bgr)
        self.depth_image = self.draw_box_point_z_distance_depth(self.depth_image,z_distances[1],left_down_x,left_down_y,self.bgr)
        self.depth_image = self.draw_box_point_z_distance_depth(self.depth_image,z_distances[2],right_top_x,right_top_y,self.bgr)
        self.depth_image = self.draw_box_point_z_distance_depth(self.depth_image,z_distances[3],right_down_x,right_down_y,self.bgr)
        self.depth_image = self.draw_box_point_z_distance_depth(self.depth_image,z_distances[4],center_x,center_y,self.bgr)

        return self.rgb_image, self.depth_image


    def euler_from_quaternion(self, rot):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """

        self.euler_x, self.euler_y, self.euler_z, self.euler_w = rot
        self.t0 = +2.0 * (self.euler_w * self.euler_x + self.euler_y * self.euler_z)
        self.t1 = +1.0 - 2.0 * (self.euler_x * self.euler_x + self.euler_y * self.euler_y)
        self.roll_x = math.atan2(self.t0, self.t1)
            
        self.t2 = +2.0 * (self.euler_w * self.euler_y - self.euler_z * self.euler_x)
        self.t2 = +1.0 if self.t2 > +1.0 else self.t2
        self.t2 = -1.0 if self.t2 < -1.0 else self.t2
        self.pitch_y = math.asin(self.t2)
            
        self.t3 = +2.0 * (self.euler_w * self.euler_z + self.euler_x * self.euler_y)
        self.t4 = +1.0 - 2.0 * (self.euler_y * self.euler_y + self.euler_z * self.euler_z)
        self.yaw_z = math.atan2(self.t3, self.t4)

        return self.roll_x, self.pitch_y, self.yaw_z 

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

    # Create marker in rviz
    def create_marker(self,x1,y1,z1,x2,y2,z2,r,g,b,id):
        self.marker = Marker()
        self.marker.header.frame_id = 'world'
        self.marker.header.stamp = rospy.Time.now()
        self.marker.ns = 'realsense_marker'
        self.marker.action = self.marker.ADD
        self.marker.id = id
        self.marker.type = Marker.SPHERE_LIST
        self.marker.pose.orientation.w = 1.0
        self.start = Point(x1,y1,z1)
        self.end = Point(x2,y2,z2)
        self.marker.points.append(self.start)
        self.marker.points.append(self.end)
        self.marker.scale.x = 0.003
        self.marker.scale.z = 0.003
        self.marker.scale.y = 0.003
        self.marker.color.a = 1.0
        self.marker.color.r = r
        self.marker.color.g = g
        self.marker.color.b = b
        return self.marker

    # Algorithm Core
    def core(self,rgb_image,depth_image,intrinsic,left_top_x, left_down_x, right_top_x, right_down_x, center_x, left_top_y, left_down_y, right_top_y, right_down_y, center_y):

        # TF Calculate from world to camera_link (realsense)
        self.realsense_trans,self.realsense_rot = self.get_tf_transform('base_link','d435_right_ir_optical_frame')

        # Image to world transformation for all markers
        # Find all box point z distance (in m)
        # Order: Left top / Left down / Right top / Right down / Center
        self.markers_z = self.box_point_z_distance(self.depth_image,left_top_x,left_down_x,right_top_x,right_down_x,center_x,left_top_y,left_down_y,right_top_y,right_down_y,center_y)
        print("markers_z: {}".format(self.markers_z))
        # Left top Point
        self.left_top_marker_world = rs2.rs2_deproject_pixel_to_point(intrinsic,[left_top_x,left_top_y],self.markers_z[0])
        # Left down Point
        self.left_down_marker_world = rs2.rs2_deproject_pixel_to_point(intrinsic,[left_down_x,left_down_y],self.markers_z[1])
        # Right top Point
        self.right_top_marker_world = rs2.rs2_deproject_pixel_to_point(intrinsic,[right_top_x,right_top_y],self.markers_z[2])
        # Right down Point
        self.right_down_marker_world = rs2.rs2_deproject_pixel_to_point(intrinsic,[right_down_x,right_down_y],self.markers_z[3])
        # Center Point
        self.center_world = rs2.rs2_deproject_pixel_to_point(intrinsic,[center_x,center_y],self.markers_z[4])
        print("center_world : {}".format(self.center_world))

        # Left top Point
        if(self.markers_z[0]!=0):
            self.left_top_marker_world_x = self.left_top_marker_world[1]/100 + self.realsense_trans[0]
            self.left_top_marker_world_y = self.left_top_marker_world[0]/100 + self.realsense_trans[1]
            self.left_top_marker_world_z = self.left_top_marker_world[2]/100*(-1) + self.realsense_trans[2]
            #print("shit : {}".format(self.left_top_marker_world[2]/100*-1))
            #print("left_top_marker_world[0] : {}".format(self.left_top_marker_world[0]))
            #print("left_top_marker_world[1] : {}".format(self.left_top_marker_world[1]))
            #print("left_top_marker_world[2] : {}".format(self.left_top_marker_world[2]))            
            #print("realsense_trans[0] : {}".format(self.realsense_trans[0]))
            #print("realsense_trans[1] : {}".format(self.realsense_trans[1]))
            #print("realsense_trans[2] : {}".format(self.realsense_trans[2]))
            #print("left_top_marker_world_x : {}".format(self.left_top_marker_world_x))
            #print("left_top_marker_world_y : {}".format(self.left_top_marker_world_y))
            #print("left_top_marker_world_z : {}".format(self.left_top_marker_world_z))

            # Publish center marker
            # z2 set to 0 because the arrow needs to point to the ground
            self.left_top_marker_rviz = self.create_marker(self.realsense_trans[0],self.realsense_trans[1],self.realsense_trans[2],
                                            self.left_top_marker_world_x,self.left_top_marker_world_y,self.left_top_marker_world_z,1.0,0.0,1.0,id=0)
            #print ("left_top_marker_rviz : {}".format(self.left_top_marker_rviz))
            #self.marker_vis.publish(self.left_top_marker_rviz)
            
        # Left down Point
        if(self.markers_z[1]!=0):
            self.left_down_marker_world_x = self.left_down_marker_world[1]/100 + self.realsense_trans[0]
            self.left_down_marker_world_y = self.left_down_marker_world[0]/100 + self.realsense_trans[1]
            self.left_down_marker_world_z = self.left_down_marker_world[2]/100*-1 + self.realsense_trans[2]

            # Publish center marker
            # z2 set to 0 because the arrow needs to point to the ground
            self.left_down_marker_rviz = self.create_marker(self.realsense_trans[0],self.realsense_trans[1],self.realsense_trans[2],
                                            self.left_down_marker_world_x,self.left_down_marker_world_y,self.left_down_marker_world_z,0.2,1.0,0.2,id=1)
            #self.marker_vis.publish(self.left_down_marker_rviz)

        # Right top Point
        if(self.markers_z[2]!=0):
            self.right_top_marker_world_x = self.right_top_marker_world[1]/100 + self.realsense_trans[0]
            self.right_top_marker_world_y = self.right_top_marker_world[0]/100 + self.realsense_trans[1]
            self.right_top_marker_world_z = self.right_top_marker_world[2]/100*-1 + self.realsense_trans[2] 

            # Publish center marker
            # z2 set to 0 because the arrow needs to point to the ground
            self.right_top_marker_rviz = self.create_marker(self.realsense_trans[0],self.realsense_trans[1],self.realsense_trans[2],
                                            self.right_top_marker_world_x,self.right_top_marker_world_y,self.right_top_marker_world_z,0.2,1.0,1.0,id=2)
            #self.marker_vis.publish(self.right_top_marker_rviz)    

        # Right down Point
        if(self.markers_z[3]!=0):
            self.right_down_marker_world_x = self.right_down_marker_world[1]/100 + self.realsense_trans[0]
            self.right_down_marker_world_y = self.right_down_marker_world[0]/100 + self.realsense_trans[1]
            self.right_down_marker_world_z = self.right_down_marker_world[2]/100*-1 + self.realsense_trans[2] 

            # Publish center marker
            # z2 set to 0 because the arrow needs to point to the ground
            self.right_down_marker_rviz = self.create_marker(self.realsense_trans[0],self.realsense_trans[1],self.realsense_trans[2],
                                            self.right_down_marker_world_x,self.right_down_marker_world_y,self.right_down_marker_world_z,1.0,0.2,0.2,id=3)
            #self.marker_vis.publish(self.right_down_marker_rviz)    

        # Center Point
        if(self.markers_z[4]!=0):

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

            # Publish center marker
            # z2 set to 0 because the arrow needs to point to the ground
            self.center_rviz = self.create_marker(self.realsense_trans[0],self.realsense_trans[1],self.realsense_trans[2],
                                            self.center_world_x,self.center_world_y,self.center_world_z,1.0,1.0,0.6,id=4)
            self.marker_vis.publish(self.center_rviz) 

        # Draw markers on rgb and depth images
        self.draw_box_point_into_image(self.rgb_image,self.depth_image,self.markers_z,left_top_x,left_down_x,right_top_x,right_down_x,center_x,left_top_y,left_down_y,right_top_y,right_down_y,center_y)

        # Marker should only transform once.
        # As it is physically exist in the world, it should not change anymore
        self.marker_transform_ok = True   



        #Apply color map to depth image
        self.depth_image = cv2.applyColorMap(cv2.convertScaleAbs(self.depth_image, alpha=0.065), cv2.COLORMAP_JET)


        return self.rgb_image,self.depth_image

    def main(self):
        self.desired_size = [0,0,0]
        self.desired_size[0] = 480
        self.desired_size[1] = 640
        self.desired_size[2] = 3

        # 2D/3D Image processing
        while not rospy.is_shutdown():
            rospy.wait_for_message("/d435/color/image_raw",Image)
            rospy.wait_for_message("/d435/depth/image_raw",Image)
            rospy.wait_for_message("/d435/depth/camera_info",CameraInfo)

            try:
                # Processing Core
                self.color_output, self.depth_output = self.core(self.rgb_image,self.depth_image,self.intrinsic,270,270,370,370,320,90,190,90,190,140)
                self.desired_size[0] = self.color_output.shape[0]
                self.desired_size[1] = self.color_output.shape[1]
                self.desired_size[2] = self.color_output.shape[2]
                self.desired_size[0] = self.depth_output.shape[0]
                self.desired_size[1] = self.depth_output.shape[1]
                self.desired_size[2] = self.depth_output.shape[2]
            except:
                print("wait")
            self.t1 = time.time()
            rospy.sleep(rospy.Duration(0.2))
            self.fps  = ( self.fps + (1.0/(time.time()-self.t1)) ) / 2.0
            print("fps= %.2f"%(self.fps))

            # Visulaiztion in RVIZ
            print ("depth outout: {}".format(self.depth_output.shape))
            print ("color outout: {}".format(self.color_output.shape))

            self.depth_output = cv2.putText(self.depth_output, "fps= %.2f"%(self.fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.color_output = cv2.putText(self.color_output, "fps= %.2f"%(self.fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #self.color_with_marker_pub.publish(self.bridge.cv2_to_imgmsg(self.color_output, "bgr8"))
            #self.depth_with_marker_pub.publish(self.bridge.cv2_to_imgmsg(self.depth_output, "bgr8"))
            # Visualization in new window
            cv2.imshow("Realsense [RGB]",self.color_output)
            cv2.imshow("Realsense [Depth]",self.depth_output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                    
        cv2.waitKey(1)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Delay for tf capture
    print("Start")
    transformer = rgbd_to_pointcloud_xy_coordinate_transform()
    transformer.main()

