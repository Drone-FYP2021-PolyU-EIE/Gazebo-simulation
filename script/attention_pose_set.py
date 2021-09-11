#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
rospy.init_node("attention_pose_set")
boxarray_pub = rospy.Publisher("/desired/input/box_array", BoundingBoxArray, queue_size=1)
box_pub = rospy.Publisher("/desired/input/box", BoundingBox, queue_size=1)
r = rospy.Rate(10)
theta = 0
while not rospy.is_shutdown():
    boxes = BoundingBoxArray()
    box = BoundingBox()
    box.header.stamp = rospy.Time.now()
    box.header.frame_id = "d435_depth_optical_frame"
    box.pose.orientation.w = 1

    box.pose.position.x = 0.0265447860956 # increase in program  = move to right in rviz (directly use center_world_x)
    box.pose.position.y = 0.0913378709553 # increase in program  = downward in rviz (directly use center_world_y)
    box.pose.position.z = 1.8839999437332153 # increase in program  = move away in rviz (directly use the depth distance)
    box.dimensions.x = 0.5
    box.dimensions.y = 0.5
    box.dimensions.z = 0.5
    box2 = BoundingBox()
    box2.header.stamp = rospy.Time.now()
    box2.header.frame_id = "d435_depth_optical_frame"
    box2.pose.orientation.w = 1
    box2.pose.position.x = 0.1
    box2.pose.position.y = 0.1
    box2.pose.position.z = 0.5
    box2.dimensions.x = 0.5
    box2.dimensions.y = 0.5
    box2.dimensions.z = 0.5
    boxes.boxes = [box, box2]
    boxes.header.frame_id = "d435_depth_optical_frame"
    boxes.header.stamp = rospy.Time.now()
    box_pub.publish(box)
    #boxarray_pub.publish(boxes)
    print ("finish one round")
    r.sleep()
