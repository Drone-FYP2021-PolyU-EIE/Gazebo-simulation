#!/usr/bin/env python2.7
import rospy
import numpy as np

#import cv2 package
import cv2
from cv_bridge import CvBridge

#import ros package
from sensor_msgs.msg import Image
from gazebo_msgs.msg import ModelState

line_thickness = 2
desired_box_size = 50
desired_box_x_offset = 0
desired_box_y_offset = -100

class test_depth_distance():
    def __init__(self):
        rospy.init_node('test_depth_distance')
        rospy.Subscriber("/d435/depth/image_raw", Image, self.depth_cb)   #subscribe the rostopic "/d435/depth/image_raw"
        self.set_model_state = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        print("init finish")
        
    def depth_cb(self, msg):
        
        self.bridge = CvBridge()                  #prepare the bridge
        self.image = self.bridge.imgmsg_to_cv2(msg)
        self.depth_image = cv2.resize(self.image,(640,480))   # 640 width, 360 height
        self.depth_array = np.array(self.depth_image, dtype=np.float32)
        self.depth_array = self.depth_array/1000                # convert to meter

        self.height = self.depth_array.shape[0]
        self.width = self.depth_array.shape[1]
        
        # calculate the box XY
        self.depth_image_centre = [(self.height/2)+desired_box_y_offset, self.width/2]
        self.x1 = self.depth_image_centre[1]-desired_box_size
        self.y1 = self.depth_image_centre[0]-desired_box_size
        self.x2 = self.depth_image_centre[1]+desired_box_size
        self.y2 = self.depth_image_centre[0]+desired_box_size
        print (self.depth_image_centre)
        print (self.x1)
        print (self.x2)
        print (self.y1)
        print (self.y2)

        self.lefttop_corner_distance = self.depth_array[self.y1, self.x1]
        self.righttop_corner_distance = self.depth_array[self.y1, self.x2]
        self.leftdown_corner_distance = self.depth_array[self.y2, self.x1]
        self.rightdown_corner_distance = self.depth_array[self.y2, self.x2]

        print("original shape: {}".format(self.depth_array.shape))
        print ('box center depth: {}'.format(self.depth_array[self.depth_image_centre[0], self.depth_image_centre[1]]))

        # draw the box
        self.depth_array = cv2.rectangle(self.depth_array, (self.x1,self.y1), (self.x2,self.y2), (255,0,0),thickness=line_thickness)

        print ('left top corner depth: {}'.format(self.lefttop_corner_distance))
        print ('right top corner depth: {}'.format(self.righttop_corner_distance))
        print ('left down corner depth: {}'.format(self.leftdown_corner_distance))
        print ('right down corner depth: {}'.format(self.rightdown_corner_distance))
        #print ("depth callback finish")

    def show_image(self):
        cv2.imshow("depth camera", self.depth_array)
        cv2.waitKey(1)

    def set_model(self):
        self.maintain_state = ModelState()
        self.maintain_state.model_name = 'Box'
        # pose
        self.maintain_state.pose.position.x = 0.0
        self.maintain_state.pose.position.y = 1.5
        self.set_model_state.publish(self.maintain_state)

        self.maintain_state = ModelState()
        self.maintain_state.model_name = 'quadrotor'
        # pose
        self.maintain_state.pose.position.x = 0.0
        self.maintain_state.pose.position.y = 0.0
        self.maintain_state.pose.position.z = 0.183
        self.maintain_state.pose.orientation.z = 0.0
        self.set_model_state.publish(self.maintain_state)
        #print("set model position finish")

if __name__ == "__main__":
        try:
            test = test_depth_distance()
            rospy.sleep(rospy.Duration(0.5)) # buffer time
            while not rospy.is_shutdown():
                test.show_image()
                test.set_model()
        except rospy.ROSInterruptException:
            pass
