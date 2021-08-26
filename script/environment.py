# import ros package
import rospy
import rospkg
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import *

# import python package
import random
import math
import numpy as np
from datetime import datetime


class environment():
    def __init__(self):
        self.model_state_x = {}
        self.model_state_y = {}
        # get the model name from the model name text file and store into list
        self.text_file = open("/home/iastaff/catkin_ws/src/drone/model_name.txt", "r")
        self.model = self.text_file.readlines()
        self.model_name = [x.strip() for x in self.model]
        self.text_file.close()
        rospy.init_node('generate_model_pos')
        self.update_model_pos = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.get_model_pos = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def get_obstacles_pos(self):
        self.model_pos = GetModelStateRequest()
        for i in range (len(self.model_name)):
            self.model_pos.model_name = self.model_name[i]
            self.model_state_all = self.get_model_pos(self.model_pos)
            self.model_state_x[i] = self.model_state_all.pose.position.x
            self.model_state_y[i] = self.model_state_all.pose.position.y

    def set_obstacles_pos(self):
        self.set_state = ModelState()
        for i in range (len(self.model_name)):
            self.set_state.model_name = self.model_name[i]
            # set pose
            self.set_state.pose.position.x = random.uniform(-50.0,50.0)
            self.set_state.pose.position.y = random.uniform(-50.0,50.0)
            self.set_state.pose.orientation.z = random.uniform(-50.0,50.0)
            self.update_model_pos.publish(self.set_state)
            self.model_state_x[i] = self.set_state.pose.position.x          # update dict
            self.model_state_y[i] = self.set_state.pose.position.y          # update dict
            print ("update dict finish")
            rospy.sleep(0.03)            # provide buffer time to set model new position
            print ("Finish reset {} position".format(self.model_name[i]))
            print ("{}    x:{}    y:{}".format(self.model_name[i],self.set_state.pose.position.x,self.set_state.pose.position.y ))


if __name__ == "__main__":
        try:
            set_environment = environment()
            set_environment.get_obstacles_pos()
            set_environment.set_obstacles_pos()
        except rospy.ROSInterruptException:
            pass
