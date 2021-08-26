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

# get the model name from the model name text file and store into list
text_file = open("/home/iastaff/catkin_ws/src/drone/model_name.txt", "r")
lines = text_file.readlines()
lines = [x.strip() for x in lines] 
print type(lines)
print len(lines)
text_file.close()