#!/usr/bin/env python  
from tokenize import String
import roslib
roslib.load_manifest('offboard_test')
import cmath
import math
import numpy
import scipy
import scipy.signal
import argparse
import rospy
import threading
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import  PoseStamped, Quaternion, Point
from std_msgs.msg import Int32,Header
from visualization_msgs.msg import *
from mavros_msgs.srv import CommandBool, SetMode
from mavros_msgs.msg import State
from tf.transformations import quaternion_from_euler
from gazebo_msgs.msg import *
from nav_msgs.msg import *
from scipy.interpolate import *
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint # for geometric_controller
import matplotlib.pyplot as plt
import itertools
import pcl
import skfmm
import time
import random
from scipy.spatial import distance
import scipy.interpolate as scipy_interpolate


service_timeout=5.0
camera_frame_name="D435i::camera_depth_frame"
world_frame_name="Gazebo"
base_link_name="base_link"

# k means parameters
MAX_LOOP = 10
DCOST_TH = 0.1

class Clusters:

    def __init__(self, x, y, n_label):
        self.x = x
        self.y = y
        self.n_data = len(self.x)
        self.n_label = n_label
        self.labels = [random.randint(0, n_label - 1)
                       for _ in range(self.n_data)]
        self.center_x = [0.0 for _ in range(n_label)]
        self.center_y = [0.0 for _ in range(n_label)]

    def plot_cluster(self):
        cluster_x_list=[]
        cluster_y_list=[]
        for label in set(self.labels):
            x, y = self._get_labeled_x_y(label)
            cluster_x_list.append(x)
            cluster_y_list.append(y)
        return cluster_x_list, cluster_y_list

    def calc_centroid(self,height):
        cluster_dict=dict()
        cluster_euclidean_distance_list=[]

        for label in set(self.labels):
            x, y = self._get_labeled_x_y(label)
            n_data = len(x)
            self.center_x[label] = sum(x) / n_data
            self.center_y[label] = sum(y) / n_data
            for i in range (n_data):
                cluster_euclidean_distance_list.append(((x[i]-self.center_x[label])**2+(y[i]-self.center_y[label])**2)**0.5)
                max_cluster_euclidean_distance_list=max(cluster_euclidean_distance_list)
            cluster_dict["cluster_"+str(label)]=(self.center_x[label],self.center_y[label],height,max_cluster_euclidean_distance_list)

        return cluster_dict

    def update_clusters(self):
        cost = 0.0

        for ip in range(self.n_data):
            px = self.x[ip]
            py = self.y[ip]

            dx = [icx - px for icx in self.center_x]
            dy = [icy - py for icy in self.center_y]

            dist_list = [math.hypot(idx, idy) for (idx, idy) in zip(dx, dy)]
            min_dist = min(dist_list)
            min_id = dist_list.index(min_dist)
            self.labels[ip] = min_id
            cost += min_dist

        return cost

    def _get_labeled_x_y(self, target_label):
        x = [self.x[i] for i, label in enumerate(self.labels) if label == target_label]
        y = [self.y[i] for i, label in enumerate(self.labels) if label == target_label]
        return x, y



class moving():
    def __init__(self):
        # start ros node
        rospy.init_node('PX4_AuotFLy')
        self.finish_ready_position = False
        self.update_new_path = False
        self.use_the_second_part = False
        self.shift = False

        self.pointcloud_x_list = []
        self.pointcloud_y_list = []
        self.pointcloud_z_list = []
        self.past_passthrough_height=[]
        self.past_cluster_center_list=[]
        self.past_nearest_point_circle=[]
        self.past_local_path_x_leftside=[]
        self.past_local_path_y_leftside=[]
        self.past_local_path_x_rightside=[]
        self.past_local_path_y_rightside=[]
        self.past_concat_x=[]
        self.past_concat_y=[]
        self.past_concat_z=[]
        self.past_local_path_x_after_Bspline=[]
        self.past_local_path_y_after_Bspline=[]
        self.past_local_path_z_after_Bspline=[]
        self.past_global_x_trajectory=[]
        self.past_global_y_trajectory=[]
        self.past_global_z_trajectory=[]
        
        # prepare the publisher and subscriber
        self.position_pub = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=1)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self.mavros_state_callback)
        #self.local_position_sub = rospy.Subscriber("/mavros/local_position/pose",PoseStamped, self.callback_local_position)

        self.point_cloud2_sub  = rospy.Subscriber("/extract_indices/output",PointCloud2,self.callback_pointcloud)
        self.number_of_obstacle_sub  = rospy.Subscriber("/detection_result/number_of_obstacle", Int32, self.callback_number_of_obstacle)
        self.number_of_human_sub  = rospy.Subscriber("/detection_result/number_of_human", Int32, self.callback_number_of_human)
        self.number_of_injury_sub  = rospy.Subscriber("/detection_result/number_of_injury", Int32, self.callback_number_of_injury)

        self.marker_vis = rospy.Publisher("/desired_path/position",MarkerArray,queue_size=1)
        self.validation_marker_vis = rospy.Publisher("/desired_path/validation_position",MarkerArray,queue_size=1)

        self.global_marker_vis = rospy.Publisher("/desired_path/global_marker",MarkerArray,queue_size=1)
        self.global_trajectory_vis = rospy.Publisher("/desired_path/global_trajectory",Path,queue_size=1)
        self.local_marker_vis = rospy.Publisher("/desired_path/local_marker",MarkerArray,queue_size=1)
        self.local_trajectory_vis = rospy.Publisher("/desired_path/local_trajectory",Path,queue_size=1)

        # Trajectory config
        self.local_marker_array=MarkerArray()

        # TF config
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.transform = tf2_geometry_msgs.PointStamped()

        # customize parameter
        self.pi = 3.141592654
        self.drone_width_for_the_obstacle = 0.5
        self.updated_path_time = 0
        self.desired_position_x = 0.0
        self.desired_position_y = 0.0
        self.desired_position_z = 0.0
        self.local_position_x = 0.0
        self.local_position_y = 0.0
        self.local_position_z = 0.0
        self.number_of_obstacle = 0
        self.number_of_human = 0
        self.number_of_injury = 0
        self.r_ready_position_quatrernion=[0.0,0.0,0.0,1.0]

        # arm and change to offboard mode
        rospy.wait_for_service('/mavros/cmd/arming', service_timeout)
        rospy.wait_for_service('/mavros/set_mode', service_timeout)
        self.set_arming_srv = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
        self.mavros_state = State()
        print("success init!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def callback_local_position(self):
        for i in range(len(self.model.name)):
            if self.model.name[i] == 'iris':
                iris = ModelState()
                iris.pose = self.model.pose[i]
                self.global_x_trajectory = numpy.linspace(self.local_position_x,self.arg.goal_x,self.display_point)
                self.global_y_trajectory = numpy.linspace(self.local_position_y,self.arg.goal_y,self.display_point)
                self.global_z_trajectory = numpy.full((1, len(self.global_x_trajectory)), self.arg.height)
                self.local_position_x = round(iris.pose.position.x,3)
                self.local_position_y = round(iris.pose.position.y,3)
                self.local_position_z = round(iris.pose.position.z,3)
                self.local_orientation_x = iris.pose.orientation.x
                self.local_orientation_y = iris.pose.orientation.y
                self.local_orientation_z = iris.pose.orientation.z
                self.local_orientation_w = iris.pose.orientation.w
    def callback_number_of_obstacle(self,data):
        self.number_of_obstacle = int(data.data)
    def callback_number_of_human(self,data):
        self.number_of_human = int(data.data) 
    def callback_number_of_injury(self,data):
        self.number_of_injury = int(data.data) 
    def mavros_state_callback(self, msg):
        self.mavros_state = msg
    def callback_pointcloud(self, data):
        assert isinstance(data, PointCloud2)
        gen = point_cloud2.read_points(data,field_names=("x","y","z"), skip_nans=True)
        for p in gen:
            self.pointcloud_x_list.append(p[0])
            self.pointcloud_y_list.append(p[1])
            self.pointcloud_z_list.append(p[2])

    def kmeans_clustering(self,rx, ry, nc):
        clusters = Clusters(rx, ry, nc)
        clusters.calc_centroid(self.arg.height)
        pre_cost = float("inf")
        for loop in range(MAX_LOOP):
            cost = clusters.update_clusters()
            cluster_dict=clusters.calc_centroid(self.arg.height)
            d_cost = abs(cost - pre_cost)
            if d_cost < DCOST_TH:
                break
            pre_cost = cost
        return clusters, cluster_dict

    def interpolate_b_spline_path(self,x: list, y: list, n_path_points: int, degree: int = 3) -> tuple:
        ipl_t = numpy.linspace(0.0, len(x)-1, len(x))
        spl_i_x = scipy_interpolate.make_interp_spline(ipl_t, x, k=degree)
        spl_i_y = scipy_interpolate.make_interp_spline(ipl_t, y, k=degree)
        travel = numpy.linspace(0.0, len(x) - 1, n_path_points)
        return spl_i_x(travel), spl_i_y(travel)


    def store_path_planner_result(self,passthrough_height,cluster_center_list,nearest_point_circle,local_path_x_leftside,local_path_y_leftside,local_path_x_rightside,local_path_y_rightside):
        if passthrough_height is not None:
            if len(passthrough_height)!=0:
                if self.update_new_path==False:
                    self.past_passthrough_height = list(self.past_passthrough_height)
                    self.past_cluster_center_list = list(self.past_cluster_center_list)
                    self.past_nearest_point_circle = list(self.past_nearest_point_circle)
                    self.past_local_path_x_leftside = list(self.past_local_path_x_leftside)
                    self.past_local_path_y_leftside = list(self.past_local_path_y_leftside)
                    self.past_local_path_x_rightside = list(self.past_local_path_x_rightside)
                    self.past_local_path_y_rightside = list(self.past_local_path_y_rightside)
                    del self.past_passthrough_height[:]
                    del self.past_cluster_center_list[:]
                    del self.past_nearest_point_circle[:]
                    del self.past_local_path_x_leftside[:]
                    del self.past_local_path_y_leftside[:]
                    del self.past_local_path_x_rightside[:]
                    del self.past_local_path_y_rightside[:]

                    self.past_passthrough_height = passthrough_height
                    self.past_cluster_center_list = cluster_center_list
                    self.past_nearest_point_circle = nearest_point_circle
                    self.past_local_path_x_leftside = local_path_x_leftside
                    self.past_local_path_y_leftside = local_path_y_leftside
                    self.past_local_path_x_rightside = local_path_x_rightside
                    self.past_local_path_y_rightside = local_path_y_rightside
                    print("store path planner result")
                elif self.update_new_path==True:
                    print("output path planner result when update new path is True")
                    return self.past_passthrough_height,self.past_cluster_center_list,self.past_nearest_point_circle,self.past_local_path_x_leftside,self.past_local_path_y_leftside,self.past_local_path_x_rightside,self.past_local_path_y_rightside
            elif len(passthrough_height)==0:
                print("output path planner result when no obstacle pointcloud")
                return self.past_passthrough_height,self.past_cluster_center_list,self.past_nearest_point_circle,self.past_local_path_x_leftside,self.past_local_path_y_leftside,self.past_local_path_x_rightside,self.past_local_path_y_rightside
        else:
            print("just want to get output path planner result")
            return self.past_passthrough_height,self.past_cluster_center_list,self.past_nearest_point_circle,self.past_local_path_x_leftside,self.past_local_path_y_leftside,self.past_local_path_x_rightside,self.past_local_path_y_rightside

    def shortest_distance_from_point_to_line(self,start,end,point):
        d = numpy.linalg.norm(numpy.cross(end-start, start-point))/numpy.linalg.norm(end-start)
        return d

    def path_planner(self):
        rospy.wait_for_message('/gazebo/model_states', ModelStates)
        rospy.wait_for_message('/extract_indices/output', PointCloud2)
        self.callback_local_position()
        self.auto_tf()
        ripple_filter = pcl.PointCloud()
        pointcloud = numpy.array((self.pointcloud_x_list,self.pointcloud_y_list,self.pointcloud_z_list),dtype=numpy.float32)
        pointcloud = pointcloud.T
        ripple_filter.from_array(pointcloud)
        if pointcloud.size!=0:
            passthrough = ripple_filter.make_passthrough_filter()
            passthrough.set_filter_field_name("z")
            passthrough.set_filter_limits(self.arg.height-0.5, self.arg.height+0.5)
            passthrough_height = passthrough.filter()
            if passthrough_height.size!=0:
                passthrough_height = numpy.array(passthrough_height).T
                #passthrough_height = numpy.delete(passthrough_height, 2, 0)
                clusters, cluster_dict = self.kmeans_clustering(passthrough_height[0], passthrough_height[1], self.number_of_obstacle)
                cluster_centroid_x_list=[]
                cluster_centroid_y_list=[]
                cluster_centroid_z_list=[]
                safety_region=[]
                tmp_list=[]
                cluster_center_list=[]
                nearest_point_circle=[]

                for i in range(len(cluster_dict)):
                    tmp_list=cluster_dict.get(("cluster_"+str(i)),0)
                    if tmp_list!=0:
                        cluster_centroid_x_list.append(tmp_list[0])
                        cluster_centroid_y_list.append(tmp_list[1])
                        cluster_centroid_z_list.append(tmp_list[2])
                        safety_region.append(tmp_list[3])

                try:
                    cluster_centroid_x_list.remove(0)
                    cluster_centroid_y_list.remove(0)
                    cluster_centroid_z_list.remove(0)
                    safety_region.remove(0)
                    print("cluster missing!!!!!!!!!!!!!!!!")
                except:
                    print("no cluster missing")

                cluster_centroid_x_list = [round(num, 3) for num in cluster_centroid_x_list]
                cluster_centroid_y_list = [round(num, 3) for num in cluster_centroid_y_list]
                
                tmp_cluster_center_list = numpy.array((cluster_centroid_x_list,cluster_centroid_y_list,cluster_centroid_z_list),dtype=numpy.float32)
                for i in range(len(cluster_centroid_x_list)):
                    cluster_center_list.append([cluster_centroid_x_list[i],cluster_centroid_y_list[i],cluster_centroid_z_list[i]])


                ripple_filter.from_array(tmp_cluster_center_list.T)
                resolution = 0.2
                octree = ripple_filter.make_octreeSearch(resolution)
                octree.add_points_from_input_cloud()
                searchPoint = pcl.PointCloud()
                searchPoints = numpy.zeros((1, 3), dtype=numpy.float32)
                searchPoints[0][0] = self.local_position_x
                searchPoints[0][1] = self.local_position_y
                searchPoints[0][2] = self.local_position_z
                searchPoint.from_array(searchPoints)
                [ind, sqdist_nearest_point] = octree.nearest_k_search_for_cloud(searchPoint, 1)
                nearest_point=numpy.array((ripple_filter[ind[0][0]][0],ripple_filter[ind[0][0]][1],ripple_filter[ind[0][0]][2]))
                euclidean_distance=sqdist_nearest_point[0][0]

                self.desired_position_x = ripple_filter[ind[0][0]][0]
                self.desired_position_y = ripple_filter[ind[0][0]][1]
                self.desired_position_z = ripple_filter[ind[0][0]][2]
                
                # find out the desired position under which cluster
                desired_position_cluster = cluster_centroid_x_list.index(round(self.desired_position_x,3))

                for i in range(6):
                    nearest_point_circle.append([(nearest_point[0]+(safety_region[desired_position_cluster]+self.drone_width_for_the_obstacle)*math.cos(math.radians((60)*i))), (nearest_point[1]+(safety_region[desired_position_cluster]+self.drone_width_for_the_obstacle)*math.sin(math.radians((60)*i))), ripple_filter[ind[0][0]][2]])
                nearest_point_circle.sort()
                nearest_point_circle = numpy.array(nearest_point_circle)

                # way points
                local_path_x_leftside = [self.local_position_x, nearest_point_circle[0][0], nearest_point_circle[2][0],nearest_point_circle[4][0]]
                local_path_y_leftside = [self.local_position_y, nearest_point_circle[0][1], nearest_point_circle[2][1],nearest_point_circle[4][1]]
                local_path_x_rightside = [self.local_position_x, nearest_point_circle[1][0], nearest_point_circle[3][0],nearest_point_circle[5][0]]
                local_path_y_rightside = [self.local_position_y, nearest_point_circle[1][1], nearest_point_circle[3][1],nearest_point_circle[5][1]]

                if self.updated_path_time==0:
                    self.store_path_planner_result(passthrough_height,cluster_center_list,nearest_point_circle,local_path_x_leftside,local_path_y_leftside,local_path_x_rightside,local_path_y_rightside)
                if self.updated_path_time!=0:
                    #past_passthrough_height,past_cluster_center_list,past_nearest_point_circle,past_local_path_x_leftside,past_local_path_y_leftside,past_local_path_x_rightside,past_local_path_y_rightside=self.store_path_planner_result(None,None,None,None,None,None,None)
                    concat_x,concat_y,concat_z,local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline,global_x_trajectory,global_y_trajectory,global_z_trajectory=self.store_local_marker_array_result(None,None,None,None,None,None,None,None,None)
                    path_pointcloud = pcl.PointCloud()
                    path_point_list = numpy.array((concat_x,concat_y,concat_z),dtype=numpy.float32)
                    path_point_list = path_point_list.T
                    passthrough_height = passthrough_height.T
                    path_pointcloud.from_array(passthrough_height)
                    resolution = 0.2
                    octree = path_pointcloud.make_octreeSearch(resolution)
                    octree.add_points_from_input_cloud()
                    path_point = pcl.PointCloud()
                    path_points = numpy.zeros((1, 3), dtype=numpy.float32)
                    update_path=0
                    not_update_path=0
                    for i in range(len(concat_x)):
                        path_points[0][0] = path_point_list[i][0]
                        path_points[0][1] = path_point_list[i][1]
                        path_points[0][2] = path_point_list[i][2]
                        path_point.from_array(path_points)
                        [ind, sqdist] = octree.nearest_k_search_for_cloud(path_point, 1)
                        distance=sqdist[0][0]
                        if distance>self.drone_width_for_the_obstacle:
                            update_path = update_path+1
                        else:
                            not_update_path = not_update_path+1

                    rightside_point = numpy.array((local_path_x_rightside[3],local_path_y_rightside[3]),dtype=numpy.float32)
                    leftside_point = numpy.array((local_path_x_leftside[3],local_path_y_leftside[3]),dtype=numpy.float32)
                    start_point = numpy.array((self.local_position_x,self.local_position_y),dtype=numpy.float32)
                    end_point = numpy.array((self.arg.goal_x,self.arg.goal_y),dtype=numpy.float32)
                    distance_from_point_to_line_rightside = self.shortest_distance_from_point_to_line(start_point,end_point,rightside_point)
                    distance_from_point_to_line_leftside = self.shortest_distance_from_point_to_line(start_point,end_point,leftside_point)

                    if distance_from_point_to_line_rightside>distance_from_point_to_line_leftside:
                        self.shift = False
                        print("Leftside choosed")
                    else:
                        self.shift = True
                        print("Rightside choosed")

                    if update_path>not_update_path:
                        self.update_new_path = True
                        self.use_the_second_part = False
                        self.store_path_planner_result(passthrough_height,cluster_center_list,nearest_point_circle,local_path_x_leftside,local_path_y_leftside,local_path_x_rightside,local_path_y_rightside)
                        print("Choose to update the path")
                    else:
                        self.update_new_path = False
                        self.use_the_second_part = True
                        print("Choose to not update the path")
                    print("Finish check the current remaining path is near to the obstacle or not in after updated path one time situation")

                self.updated_path_time = self.updated_path_time+1
                return passthrough_height,cluster_center_list,nearest_point_circle,local_path_x_leftside,local_path_y_leftside,local_path_x_rightside,local_path_y_rightside
            else:
                #print("No obstacle detected")
                past_passthrough_height,past_cluster_center_list,past_nearest_point_circle,past_local_path_x_leftside,past_local_path_y_leftside,past_local_path_x_rightside,past_local_path_y_rightside=self.store_path_planner_result(None,None,None,None,None,None,None)
                print("Choose to not update the path")
                print("Finish check the current remaining path is near to the obstacle or not in No obstacle detected situation")
                self.updated_path_time = self.updated_path_time+1
                return past_passthrough_height,past_cluster_center_list,past_nearest_point_circle,past_local_path_x_leftside,past_local_path_y_leftside,past_local_path_x_rightside,past_local_path_y_rightside
        else:
            #print("No obstacle pointcloud")
            #self.use_the_second_part = True
            past_passthrough_height,past_cluster_center_list,past_nearest_point_circle,past_local_path_x_leftside,past_local_path_y_leftside,past_local_path_x_rightside,past_local_path_y_rightside=self.store_path_planner_result(None,None,None,None,None,None,None)
            self.update_new_path = False
            self.use_the_second_part = True
            print("Choose to not update the path")
            print("Finish check the current remaining path is near to the obstacle or not in No obstacle pointcloud situation")
            self.updated_path_time = self.updated_path_time+1
            return past_passthrough_height,past_cluster_center_list,past_nearest_point_circle,past_local_path_x_leftside,past_local_path_y_leftside,past_local_path_x_rightside,past_local_path_y_rightside
    
    # Create marker in rviz
    def create_marker(self,cluster_center_list):
        marker = Marker()
        marker_array = MarkerArray()
        for i in range(len(cluster_center_list)):
            marker.header.frame_id = world_frame_name
            marker.header.stamp = rospy.Time.now()+rospy.Duration(0.01*i)
            marker.ns = "nearest_position_with_kmean_cluster_centroid_marker"
            marker.action = marker.ADD
            marker.type = Marker.SPHERE_LIST
            marker.pose.orientation.w = 1.0
            point = Point(cluster_center_list[i][0],cluster_center_list[i][1],cluster_center_list[i][2])
            #print(point)
            marker.id = i
            marker.points.append(point)
            marker.scale.x = 0.1
            marker.scale.z = 0.1
            marker.scale.y = 0.1
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        self.marker_vis.publish(marker_array)
    
    # Create validation marker in rviz
    def create_validation_marker(self,nearest_point_circle):
        validation_marker = Marker()
        validation_marker_array = MarkerArray()
        for i in range(len(nearest_point_circle)):
            #local marker config
            validation_marker.header.frame_id = world_frame_name
            validation_marker.header.stamp = rospy.Time.now()+rospy.Duration(0.01*i)
            validation_marker.ns = "path_validation_marker"
            validation_marker.action = validation_marker.ADD
            validation_marker.type = Marker.SPHERE_LIST
            validation_marker.pose.orientation.w = 1.0
            point = Point(nearest_point_circle[i][0],nearest_point_circle[i][1],nearest_point_circle[i][2])
            validation_marker.id = i
            validation_marker.points.append(point)
            validation_marker.scale.x = 0.1
            validation_marker.scale.z = 0.1
            validation_marker.scale.y = 0.1
            validation_marker.color.a = 1.0
            validation_marker.color.r = 0.0
            validation_marker.color.g = 1.0
            validation_marker.color.b = 0.0
            validation_marker_array.markers.append(validation_marker)
    
        self.validation_marker_vis.publish(validation_marker_array)
        

    # Create local marker in rviz
    def create_local_marker_array(self,passthrough_height,local_path_x_leftside, local_path_y_leftside,local_path_x_rightside,local_path_y_rightside):

        if self.shift==False:
            local_marker = Marker()
            local_path_x_after_Bspline, local_path_y_after_Bspline = self.interpolate_b_spline_path(local_path_x_leftside, local_path_y_leftside,self.display_point)
            nearest_global_x_index = (numpy.abs(self.global_x_trajectory-local_path_x_after_Bspline[-1])).argmin()
            nearest_global_y_index = (numpy.abs(self.global_y_trajectory-local_path_y_after_Bspline[-1])).argmin()
            nearest_global_x=self.global_x_trajectory[nearest_global_x_index]
            nearest_global_y=self.global_y_trajectory[nearest_global_y_index]
            global_x_trajectory = numpy.linspace(nearest_global_x,self.arg.goal_x,self.display_point)
            global_y_trajectory = numpy.linspace(nearest_global_y,self.arg.goal_y,self.display_point)
            global_z_trajectory = numpy.full((1, len(global_x_trajectory)), self.arg.height)
            local_path_x_after_Bspline=numpy.array(local_path_x_after_Bspline)
            local_path_y_after_Bspline=numpy.array(local_path_y_after_Bspline)
            local_path_z_after_Bspline = numpy.full((1, len(local_path_x_after_Bspline)), self.arg.height)
            concat_x = numpy.concatenate([local_path_x_after_Bspline, global_x_trajectory])
            concat_y = numpy.concatenate([local_path_y_after_Bspline, global_y_trajectory])
            concat_z = numpy.full((1, len(concat_x)), self.arg.height)

            for i in range(len(concat_x)):
                #local marker config
                local_marker.header.frame_id = world_frame_name
                local_marker.header.stamp = rospy.Time.now()+rospy.Duration(0.01*i)
                local_marker.ns = "local_marker"
                local_marker.action = local_marker.ADD
                local_marker.type = Marker.SPHERE_LIST
                local_marker.pose.orientation.w = 1.0
                point = Point(concat_x[i],concat_y[i],concat_z[0][i])
                local_marker.id = i
                local_marker.points.append(point)
                local_marker.scale.x = 0.1
                local_marker.scale.z = 0.1
                local_marker.scale.y = 0.1
                local_marker.color.a = 1.0
                local_marker.color.r = 1.0
                local_marker.color.g = 0.0
                local_marker.color.b = 0.0
                self.local_marker_array.markers.append(local_marker)

            passthrough_height=list(passthrough_height)
            x_begin = max(concat_x[0], passthrough_height[0][0])     # 3
            x_end = min(concat_x[-1], passthrough_height[0][-1])     # 8
            points1 = [t for t in zip(concat_x, concat_y) if x_begin<=t[0]<=x_end]
            points2 = [t for t in zip(passthrough_height[0], passthrough_height[1]) if x_begin<=t[0]<=x_end]
            idx = 0
            nrof_points = len(points1)
            if len(points2)>=len(points1):
                while idx < nrof_points-1:
                    # Iterate over two line segments
                    y1_new = numpy.linspace(points1[idx][1], points1[idx+1][1], 1000)  # e.g., (6, 7) corresponding to (240, 50) in y1
                    y2_new = numpy.linspace(points2[idx][1], points2[idx+1][1], 1000)  # e.g., (6, 7) corresponding to (67, 88) in y2
                    tmp_idx = numpy.argwhere(numpy.isclose(y1_new, y2_new, atol=0.1)).reshape(-1)
                    if tmp_idx.size!=0:
                        self.shift = True
                        
                    idx += 1
            self.store_local_marker_array_result(concat_x,concat_y,concat_z[0],local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline[0],global_x_trajectory,global_y_trajectory,global_z_trajectory[0])

        if self.shift==True:
            local_marker = Marker()
            local_path_x_after_Bspline, local_path_y_after_Bspline = self.interpolate_b_spline_path(local_path_x_rightside, local_path_y_rightside,self.display_point)
            nearest_global_x_index = (numpy.abs(self.global_x_trajectory-local_path_x_after_Bspline[-1])).argmin()
            nearest_global_y_index = (numpy.abs(self.global_y_trajectory-local_path_y_after_Bspline[-1])).argmin()
            nearest_global_x=self.global_x_trajectory[nearest_global_x_index]
            nearest_global_y=self.global_y_trajectory[nearest_global_y_index]
            global_x_trajectory = numpy.linspace(nearest_global_x,self.arg.goal_x,self.display_point)
            global_y_trajectory = numpy.linspace(nearest_global_y,self.arg.goal_y,self.display_point)
            global_z_trajectory = numpy.full((1, len(global_x_trajectory)), self.arg.height)
            local_path_x_after_Bspline=numpy.array(local_path_x_after_Bspline)
            local_path_y_after_Bspline=numpy.array(local_path_y_after_Bspline)
            local_path_z_after_Bspline = numpy.full((1, len(local_path_x_after_Bspline)), self.arg.height)
            concat_x = numpy.concatenate([local_path_x_after_Bspline, global_x_trajectory])
            concat_y = numpy.concatenate([local_path_y_after_Bspline, global_y_trajectory])
            concat_z = numpy.full((1, len(concat_x)), self.arg.height)

            for i in range(len(concat_x)):
                #local marker config
                local_marker.header.frame_id = world_frame_name
                local_marker.header.stamp = rospy.Time.now()+rospy.Duration(0.01*i)
                local_marker.ns = "local_marker"
                local_marker.action = local_marker.ADD
                local_marker.type = Marker.SPHERE_LIST
                local_marker.pose.orientation.w = 1.0
                point = Point(concat_x[i],concat_y[i],concat_z[0][i])
                local_marker.id = i
                local_marker.points.append(point)
                local_marker.scale.x = 0.1
                local_marker.scale.z = 0.1
                local_marker.scale.y = 0.1
                local_marker.color.a = 1.0
                local_marker.color.r = 1.0
                local_marker.color.g = 0.0
                local_marker.color.b = 0.0
                self.local_marker_array.markers.append(local_marker)

            passthrough_height=list(passthrough_height)
            x_begin = max(concat_x[0], passthrough_height[0][0])     # 3
            x_end = min(concat_x[-1], passthrough_height[0][-1])     # 8
            points1 = [t for t in zip(concat_x, concat_y) if x_begin<=t[0]<=x_end]
            points2 = [t for t in zip(passthrough_height[0], passthrough_height[1]) if x_begin<=t[0]<=x_end]
            idx = 0
            nrof_points = len(points1)
            if len(points2)>=len(points1):
                while idx < nrof_points-1:
                    # Iterate over two line segments
                    y1_new = numpy.linspace(points1[idx][1], points1[idx+1][1], 1000)  # e.g., (6, 7) corresponding to (240, 50) in y1
                    y2_new = numpy.linspace(points2[idx][1], points2[idx+1][1], 1000)  # e.g., (6, 7) corresponding to (67, 88) in y2
                    tmp_idx = numpy.argwhere(numpy.isclose(y1_new, y2_new, atol=0.1)).reshape(-1)
                    idx += 1
            self.store_local_marker_array_result(concat_x,concat_y,concat_z[0],local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline[0],global_x_trajectory,global_y_trajectory,global_z_trajectory[0])

        
        return concat_x,concat_y,concat_z[0],local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline[0],global_x_trajectory,global_y_trajectory,global_z_trajectory[0]


    def store_local_marker_array_result(self,concat_x,concat_y,concat_z,local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline,global_x_trajectory,global_y_trajectory,global_z_trajectory):
        if concat_x is not None:
            if (self.update_new_path==False)&(self.use_the_second_part==False):
                self.past_concat_x = list(self.past_concat_x)
                self.past_concat_y = list(self.past_concat_y)
                self.past_concat_z = list(self.past_concat_z)
                self.past_local_path_x_after_Bspline = list(self.past_local_path_x_after_Bspline)
                self.past_local_path_y_after_Bspline = list(self.past_local_path_y_after_Bspline)
                self.past_local_path_z_after_Bspline = list(self.past_local_path_z_after_Bspline)
                self.past_global_x_trajectory = list(self.past_global_x_trajectory)
                self.past_global_y_trajectory = list(self.past_global_y_trajectory)
                self.past_global_z_trajectory = list(self.past_global_z_trajectory)
                del self.past_concat_x[:]
                del self.past_concat_y[:]
                del self.past_concat_z[:]
                del self.past_local_path_x_after_Bspline[:]
                del self.past_local_path_y_after_Bspline[:]
                del self.past_local_path_z_after_Bspline[:]
                del self.past_global_x_trajectory[:]
                del self.past_global_y_trajectory[:]
                del self.past_global_z_trajectory[:]

                self.past_concat_x = concat_x
                self.past_concat_y = concat_y
                self.past_concat_z = concat_z
                self.past_local_path_x_after_Bspline = local_path_x_after_Bspline
                self.past_local_path_y_after_Bspline = local_path_y_after_Bspline
                self.past_local_path_z_after_Bspline = local_path_z_after_Bspline

                self.past_global_x_trajectory = global_x_trajectory
                self.past_global_y_trajectory = global_y_trajectory
                self.past_global_z_trajectory = global_z_trajectory
                print("store marker result!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            elif (self.update_new_path==False)&(self.use_the_second_part==True):
                self.past_concat_x = list(self.past_concat_x)
                self.past_concat_y = list(self.past_concat_y)
                self.past_concat_z = list(self.past_concat_z)
                self.past_local_path_x_after_Bspline = list(self.past_local_path_x_after_Bspline)
                self.past_local_path_y_after_Bspline = list(self.past_local_path_y_after_Bspline)
                self.past_local_path_z_after_Bspline = list(self.past_local_path_z_after_Bspline)
                self.past_global_x_trajectory = list(self.past_global_x_trajectory)
                self.past_global_y_trajectory = list(self.past_global_y_trajectory)
                self.past_global_z_trajectory = list(self.past_global_z_trajectory)
                del self.past_concat_x[:]
                del self.past_concat_y[:]
                del self.past_concat_z[:]
                del self.past_local_path_x_after_Bspline[:]
                del self.past_local_path_y_after_Bspline[:]
                del self.past_local_path_z_after_Bspline[:]
                del self.past_global_x_trajectory[:]
                del self.past_global_y_trajectory[:]
                del self.past_global_z_trajectory[:]

                self.past_concat_x = concat_x
                self.past_concat_y = concat_y
                self.past_concat_z = concat_z
                self.past_local_path_x_after_Bspline = local_path_x_after_Bspline
                self.past_local_path_y_after_Bspline = local_path_y_after_Bspline
                self.past_local_path_z_after_Bspline = local_path_z_after_Bspline

                self.past_global_x_trajectory = global_x_trajectory
                self.past_global_y_trajectory = global_y_trajectory
                self.past_global_z_trajectory = global_z_trajectory
                print("update marker result as the drone moving along the second part of path")
                return self.past_concat_x,self.past_concat_y,self.past_concat_z,self.past_local_path_x_after_Bspline,self.past_local_path_y_after_Bspline,self.past_local_path_z_after_Bspline,self.past_global_x_trajectory,self.past_global_y_trajectory,self.past_global_z_trajectory

            elif (self.update_new_path==True)&(self.use_the_second_part==False):
                self.past_concat_x = list(self.past_concat_x)
                self.past_concat_y = list(self.past_concat_y)
                self.past_concat_z = list(self.past_concat_z)
                self.past_local_path_x_after_Bspline = list(self.past_local_path_x_after_Bspline)
                self.past_local_path_y_after_Bspline = list(self.past_local_path_y_after_Bspline)
                self.past_local_path_z_after_Bspline = list(self.past_local_path_z_after_Bspline)
                self.past_global_x_trajectory = list(self.past_global_x_trajectory)
                self.past_global_y_trajectory = list(self.past_global_y_trajectory)
                self.past_global_z_trajectory = list(self.past_global_z_trajectory)
                del self.past_concat_x[:]
                del self.past_concat_y[:]
                del self.past_concat_z[:]
                del self.past_local_path_x_after_Bspline[:]
                del self.past_local_path_y_after_Bspline[:]
                del self.past_local_path_z_after_Bspline[:]
                del self.past_global_x_trajectory[:]
                del self.past_global_y_trajectory[:]
                del self.past_global_z_trajectory[:]

                self.past_concat_x = concat_x
                self.past_concat_y = concat_y
                self.past_concat_z = concat_z
                self.past_local_path_x_after_Bspline = local_path_x_after_Bspline
                self.past_local_path_y_after_Bspline = local_path_y_after_Bspline
                self.past_local_path_z_after_Bspline = local_path_z_after_Bspline

                self.past_global_x_trajectory = global_x_trajectory
                self.past_global_y_trajectory = global_y_trajectory
                self.past_global_z_trajectory = global_z_trajectory
                print("update marker result as the drone moving along the path")
                return self.past_concat_x,self.past_concat_y,self.past_concat_z,self.past_local_path_x_after_Bspline,self.past_local_path_y_after_Bspline,self.past_local_path_z_after_Bspline,self.past_global_x_trajectory,self.past_global_y_trajectory,self.past_global_z_trajectory
        else:
            print("update new path: {}".format(self.update_new_path))
            print("use the second part: {}".format(self.use_the_second_part))
            print("updated path time: {}".format(self.updated_path_time))
            print("output marker result!")
            return self.past_concat_x,self.past_concat_y,self.past_concat_z,self.past_local_path_x_after_Bspline,self.past_local_path_y_after_Bspline,self.past_local_path_z_after_Bspline,self.past_global_x_trajectory,self.past_global_y_trajectory,self.past_global_z_trajectory


    # Moving in local position
    def local_position(self,concat_x,concat_y,concat_z,local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline,global_x_trajectory,global_y_trajectory,global_z_trajectory):
        self.model = rospy.wait_for_message('/gazebo/model_states', ModelStates)
        self.callback_local_position()
        self.auto_tf()        
        
        local_trajectory = Path()
        local_path_x_after_Bspline=list(local_path_x_after_Bspline)
        local_path_y_after_Bspline=list(local_path_y_after_Bspline)
        local_path_z_after_Bspline=list(local_path_z_after_Bspline)
        local_path_x_after_Bspline = [round(num, 3) for num in local_path_x_after_Bspline]
        local_path_y_after_Bspline = [round(num, 3) for num in local_path_y_after_Bspline]
        local_path_z_after_Bspline = [round(num, 3) for num in local_path_z_after_Bspline]

        global_x_trajectory=list(global_x_trajectory)
        global_y_trajectory=list(global_y_trajectory)
        global_z_trajectory=list(global_z_trajectory)
        global_x_trajectory = [round(num, 3) for num in global_x_trajectory]
        global_y_trajectory = [round(num, 3) for num in global_y_trajectory]
        global_z_trajectory = [round(num, 3) for num in global_z_trajectory]


        if self.use_the_second_part == False:
            local_trajectory_pose = PoseStamped()
            local_trajectory_pose.header.frame_id = world_frame_name
            local_trajectory_pose.header.stamp = rospy.Time.now()
            local_trajectory_pose.pose.position.x=local_path_x_after_Bspline[0]
            local_trajectory_pose.pose.position.y=local_path_y_after_Bspline[0]
            local_trajectory_pose.pose.position.z=local_path_z_after_Bspline[0]
            local_trajectory_pose.pose.orientation.w = 1.0
            local_trajectory.header.frame_id = world_frame_name
            local_trajectory.header.stamp = rospy.Time.now()
            local_trajectory.poses.append(local_trajectory_pose)
            self.local_trajectory_vis.publish(local_trajectory)

            posestamped = PoseStamped()
            posestamped.header.stamp = rospy.Time.now()
            posestamped.header.frame_id = base_link_name
            posestamped.pose.position.x = local_path_x_after_Bspline[0]
            posestamped.pose.position.y = local_path_y_after_Bspline[0]
            posestamped.pose.position.z = local_path_z_after_Bspline[0]

            tmp_y=self.arg.goal_y-local_path_y_after_Bspline[0]
            tmp_x=self.arg.goal_x-local_path_x_after_Bspline[0]
            quatrernion=[]        
            yaw_degree =  math.atan(tmp_y/tmp_x)
            if (tmp_y<0.0) & (tmp_x>0.0):
                yaw_degree=((self.pi/2)+yaw_degree)*-1
            elif (tmp_y<0.0) & (tmp_x<0.0):
                yaw_degree=self.pi+yaw_degree
            elif (tmp_y>0.0) & (tmp_x>0.0):
                yaw_degree=yaw_degree
            elif (tmp_y>0.0) & (tmp_x<0.0):
                yaw_degree=(self.pi/2)-yaw_degree
            quatrernion = quaternion_from_euler(0, 0, yaw_degree)
            posestamped.pose.orientation.x = quatrernion[0]
            posestamped.pose.orientation.y = quatrernion[1]
            posestamped.pose.orientation.z = quatrernion[2]
            posestamped.pose.orientation.w = quatrernion[3]

            self.position_pub.publish(posestamped)
            self.update_new_path = True
            rospy.sleep(rospy.Duration(0.2))
            local_path_x_after_Bspline.pop(0)
            local_path_y_after_Bspline.pop(0)
            local_path_z_after_Bspline.pop(0)
            print("local_path_x_after_Bspline len: {}".format(len(local_path_x_after_Bspline)))
            print("local_path_y_after_Bspline len: {}".format(len(local_path_y_after_Bspline)))
            print("local_path_z_after_Bspline len: {}".format(len(local_path_z_after_Bspline)))
            if len(local_path_x_after_Bspline)==0:
                self.update_new_path = False
                self.use_the_second_part = False
            self.store_local_marker_array_result(concat_x,concat_y,concat_z,local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline,global_x_trajectory,global_y_trajectory,global_z_trajectory)
            print("moving!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        elif self.use_the_second_part == True:
            local_trajectory_pose = PoseStamped()
            local_trajectory_pose.header.frame_id = world_frame_name
            local_trajectory_pose.header.stamp = rospy.Time.now()
            local_trajectory_pose.pose.position.x=global_x_trajectory[0]
            local_trajectory_pose.pose.position.y=global_y_trajectory[0]
            local_trajectory_pose.pose.position.z=global_z_trajectory[0]
            local_trajectory_pose.pose.orientation.w = 1.0
            local_trajectory.header.frame_id = world_frame_name
            local_trajectory.header.stamp = rospy.Time.now()
            local_trajectory.poses.append(local_trajectory_pose)
            self.local_trajectory_vis.publish(local_trajectory)

            posestamped = PoseStamped()
            posestamped.header.stamp = rospy.Time.now()
            posestamped.header.frame_id = base_link_name
            posestamped.pose.position.x = global_x_trajectory[0]
            posestamped.pose.position.y = global_y_trajectory[0]
            posestamped.pose.position.z = global_z_trajectory[0]
            tmp_y=self.arg.goal_y-global_y_trajectory[0]
            tmp_x=self.arg.goal_x-global_x_trajectory[0]
            quatrernion=[]        
            yaw_degree =  math.atan(tmp_y/tmp_x)
            if (tmp_y<0.0) & (tmp_x>0.0):
                yaw_degree=((self.pi/2)+yaw_degree)*-1
            elif (tmp_y<0.0) & (tmp_x<0.0):
                yaw_degree=self.pi+yaw_degree
            elif (tmp_y>0.0) & (tmp_x>0.0):
                yaw_degree=yaw_degree
            elif (tmp_y>0.0) & (tmp_x<0.0):
                yaw_degree=(self.pi/2)-yaw_degree
            quatrernion = quaternion_from_euler(0, 0, yaw_degree)
            posestamped.pose.orientation.x = quatrernion[0]
            posestamped.pose.orientation.y = quatrernion[1]
            posestamped.pose.orientation.z = quatrernion[2]
            posestamped.pose.orientation.w = quatrernion[3]
            self.position_pub.publish(posestamped)
            self.update_new_path = False
            rospy.sleep(rospy.Duration(0.2))
            global_x_trajectory.pop(0)
            global_y_trajectory.pop(0)
            global_z_trajectory.pop(0)
            print("global_x_trajectory len: {}".format(len(global_x_trajectory)))
            print("global_y_trajectory len: {}".format(len(global_y_trajectory)))
            print("global_z_trajectory len: {}".format(len(global_z_trajectory)))
            if len(global_x_trajectory)==0:
                self.update_new_path = False
                self.use_the_second_part = False
            self.store_local_marker_array_result(concat_x,concat_y,concat_z,local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline,global_x_trajectory,global_y_trajectory,global_z_trajectory)
        
            print("moving with second part!!!!!!!!!!!!!!!!!")


    def cmd(self):
        self.args = argparse.ArgumentParser(description="User choice")
        self.args.add_argument("-height", type=float, help="ready position height", default=1.0)
        self.args.add_argument("-s", type=float, help="start point")
        self.args.add_argument("-goal_x", type=float, help="x-axis end point", default=0.0)
        self.args.add_argument("-goal_y", type=float, help="y-axis end point", default=0.0)
        self.arg = self.args.parse_args()
        #squad_distance = (((self.arg.goal_x)**2)+((self.arg.goal_y)**2))**0.5
        #self.display_point = int(squad_distance/0.25)
        self.display_point = int(abs(self.arg.goal_x)*abs(self.arg.goal_y))

    def euler_to_quatrernion_ready_position(self):
        if (self.arg.goal_y==0.0) & (self.arg.goal_x==0.0):
            self.r_ready_position_quatrernion[0]=0.0
            self.r_ready_position_quatrernion[1]=0.0
            self.r_ready_position_quatrernion[2]=0.0
            self.r_ready_position_quatrernion[3]=1.0
        else:
            self.yaw_degree =  math.atan(self.arg.goal_y/self.arg.goal_x)
            if (self.arg.goal_y<0.0) & (self.arg.goal_x>0.0):
                self.yaw_degree=((self.pi/2)+self.yaw_degree)*-1    #correct
            elif (self.arg.goal_y<0.0) & (self.arg.goal_x<0.0):
                self.yaw_degree=self.pi+self.yaw_degree     #correct
            elif (self.arg.goal_y>0.0) & (self.arg.goal_x>0.0):
                self.yaw_degree=self.yaw_degree             #correct
            elif (self.arg.goal_y>0.0) & (self.arg.goal_x<0.0):
                self.yaw_degree=(self.pi/2)-self.yaw_degree #correct
            self.r_ready_position_quatrernion = quaternion_from_euler(0, 0, self.yaw_degree)

    def ready_position(self):
        self.posestamped = PoseStamped()
        self.posestamped.header.stamp = rospy.Time.now()
        self.posestamped.header.frame_id = base_link_name
        self.posestamped.pose.position.x = 0.0
        self.posestamped.pose.position.y = 0.0
        self.posestamped.pose.position.z = self.arg.height
        self.posestamped.pose.orientation.x = self.r_ready_position_quatrernion[0]
        self.posestamped.pose.orientation.y = self.r_ready_position_quatrernion[1]
        self.posestamped.pose.orientation.z = self.r_ready_position_quatrernion[2]
        self.posestamped.pose.orientation.w = self.r_ready_position_quatrernion[3]
        self.position_pub.publish(self.posestamped)

    def auto_tf(self):
        br = tf2_ros.TransformBroadcaster()
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = world_frame_name
        t.child_frame_id = base_link_name
        t.transform.translation.x = self.local_position_x
        t.transform.translation.y = self.local_position_y
        t.transform.translation.z = self.local_position_z
        t.transform.rotation.x = self.local_orientation_x
        t.transform.rotation.y = self.local_orientation_y
        t.transform.rotation.z = self.local_orientation_z
        t.transform.rotation.w = self.local_orientation_w
        br.sendTransform(t)
        #R.sleep()

    def arm_and_offboard(self):
        self.model = rospy.wait_for_message('/gazebo/model_states', ModelStates)
        self.callback_local_position()
        self.auto_tf()
        if (self.mavros_state.armed == False):
            self.set_arming_srv(True)
            rospy.sleep(rospy.Duration(1.0))
        elif (self.mavros_state.armed == True):
            if (self.mavros_state.mode != 'OFFBOARD'):
                self.euler_to_quatrernion_ready_position()
                self.ready_position()
                self.set_mode_srv(custom_mode='OFFBOARD')       #return true or false
                print("turn to offboard")
            elif (self.mavros_state.mode == 'OFFBOARD'):
                if self.finish_ready_position == False:
                    self.euler_to_quatrernion_ready_position()
                    self.ready_position()
                    if self.local_position_z>=self.arg.height-0.2:
                        self.finish_ready_position = True
                        rospy.sleep(rospy.Duration(1.0))
                elif self.finish_ready_position == True:
                    if (self.update_new_path == False)&(self.use_the_second_part == False):
                        passthrough_height,cluster_center_list,nearest_point_circle,local_path_x_leftside,local_path_y_leftside,local_path_x_rightside,local_path_y_rightside=self.path_planner()
                        self.create_marker(cluster_center_list)
                        self.create_validation_marker(nearest_point_circle)
                        concat_x,concat_y,concat_z,local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline,global_x_trajectory,global_y_trajectory,global_z_trajectory=self.create_local_marker_array(passthrough_height,local_path_x_leftside, local_path_y_leftside,local_path_x_rightside,local_path_y_rightside)
                        self.local_marker_vis.publish(self.local_marker_array)
                    
                    concat_x,concat_y,concat_z,local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline,global_x_trajectory,global_y_trajectory,global_z_trajectory=self.store_local_marker_array_result(None,None,None,None,None,None,None,None,None)
                    self.local_position(concat_x,concat_y,concat_z,local_path_x_after_Bspline,local_path_y_after_Bspline,local_path_z_after_Bspline,global_x_trajectory,global_y_trajectory,global_z_trajectory)        

            del self.pointcloud_x_list[:]
            del self.pointcloud_y_list[:]
            del self.pointcloud_z_list[:]

if __name__=="__main__":
    start = moving()
    start.cmd()
    while not rospy.is_shutdown():
        start.model = rospy.wait_for_message('/gazebo/model_states', ModelStates)

        # Control the drone
        start.arm_and_offboard()

