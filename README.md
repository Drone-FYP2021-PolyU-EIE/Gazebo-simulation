# Gazebo-simulation
### Subscriber (in navigation node)
`"/drone/nagvation/pos"`, `PoseStamped`, come from `control` node and refers to the target position <br />
`"/auto_mode/status"`, `Bool`, come from `control` node and refers to whether turns to auto mode <br />
`"/detection_result/number_of_obstacle"`, `Int32`, come from `detection` node and refers to number of obstacle <br />
`"/detection_result/number_of_human"`, `Int32`, come from `detection` node and refers to number of human <br />
`"/detection_result/number_of_injury"`, `Int32`, come from `detection` node and refers to number of injury <br />
`"/extract_indices/output"`, `Int32`, come from `detection` node and refers to the pointcloud that processed by filter <br />

### Publisher (in navigation node)
`"/drone/input_postion/pose"`, `PoseStamped`, send to `control` node and refers to the position that the drone should be go
`"/desired_path/position"`, `MarkerArray`, for observe purpose and refers to the approximated obstacle centroid
`"/desired_path/validation_position"`, `MarkerArray`, for observe purpose and refers to the point around the approximated obstacle centroid
`"/desired_path/local_marker"`, `MarkerArray`, for observe purpose and refers to the drone path

### Update Blog
2021-8-26 upload ```environment.py``` which can allow the obstacle to place into the game field randomly

2021-8-27 upload ```motion.py``` which allow to maintain particular height and proivde rgb, depth, infra1 and infra2 drone view

2021-8-30 upload ```empty_world.world```, ```calibrate_world.world``` and ```camera_calibration.launch``` to prepare the camera calibration environment

2021-8-31 update ```motion.py``` to provide distance data via depth message

2021-9-4 update ```motion.py``` to provide dynamic filter for the cropped region pointcloud and write the record the point cloud XYZ into csv file

2021-9-4 upload ```plot_csv.py``` which can povide the visual animate 3d plot from csv file data

2021-9-5 update ```camera_calibration.launch```, ```spawn.launch``` and ```field_realsense.launch``` to visualize the drone model in rviz and correct the right tf between the drone and the world 

2021-9-6 upload ```camera_capture.py``` which allow to press keyboard button to capture the camera frame and save it into specific folder

2021-9-7 upload ```camera_image_callback.py``` which include camera callback class specially for return D435i camera data

2021-9-10 upload ```test_depth_distance.py``` which can detect the bounding box distance and show the marker on the image window

2021-9-10 upload ```test_attention_clipper.launch``` which can allow input Bounding Box topic to change the point cloud ROI size 

2021-9-10 upload ```attention_pose_set.py``` which can publish the input Bounding Box topic for ```test_attention_clipper.launch``` to change the Bounding Box size

2021-9-11 upload ```test_2d_to_3d.world```, ```test_2d_to_3d.launch``` and box (0.5m) model for testing ```2d_coordinate_to_3d_coordinate.py```

2021-9-17 upload ```publish_marker.py``` which provide that the 2d rgb or depth image point to the 3d position rely on the below matrix

```
                               [ 2d point x]    [fx     0      u      0] [3d point x]
depth distance to the 3d point [ 2d point y] =  [0      fy     v      0] [3d point y]
                               [     1     ]    [0      0      1      0] [depth distance to the 3d point]
                                                                         [1]
                                                                         
the 3d point in rviz have a little bit error, it should be the calculated 3d point is based on the world frame as origin instead of d435 camera
After testing, it should be not a big problem as the point will set in front of the trunk which means the front part of the trunk will be consider
```

2021-9-19 upload ```yolov4-tiny``` folder which responsible for detecting the tree and ```dynamic_filter.launch``` for launch the clipped point cloud

2021-9-19 upload ```dynamic_filter.rviz``` for visualize the clipped point cloud performance

2021-10-19 upload ```SETUP.md``` for PX4 Gazebo with ROS Wrapper environment setup

2021-10-27 update ```SETUP.md``` and complete the PX4 Gazebo with ROS Wrapper environment setup

2021-10-29 successfully install realsense D435i into drone model and all the camera topic seems work normally, start to study VINS-FUSION for localization component

2021-11-8 finish the vins-fusion testing in Gazebo environment and start to study octomap

2021-11-10 finish the testing for vins-fusion part and octomap with downsampled point cloud in Gazebo, start to implement the object detection into the Gazebo

2021-11-11 update ```SETUP.md``` with VINS-fusion installation, octomap installation and QGC installation

2021-11-12 update ```SETUP.md``` with nvidia-driver, CUDA, cuDNN and Pytorch installation

2021-12-17 upload ```realworld_dynamic_filter_testing``` folder which able to find obstacle and people attention point cloud by dynamic filter perfectly

2022-1-11 update ```SETUP.md``` for ROS noetic version software and confirm that noetic version default python is ```Python3```

2022-1-22 update ```SETUP.md``` with remote Jetson, Vrpn installation

2022-1-22 successfully convert the ```moving.py``` into ```moving_with_auto_tf.py``` which can change the transform between camera frame to the world frame

2022-1-25 NED or ENU system need to be confirm, the futher study on https://blog.csdn.net/qq_33641919/article/details/101003978

2022-2-11 Updated ```moving_with_static_obstacle_v8.py``` which included k-means cluster to find the cluster pointcloud size and automatically update the path with best current solution   

2022-2-21 Steve: added note to setup jetson (`SetupJetson.md`)  

2022-2-23 Steve: added note to setup jetson to boot from SSD refer readme `SetupJetsonSSD.md`     

2022-3-3 Upload ```yolov5``` in Pytorch which speed up with TensorRT

2022-3-6 Finish ```YOLOv4-tiny``` in Darknet which speed up with TensorRT, specifiy details in others repo

### Remark
```test_attention_clipper.launch``` will automatically map the input Bounding Box into the point cloud, it seems to map the original point into d435_depth_optical_frame

```attention_pose_set.py``` will set the bounding box center via ```pose.position``` but the bounding boxwill defined by ```dimensions```
```sudo apt-get install ros-melodic-geographic-msgs``` for the hector localization
```/d435/depth/color/points``` is ```organized point cloud``` and datatype is ```XYZRGB```
