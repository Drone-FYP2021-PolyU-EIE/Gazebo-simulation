# Gazebo-simulation
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







### Remark
```test_attention_clipper.launch``` will automatically map the input Bounding Box into the point cloud, it seems to map the original point into d435_depth_optical_frame

```attention_pose_set.py``` will set the bounding box center via ```pose.position``` but the bounding boxwill defined by ```dimensions```
