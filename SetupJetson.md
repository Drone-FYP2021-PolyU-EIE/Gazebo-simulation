# Setup (For Jetson)
Set fan mode to cool
```bash
#sudo /usr/sbin/nvpmodel -d <fan_mode:quiet||cool>
sudo /usr/sbin/nvpmodel -d cool
```

## Ros setup px4 
``` bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
```
## create px4 in catkin 
NOTE: ROS Melodic is installed with Gazebo9 by default.   
Your catkin (ROS build system) workspace is created at ~/catkin_ws/.(<---this important)    
The script uses instructions from the ROS Wiki "Melodic"    
```bash
bash ubuntu.sh --no-nuttx
```

## fix for 19 issues
in /build/px4_sitl_default/etc/init.d-posix/rcS   
in add after line 108 `param set MAV_SYS_ID $((px4_instance+1))`   
```bash 
param set simulator_udp_port $((14560+px4_instance))
```
in add after line 213 `. px4-rc.params`   
```bash
simulator start -u $simulator_udp_port
```
in /Tools/sitl_gazebo/models/iris/iris.sdf in line 467 change from `<use_tcp>1</use_tcp>` to    
```bash
<use_tcp>0</use_tcp>
```

## problem in install matplotlib
```bash
#put this statement in .bashrc
export OPENBLAS_CORETYPE=ARMV8 python3


```
