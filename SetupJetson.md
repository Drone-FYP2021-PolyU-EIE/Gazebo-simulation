# Setup (For Jetson)
Set fan mode to cool
```bash
#sudo /usr/sbin/nvpmodel -d <fan_mode:quiet||cool>
sudo /usr/sbin/nvpmodel -d cool
```
## Setup ROS for python3
```bash
sudo apt-get install python3-pip python3-yaml
sudo pip3 install rospkg catkin_pkg
```
## Ros setup px4     
``` bash
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx
```

## Create px4 in catkin
NOTE: ROS Melodic is installed with Gazebo9 by default.   
Your catkin (ROS build system) workspace is created at ~/catkin_ws/.(<---this important)    
The script uses instructions from the ROS Wiki "Melodic" 
```bash 
wget https://raw.githubusercontent.com/PX4/Devguide/master/build_scripts/ubuntu_sim_ros_melodic.sh
bash ubuntu_sim_ros_melodic.sh
```
in `.bashrc` find `export ROS_IP=192.168.x.xxx` remove this line, this line wil f*** ros if you change new ip
change to`export ROS_HOSTNAME=localhost`    
Add following in `.bashrc` (assume that the catkin workspace for px4 is in `~/PX4-Autopilot`)
```bash 
source ~/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/sitl_gazebo
```
Then you can test the gazbo setup by:    
```bash
roslaunch px4 mavros_posix_sitl.launch
```
if you see ` Waiting for simulator to accept connection on TCP port 4560` then do the `fix for 19 TCP issues` section la    

## fix for 19 TCP issues (use UDP instead)
in ~/PX4-Autopilot/build/px4_sitl_default/etc/init.d-posix/rcS
```bash
vim ~/PX4-Autopilot/build/px4_sitl_default/etc/init.d-posix/rcS
```

in add *after* line 108 `param set MAV_SYS_ID $((px4_instance+1))`   
```bash 
param set simulator_udp_port $((14560+px4_instance))
```
add *before* line 213 `. px4-rc.params`   
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

## Remote Destkop(xrdp) for Jetson 
```bash
sudo apt install -y xrdp
sudo apt-get install gnome-tweak-tool gnome-shell-extensions  -y
gnome-shell-extension-tool -e ubuntu-dock@ubuntu.com
```
`vim /etc/polkit-1/localauthority/50-local.d/45-allow.colord.pkla` create the following   
```bash
[Allow Colord all Users]
Identity=unix-user:*
Action=org.freedesktop.color-manager.create-device;org.freedesktop.color-manager.create-profile;org.freedesktop.color-manager.delete-device;org.freedesktop.color-manager.delete-profile;org.freedesktop.color-manager.modify-device;org.freedesktop.color-manager.modify-profile
ResultAny=no
ResultInactive=no
ResultActive=yes
```
`vim /etc/polkit-1/localauthority.conf.d/02-allow-colord.conf` create the following   
```bash
polkit.addRule(function(action, subject) {
 if ((action.id == "org.freedesktop.color-manager.create-device" ||
 action.id == "org.freedesktop.color-manager.create-profile" ||
 action.id == "org.freedesktop.color-manager.delete-device" ||
 action.id == "org.freedesktop.color-manager.delete-profile" ||
 action.id == "org.freedesktop.color-manager.modify-device" ||
 action.id == "org.freedesktop.color-manager.modify-profile") &&
 subject.isInGroup("{users}")) {
 return polkit.Result.YES;
 }
 });
```
## Fast test in cmd
```
rostopic pub -r 20 /mavros/setpoint_position/local geometry_msgs/PoseStamped "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: ''
pose:
  position:
    x: 0.0
    y: 0.0
    z: 0.5
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 0.0" 

rosrun mavros mavsafety arm
rosrun mavros mavsys mode -c OFFBOARD
```
