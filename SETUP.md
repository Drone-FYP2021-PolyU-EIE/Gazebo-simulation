PX4 ROS Gazebo environment
```
pip3 install --user empy
pip3 install --user toml
pip3 install --user numpy
pip3 install --user pyros-genmsg
pip3 install kconfiglib
pip3 install --user packaging
pip3 install --user jinja2
pip3 install --user jsonschema
sudo apt-get install ros-melodic-mavros ros-melodic-mavros-extras
sudo apt install gcc-arm-none-eabi
sudo apt install gperf
sudo apt-get install python-dev python3-dev libxml2-dev libxslt1-dev zlib1g-dev

cd ~/Desktop
git clone https://github.com/PX4/pyulog
cd pyulog   ** remember to add from __future__ import print_function into versioneer.py which inside pyulog
python setup.py build install
cd ~/Desktop
wget https://raw.githubusercontent.com/PX4/Devguide/master/build_scripts/ubuntu_sim_ros_melodic.sh
source ubuntu_sim_ros_melodic.sh
cd
mkdir src
cd ~/src
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd ~/src/Firmware
git submodule update --init --recursive

make px4_fmu-v3_default **refer to https://docs.px4.io/master/en/dev_setup/building_px4.html to check version which only for hardware setup

DONT_RUN=1 make px4_sitl_default gazebo
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"
```

type the following code into .bashrc
```
source /home/fyp/Desktop/PX4-Autopilot/Tools/setup_gazebo.bash /home/fyp/Desktop/PX4-Autopilot/ /home/fyp/Desktop/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/home/fyp/Desktop/PX4-Autopilot/
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:/home/fyp/Desktop/PX4-Autopilot/Tools/sitl_gazebo
```