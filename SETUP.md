## PX4 ROS Gazebo environment
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
sudo apt upgrade libignition-math2          **for gazebo error which cause the gazebo cannot launch

cd ~/Desktop
wget https://raw.githubusercontent.com/PX4/Devguide/master/build_scripts/ubuntu_sim_ros_melodic.sh
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
sudo bash install_geographiclib_datasets.sh
source ubuntu_sim_ros_melodic.sh
cd

git clone https://github.com/PX4/PX4-Autopilot.git
cd ~/PX4-Autopilot
git checkout v1.12.3
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx
git submodule update --init --recursive

make px4_fmu-v3_default **refer to https://docs.px4.io/master/en/dev_setup/building_px4.html to check version which only for hardware setup

cd ~/PX4-Autopilot
make px4_sitl_default gazebo
roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"   **only sitl with gazebo
roslaunch px4 mavros_posix_sitl.launch        **SITL and MAVROS
```

## QGC installation
```
sudo apt-get install speech-dispatcher libudev-dev libsdl2-dev
cd
git clone --recursive -j8 https://github.com/mavlink/qgroundcontrol.git
git submodule update --recursive

***create account for QT***
***download the online installer from https://www.qt.io/download-qt-installer?hsCtaTracking=99d9dd4f-5681-48d2-b096-470725510d34%7C074ddad0-fdef-4e53-8aa8-5e8a876d6ab4***
chmod +x qt-unified-linux-x64-4.1.1-online.run
./qt-unified-linux-x64-4.1.1-online.run
***follow https://dev.qgroundcontrol.com/master/en/getting_started/index.html QT part 2 step to install the correct version***
sudo apt install libsdl2-dev
```

## ceres-solver installation
```
***get the latest stable version of ceres-solver from http://ceres-solver.org/installation.html***
sudo apt-get install cmake
sudo apt-get install libgoogle-glog-dev libgflags-dev
sudo apt-get install libatlas-base-dev
sudo apt-get install libeigen3-dev
sudo apt-get install libsuitesparse-dev
tar zxf ceres-solver-2.0.0.tar.gz
mkdir ceres-bin
cd ceres-bin
cmake ../ceres-solver-2.0.0
make -j3
make test
sudo make install
```

## ROS Octomap installation
```
sudo apt-get install ros-melodic-octomap-ros
sudo apt-get install ros-melodic-octomap-msgs 
sudo apt-get install ros-melodic-octomap-server
sudo apt-get install ros-melodic-octomap-rviz-plugins
sudo apt-get install ros-melodic-octomap-mapping
sudo apt-get install ros-melodic-octomap
```

## jsk_pcl_ros installation
```
sudo apt-get install ros-melodic-jsk-pcl-ros
sudo apt-get install ros-melodic-jsk-rviz-plugins
sudo apt-get install ros-melodic-ros-numpy
```

## Type the following code into .bashrc
```
source ~/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/sitl_gazebo
```
