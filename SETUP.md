PX4 ROS Gazebo environment
```
pip3 install kconfiglib
pip3 install --user packaging
pip3 install --user jinja2
pip3 install --user jsonschema
sudo apt install gcc-arm-none-eabi

//go to /usr/include/newlib/math.h to add #define __ULong unsigned long at the beginning of the code

cd ~/Desktop
wget https://raw.githubusercontent.com/PX4/Devguide/master/build_scripts/ubuntu_sim_ros_melodic.sh
source ubuntu_sim_ros_melodic.sh
cd
mkdir src
cd ~/src
git clone https://github.com/PX4/Firmware.git
cd ~/src/Firmware
git submodule update --init --recursive
make px4_fmu-v3_default //refer to https://docs.px4.io/master/en/dev_setup/building_px4.html to check version
DONT_RUN=1 make px4_sitl_default gazebo
source Tools/setup_gazebo.bash $(pwd) $(pwd)/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/Tools/sitl_gazebo
sudo make px4_sitl_default gazebo
```
