## Nvidia-driver installation
```
# Remove existing CuDA versions
sudo apt --purge remove "cublas*" "cuda*"
sudo apt --purge remove "nvidia*"
sudo rm -rf /usr/local/cuda*
sudo apt-get autoremove && sudo apt-get autoclean

# Reboot to remove cached files 
reboot

# After reboot
sudo apt-get clean

# check all available nvidia driver version in the repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install ubuntu-drivers-common

# check which nvidia driver version available for your own graphic card
ubuntu-drivers devices

# Remember do not install the nvidia-driver-server
# xxx is the nvidia-driver version that you want to install
sudo apt install nvidia-driver-xxx
reboot
```

## CUDA installation
```
# Go to https://developer.nvidia.com/cuda-toolkit-archive to search the cuda version that you want to install
# Refer to https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html to check which cuda version match with nvidia-driver version
# Higher nvidia-driver version is compatible install lower CUDA version
# Remember not to choose deb(local) for Installer Type if you want to downgrade your CUDA version
# Choose runfile(local) for Installer Type if you want to downgrade your CUDA version
# Then follow the instruction from website

(Example for install CUDA 10.1 when you want to downgrade the CUDA after install nvidia-driver-460)
wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
sudo sh cuda_10.1.243_418.87.00_linux.run

Choose continue
Do you accept the above EULA? (accept/decline/quit):
accept

│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 418.87.00                                                           │
│ + [X] CUDA Toolkit 10.1                                                      │
│   [X] CUDA Samples 10.1                                                      │
│   [X] CUDA Demo Suite 10.1                                                   │
│   [X] CUDA Documentation 10.1                                                │
│   Options                                                                    │
│   Install                                                                    │

Choose install
# After installation
nvcc -V (should be show CUDA 10.1)
nvidia-smi (should be show NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2)

gedit ~/.bashrc
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda
source ~/.bashrc
```

## cuDNN installation
```
# Go to https://developer.nvidia.com/rdp/cudnn-archive to find out the correct Platform and correct CUDA version for your own situation
# Download cuDNN Runtime Library + cuDNN Developer Library by clicking them
# cuDNN Code Samples and User Guide is optional
# Navigate to your <cudnnpath> directory containing the cuDNN Debian file.
# Replace x.x and 8.x.x.x with your specific CUDAand cuDNN versions and package date

sudo dpkg -i libcudnn8_x.x.x-1+cudax.x_amd64.deb
sudo dpkg -i libcudnn8-dev_8.x.x.x-1+cudax.x_amd64.deb
sudo dpkg -i libcudnn8-samples_8.x.x.x-1+cudax.x_amd64.deb (optional)

(Example for install cuDNN 8.0.5 which compatible with CUDA 10.1 in Ubuntu 18.04)
# The following codes must be installed in order
sudo dpkg -i libcudnn8_8.0.5.39-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn8-dev_8.0.5.39-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn8-samples_8.0.5.39-1+cuda10.1_amd64.deb (optional)

sudo cp /usr/include/cudnn.h /usr/local/cuda/include
sudo chmod a+x /usr/local/cuda/include/cudnn.h
cat /usr/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# Show below information means install cuDNN success
  #define CUDNN_MAJOR 8
  #define CUDNN_MINOR 0
  #define CUDNN_PATCHLEVEL 5
  --
  #define CUDNN_VERSION (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)

  #endif /* CUDNN_VERSION_H */
```

## Pytorch 1.4 (last version compatible with python 2.7)
```
# For CUDA 10.1
pip install torch==1.4.0 torchvision==0.5.0

# Verify the installation
# Show below code means installation correctly
python2.7
Python 2.7.17 (default, Feb 27 2021, 15:10:58) 
[GCC 7.5.0] on linux2
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> exit()
```

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

## Realsense-viewer installation (PC version)
```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install librealsense2-dkms
sudo apt-get install librealsense2-utils
sudo apt-get install librealsense2-dev
sudo apt-get install librealsense2-dbg
```

## Realsense ROS-Wrapper installation (PC version)
```
cd ~/catkin_ws
cd src
git clone https://github.com/IntelRealSense/realsense-ros
git clone https://github.com/pal-robotics/ddynamic_reconfigure
cd ..
catkin_make     or      catkin build
```

## ROS bag installation
```
sudo apt install ffmpeg
sudo apt-get install ubuntu-restricted-extras

(Example for using rosbag)
rosparam set /use_sim_time false
rosbag record /camera/color/image_raw
python rosbag2video.py bag_file_name
```

## Type the following code into .bashrc
```
source ~/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/sitl_gazebo
```
