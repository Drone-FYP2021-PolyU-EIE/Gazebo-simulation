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

# runfile (local)
(Example for install CUDA 10.1 when you want to downgrade the CUDA after install nvidia-driver-460)
wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.243_418.87.00_linux.run
# If it show the gcc version fail to verify, please add --override such as sudo sh cuda_10.1.243_418.87.00_linux.run --override
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


# Deb(local) (CUDA 11.4)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.4.4/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.4-470.82.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.4-470.82.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

gedit ~/.bashrc
export PATH=/usr/local/cuda-11.4/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
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

## TensorRT installation
```
## pip install
python3 -m pip install numpy
python3 -m pip install 'pycuda<2021.1'
python3 -m pip install --upgrade setuptools pip
python3 -m pip install nvidia-pyindex
python3 -m pip install --upgrade nvidia-tensorrt

## deb install
# Go to https://developer.nvidia.com/nvidia-tensorrt-8x-download find out the suitable TensorRT version that correspond with your cuda version
# After finish download
os="ubuntuxx04"
tag="cudax.x-trt8.x.x.x-yyyymmdd"
sudo dpkg -i nv-tensorrt-repo-${os}-${tag}_1-1_amd64.deb
sudo apt-key add /var/nv-tensorrt-repo-${os}-${tag}/7fa2af80.pub

sudo apt-get update
sudo apt-get install tensorrt
sudo apt-get install python3-libnvinfer-dev
sudo apt-get install uff-converter-tf
sudo apt-get install onnx-graphsurgeon
dpkg -l | grep TensorRT
# it should be shown below if you install successfully
ii  graphsurgeon-tf	8.2.0-1+cuda11.4	amd64	GraphSurgeon for TensorRT package
ii  libnvinfer-bin		8.2.0-1+cuda11.4	amd64	TensorRT binaries
ii  libnvinfer-dev		8.2.0-1+cuda11.4	amd64	TensorRT development libraries and headers
ii  libnvinfer-doc		8.2.0-1+cuda11.4	all	TensorRT documentation
ii  libnvinfer-plugin-dev	8.2.0-1+cuda11.4	amd64	TensorRT plugin libraries
ii  libnvinfer-plugin8	8.2.0-1+cuda11.4	amd64	TensorRT plugin libraries
ii  libnvinfer-samples	8.2.0-1+cuda11.4	all	TensorRT samples
ii  libnvinfer8		8.2.0-1+cuda11.4	amd64	TensorRT runtime libraries
ii  libnvonnxparsers-dev		8.2.0-1+cuda11.4	amd64	TensorRT ONNX libraries
ii  libnvonnxparsers8	8.2.0-1+cuda11.4	amd64	TensorRT ONNX libraries
ii  libnvparsers-dev	8.2.0-1+cuda11.4	amd64	TensorRT parsers libraries
ii  libnvparsers8	8.2.0-1+cuda11.4	amd64	TensorRT parsers libraries
ii  python3-libnvinfer	8.2.0-1+cuda11.4	amd64	Python 3 bindings for TensorRT
ii  python3-libnvinfer-dev	8.2.0-1+cuda11.4	amd64	Python 3 development package for TensorRT
ii  tensorrt		8.2.0.x-1+cuda11.4 	amd64	Meta package of TensorRT
ii  uff-converter-tf	8.2.0-1+cuda11.4	amd64	UFF converter for TensorRT package
ii  onnx-graphsurgeon   8.2.0-1+cuda11.4  amd64 ONNX GraphSurgeon for TensorRT package

# if you find the below problem, you should install the correct CUDA version
The following packages have unmet dependencies:
 tensorrt : Depends: libnvinfer8 (= 8.0.3-1+cuda11.3) but 8.2.3-1+cuda11.4 is to be installed
            Depends: libnvinfer-plugin8 (= 8.0.3-1+cuda11.3) but 8.2.3-1+cuda11.4 is to be installed
            Depends: libnvparsers8 (= 8.0.3-1+cuda11.3) but 8.2.3-1+cuda11.4 is to be installed
            Depends: libnvonnxparsers8 (= 8.0.3-1+cuda11.3) but 8.2.3-1+cuda11.4 is to be installed
            Depends: libnvinfer-bin (= 8.0.3-1+cuda11.3) but it is not going to be installed
            Depends: libnvinfer-dev (= 8.0.3-1+cuda11.3) but 8.2.3-1+cuda11.4 is to be installed
            Depends: libnvinfer-plugin-dev (= 8.0.3-1+cuda11.3) but 8.2.3-1+cuda11.4 is to be installed
            Depends: libnvparsers-dev (= 8.0.3-1+cuda11.3) but 8.2.3-1+cuda11.4 is to be installed
            Depends: libnvonnxparsers-dev (= 8.0.3-1+cuda11.3) but 8.2.3-1+cuda11.4 is to be installed
            Depends: libnvinfer-samples (= 8.0.3-1+cuda11.3) but it is not going to be installed
            Depends: libnvinfer-doc (= 8.0.3-1+cuda11.3) but it is not going to be installed
E: Unable to correct problems, you have held broken packages.
```

## Pytorch 1.4 (last version compatible with python 2.7)
```
# For CUDA 10.1
sudo apt install python3-pip
pip install torch==1.4.0 torchvision==0.5.0

# Verify the installation
# Show below code means installed correctly
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
sudo apt-get install ros-melodic-mavros ros-melodic-mavros-extras     ** for melodic
sudo apt-get install ros-noetic-mavros ros-noetic-mavros-extras     ** for noetic
sudo apt install gcc-arm-none-eabi
sudo apt install gperf
sudo apt-get install python-dev python3-dev libxml2-dev libxslt1-dev zlib1g-dev
sudo apt upgrade libignition-math2          **for gazebo error which cause the gazebo cannot launch

gedit ~/.bashrc
#put this statement in .bashrc
export OPENBLAS_CORETYPE=ARMV8 python3

cd ~/Desktop
wget https://raw.githubusercontent.com/PX4/Devguide/master/build_scripts/ubuntu_sim_ros_melodic.sh
wget https://raw.githubusercontent.com/mavlink/mavros/master/mavros/scripts/install_geographiclib_datasets.sh
chmod +x install_geographiclib_datasets.sh
chmod +x ubuntu_sim_ros_melodic.sh
sudo bash install_geographiclib_datasets.sh
source ubuntu_sim_ros_melodic.sh
cd

git clone https://github.com/PX4/PX4-Autopilot.git
cd ~/PX4-Autopilot
git checkout v1.12.3
bash ~/PX4-Autopilot/Tools/setup/ubuntu.sh --no-nuttx
git submodule update --init --recursive

make px4_fmu-v3_default **refer to https://docs.px4.io/master/en/dev_setup/building_px4.html to check version which only for hardware setup

cd ~/PX4-Autopilot
make px4_sitl_default gazebo

## Type the following code into .bashrc
source ~/PX4-Autopilot/Tools/setup_gazebo.bash ~/PX4-Autopilot ~/PX4-Autopilot/build/px4_sitl_default
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/PX4-Autopilot/Tools/sitl_gazebo

roslaunch mavros px4.launch fcu_url:="udp://:14540@127.0.0.1:14557"   **only sitl with gazebo
roslaunch px4 mavros_posix_sitl.launch        **SITL and MAVROS
```

## QGC installation
```
#### build from source
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
_____________________________________________________________________________________________________________________________________________________________

#### follow the official tutorial (https://docs.qgroundcontrol.com/master/en/getting_started/download_and_install.html)
sudo usermod -a -G dialout $USER
sudo apt-get remove modemmanager -y
sudo apt install gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-gl -y
***Download QGroundControl.AppImage***
chmod +x ./QGroundControl.AppImage
./QGroundControl.AppImage
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
# Ubuntu 18.04
sudo apt-get install ros-melodic-octomap-ros
sudo apt-get install ros-melodic-octomap-msgs 
sudo apt-get install ros-melodic-octomap-server
sudo apt-get install ros-melodic-octomap-rviz-plugins
sudo apt-get install ros-melodic-octomap-mapping
sudo apt-get install ros-melodic-octomap

# Ubuntu 20.04
sudo apt-get install ros-noetic-octomap-ros
sudo apt-get install ros-noetic-octomap-msgs 
sudo apt-get install ros-noetic-octomap-server
sudo apt-get install ros-noetic-octomap-rviz-plugins
sudo apt-get install ros-noetic-octomap-mapping
sudo apt-get install ros-noetic-octomap
```

## jsk_pcl_ros installation
```
# Ubuntu 18.04
sudo apt-get install ros-melodic-jsk-pcl-ros
sudo apt-get install ros-melodic-jsk-rviz-plugins
sudo apt-get install ros-melodic-ros-numpy

# Ubuntu 20.04
sudo apt-get install ros-noetic-jsk-pcl-ros
sudo apt-get install ros-noetic-jsk-rviz-plugins
sudo apt-get install ros-noetic-ros-numpy
```

## Realsense-viewer installation (Jetson version)
```
git clone https://github.com/IntelRealSense/librealsense.git
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install librealsense2-utils
sudo apt-get install librealsense2-dev
```

## Realsense ROS-Wrapper installation (Jetson version)
```
cd ~/catkin_ws/src
git clone https://github.com/IntelRealSense/realsense-ros.git
git clone https://github.com/pal-robotics/ddynamic_reconfigure.git
cd /opt/ros/melodic/share/cv_bridge/cmake
sudo gedit cv_bridgeConfig.cmake      ## change the opencv directory to opencv4
python3 -m pip install empy

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
git clone https://github.com/IntelRealSense/realsense-ros.git
cd realsense-ros/
git checkout ***the most stable version and support with your realsense-viewer version***
cd ..

git clone https://github.com/pal-robotics/ddynamic_reconfigure
cd ddynamic_reconfigure/
git checkout ***the most stable version***
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

## Vicon_bridge installation
```
cd ~/catkin_ws/src
git clone https://github.com/ethz-asl/vicon_bridge.git
cd ..
catkin_make
roslaunch vicon_bridge vicon.launch    ***change datastream_hostport to the suitable IP address***
***The tf frame vicon_world represent the drone***
```

## pip and pip3 installation ubuntu 20.04
```
sudo apt update
sudo apt install python2
sudo apt install python3
sudo apt install python2-pip
sudo apt install python3-pip
sudo -H pip3 install --upgrade pip
sudo -H pip2 install --upgrade pip
sudo apt install curl
curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
sudo python2 get-pip.py
pip2 --version ***show pip 20.3.4 from /usr/local/lib/python2.7/dist-packages/pip (python 2.7)***
pip3 --version ***show pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)***
```

## Signal installation
```
wget -O- https://updates.signal.org/desktop/apt/keys.asc | gpg --dearmor > signal-desktop-keyring.gpg
cat signal-desktop-keyring.gpg | sudo tee -a /usr/share/keyrings/signal-desktop-keyring.gpg > /dev/null
echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/signal-desktop-keyring.gpg] https://updates.signal.org/desktop/apt xenial main' |\
  sudo tee -a /etc/apt/sources.list.d/signal-xenial.list
sudo apt update && sudo apt install signal-desktop
```

## Grub-customizer (for Dual boot to change booting priority)
```
sudo apt install grub-customizer
```

## Jetson fan mode
```
sudo /usr/sbin/nvpmodel -d cool
```

## Chinese input (倉頡速成)
```
sudo apt-get install ibus-cangjie
# follow below website 
https://medium.com/hong-kong-linux-user-group/%E5%A6%82%E4%BD%95%E5%9C%A8ubuntu%E8%8B%B1%E6%96%87%E4%BB%8B%E9%9D%A2%E4%B8%8B%E4%BD%BF%E7%94%A8%E4%B8%AD%E6%96%87%E5%80%89%E9%A0%A1%E9%80%9F%E6%88%90%E8%BC%B8%E5%85%A5%E6%B3%95-24d0f4bcf479
```

## Vins-Fusion
```
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
make install

cd ~/catkin_ws/src
git clone https://github.com/HKUST-Aerial-Robotics/VINS-Fusion.git
cd ../
catkin_make
source ~/catkin_ws/devel/setup.bash
```

## Vrpn installation
```
sudo apt install ros-melodic-vrpn-client-ros
cd ~/catkin_ws/src
git clone https://github.com/ros-drivers/vrpn_client_ros.git
cd ..
catkin_make
```

## VNC Jetson installation (not finish yet)
```
sudo apt update
sudo apt install vino
gsettings set org.gnome.Vino prompt-enabled false
gsettings set org.gnome.Vino require-encryption false
nmcli connection show

***Example***
NAME                UUID                                  TYPE      DEVICE  
VICON_5G            f6e64dcf-90d5-4bfb-a3a6-976cf208ca37  wifi      wlan0   
docker0             65e700af-e350-440f-85e3-079e032fc55b  bridge    docker0 
EIA-W311MESH        3e3e1be4-9d3c-4cc7-836e-1034638191cd  wifi      --      
TOTOLINK A1004      52635202-f2bd-4c4c-8e40-1328a12899ac  wifi      --      
Wired connection 1  313a7146-a09a-34b9-bc54-d0c539139c4b  ethernet  --      

dconf write /org/gnome/settings-daemon/plugins/sharing/vino-server/enabled-connections "[' Put the desired UUID here']"
export DISPLAY=:0
ifconfig        ***record your current IP address***
```

##  xrdp remote desktop installation
```
sudo apt update
sudo apt-get install tightvncserver xrdp
sudo reboot
sudo apt-get install xubuntu-desktop
echo xfce4-session >~/.xsession
sudo service xrdp restart

***Turn to window use remote desktop access and enter the Jetson IP address***
```

## OBS Linux installation
```
sudo apt install ffmpeg
sudo add-apt-repository ppa:obsproject/obs-studio
sudo apt update
sudo apt install obs-studio
```

## Python3 ROS melodic installation
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install ros-melodic-desktop-full
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
echo "source ~/catkin_ws/install/setup.bash --extend" >> ~/.bashrc
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
sudo apt install python-rosdep

sudo apt-get install python-pip python-yaml 
sudo apt-get install python3-pip python3-yaml 
sudo pip3 install rospkg catkin_pkg
sudo apt-get install python-catkin-tools 
sudo apt-get install python3-catkin-tools 
sudo apt-get install python3-dev python3-numpy
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
```

## Jetson System Monitor (Jtop)
```
sudo -H pip install -U jetson-stats
reboot
sudo jtop
```

## Mediapipe (Jetson version)
```
sudo apt install python3-dev
sudo apt install cmake
sudo apt install protobuf-compiler
pip3 install scikit-build
pip3 install opencv_contrib_python
sudo apt-get install curl
git clone https://github.com/PINTO0309/mediapipe-bin
cd mediapipe-bin
./v0.8.5/numpy119x/mediapipe-0.8.5_cuda102-cp36-cp36m-linux_aarch64_numpy119x_jetsonnano_L4T32.5.1_download.sh   (這個 sh 檔會下載一些檔案)
pip3 install numpy-1.19.4-cp36-none-manylinux2014_aarch64.whl
pip3 install mediapipe-0.8.5_cuda102-cp36-none-linux_aarch64.whl
pip3 install opencv-python dataclasses
```

## Opencv 4.5.2(Jetson GPU version)
```
cd ~
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout tags/4.5.2
cd ..
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout tags/4.5.2
mkdir build
cd build

##Change according to cuda/gcc version
cmake --clean-first \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
-D WITH_OPENCL=ON \
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN=7.2 \
-D WITH_CUDNN=ON \
-D CUDNN_VERSION='8.0' \
-D WITH_CUBLAS=ON \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D OPENCV_DNN_CUDA=ON \
-D ENABLE_NEON=ON \
-D BUILD_opencv_cudacodec=ON \
-D WITH_QT=ON \
-D WITH_OPENMP=ON \
-D WITH_OPENGL=ON \
-D BUILD_TIFF=ON \
-D WITH_FFMPEG=ON \
-D WITH_GSTREAMER=ON \
-D WITH_TBB=ON \
-D BUILD_TBB=ON \
-D BUILD_TESTS=OFF \
-D WITH_EIGEN=ON \
-D WITH_V4L=ON \
-D WITH_LIBV4L=ON \
-D OPENCV_ENABLE_NONFREE=ON \
-D INSTALL_C_EXAMPLES=ON \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D BUILD_NEW_PYTHON_SUPPORT=ON \
-D BUILD_opencv_python2=ON \
-D BUILD_opencv_python3=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D WITH_NVCUVID=ON\
-D BUILD_EXAMPLES=ON ..

########### make will take ~ 2 hours, also make sure ram is not occupied

sudo make -j6 && sudo make -j6 install

########### reboot first

sudo apt install -y libssl-dev libusb-1.0-0-dev pkg-config build-essential cmake cmake-curses-gui libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev qtcreator python3 python3-dev apt-utils

cd ~/catkin_ws/src
git clone https://github.com/MartinNievas/vision_opencv.git
cd vision_opencv/
git checkout compile_oCV4
cd ~/catkin_ws
catkin build -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
```

## Google Chrome(PC version)
```
sudo apt install wget
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo dpkg -i google-chrome-stable_current_amd64.deb
```

## Python-PCL (PC version)
```
## Python3 PCL (ubuntu20.04)
sudo apt install python3-pcl
pip3 install scikit-fmm
```

## YOLOv5 (Jetson version)
```
sudo apt-get install python3-pip
pip3 install tqdm
pip3 install seaborn
pip3 install tensorflow-io

gedit ~/.bashrc
export CPATH=$CPATH:/usr/local/cuda-10.2/targets/aarch64-linux/include
export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-10.2/targets/aarch64-linux/lib
export CUDA_INC_DIR=/usr/local/cuda/include
export PATH=$PATH:/usr/local/cuda/bin

sudo apt-get update
sudo apt-get install -y build-essential libatlas-base-dev
sudo apt-get install libatlas-base-dev gfortran
pip3 install Cython
pip3 install pycuda --user
```

## Mediapipe (PC version)
```
pip3 install mediapipe
```

## Ubuntu OpenCV C++ environment setting (PC)
```
make sure you have installed eigen first, and check the version number
sudo apt install libopencv-dev

sudo gedit /usr/local/lib/pkgconfig/opencv.pc

prefix=/usr
exec_prefix=${prefix}
includedir=${prefix}/include
libdir=${exec_prefix}/lib64

Name: opencv
Description: The opencv library
Version:4.2.0
Cflags: -I${includedir}/opencv4
Libs: -L${libdir} -lopencv_stitching -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

pkg-config --cflags --libs opencv
should be able to see following statement:
-I/usr/include/opencv4 -L/usr/lib64 -lopencv_stitching -lopencv_objdetect -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_video -lopencv_photo -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_core

pkg-config --modversion opencv
show: 4.2.0

g++ test2.cpp -o test2 `pkg-config --cflags --libs opencv`
./test2
```

## Ubuntu Eigen C++ environment setting (Able to run in Vscode) (PC)
```
make sure you have installed eigen first
sudo ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
or
g++ $(pkg-config --cflags eigen3) test.cpp -o test
```

## Run Geographiclib in ubuntu command line
```
g++ test2.cpp -o test2 -l Geographic
```

## ROS TF in python2 problem
```
wstool init
wstool set -y src/geometry2 --git https://github.com/ros/geometry2 -v 0.6.5
wstool up
rosdep install --from-paths src --ignore-src -y -r
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/aarch64-linux-gnu/libpython3.6m.so -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release

#raspberry pi version
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.7m -DPYTHON_LIBRARY=/usr/lib/arm-linux-gnueabihf/libpython3.7m.so -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release
```

## ROS imu-tools (Complementary filter, Madgwick filter, rviz imu plugin etc.)
```
sudo apt-get install ros-<YOUR_ROSDISTO>-imu-tools
```

## Install C++ plugin in Window VS code
```
Just follow this https://blog.csdn.net/LeonTom/article/details/123015501
```

## Docker Usage
```
# Create new docker container
docker create -it --name <desired_name> <your_docker_image> -bash

# Delete docker container
docker rm <container id>

# Display all docker container
docker ps -a

# Start docker container
docker start <container_id> or <container_name>

# Get into container
docker attach <container_id> or <container_name>

# Exit and shut down container
exit

# Open the same container with more than one terminal
docker exec -it <container_id> or <container_name> bash

# Create docker that can access serial port with below statment
-v /dev/bus/usb:/dev/bus/usb --privileged

# Create docker that can share the same ip address with host with below statment
--network=host
```

## Extend disk space in liunx
```
https://blog.csdn.net/u010801439/article/details/77676668
```

## Install CH340 TTL to USB driver in linux
```
wget https://www.wch.cn/downloads/file/5.html?time=2023-03-17%2016:48:51&code=BBZJzJ2kw24QbUa8LUedUzVx4PRkjlwz9evWqUCb?time=2023-04-18%2017:56:51&code=Nij4leSn7aKQriDQTsGnjd1qnGHdu2lpD7qVcffe
unzip CH340SER_LINUX
cd CH340SER_LINUX/driver
make
sudo make load
```

## Install jupyter notebook
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3 python3-dev python3-pip -y
pip3 install --upgrade pip
pip3 install jupyter
apt-get install vim -y
jupyter notebook --generate-config
ipython

from notebook.auth import passwd
passwd()
input[1] Enter password: xxx
input[2] Verify password: xxx
output[3] 'example of key....................................'

vi /root/.jupyter/jupyter_notebook_config.py

c.NotebookApp.password=u'example of key....................................'
c.NotebookApp.open_browser=False
c.NotebookApp.allow_remote_access=True
c.NotebookApp.ip="xxx.xxx.xxx.xxx"
c.NotebookApp.port=8888
c.NotebookApp.notebook_dir="/home/xxx/xxx"

pip3 install jupyter_nbextensions_configurator
jupyter nbextensions_configurator enable --user
jupyter notebook
```

## Install onnxruntime, pytorch in raspberry pi armv7l system
```
# find out your debian version (raspberrian OS is kind of debian)

# Onnxruntime installation
cat /etc/debian_version
# 11.x bullseye
# 10.x buster
# 9.x stretch
Go to [https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux](https://github.com/nknytk/built-onnxruntime-for-raspberrypi-linux/tree/master/wheels)
Then, download your desired onnxruntime version wheel, and pip3 install it


# Pytorch installation
Go to https://github.com/KumaTea/pytorch-arm/releases to find your desired wheel and then just pip3 install it
```

## git process
```
# To git clone complete repo without losing the submodule
git clone --recursive <repo>

# To git clone and add the submodule with specific branch
git submodules add -b <branch-name> <repo>

# To switch to other local branch
git checkout <branch-name>/<head>

# To delete the branch locally
git branch --delete <branch-name>

# To show all the branch include local and remote
git branch -a

# To show local branch
git branch

# To delete local branch
git branch -d <branch-name>

# To show remote branch
git branch -r

# To delete remote branch
git branch -d -r <branch-name>

# To create local branch
git branch <branch-name>

# To rename the local branch
git branch -m <old-name> <new-name>

# To synchronize local git with github
git pull
```
