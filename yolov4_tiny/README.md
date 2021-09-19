### Base on Digital Twins IC382 from https://github.com/vincent51689453/Digital_Twins_IC382

### 1 Configure ROS master uri settings
### 1.1 Find out the host IP address
Type ```ifconifg``` in command line for both master and slave host
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/ifconfig.png)

### 1.2 Find out the host name
Type ```hostname``` in command line

![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/hostname.png)

### 1.3 Change the bashrc file configuration
Open new terminal and type ```sudo gedit .bashrc```in command line to open bashrc file
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/open_bashrc.png)

Demo picture for slave bashrc file:
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/slaver_bashrc.png)
* Add/Change the ROS_MASTER_URI with master IP address such as```http://192.168.1.1:11311```
* Add/Change the ROS_IP with slave IP address such as ```ROS_IP=192.168.1.2```

Demo picture for master bashrc file:
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/master_bashrc.png)
* Add/Change the ROS_MASTER_URI with master IP address such as```http://192.168.1.1:11311```
* Add/Change the ROS_IP with master IP address such as ```ROS_IP=192.168.1.1```

### 1.4 Change the host file configuration
Open new terminal and type following command to change host file setting
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/open_hosts.png)

Demo picture for slave host file:
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/slaver_host.png)
* Add/Change the slave IP address with master host name such as ```192.168.1.1    master_host_name```
* Add/Change the master IP address with slave host name such as ```192.168.1.2    slave_host_name```

Demo picture for master host file:
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/master_host.png)
* Add/Change the master IP address with slave host name such as ```192.168.1.2    slave_host_name```
* Add/Change the slave IP address with master host name such as ```192.168.1.1    master_host_name```

### 1.5 Connect master and slave host with Lan connection and type ```source ~/.bashrc```
### Friendly reminder for ros master setting, don't turn on roscore in slave otherwise it will error

### 2 YOLO installation
### 2.1 Python2.7 Package preparation before using YOLOv4-tiny
Please review the requirement text file https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/yolov4-tiny/YOLO_requirement.txt
Friendly reminder for using PIL python2.7 package, Please ```sudo apt-get install libjpeg8-dev zlib1g-dev libfreetype6-dev``` first otherwise the PIL will not function as normal
### 2.2 Download YOLO script from this repository
Type ```git clone https://github.com/laitathei/Gazebo-rosserial-rescue-robot``` to download the script
### 3 YOLO training
### 3.1 Prepare the dataset via labelImg
* YOLO detection requirement VOC format dataset. labelImg can generate two kind of dataset which are VOC format and YOLO format. 
* Please use the window OS to download labelImg because the Qt version have conflict with python2 and python3. 
* The XML files will mix up with JPEG files in the same directory. Please seperate into two holder which are Annotations and JPEGImages. 
* Annotations will store all the XML files and JPEGImages will store all the JPEG file
* For the tutorial of using labelImg, please refer to https://github.com/tzutalin/labelImg

### 3.2 Generate train.txt file with bounding box XY coordinate
Before running the python script, please change the path with your own path
1. Using ```voc2yolov4.py``` to shuffle the dataset into different data such as train, test, validation randomly
2. Those train, test, validation data will store into train.txt, test.txt and validation.txt and trainval.txt respectively
3. These files directory is```/workspace/yolov4-tiny/VOCdevkit/VOC2007/ImageSets/Main```
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/text_file_directory.png)
4. Using ```voc_annotation.py``` to generate text file with with bounding box XY coordinate base on ImageSets text file information
If you have more than one object have been labelled, please add it into voc_annotation.py `classes = ["xx", "xx"]`
Also, add the class name into ```model_data/voc_classes.txt```with same order
5. The output of the voc_annotation.py will become ```2007_test.txt```, ```2007_train.txt```, ```2007_val.txt``` file
6. These three files will store in the same directory with ```voc_annotation.py```
The final dataset structure:
```
--workspace
          --VOCdevkit
                      --VOC2007
                                --Annotations (XML files)
                                --ImageSets (text file)
                                            --Main
                                                   --test.txt
                                                   --train.txt
                                                   --trainval.txt
                                                   --val.txt
                                                   
                                --JPEGImages (JPEG files)
          --2007_test.txt
          --2007_train.txt
          --2007_val.txt
          --voc_annotation.py
```
### 3.3 Config hyperparameters (optional)
In train.py, you can change:
* cuda (True/False)
* normalize (True/False)
* input_shape
* mosaic (True/False)
* cosine_scheduler (True/False)
* label_smoothing
* lr (learning rate)
* batch_size
* Init_Epoch
* Freeze_Epoch

### 3.4 Training
* Type ```python2.7 train.py``` in command line
* The validation loss and total loss will shown on the command line
* Also, the final model will also shown the validation loss and total loss
* After finish training, the model will be store into logs file
* Please choose the model with lowest validation loss and total loss which can result better performance

### 4 YOLO Deployment
### 4.1 yolo.py modification before inferencing
Please change the model path to your own model path
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/yolo_load_path.png)
### 4.2 Get the class label, scores and boundary boxes XY coordinates
* ```top_conf``` means the detected confidence level and the type is numpy.ndarray
* ```top_label```means the detected class and the type is numpy.ndarray
* ```boxes```means the XY coordinates of boundary boxes from detected class and the type is numpy.ndarray
Class label have the constant order which means the class label generated from detection also have constant order
For example, this is the my defined class in ```model_data/voc_classes.txt```
```
turn_left
turn_right
```
turn_left and turn_right will have an assumed index. The first class will have index 0 to refer to this class and the second class will have index 1.
more class that you have, higher index will provided to refer the class
If multi object detected in a single image, the ```top_conf``` will become ```[0 1 2 3...]```
According to the index for each class, we can easily to find out the class have been detected in the image
```
        global turn_left
        global turn_right
        global detection_score
        global detection_top_ymin
        global detection_top_xmin
        global detection_top_ymax
        global detection_top_xmax
        ......
        if len(top_conf) == 1:
            if top_label[0] == 1: # turn right detected
                turn_right = True
                turn_left = False
                detection_score=[top_conf[0]] # get the confidence level for the class
                detection_top_ymin=[boxes[0][0]]
                detection_top_xmin=[boxes[0][1]]
                detection_top_ymax=[boxes[0][2]]
                detection_top_xmax=[boxes[0][3]]

            if top_label[0] == 0: # turn left detected
                turn_left = True
                turn_right = False
                detection_score=[top_conf[0]] # get the confidence level for the class
                detection_top_ymin=[boxes[0][0]]
                detection_top_xmin=[boxes[0][1]]
                detection_top_ymax=[boxes[0][2]]
                detection_top_xmax=[boxes[0][3]]

        elif len(top_conf) == 2:    # more than one object is detected
            #compare two object score
            if top_conf[0] > top_conf[1]:    # left confidence level larger than right confidence level
                detection_score=[top_conf[0]]
                top_label=[top_label[0]]
                turn_left = True
                turn_right = False
                detection_top_ymin=[boxes[0][0]]
                detection_top_xmin=[boxes[0][1]]
                detection_top_ymax=[boxes[0][2]]
                detection_top_xmax=[boxes[0][3]]
                print ("{} is detected : {}".format("turn_left",score))

            elif top_conf[1] > top_conf[0]:    # right confidence level larger than left confidence level
                detection_score=[top_conf[1]]
                top_label=[top_label[1]]
                turn_right = True
                turn_left = False
                detection_top_ymin=[boxes[0][0]]
                detection_top_xmin=[boxes[0][1]]
                detection_top_ymax=[boxes[0][2]]
                detection_top_xmax=[boxes[0][3]]
            ......
    def detection_result(self):
        left=turn_left
        right=turn_right
        transfer_score=detection_score
        transfer_top_ymin=detection_top_ymin
        transfer_top_xmin=detection_top_xmin
        transfer_top_ymax=detection_top_ymax
        transfer_top_xmax=detection_top_xmax
        return left, right,transfer_score,transfer_top_ymin,transfer_top_xmin,transfer_top_ymax,transfer_top_xmax
```
* Base on this two class example in```yolo.py```, more than two class can be further develop
* The detection_result function will store the required variable and just need to call it via ```yolo.detection_result```
### 4.2 YOLO inferencing
* Type ```python2.7 inference_ros.py``` to inferencing
* The subscriber will keep listening ```'/robot/camera1/image_raw'``` this topic
* Change this camera topic to other camera if you want
![image](https://github.com/laitathei/Gazebo-rosserial-rescue-robot/blob/main/photo/inference.png)
