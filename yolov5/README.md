### YOLOv5 object detection with TensorRT and Realsense D455 in ROS
### Base on YOLOv5 from https://github.com/ultralytics/yolov5 and TensorRTx https://github.com/wang-xinyu/tensorrtx.git

### 1. Download the required project
```
git clone https://github.com/ultralytics/yolov5
git clone https://github.com/wang-xinyu/tensorrtx.git
```

### 2. Prepare your datasets
```

```

### 2.1 Datasets folder structures
```
├── dataset
   ├── annotations
   │   ├── xxx.xml
   │   
   ├── ImageSets
   │   ├── Main
   │       ├── test.txt
   │       ├── train.txt
   │       ├── trainval.txt
   │       ├── val.txt
   │   
   ├── labels
   │   ├── val 
   │       ├── xxx.txt
   |       ├──   ...
   │   ├── train
   │       ├── xxx.txt
   |       ├──   ...
   │   
   ├── images
   │   ├── val 
   │       ├── xxx.jpg
   |       ├──   ...
   │   ├── train
   │       ├── xxx.jpg
   |       ├──   ...
   ├── train.txt
   ├── test.txt
   └── valid.txt
   ├── split_train_val.py
   └── voc2yolo_label.py
```
