### YOLOv5 object detection with TensorRT and Realsense D455 in ROS
### Base on YOLOv5 from https://github.com/ultralytics/yolov5 and TensorRTx https://github.com/wang-xinyu/tensorrtx.git

### 1. Download the required project
```
cd ~/Desktop
mkdir workspace
cd workspace
git clone https://github.com/ultralytics/yolov5
git clone https://github.com/wang-xinyu/tensorrtx.git
```

### 2. Prepare your datasets
```
cd ~/Desktop/workspace
mkdir dataset
cd dataset
mkdir annotations
mkdir images
mkdir ImageSets
cd ImageSets
mkdir Main
```

### 2.1 Place your images and annotations into images and annotations folder respectively
### 2.2 Put the ```split_train_val.py``` and ```voc2yolo_label.py```into dataset folder
### 2.3 Change ```voc2yolo_label.py``` line 8-13 to your config
```
classes = ["obstacle", "human", "injury"]   # 改成自己的类别
image_dir = "/home/laitathei/Desktop/workspace/dataset/images/"
labels_dir = "/home/laitathei/Desktop/workspace/dataset/labels/"
annotations_dir = "/home/laitathei/Desktop/workspace/dataset/annotations/"
ImageSets_Main_dir = "/home/laitathei/Desktop/workspace/dataset/ImageSets/Main/"
dataset_dir = "/home/laitathei/Desktop/workspace/dataset/"
```

### 2.4 Datasets folder structures
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
   │   ├── test
   │       ├── xxx.txt
   |       ├──   ...
   │   ├── val 
   │       ├── xxx.txt
   |       ├──   ...
   │   ├── train
   │       ├── xxx.txt
   |       ├──   ...
   │   
   ├── images
   │   ├── test
   │       ├── xxx.jpg
   |       ├──   ...
   │   ├── val 
   │       ├── xxx.jpg
   |       ├──   ...
   │   ├── train
   │       ├── xxx.jpg
   |       ├──   ...
   ├── train.txt
   ├── train_dummy.txt
   ├── test.txt
   ├── test_dummy.txt
   └── valid.txt
   └── valid_dummy.txt
   ├── split_train_val.py
   └── voc2yolo_label.py
```

### 3. Train your dataset in YOLOv5
### 3.1 Prepare your dataset yaml file, download the official weight from https://github.com/ultralytics/yolov5/releases/tag/v6.1 and place it into yolov5/weight folder
```
cd ~/Desktop/workspace/yolov5
mkdir weight
cd weight
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt
cd ~/Desktop/workspace/yolov5/data
gedit custom_dataset.yaml
```

### 3.2 custom_dataset.yaml content and change your configuration such as number of class and class names
```
train: /home/laitathei/Desktop/workspace/dataset/train.txt
val: /home/laitathei/Desktop/workspace/dataset/val.txt
test: /home/laitathei/Desktop/workspace/dataset/test.txt
test_xml: /home/laitathei/Desktop/workspace/dataset/annotations

# Classes
nc: 3  # number of classes
names: ['obstacle','human', 'injury']  # class names
```

### 3.3 Train your dataset
```
cd ~/Desktop/workspace/yolov5/
python3 train.py --weights /home/laitathei/Desktop/workspace/yolov5/weight/yolov5n.pt --cfg /home/laitathei/Desktop/workspace/yolov5/models/yolov5n.yaml --data /home/laitathei/Desktop/workspace/yolov5/data/custom_dataset.yaml --epochs 100
```

### 3.4 Train result, best.pt and last.pt are FP32 format
```
100 epochs completed in 0.091 hours.
Optimizer stripped from runs/train/exp/weights/last.pt, 3.9MB
Optimizer stripped from runs/train/exp/weights/best.pt, 3.9MB

Validating runs/train/exp/weights/best.pt...
Fusing layers... 
Model Summary: 213 layers, 1763224 parameters, 0 gradients, 4.2 GFLOPs
               Class     Images     Labels          P          R     mAP@.5 mAP@.5:.95: 100%|██████████| 2/2 [00:00<00:00,  3.06it/s]                                                                      
                 all         55        215      0.986      0.977      0.994      0.916
            obstacle         55        215      0.986      0.977      0.994      0.916
Results saved to runs/train/exp
```

### 4. Convert best.pt->best.wts->best.engine in TensorRTx
### 4.1 Convert best.pt->best.wts by ```gen_wts.py```
```
~/Desktop/workspace/tensorrtx/yolov5
```
