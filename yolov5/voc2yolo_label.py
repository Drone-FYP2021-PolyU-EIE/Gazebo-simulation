# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import os
from os import getcwd
import shutil

sets = ['train', 'val', 'test']
classes = ["obstacle", "human", "injury"]   # 改成自己的类别
image_dir = "/home/laitathei/Desktop/workspace/dataset/images/"
labels_dir = "/home/laitathei/Desktop/workspace/dataset/labels/"
annotations_dir = "/home/laitathei/Desktop/workspace/dataset/annotations/"
ImageSets_Main_dir = "/home/laitathei/Desktop/workspace/dataset/ImageSets/Main/"
dataset_dir = "/home/laitathei/Desktop/workspace/dataset/"

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h
 
def convert_annotation_train(image_id):
    in_file = open(annotations_dir+'%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(labels_dir+'train/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def convert_annotation_test(image_id):
    in_file = open(annotations_dir+'%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(labels_dir+'test/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def convert_annotation_val(image_id):
    in_file = open(annotations_dir+'%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(labels_dir+'val/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        # difficult = obj.find('difficult').text
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

images_train_dir = os.path.join(image_dir, "train/")
if not os.path.isdir(images_train_dir):                                                 # create image train folder if not exist
        os.mkdir(images_train_dir)

images_val_dir = os.path.join(image_dir, "val/")                                       # create image val folder if not exist
if not os.path.isdir(images_val_dir):
        os.mkdir(images_val_dir)


if not os.path.exists('labels_dir'):                                                    # create label folder
        os.makedirs(labels_dir)
labels_train_dir = os.path.join(labels_dir, "train/")                                   # create label train folder if not exist
if not os.path.isdir(labels_train_dir):
        os.mkdir(labels_train_dir)
labels_val_dir = os.path.join(labels_dir, "val/")                                      # create label train folder if not exist
if not os.path.isdir(labels_val_dir):
        os.mkdir(labels_val_dir)

for image_set in sets:
    print(image_set)
    image_ids = open(ImageSets_Main_dir+'%s.txt' % (image_set)).read().strip().split()
    list_file = open(dataset_dir+'%s.txt' % (image_set), 'w')
    for image_id in image_ids:
        if image_set=="train":
            list_file.write(image_dir+"train/"+'%s.jpg\n' % (image_id))     # move the image into image train folder
            convert_annotation_train(image_id)
        if image_set=="val":
            list_file.write(image_dir+"val/"+'%s.jpg\n' % (image_id))       # move the image into image val folder
            convert_annotation_val(image_id)
        if image_set=="test":
            list_file.write(image_dir+"test/"+'%s.jpg\n' % (image_id))      # move the image into image test folder
            convert_annotation_test(image_id)
    list_file.close()
