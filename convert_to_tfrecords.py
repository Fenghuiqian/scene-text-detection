#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import os
import cv2
from dataset_utils import int64_feature, float_feature, bytes_feature, convert_to_example

IMG_PATH = "./image_train/"
TXT_PATH = "./txt_train/"
OUT_PATH = "./tfr_train/"

# 训练集图片名称有多重重复后缀，重命名
new_name = 0
for f in os.listdir(PATH_IMAGE):
    os.rename(os.path.join(PATH_IMAGE, f), os.path.join(PATH_IMAGE, str(new_name) + ".jpg"))
    os.rename(os.path.join(PATH_TXT, f[:-4]+".txt"), os.path.join(PATH_TXT, str(new_name) + ".txt"))
    new_name += 1


# 转换成tfrecords
def load_file(file_path, ext_name=["jpg"]):
    files_paths = []
    for root, dirs, files in os.walk(file_path):
        for tmp_file in files:
            if tmp_file.split(".")[-1] in ext_name:
                files_paths.append(os.path.join(root, tmp_file))
    return files_paths

def read_img(path):
    img=cv2.imread(path)
    if img is None:
        return None,None,None
    return img, float(img.shape[0]), float(img.shape[1])
    
def convert_to_tfrecords(file_path,text_path,out_path):

    output_dir = os.path.dirname(out_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_paths=load_file(file_path)
    print("the total img nums is:{}".format(len(img_paths)))
    with tf.python_io.TFRecordWriter(out_path + "train_tfrecords") as tfrecord_writer:
        for idx, path in enumerate(img_paths):
            with tf.gfile.FastGFile(path, 'rb') as f:
                image_data = f.read()
            # read img
            img, h, w = read_img(path)
            if img is None:
                print("the current img is empty")
                continue
            # read txt info
            oriented_bboxes=[]
            bboxes=[]
            labels=[]
            labels_text=[]
            ignored=[]
            img_name=str(path.split("/")[-1][:-4])
            txt_path=os.path.join(text_path,img_name+".txt")
            with open(txt_path,"r", encoding="utf-8") as f:
                lines=f.readlines()
            for line in lines:
                line_array=np.array(line.strip().split(","))[:8]
                line_array = line_array.astype(float)
                oriented_box=line_array/([w,h]*4)
                oriented_bboxes.append(oriented_box)
                xy_list = np.reshape(oriented_box, (4, 2))
                xmin = xy_list[:,0].min()
                xmax = xy_list[:,0].max()
                ymin = xy_list[:,1].min()
                ymax = xy_list[:,1].max()
                bboxes.append([max(0.,xmin), max(0.,ymin), min(xmax,1.), min(ymax,1.)])
                ignored.append(0)
                labels_text.append(b"text")
                labels.append(1)
            img_name=str.encode(str(img_name))
            example = convert_to_example(image_data, img_name, labels, ignored, labels_text, bboxes,
                                                 oriented_bboxes, img.shape)
            tfrecord_writer.write(example.SerializeToString())
            if idx%100 == 0:
                print(idx)


if __name__ == '__main__':
    convert_to_tfrecords(IMG_PATH, TXT_PATH, OUT_PATH)