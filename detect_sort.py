#! /usr/bin/env python
from __future__ import print_function
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from os.path import basename, splitext
import glob
import re

import sys
sys.path.append(os.path.join(os.getcwd(),'sort/'))
from utils import BoundBox
from sort_utils import sort_nicely
sys.path.append(os.path.join(os.getcwd(),'python/'))
import darknet as dn
import pdb


argparser = argparse.ArgumentParser(
    description='Detect videos and store in MOT format')

argparser.add_argument(
    '-c',
    '--conf',
    default='config_aerial.json',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    default='yolo_coco_aerial_person.h5',
    help='path to pretrained weights')


def _main_(args):

    ###############################
    #   Prepare data to be detected
    ###############################

    # data_folder = "/home/peng/data/good_rolo_data/"
    data_folder = "/home/peng/data/sort_data/images/"
    # data_folder = "/home/peng/data/sort_data/images/"
    video_folders_list = sorted(glob.glob(data_folder + '*'))
    sort_nicely(video_folders_list)
 
    ###############################
    #   Make the model and Load trained weights
    ###############################

    dn.set_gpu(1)
    # net = dn.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
    net = dn.load_net("cfg/yolov3.cfg", "yolov3-aerial.weights", 0)
    meta = dn.load_meta("cfg/voc.data")
    ###############################
    #   Predict bounding boxes 
    ###############################

    for video_folder in video_folders_list:
        video_name = basename(video_folder)

        #if video_name != "person14_3":
        #    continue

        print("Processing %s." % video_name)
        image_paths = sorted(glob.glob(os.path.join(video_folder, '*jpg')))
        sort_nicely(image_paths)

        with open('det_mot_thr45/' + video_name + '.txt', 'w') as out_file:
            for i in tqdm(range(len(image_paths))):
                # image = cv2.imread(image_paths[i])
                results = dn.detect(net, meta, image_paths[i], thresh=0.45, nms=0.5)

                for r in results:
                    if r[0] == 'person' and r[1] > 0.88:
                        box = BoundBox(r[2][0], r[2][1], r[2][2], r[2][3], r[1], r[0])
                        x1 = (box.x - box.w/2)
                        y1 = (box.y - box.h/2)
                        print('%d,-1,%.2f,%.2f,%.2f,%.2f,%.6f,-1,-1,-1' % (i+1, x1, y1, box.w, box.h, box.c), file=out_file)
        
                    

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
