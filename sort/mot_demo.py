import os
import cv2
import glob
import math
import numpy as np
import matplotlib
from os.path import basename, splitext

from bbox_utils import *
from sort_utils import sort_nicely


def mot(data_dir, sort_result_path, STORE=False):
    if not os.path.exists(data_dir):
        raise IOError("Invalid data path:", data_dir)
    if not os.path.exists(sort_result_path):
        raise IOError("Invalid annotation path:", sort_result_path)
    
    colours = np.round(np.random.rand(32, 3) * 255)

    if STORE:
        video_name = data_dir.split('/')[-2]
        FPS = 30
        # remember to modify frame width and height before testing video
        frame_width = 1280
        frame_height = 720
        video_writer = cv2.VideoWriter(video_name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))

    image_paths = sorted(glob.glob(os.path.join(data_dir, '*jpg')))
    sort_nicely(image_paths)

    sort_result = np.loadtxt(sort_result_path, delimiter=',')
    sort_result[:, 2:6] = xywh_to_xyxy(sort_result[:, 2:6])
    
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)

        index_list = np.argwhere(sort_result[:, 0] == (i+1))
        if index_list.shape[0] != 0:
            for index in index_list[:, 0]:
                cv2.rectangle(image, 
                    (int(sort_result[index, 2]),int(sort_result[index, 3])), 
                    (int(sort_result[index, 4]),int(sort_result[index, 5])), 
                    colours[int(sort_result[index, 1])%32], 3)

        if STORE:
            video_writer.write(image)
        else:
            cv2.imshow("output", image)
            try:
                if index_list.shape[0] == 0:
                    cv2.waitKey(0)
                else:
                    cv2.waitKey(30)
            except ValueError:
                cv2.waitKey(30)

        if i > 20:
            break


if __name__ == '__main__':
    data_dir         = '/home/peng/data/sort_data/images/person23/'
    sort_result_path = '/home/peng/basic-yolo-keras/sort/output/person23.txt'
    mot(data_dir, sort_result_path, STORE=False)