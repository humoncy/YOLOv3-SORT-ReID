from __future__ import print_function
import os
import cv2
import glob
import math
import numpy as np
import matplotlib
import time
from tqdm import tqdm
from os.path import basename, splitext

from sort import Sort
# from reid_sort import Sort
from bbox_utils import *
from sort_utils import sort_nicely
from utils import BoundBox
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), 'python')))
import darknet as dn
import sys
sys.path.append('/home/peng/Documents/scncd')
import scncd


def results2dets(results, image_shape):
    seq_dets = []

    for r in results:
        # Enough confidence of person
        if r[0] == 'person' and r[1] > 0.87:
            box = BoundBox(r[2][0], r[2][1], r[2][2], r[2][3], r[1], r[0])
            x1 = (box.x - box.w/2)
            y1 = (box.y - box.h/2)
            x2 = (box.x + box.w/2)
            y2 = (box.y + box.h/2)
            seq_dets.append([x1, y1, x2, y2, box.c])

    return np.array(seq_dets)


def online_tracking(data_dir, STORE=False):
    if not os.path.exists(data_dir):
        raise IOError("Invalid data path:", data_dir)

    dn.set_gpu(0)
    # net = dn.load_net("../cfg/yolov3.cfg", "../yolov3.weights", 0)
    net = dn.load_net("../cfg/yolov3.cfg", "../yolov3-aerial.weights", 0)
    meta = dn.load_meta("../cfg/voc.data")
    
    cmap = matplotlib.cm.get_cmap('tab20')
    ci = np.linspace(0,1,20)
    colours = cmap(ci)[:,:3]
    colours = colours[:,::-1] * 255

    scn = scncd.SCNCD()

    frame_width = 1280
    frame_height = 720

    if STORE:
        video_name = data_dir.split('/')[-2]
        FPS = 30
        # remember to modify frame width and height before testing video
        video_writer = cv2.VideoWriter('mot_video/' + video_name + '.avi', cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))

    image_paths = sorted(glob.glob(os.path.join(data_dir, '*jpg')))
    sort_nicely(image_paths)

    ####################################################################################################3
    mot_tracker = Sort(max_age=1, min_hits=3) # create instance of the SORT tracker
    # mot_tracker = Sort(max_age=150, min_hits=3) # create instance of the SORT tracker
    ####################################################################################################3

    total_time = 0.0
    total_det_time = 0.0
    total_sort_time = 0.0

    for i, image_path in enumerate(tqdm(image_paths)):
        image = cv2.imread(image_path)
        
        track_start_time = time.time()

        results = dn.detect(net, meta, image_paths[i])

        detect_time = time.time() - track_start_time
        total_det_time += detect_time
        
        sort_start_time = time.time()

        dets = results2dets(results, image.shape)
        # feats = np.zeros((dets.shape[0], 16))
        # for det_id in range(len(feats)):
        #     x1 = int(dets[det_id, 0])
        #     y1 = int(dets[det_id, 1])
        #     x2 = int(dets[det_id, 2])
        #     y2 = int(dets[det_id, 3])
        #     feat = scn.compute(image[y1:y2, x1:x2])
        #     feats[det_id] = feat
        # trackers = mot_tracker.update(dets, feats)
        trackers = mot_tracker.update(dets)

        end_time = time.time()
        cycle_time = end_time - track_start_time
        total_time += cycle_time
        sort_time = end_time - sort_start_time
        total_sort_time += sort_time

        # print("#{} frame".format(i))
        # print(results)

        for d in trackers:
            color = colours[int(d[4]-1)%20]
            cv2.rectangle(image, 
                         (int(d[0]), int(d[1])), 
                         (int(d[2]), int(d[3])),
                         color, 3)
            # cv2.putText(image, 
            #             'id = ' + str(int(d[4])), 
            #             (int(d[0]), int(d[1]) - 13), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 
            #             1e-3 * image.shape[0], 
            #             color, 2)

            cv2.rectangle(image, 
                        (int(d[0])-2,int(d[1]-20)), 
                        (int(d[0])+20+int(d[4])//10 ,int(d[1])+1), 
                        color, -1)

            cv2.putText(image, 
                        str(int(d[4])), 
                        (int(d[0]+2), int(d[1])-3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0,0,0), 2)

        #########################################################
        # FPS information
        cv2.putText(image, 
                    '   Tracking FPS = {:.2f}'.format(1 / cycle_time),
                    (frame_width - 350, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,250,0), 2)

        cv2.putText(image, 
                    '      YOLO FPS = {:.2f}'.format(1 / detect_time),
                    (frame_width - 350, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,250,0), 2)

        cv2.putText(image, 
                    'SORT(Reid) FPS = {:.2f}'.format(1 / sort_time),
                    (frame_width - 350, 75), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1e-3 * image.shape[0], 
                    (0,250,0), 2)

        #########################################################

        if STORE:
            video_writer.write(image)
        else:
            cv2.imshow("output", image)
            cv2.waitKey(0)

        if i > 600:
            break

    total_frames = i + 1
    print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time, total_frames, total_frames/total_time))    
    print("Total Detection took: %.3f for %d frames or %.1f FPS"%(total_det_time, total_frames, total_frames/total_det_time))    
    print("Total SORT took: %.3f for %d frames or %.1f FPS"%(total_sort_time, total_frames, total_frames/total_sort_time))    


if __name__ == '__main__':
    data_dir = '/home/peng/data/sort_data/images/person14_1/'
    # data_dir = '/home/peng/data/UAV123/data_seq/UAV123/group1/'
    online_tracking(data_dir, STORE=True)

