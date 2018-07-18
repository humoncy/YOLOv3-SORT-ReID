import os
import glob
from os.path import basename, splitext
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.cm
from collections import deque

from sort_utils import sort_nicely


# [x_min y_min w h] to [x_min y_min x_max y_max]
def xywh_to_xyxy(bboxes):
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes)
    xyxy_bboxes = np.zeros_like(bboxes)
    for i in range(len(bboxes)):
        x = bboxes[i][0]
        y = bboxes[i][1]
        w = bboxes[i][2]
        h = bboxes[i][3]
        xyxy_bboxes[i][0] = x
        xyxy_bboxes[i][1] = y
        xyxy_bboxes[i][2] = x + w
        xyxy_bboxes[i][3] = y + h

    return xyxy_bboxes
    

# This file is based on output2video.py and online_tracking_trajectory.py, see them to find more comments
def store_output(result_dir, video_dir):
    """ Store tracking results to video (with velocity)
        Params:
            result_dir: string, path to tracking result directory
            video_dir:  string, path to video frames
        Return:
            None, video will be stored in the store_path
    """
    cmap = matplotlib.cm.get_cmap('tab20')
    ci = np.linspace(0,1,20)
    colours = cmap(ci)[:,:3]
    colours = colours[:,::-1] * 255
    # print(colours)
    # exit()

    PT_BUFFER = 64
    
    r_paths = sorted(glob.glob(os.path.join(result_dir, '*.txt')))
    sort_nicely(r_paths)

    for r_path in r_paths:
        FPS = 30
        # remember to modify frame width and height before testing video
        frame_width = 1280
        frame_height = 720
        video_name = splitext(basename(r_path))[0]


        ##############################################################################
        # Modify here to ouput only one video.
        if video_name != 'person14_1':
            continue
        ##############################################################################

        store_path = 'velocity_output_video/' + video_name + '_velocity.avi'
        print("Store path:", store_path)
        
        video_writer = cv2.VideoWriter(store_path, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))
        video_folder = os.path.join(video_dir, video_name)

        image_paths = sorted(glob.glob(os.path.join(video_folder, '*.jpg')))
        sort_nicely(image_paths)

        trk_result = np.loadtxt(r_path, delimiter=',')
        # trk_result[:, 2:6] = xywh_to_xyxy(trk_result[:, 2:6])

        print("Stroing " + video_name)

        target_pts_dict = {}
        target_pts_appear_dict = {}

        for i, image_path in enumerate(tqdm(image_paths)):
            image = cv2.imread(image_path)

            for key in target_pts_appear_dict:
                target_pts_appear_dict[key] += 1

            index_list = np.argwhere(trk_result[:, 0] == (i+1))
            if index_list.shape[0] != 0:
                for index in index_list[:, 0]:
                    target_id = trk_result[index,1]
                    color_index = int(target_id-1)%20

                    if target_pts_dict.get(target_id) is None:
                        # initialize the list of tracked points
                        target_pts_dict[target_id] = deque(maxlen=PT_BUFFER)
                        target_pts_appear_dict[target_id] = 0

                    color = colours[color_index]
                    bbox = trk_result[index, 2:6]

                    cv2.rectangle(image, 
                        (int(bbox[0]),int(bbox[1])), 
                        (int(bbox[0]+bbox[2]),int(bbox[1]+bbox[3])), 
                        color, 3)

                    x_center = bbox[0] + bbox[2]/2
                    y_center = bbox[1] + bbox[3]/2

                    if target_pts_dict.get(target_id) is not None:
                        target_pts_dict[target_id].appendleft((int(x_center), int(y_center)))
                        target_pts_appear_dict[target_id] = 0

                    pts = target_pts_dict[target_id]
                    # loop over the set of tracked points
                    for pti in range(1, len(pts)):
                        # if either of the tracked points are None, ignore them
                        if pts[pti - 1] is None or pts[pti] is None:
                            continue

                        # otherwise, compute the thickness of the line and draw the connecting lines
                        thickness = int(np.sqrt(PT_BUFFER / float(pti+1)) * 2.5)
                        cv2.line(image, pts[pti - 1], pts[pti], color, thickness)

                    velocity_x = trk_result[index, 10] * bbox[2] * 0.5 + x_center
                    velocity_y = trk_result[index, 11] * bbox[3] * 0.5 + y_center

                    cv2.arrowedLine(image, 
                        (int(x_center),int(y_center)), 
                        (int(velocity_x),int(velocity_y)), 
                        color, 2)

                    # alpha = 0.5
                    # overlay = image.copy()
                    # v_bbox = trk_result[index, -4:]
                    # cv2.rectangle(overlay, 
                    #     (int(v_bbox[0]),int(v_bbox[1])), 
                    #     (int(v_bbox[2]),int(v_bbox[3])), 
                    #     color, -1)
                    # cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

                    # Tag rectangle (filled rect)
                    cv2.rectangle(image, 
                        (int(bbox[0]-2),int(bbox[1]-20)), 
                        (int(bbox[0]+20+int(trk_result[index,1])//10) ,int(bbox[1]+1)), 
                        color, -1)

                    # Index tag
                    cv2.putText(image, 
                        str(int(trk_result[index, 1])), 
                        (int(bbox[0]+2), int(bbox[1]-3)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0,0,0), 2)

                    # cv2.imshow('g',image)
                    # cv2.waitKey(0)
                    # exit()

            for key, value in target_pts_appear_dict.items():
                if value > 3:
                    target_pts_dict.pop(key, None)
                    target_pts_appear_dict.pop(key, None)


            video_writer.write(image)

            # if i > 100:
            #     break


if __name__ == '__main__':
    result_dir = "/home/peng/darknet/sort/velocity_output/"
    # result_dir = "/home/peng/darknet/sort/kf_output/"
    video_dir = "/home/peng/data/sort_data/images/"
    store_output(result_dir, video_dir)
