import os
import glob
from os.path import basename, splitext
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.cm

from sort_utils import sort_nicely


def xywh_to_xyxy(bboxes):
    """ Convert bboxes of [x_min y_min w h] into [x_min y_min x_max y_max]
        Params:
            bboxes: ndarray of shape (#bbox, 4), bbox format: [x_min y_min w h]
        Return:
            bboxes: ndarray of shape (#bbox, 4), bbox format: [x_min y_min x_max y_max]
    """
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
    

def store_output(result_dir, video_dir):
    """ Store tracking results to video
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

        store_path = 'output_video/' + video_name + '_pretty.avi'
        print("Store path:", store_path)
        
        video_writer = cv2.VideoWriter(store_path, cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))
        video_folder = os.path.join(video_dir, video_name)

        image_paths = sorted(glob.glob(os.path.join(video_folder, '*.jpg')))
        sort_nicely(image_paths)

        trk_result = np.loadtxt(r_path, delimiter=',')
        trk_result[:, 2:6] = xywh_to_xyxy(trk_result[:, 2:6])

        print("Stroing " + video_name)

        for i, image_path in enumerate(tqdm(image_paths)):
            image = cv2.imread(image_path)

            index_list = np.argwhere(trk_result[:, 0] == (i+1))
            if index_list.shape[0] != 0:
                for index in index_list[:, 0]:
                    target_index = trk_result[index,1]
                    color = colours[int(target_index-1)%20]
                    cv2.rectangle(image, 
                        (int(trk_result[index, 2]),int(trk_result[index, 3])), 
                        (int(trk_result[index, 4]),int(trk_result[index, 5])), 
                        color, 3)

                    if target_index//10 > 0:
                        digit = 2
                    else:
                        digit = 1

                    # Tag rectangle (filled rect)
                    cv2.rectangle(image, 
                        (int(trk_result[index, 2])-2,int(trk_result[index, 3]-20)), 
                        (int(trk_result[index, 2])+10*digit+10 ,int(trk_result[index, 3])+1), 
                        color, -1)

                    # Index tag
                    cv2.putText(image, 
                        str(int(trk_result[index, 1])), 
                        (int(trk_result[index, 2]+2), int(trk_result[index, 3])-3), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0,0,0), 2)

                    # cv2.imshow('g',image)
                    # cv2.waitKey(0)
                    # exit()

            video_writer.write(image)
            # dir_path = 'output_video/person22/detected/'
            # frame_name = "{0:0=3d}".format(i) + ".jpg"
            # cv2.imwrite(dir_path + frame_name, image)


if __name__ == '__main__':
    # result_dir = "/home/peng/darknet/det_mot/"
    result_dir = "/home/peng/darknet/sort/reid_output/"
    video_dir = "/home/peng/data/sort_data/images/"
    store_output(result_dir, video_dir)
