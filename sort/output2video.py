import os
import glob
from os.path import basename, splitext
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.cm

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
    

def store_output(result_dir, video_dir):
    cmap = matplotlib.cm.get_cmap('tab20')
    ci = np.linspace(0,1,20)
    colours = cmap(ci)[:,:3]
    colours = colours[:,::-1] * 255
    # print(colours)
    # exit()
    
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
        
        video_writer = cv2.VideoWriter('reid_output_video/' + video_name + '_.avi', cv2.VideoWriter_fourcc('M','J','P','G'), FPS, (frame_width, frame_height))
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
                    color = colours[int(trk_result[index,1]-1)%20]
                    cv2.rectangle(image, 
                        (int(trk_result[index, 2]),int(trk_result[index, 3])), 
                        (int(trk_result[index, 4]),int(trk_result[index, 5])), 
                        color, 3)

                    # Tag rectangle (filled rect)
                    cv2.rectangle(image, 
                        (int(trk_result[index, 2])-2,int(trk_result[index, 3]-20)), 
                        (int(trk_result[index, 2])+20+int(trk_result[index,1])//10 ,int(trk_result[index, 3])+1), 
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


if __name__ == '__main__':
    result_dir = "/home/peng/darknet/sort/reid_output/"
    video_dir = "/home/peng/data/sort_data/images/"
    store_output(result_dir, video_dir)