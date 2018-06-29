# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import cv2
import time
import numpy as np


dn.set_gpu(1)
# net = dn.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
net = dn.load_net("cfg/yolov3.cfg", "yolov3-aerial.weights", 0)
# net = dn.load_net("cfg/yolov3.cfg", "backup/yolov3-aerial_final.weights", 0)
meta = dn.load_meta("cfg/aerial.data")

# img_path = "data/original_.jpg"
# # And then down here you could detect a lot more images like:
# result = dn.detect(net, meta, img_path)
# img = cv2.imread(img_path)

# if img is None:
#     raise Exception("No input image!")
# for r in result:
#     if r[0] == 'person' and r[1] > 0.95:
#         x, y, w, h = r[2][0], r[2][1], r[2][2], r[2][3]
#         x1 = int(x - w / 2)
#         y1 = int(y - h / 2)
#         x2 = int(x + w / 2)
#         y2 = int(y + h / 2)

#         cv2.rectangle(img, 
#             (int(x1),int(y1)),
#             (int(x2),int(y2)), 
#             (255,0,0), 3)
# cv2.imshow("output", img)
# # cv2.imwrite("100000.jpg", img)
# cv2.waitKey(0)


img_paths = ["8right.jpg", "10left.jpg", "12left+mid.jpg", "18left.jpg"]
for img_path in img_paths:
    img_path = 'data/thesis/' + img_path
    print("Processing:", img_path)
    img = cv2.imread(img_path)
    start_time = time.time()
    result = dn.detect(net, meta, img_path, thresh=.45)
    clock_time = time.time()
    print("Processing time:", clock_time - start_time)
    print("Result", result)

    if img is None:
        raise Exception("No input image!")
    for r in result:
        x, y, w, h = r[2][0], r[2][1], r[2][2], r[2][3]
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(img, 
            (int(x1),int(y1)),
            (int(x2),int(y2)), 
            (255,0,0), 3)
    cv2.imshow("output", img)
    # cv2.imwrite(img_path[:-4] + "_v3.jpg", img)
    cv2.waitKey(0)



# r = dn.detect(net, meta, "data/giraffe.jpg")
# print r
# r = dn.detect(net, meta, "data/horses.jpg")
# print r
# r = dn.detect(net, meta, "data/person.jpg")
# print r
