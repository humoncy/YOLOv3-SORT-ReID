# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys, os
sys.path.append(os.path.join(os.getcwd(),'python/'))

import darknet as dn
import pdb
import cv2

dn.set_gpu(0)
# net = dn.load_net("cfg/yolov3.cfg", "yolov3.weights", 0)
net = dn.load_net("cfg/yolov3.cfg", "backup/yolov3-aerial_10000.weights", 0)
meta = dn.load_meta("cfg/aerial.data")

# img_path = "/home/peng/data/sort_data/images/person14_2/001123.jpg"
img_path = "/home/peng/data/aerial/crop.png"

# And then down here you could detect a lot more images like:
result = dn.detect(net, meta, img_path)
print(result)

img = cv2.imread(img_path)

if img is None:
    raise Exception("No input image!")

for r in result:
    x, y, w, h = r[2][0], r[2][1], r[2][2], r[2][3]
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    cv2.rectangle(img, 
        (int(x1),int(y1)),
        (int(x2),int(y2)), 
        (255,0,0), 3)

cv2.imshow("output", img)
# cv2.imwrite("yolov3.jpg", img)
# cv2.imwrite("bad_detection.jpg", img)
cv2.imwrite("crop_detection.jpg", img)
cv2.waitKey(0)

# r = dn.detect(net, meta, "data/giraffe.jpg")
# print r
# r = dn.detect(net, meta, "data/horses.jpg")
# print r
# r = dn.detect(net, meta, "data/person.jpg")
# print r
