import numpy as np
import cv2

import os.path
import sys


class BoundBox:
    def __init__(self, x, y, w, h, c = None, class_name = None):
        self.x     = x
        self.y     = y
        self.w     = w
        self.h     = h
        
        self.c     = c
        self.class_name = class_name

    def print_box(self):
        print("Class name: {}".format(self.class_name))
        print("(x_center, y_center, w, h, c): (%f, %f, %f, %f, %f)" % (self.x, self.y, self.w, self.h, self.c))