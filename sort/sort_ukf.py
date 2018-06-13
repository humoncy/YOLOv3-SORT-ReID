"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016 Alex Bewley alex@dynamicdetection.com

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

from numba import jit
import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
from sort_utils import get_data_lists
import evaluation
from os.path import basename, splitext
import csv


@jit
def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r])

def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0 # How many object been tracked
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    def fx(x, dt):
      # state transition function - predict next state based on constant velocity model x = x_0 + vt
      F = np.array([[1, 0, 0, 0, dt, 0, 0],
                    [0, 1, 0, 0, 0, dt, 0],
                    [0, 0, 1, 0, 0, 0, dt],
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 1]], dtype=float)
      
      return np.dot(F, x)

    def hx(x):
      # measurement function - convert state into a measurement where measurements are [u, v, s, r]
      return np.array([x[0],x[1],x[2],x[3]])

    dt = 0.1
    # create sigma points to use in the filter. This is standard for Gaussian processes
    points = MerweScaledSigmaPoints(7, alpha=.1, beta=2., kappa=-1)

    #define constant velocity model
    self.kf = UnscentedKalmanFilter(dim_x=7, dim_z=4, dt=dt, fx=fx, hx=hx, points=points)

    # Assign the initial value of the state
    self.kf.x[:4] = convert_bbox_to_z(bbox)
    # Covariance matrix (dim_x, dim_x), default I
    self.kf.P[4:,4:] *= 1000. # Give high uncertainty to the unobservable initial velocities
    self.kf.P *= 0.2
    # Measuerment uncertainty (dim_z, dim_z), default I 
    self.kf.R *= 0.01
    # Process noise (dim_x, dim_x), default I
    self.kf.Q[-1,-1] *= 0.0001
    self.kf.Q[4:,4:] *= 0.0001

    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)



def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix)
  
  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  # Filter out matched with low IOU
  matches = []
  for m in matched_indices:
    # print(iou_matrix[m[0],m[1]])
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)



class Sort(object):
  def __init__(self, max_age=1, min_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits  # Probationary period
    self.trackers = []
    self.frame_count = 0

  def update(self, dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # Get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)

    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      # print("------------------------- 1")
      self.trackers.pop(t)

    if self.frame_count < 10:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, iou_threshold=0.0)
    else:
      matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, iou_threshold=0.2)

    # Update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0], 0]
        trk.update(dets[d,:][0])

    # Create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
        # print("~~~~~~~~~~~~~~~~~~", trk.id+1)

    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if ( (trk.time_since_update < self.max_age)  and  (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits) ):
          ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1

        # Remove dead tracklet
        if (trk.time_since_update > self.max_age):
          # print("------------------------- 2")
          self.trackers.pop(i)
    
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
    


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--evaluate', dest='evaluate', help='Evaluate the tracker output',action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
  args = parse_args()
  evaluate = args.evaluate

  data = {
    'sort': {
      'image_folder': '/home/peng/data/sort_data/images/',
      'annot_folder': '/home/peng/data/sort_data/annotations/',
      'mot_det_folder': '/home/peng/darknet/det_mot/'
    }
  }
  annots, videos, detections = get_data_lists(data['sort'])

  outputs = []

  total_time = 0.0
  total_frames = 0
  
  if not os.path.exists('output'):
    os.makedirs('output')
  
  for det in detections:
    mot_tracker = Sort(max_age=10, min_hits=0) # create instance of the SORT tracker
    seq_dets = np.loadtxt(det, delimiter=',') # load detections
    video_name = splitext(basename(det))[0]

    # if video_name != "person19_3":
      # continue

    with open('output/' + video_name + '.txt', 'w') as out_file:
      print("Processing %s." % video_name)
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 # detection and frame numbers begin at 1

        # print("\n------------ {}th frmae start ---------------\n".format(frame))

        dets = seq_dets[seq_dets[:,0]==frame, 2:7]  # seq_dets format: [x1,y1,x2,y2,score]

        dets[:,2:4] += dets[:,0:2] # convert [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        # if len(trackers) == 0 and len(dets) == 0:
        #   print("!!!!!!!!!!!!!!! 1")
        #   print(frame, trackers)
        #   print(dets)

        # if len(trackers) == 0 and len(dets) > 0:
        #   print("!!!!!!!!!!!!!!! 2")
        #   print(frame, trackers)
        #   print(dets)

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4] ,d[0] ,d[1], d[2]-d[0], d[3]-d[1]), file=out_file)

        # print("\n------------ {}th frmae done ---------------".format(frame))
        # if frame == 100 and video_name == "person4_1":
        #   exit()
        
    outputs += [os.getcwd() + '/output/' + video_name + '.txt']
    KalmanBoxTracker.count = 0

  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
  
  if evaluate:
    print()
    print("========================= Tracking by YOLO+SORT ===============================")
    total_avg_iou, total_lost = evaluation.average_IOU(annots, outputs, "tracker")
    print("Total average IOU = {}".format(total_avg_iou))
    print("Total lost track  = {}".format(total_lost))

    # print("========================= Tracking by YOLO only ===============================")
    # total_avg_iou, total_lost = evaluation.average_IOU(annots, detections, "detector")
    # print("Total average IOU = {}".format(total_avg_iou))
    # print("Total lost track  = {}".format(total_lost))

    # if os.path.exists("detector.csv") and os.path.exists("tracker.csv"):
    #   d_file = open("detector.csv")
    #   t_file = open("tracker.csv")

    #   data = []
    #   data.append(["Video", "Avg IOU(D)", "Avg IOU(T)", "#Frame", "#Lost(D)", "#Lost(T)"])
    #   for d, t in zip(csv.DictReader(d_file), csv.DictReader(t_file)):
    #     data.append([d["video"], d["avg_iou"], t["avg_iou"], d["#frame"], d["#lost"], t["#lost"]])

    #   d_file.close()
    #   t_file.close()

    #   with open("ouput.csv", "w") as f:
    #     w = csv.writer(f)
    #     w.writerows(data)
    
    # evaluation.success_plot_auc(ann=annots, tra=outputs, det=detections)
