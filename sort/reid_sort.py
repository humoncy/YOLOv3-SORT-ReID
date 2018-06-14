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
import cv2
import argparse
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from sort_utils import get_data_lists, sort_nicely
import evaluation
from os.path import basename, splitext
import csv
import sys
sys.path.append('/home/peng/Documents/scncd')
import scncd


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

@jit
def cosine_distance(det_feature, trk_features):
  """ Compute the maximum cosine distance between measurement feature and tracker features
  """
  max_cosine = 0.0
  for trk_feature in trk_features:
    cosine = np.dot(det_feature, trk_feature)
    if cosine > max_cosine:
      max_cosine = cosine

  return max_cosine


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
  r = w/float(h) # aspect ratio
  return np.array([x,y,s,r]).reshape((4,1))

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
  def __init__(self, bbox, feature, trk_id=None):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    # State transition matrix
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    # Measurement function
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    # Measuerment uncertainty (dim_z, dim_z), default I 
    self.kf.R[2:,2:] *= 10.
    # Covariance matrix (dim_x, dim_x), default I
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    # Process noise (dim_x, dim_x), default I
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    # Assign the initial value of the state
    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    if trk_id is None:
      self.id = KalmanBoxTracker.count
      KalmanBoxTracker.count += 1
    else:
      self.id = trk_id
    self.history = []
    self.hits = 0 # total number of updates
    self.hit_streak = 0 # consective number of updates (probationary perirod)
    self.age = 0

    self.features = []
    if feature is not None:
      self.features.append(feature)

  def update(self,bbox,feature):
    """
    Updates the state vector with observed bbox.
    (Unnecessarily update in every frame)
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))

    self.features.append(feature)
    if len(self.features) > 100:
      self.features.pop(0)

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    (Predict in every frame)
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



def associate_detections_to_trackers(detections, features, trackers, iou_threshold=0.3):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  # Get predicted locations from existing trackers.
  trks = np.zeros((len(trackers),5))
  to_del = []
  for t,trk in enumerate(trks):
    pos = trackers[t].predict()[0]
    trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
    if(np.any(np.isnan(pos))):
      print("predict nan")
      to_del.append(t)
  trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # invalid: nan, inf
  for t in reversed(to_del):
    trackers.pop(t)
  if len(trks) != len(trackers):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

  if(len(trks)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

  iou_matrix = np.zeros((len(detections),len(trks)),dtype=np.float32)
  fcd_matrix = np.zeros((len(detections),len(trks)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trks):
      iou_matrix[d,t] = iou(det,trk)
      fcd_matrix[d,t] = cosine_distance(features[d,:], trackers[t].features)
  cost_matrix = iou_matrix + fcd_matrix  
  matched_indices = linear_assignment(-cost_matrix)
  
  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trks):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  # Filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold) and (fcd_matrix[m[0],m[1]]<0.9):
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
    self.trks_feat = {} # {trk_id: [[feat1, feat2]], ...}
    self.trks_feat_sinceused = {} # {trk_id: time_since_used, ...}
    self.frame_count = 0

  def update(self, dets, feats):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    ret = []

    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, feats, self.trackers, iou_threshold=0.3)

    # Update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0], 0]
        trk.update(dets[d,:][0], feats[d,:][0])

    # Create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
      trk = KalmanBoxTracker(dets[i,:], feats[i,:])
      self.trackers.append(trk)

    i = len(self.trackers)
    for trk in reversed(self.trackers):  # why reverse? pop later!
      d = trk.get_state()[0]
        
      i -= 1

      # Remove dead tracklet
      if (trk.time_since_update > self.max_age):
        self.trackers.pop(i)
      else:
        # Pass probationary period
        if (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        elif trk.hits >= 30 and trk.time_since_update < self.min_hits:
          ret.append(np.concatenate((d, [trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
          
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

  scncd = scncd.SCNCD()

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
  
  for vid, det in enumerate(detections):
    mot_tracker = Sort(max_age=150, min_hits=3) # create instance of the SORT tracker
    seq_dets = np.loadtxt(det, delimiter=',') # load detections
    video_name = splitext(basename(det))[0]

    # if video_name != "person14_1":
    #   continue

    with open('reid_output/' + video_name + '.txt', 'w') as out_file:
      print("Processing %s." % video_name)
      data_dir = videos[vid]
      image_paths = sorted(glob.glob(os.path.join(data_dir, '*jpg')))
      sort_nicely(image_paths)
      for frame in range(int(seq_dets[:,0].max())):
        start_time = time.time()
      
        img = cv2.imread(image_paths[frame])

        cycle_time = time.time() - start_time
        # print("imread: {:.3f}s".format(cycle_time))

        frame += 1 # detection and frame numbers begin at 1
        total_frames += 1

        # print("\n------------ {}th frmae start ---------------\n".format(frame))

        dets = seq_dets[seq_dets[:,0]==frame, 2:7]
        dets[:,2:4] += dets[:,0:2] # convert [x1,y1,w,h] to [x1,y1,x2,y2]

        fe_s = time.time()
        feats = np.zeros((dets.shape[0], 16))
        for det_id in range(len(feats)):
          x1 = max(int(dets[det_id, 0]), 0)
          y1 = max(int(dets[det_id, 1]), 0)
          x2 = min(int(dets[det_id, 2]), img.shape[1]-1)
          y2 = min(int(dets[det_id, 3]), img.shape[0]-1)
          feat = scncd.compute(img[y1:y2, x1:x2])
          feats[det_id] = feat
        cycle_time = time.time() - fe_s
        # print("Feature extraction: {:.3f}s".format(cycle_time))

        up_s = time.time()
        trackers = mot_tracker.update(dets, feats)
        cycle_time = time.time() - up_s
        # print("Update: {:.3f}s".format(cycle_time))

        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4] ,d[0] ,d[1], d[2]-d[0], d[3]-d[1]), file=out_file)

        # print("\n------------ {}th frmae done ---------------".format(frame))
        # if frame == 10:
        #   exit()
        
    outputs += [os.getcwd() + '/reid_output/' + video_name + '.txt']
    KalmanBoxTracker.count = 0

  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
  
  if evaluate:
    print()
    print("========================= Tracking by YOLO+SORT ===============================")
    total_avg_iou, total_lost = evaluation.average_IOU(annots, outputs, "tracker")
    print("Total average IOU = {}".format(total_avg_iou))
    print("Total lost track  = {}".format(total_lost))

    print("========================= Tracking by YOLO only ===============================")
    total_avg_iou, total_lost = evaluation.average_IOU(annots, detections, "detector")
    print("Total average IOU = {}".format(total_avg_iou))
    print("Total lost track  = {}".format(total_lost))

    if os.path.exists("detector.csv") and os.path.exists("tracker.csv"):
      d_file = open("detector.csv")
      t_file = open("tracker.csv")

      data = []
      data.append(["Video", "Avg IOU(D)", "Avg IOU(T)", "#Frame", "#Lost(D)", "#Lost(T)"])
      for d, t in zip(csv.DictReader(d_file), csv.DictReader(t_file)):
        data.append([d["video"], d["avg_iou"], t["avg_iou"], d["#frame"], d["#lost"], t["#lost"]])

      d_file.close()
      t_file.close()

      with open("ouput.csv", "w") as f:
        w = csv.writer(f)
        w.writerows(data)
    
    evaluation.success_plot_auc(ann=annots, tra=outputs, det=detections)
