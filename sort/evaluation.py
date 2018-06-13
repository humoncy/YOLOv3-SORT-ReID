from os.path import basename, splitext
import numpy as np
import glob
import math
import matplotlib.pyplot as plt
import csv


def isNAN(bbox):
    for value in bbox.flatten():
        if math.isnan(value):
            return True

def interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3  

def bbox_iou(box1, box2):
    """ Compute IOU between two bboxes in the form [x1,y1,w,h]
    """
    x1_min = box1[0]
    x1_max = x1_min + box1[2]
    y1_min = box1[1]
    y1_max = y1_min + box1[3]

    x2_min = box2[0]
    x2_max = x2_min + box2[2]
    y2_min = box2[1]
    y2_max = y2_min + box2[3]

    intersect_w = interval_overlap([x1_min, x1_max], [x2_min, x2_max])
    intersect_h = interval_overlap([y1_min, y1_max], [y2_min, y2_max])
    
    intersect = intersect_w * intersect_h
    
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersect
    
    return float(intersect) / union

def average_IOU(annots, outputs, file_name):
    nb_video = len(annots)
    total_frames = 0
    total_avg_iou = 0.0
    total_lost = 0.0
    data = []
    data.append(["video", "avg_iou", "#frame", "#lost"])
    
    for i, annot in enumerate(annots):
        if basename(annot)[:-4] != basename(outputs[i])[:-4]:
            print("Annotations:", annot)
            print("Output:", outputs)
            raise ValueError("Wrong annotation and track correspondence.")

        print("Evaluating %s." % basename(annot))

        labels = np.loadtxt(annot, delimiter=',')
        mot_results = np.loadtxt(outputs[i], delimiter=',')

        nb_frame = len(labels)
        total_frames += nb_frame

        avg_iou = 0.0
        num_lost = 0.0
        for fi, label in enumerate(labels):
            if fi == 0:
                # Tracking start
                continue

            if isNAN(labels[fi]) is True:
                # No target in the frame
                continue

            index_list = np.argwhere(mot_results[:, 0] == (fi+1))

            if index_list.shape[0] != 0:
                max_iou = 0.0
                for index in index_list[:, 0]:
                    bbox = mot_results[index, 2:6]
                    iou = bbox_iou(bbox, label)
                    if iou > max_iou:
                        max_iou = iou
                avg_iou += max_iou

                if max_iou == 0.0:
                    num_lost += 1
            else:
                # print("Lost frame:", fi+1)
                num_lost += 1

        print("\tLost target = {} / {}".format(int(num_lost), nb_frame - 1))

        avg_iou /= (nb_frame - 1)

        print("\tAverage IOU = {:.2f}%".format(avg_iou * 100))
        print("\tNumber of frames = {}".format(nb_frame))

        data.append([splitext(basename(annot))[0], "{:.2f}%".format(avg_iou * 100), nb_frame, int(num_lost)])

        total_avg_iou += avg_iou
        total_lost += num_lost

        print("==================================================")
    
    with open(file_name +'.csv', "w") as f:
        w = csv.writer(f)
        w.writerows(data)
    
    total_avg_iou /= nb_video

    print("Total frames: {}".format(total_frames))
    print("==================================================")
    
    return total_avg_iou, total_lost

def overlap_precision(annots, outputs, threshold):
    nb_video = len(annots)
    total_precision = 0.0
    for i, annot in enumerate(annots):
        if basename(annot)[:-4] != basename(outputs[i])[:-4]:
            raise ValueError("Wrong annotation and track correspondence.")

        labels = np.loadtxt(annot, delimiter=',')
        mot_results = np.loadtxt(outputs[i], delimiter=',')

        nb_frame = len(labels)

        precision = 0.0

        for fi, label in enumerate(labels):
            if fi == 0:
                # Tracking start
                continue

            if isNAN(labels[fi]) is True:
                # No target in the frame
                continue

            index_list = np.argwhere(mot_results[:, 0] == (fi+1))

            if index_list.shape[0] != 0:
                max_iou = 0.0
                for index in index_list[:, 0]:
                    bbox = mot_results[index, 2:6]
                    iou = bbox_iou(bbox, label)
                    if iou > max_iou:
                        max_iou = iou
                if max_iou > threshold:
                    precision += 1

        precision /= (nb_frame - 1)

        total_precision += precision
    
    total_precision /= nb_video
    
    return total_precision


def success_plot_auc(ann, tra, det=None):
    fig = plt.figure("Success plot")
    t = np.linspace(0.0, 1.0, 30)
    s = np.zeros_like(t)
    for i, threshold in enumerate(t):
        s[i] = overlap_precision(ann, tra, threshold)
    plt.plot(t, s)
    auc_score = np.mean(s)

    if det is not None:
        det_s = np.zeros_like(t)
        for i, threshold in enumerate(t):
            det_s[i] = overlap_precision(ann, det, threshold)
        plt.plot(t, det_s)
        det_auc_score = np.mean(det_s)
        plt.legend(('YOLO+SORT [{:.3f}]'.format(auc_score), 'YOLO [{:.3f}]'.format(det_auc_score)), loc='upper right')
    else:
        plt.legend(('YOLO+SORT [{:.3f}]'.format(auc_score), ''), loc='upper right')

    plt.xlabel('Overlap threshold (IoU)')
    plt.ylabel('Success rate')
    plt.title('Success plots of OPE on UAV123 - Person only')
    plt.grid(True)

    plt.savefig('YOLO_SORT.png')
    plt.show()

