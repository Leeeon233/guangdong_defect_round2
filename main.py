import os
import json
from detection_single_img import Detector
import numpy as np
from mmdetection.utils import merge_results

def non_max_suppression_fast(boxes, scores, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the score of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        # 计算重叠区域的左上与右下坐标
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        # 计算重叠区域的长宽
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        # 计算重叠区域占原区域的面积比（重叠率）
        overlap = (w * h) / area[idxs[:last]]

        # 删除所有重叠率大于阈值的边界框
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


root = '/tcdata/guangdong1_round2_testA_20190924'
result = []
detector = Detector()
for dir_name in os.listdir(root):
    for file in os.listdir(os.path.join(root, dir_name)):
        if not file.startswith('template'):
            res = detector.detect_single_img(os.path.join(root, dir_name, file))
            result += res

with open('result.json', 'w') as fp:
    json.dump(result, fp, indent=4, separators=(',', ': '))
