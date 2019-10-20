import json
import os
import time

import numpy as np
from mmdet.apis import init_detector, inference_detector, inference_detector_batch


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

def merge_result(predict1, predict2, file_path):
    image_name = os.path.basename(file_path)
    result = []
    for i, (bboxes, bboxes2) in enumerate(zip(predict1, predict2), 1):
        bboxes = np.reshape(bboxes, (-1, 5))
        bboxes2 = np.reshape(bboxes2, (-1, 5))
        if len(bboxes) > 0 and len(bboxes2) > 0:
            defect_label = i

            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox.tolist()
                x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                result.append(
                    {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})
    return result

class MultiDetector:
    def __init__(self):
        self.model = init_detector(
            '/competition/mmdetection/myconfig/101/grid_rcnn_gn_head_x101_32x4d_fpn_2x.py',
            '/competition/epoch_1.pth', device='cuda:0')
        self.batch_model = init_detector(
            '/competition/mmdetection/myconfig/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_round2_aug_blur.py',
            '/competition/epoch_2.pth', device='cuda:0')

    def detect_batch_imgs(self, file_paths, batch_size):
        result = []
        for j in range(0, len(file_paths), batch_size):
            paths = file_paths[j:j + batch_size]
            predicts = inference_detector_batch(self.batch_model, paths) if len(paths) > 1 \
                else inference_detector(self.batch_model, paths[0])
            for (file_path, template_path), predict in zip(paths, predicts):
                pred2 = inference_detector(self.model, [file_path, template_path])
                result += merge_result(predict, pred2, file_path)
        return result


if __name__ == '__main__':
    s = time.time()
    root = '/tcdata/guangdong1_round2_testA_20190924'
    result = []
    detector = MultiDetector()
    paths = []
    for dir_name in os.listdir(root):
        files = os.listdir(os.path.join(root, dir_name))
        if files[0].startswith('template'):
            files = [files[1], files[0]]
        file = files[0]
        template_file = files[1]
        paths.append([os.path.join(root, dir_name, file), os.path.join(root, dir_name, template_file)])

    # res = detector.detect_single_img(os.path.join(root, dir_name, file), os.path.join(root, dir_name, template_file))
    res = detector.detect_batch_imgs(paths, 10)
    result += res

    with open('result.json', 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))
    print("time use", time.time() - s)

