import json
import os
import time
import numpy as np

from mmdet.apis import init_detector, inference_detector, inference_detector_batch


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = np.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = np.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = np.clip(rb - lt + 1, 0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = np.clip(rb - lt + 1, 0, np.max(rb - lt + 1))  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


def merge_results(result1, result2, mode='inter'):
    if not isinstance(result1, np.ndarray):
        result1 = np.array(result1)
    if not isinstance(result2, np.ndarray):
        result2 = np.array(result2)
    if mode == 'inter':
        ious = bbox_overlaps(result1, result2)  # n, k
        # print("ious", ious)
        max_iou = np.max(ious, axis=1)
        # print("max ious", max_iou)
        picks = np.where(max_iou > 0.5)
        print("picks", picks[0])
        return picks[0]


def merge_result(predict1, predict2, file_path):
    image_name = os.path.basename(file_path)
    result = []
    for i, (bboxes, bboxes2) in enumerate(zip(predict1, predict2), 1):
        bboxes = np.reshape(bboxes, (-1, 5))
        bboxes2 = np.reshape(bboxes2, (-1, 5))
        if len(bboxes) > 0 and len(bboxes2) > 0:
            defect_label = i
            if i == 1 or i == 4:
                pick = merge_results(bboxes[:, :4], bboxes2[:, :4]).tolist()
                # pick2 = np.where(bboxes[:, 4] > 0.2)[0].tolist()
                # pick = list(set(pick + pick2))
                bboxes = bboxes[pick]
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
