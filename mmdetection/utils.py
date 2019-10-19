from mmdet.core import multiclass_nms, bbox_overlaps
import numpy as np


def merge_results(result1, result2, mode='inter'):
    if not isinstance(result1, np.ndarray):
        result1 = np.array(result1)
    if not isinstance(result2, np.ndarray):
        result2 = np.array(result2)
    if mode == 'inter':
        ious = bbox_overlaps(result1, result2)  # n, k
        max_iou = np.max(ious, axis=1)
        picks = np.where(max_iou > 0.7)
        return picks