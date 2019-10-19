import json
import os
import time

from mmdet.apis import init_detector, inference_detector, inference_detector_batch

count = 0


def get_result(predict, file_path):
    global count
    # score_map = {
    #     1: 0.33,
    #     2: 0,
    #     3: 0.2,
    #     4: 0.4,
    #     5: 0.7,
    #     6: 0.5,
    #     7: 0.3,
    #     8: 0.2,
    #     9: 0.1,
    #     10: 0.14,
    #     11: 0.1,
    #     12: 0.05,
    #     13: 0.15,
    #     14: 0.001,
    #     15: 0.005
    # }
    result = []
    scores = []
    defects = []
    image_name = os.path.basename(file_path)
    for i, bboxes in enumerate(predict, 1):
        if len(bboxes) > 0:
            defect_label = i
            for bbox in bboxes:
                x1, y1, x2, y2, score = bbox.tolist()
                x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                scores.append(score)
                defects.append(i)
                # if score > score_map[i]:
                result.append(
                    {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})
    return result
    if len(scores) > 0 and max(scores) > 0.05:
        if len(scores) == 1 and max(scores) < 0.1:
            count += 1
            return []
        return result
    else:
        return []


class Detector:
    def __init__(self):
        self.model = init_detector(
            '/competition/mmdetection/myconfig/101/grid_rcnn_gn_head_x101_32x4d_fpn_2x.py',
            '/competition/epoch_1.pth', device='cuda:0')

    def detect_single_img(self, file_path, template_path):
        predict = inference_detector(self.model, [file_path, template_path])
        result = get_result(predict, file_path)
        return result

    def detect_batch_imgs(self, file_paths, batch_size):
        result = []
        for j in range(0, len(file_paths), batch_size):
            paths = file_paths[j:j + batch_size]
            predicts = inference_detector_batch(self.model, paths) if len(paths) > 1 else inference_detector(self.model,
                                                                                                             paths)
            for (file_path, _), predict in zip(paths, predicts):
                result += get_result(predict, file_path)
        return result


def batch_inference():
    s = time.time()
    root = '/tcdata/guangdong1_round2_testA_20190924'
    result = []
    detector = Detector()
    paths = []
    for dir_name in os.listdir(root):
        files = os.listdir(os.path.join(root, dir_name))
        if files[0].startswith('template'):
            files = [files[1], files[0]]
        file = files[0]
        template_file = files[1]
        paths.append([os.path.join(root, dir_name, file), os.path.join(root, dir_name, template_file)])

    # res = detector.detect_single_img(os.path.join(root, dir_name, file), os.path.join(root, dir_name, template_file))
    res = detector.detect_batch_imgs(paths, 12)
    result += res

    with open('result.json', 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))
    print("time use", time.time() - s)


def single_inference():
    s = time.time()
    root = '/tcdata/guangdong1_round2_testA_20190924'
    result = []
    detector = Detector()
    for dir_name in os.listdir(root):
        files = os.listdir(os.path.join(root, dir_name))
        if files[0].startswith('template'):
            files = [files[1], files[0]]
        file = files[0]
        template_file = files[1]
        res = detector.detect_single_img(os.path.join(root, dir_name, file),
                                         os.path.join(root, dir_name, template_file))
        result += res
    with open('result.json', 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))
    print("time use", time.time() - s)


single_inference()
print("count ", count)
