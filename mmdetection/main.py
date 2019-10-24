import json
import os
import time

from mmdet.apis import init_detector, inference_detector, inference_detector_batch


def get_result(predict, file_path):
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
    if len(scores) > 0 and max(scores) > 0.03:
        # if len(scores) == 1 and max(scores) < 0.1:
        #     return []
        return result
    else:
        return []


class Detector:
    def __init__(self):
        self.model = init_detector(
            # '/competition/mmdetection/myconfig/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_libra_blur.py',
            '/competition/mmdetection/myconfig/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_round2_aug_blur.py',
            '/competition/model_blur.pth', device='cuda:0')

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
    root = '/tcdata/guangdong1_round2_testB_20191024'
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
    root = '/tcdata/guangdong1_round2_testB_20191024'
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


batch_inference()
