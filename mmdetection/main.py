from mmdet.apis import init_detector, inference_detector
import os
import json


class Detector:
    def __init__(self):
        self.model = init_detector(
            '/competition/mmdetection/myconfig/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_round2_aug2.py',
            '/competition/epoch_15.pth', device='cuda:0')

    def detect_single_img(self, file_path, template_path):
        predict = inference_detector(self.model, [file_path, template_path])
        result = []
        for i, bboxes in enumerate(predict, 1):
            rs = []
            scores = []
            if len(bboxes) > 0:
                defect_label = i
                image_name = os.path.basename(file_path)
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                    scores.append(score)
                    rs.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})
            if len(scores) > 0 and max(scores) > 0.01:
                result += rs
        return result


root = '/tcdata/guangdong1_round2_testA_20190924'
result = []
detector = Detector()
from collections import defaultdict
tmp = defaultdict(int)
for dir_name in os.listdir(root):
    files = os.listdir(os.path.join(root, dir_name))
    if files[0].startswith('template'):
        files = [files[1], files[0]]
    file = files[0]
    template_file = files[1]
    tmp[template_file]+=1
    res = detector.detect_single_img(os.path.join(root, dir_name, file), os.path.join(root, dir_name, template_file))
    result += res
print(tmp)
with open('result.json', 'w') as fp:
    json.dump(result, fp, indent=4, separators=(',', ': '))
