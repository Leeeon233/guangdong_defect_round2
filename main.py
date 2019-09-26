import os
import json
from detection_single_img import Detector

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