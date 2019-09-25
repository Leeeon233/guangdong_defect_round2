import os
import json

root = '/tcdata/guangdong1_round2_testA_20190924'
result = []
for dir_name in os.listdir(root):
    for file in os.listdir(os.path.join(root, dir_name)):
        if not file.startswith('template'):
            result.append(
                {'name': file, 'category': 1, 'bbox': [100, 200, 100, 200], 'score': 1.0})

with open('result.json', 'w') as fp:
    json.dump(result, fp, indent=4, separators=(',', ': '))