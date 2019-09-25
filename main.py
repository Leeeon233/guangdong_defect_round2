import os
import json

root = '/tcdata/guangdong1_round2_testA_20190924'
result = []
for file in os.listdir(root):
    result.append(
        {'name': file, 'category': 1, 'bbox': [100, 200, 100, 200], 'score': 1.0})

with open('result.json', 'w') as fp:
    json.dump(result, fp, indent=4, separators=(',', ': '))
