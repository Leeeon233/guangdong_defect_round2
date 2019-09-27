import os
import json
from mmdet.apis import init_detector, inference_detector
import os


class Detector:
    def __init__(self):
        self.model = init_detector('/home/zhaoliang/project/build_model/mmdetection/myconfig/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_round2.py', '/home/zhaoliang/project/build_model/mmdetection/work_dirs/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_round2/epoch_12.pth', device='cuda:0')

    def detect_single_img(self, file_path, template_path):
        predict = inference_detector(self.model, [file_path, template_path])
        result = []
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                defect_label = i
                image_name = os.path.basename(file_path)
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                    result.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})

        return result

# root = '/tcdata/guangdong1_round2_testA_20190924'
# result = []
# detector = Detector()
# for dir_name in os.listdir(root):
#     for file in os.listdir(os.path.join(root, dir_name)):
#         if not file.startswith('template'):
#             res = detector.detect_single_img(os.path.join(root, dir_name, file))
#             result += res
detector = Detector()
root = '/shared_disk/zhaoliang/datasets/guangdong_round2/siamese_coco/images/val'
filepath = os.path.join(root,'08272_de80be072bfd5c580201908270948131OK.jpg')
template = os.path.join(root,'template_08272.jpg')

res = detector.detect_single_img(filepath, template)
print(res)
# with open('result.json', 'w') as fp:
#     json.dump(result, fp, indent=4, separators=(',', ': '))