from mmdet.apis import init_detector, inference_detector
import os


class Detector:
    def __init__(self):
        self.model = init_detector('cascade_rcnn_dconv_c3-c5_r50_fpn_1x_round2.py', 'epoch_12.pth', device='cuda:0')

    def detect_single_img(self, file_path):
        predict = inference_detector(self.model, file_path)
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
