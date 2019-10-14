import json
import os
import time

from mmdet.apis import init_detector, inference_detector, inference_detector_batch

class Detector:
    def __init__(self):
        self.model = init_detector(
            '/competition/mmdetection/myconfig/cascade_rcnn_dconv_c3-c5_r50_fpn_1x_round2_aug_blur.py',
            '/competition/epoch_1.pth', device='cuda:0')

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
            if len(scores) > 0 and max(scores) > 0.05:
                # if len(scores) == 1 and max(scores) < 0.05:
                #     continue
                result += rs
        return result

    def detect_batch_imgs(self, file_paths, batch_size):
        result = []
        for j in range(0, len(file_paths), batch_size):
            paths = file_paths[j:j + batch_size]
            predicts = inference_detector_batch(self.model, paths)
            for (file_path, _), predict in zip(paths, predicts):
                # vis_img = cv2.imread(file_path)
                image_name = os.path.basename(file_path)
                for i, bboxes in enumerate(predict, 1):
                    rs = []
                    scores = []
                    if len(bboxes) > 0:
                        defect_label = i
                        for bbox in bboxes:
                            x1, y1, x2, y2, score = bbox.tolist()
                            x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                            # cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                            # cv2.putText(vis_img, str(defect_label) + f": {round(score, 2)}",
                            #             (int(x1) + 20, int(y1) + 20),
                            #             cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                            scores.append(score)
                            rs.append(
                                {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2],
                                 'score': score})
                    if len(scores) > 0 and max(scores) > 0.05:
                        # if len(scores) == 1 and max(scores) < 0.05:
                        #     continue
                        result += rs
                # img_anno = self.anno[self.anno["name"] == image_name]
                # bboxs = img_anno["bbox"].tolist()
                # defect_names = img_anno["defect_name"].tolist()
                # for b, defect_name in zip(bboxs, defect_names):
                #     cv2.rectangle(vis_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 3)
                #     cv2.putText(vis_img, str(defect_name2label[defect_name]), (int(b[0]) + 40, int(b[1]) + 40),
                #                 cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
                # cv2.imwrite(f"predict/pred_{image_name}", vis_img)
        return result

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
res = detector.detect_batch_imgs(paths, 10)
result += res

with open('result.json', 'w') as fp:
    json.dump(result, fp, indent=4, separators=(',', ': '))
print("time use", time.time() - s)