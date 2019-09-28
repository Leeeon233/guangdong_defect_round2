import time, os
import json
import mmcv
from mmdet.apis import init_detector, inference_detector
import cv2


def main():
    config_file = '/competition/mmdetection/myconfig/0928_cascade_rcnn_r50_fpn.py'  # 修改成自己的配置文件
    checkpoint_file = '/competition/epoch_16.pth'  # 修改成自己的训练权重
    test_path = '/tcdata/guangdong1_round2_testA_20190924'  # 官方测试集图片路径

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    result = []
    couple_files = os.listdir(test_path)
    for couple_file in couple_files:
        for file in os.listdir(os.path.join(test_path, couple_file)):
            if file.startswith('template'):
                pattern = cv2.imread(os.path.join(test_path, couple_file, file))
            else:
                test = cv2.imread(os.path.join(test_path, couple_file, file))
        diff = cv2.absdiff(pattern, test)

        predict = inference_detector(model, diff)
        for i, bboxes in enumerate(predict, 1):
            if len(bboxes) > 0:
                defect_label = i
                # print(i)
                image_name = couple_file + '.jpg'
                for bbox in bboxes:
                    x1, y1, x2, y2, score = bbox.tolist()
                    x1, y1, x2, y2 = round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)  # save 0.00
                    result.append(
                        {'name': image_name, 'category': defect_label, 'bbox': [x1, y1, x2, y2], 'score': score})

    with open("result.json", 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '))


if __name__ == "__main__":
    main()