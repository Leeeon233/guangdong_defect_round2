def main():
    #gen coco pretrained weight
    import torch
    num_classes = 16
    model_coco = torch.load("/disk4/zhaoliang/project/build_model/mmdetection/cascade_rcnn_dconv_c3-c5_r101_fpn_1x_20190125-aaa877cc.pth")

    # weight
    model_coco["state_dict"]["bbox_head.0.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.0.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.1.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.1.fc_cls.weight"][
                                                            :num_classes, :]
    model_coco["state_dict"]["bbox_head.2.fc_cls.weight"] = model_coco["state_dict"]["bbox_head.2.fc_cls.weight"][
                                                            :num_classes, :]
    # bias
    model_coco["state_dict"]["bbox_head.0.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.0.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.1.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.1.fc_cls.bias"][
                                                          :num_classes]
    model_coco["state_dict"]["bbox_head.2.fc_cls.bias"] = model_coco["state_dict"]["bbox_head.2.fc_cls.bias"][
                                                          :num_classes]
    # save new model
    torch.save(model_coco, "cascade_rcnn_dconv_c3-c5_r101_fpn_weights_classes_%d.pth" % num_classes)

if __name__ == "__main__":
    main()