# model settings
model = dict(
    type='GridRCNN',
    # pretrained='open-mmlab://resnext101_32x4d',
    backbone=dict(
        type='SiameseResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        num_stages=4,
        # with_blur=True,
        # gcb=dict(ratio=1. / 4., ),
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[4, 8, 16],
        anchor_ratios=[0.04, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0, 25.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHead',
        with_reg=False,
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=16,
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2],
        reg_class_agnostic=False),
    grid_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=14, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    grid_head=dict(
        type='GridHead',
        grid_points=9,
        num_convs=8,
        in_channels=256,
        point_feat_channels=64,
        norm_cfg=dict(type='GN', num_groups=36),
        loss_grid=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=15)))
# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='OHEMSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_radius=1,
        pos_weight=-1,
        max_num_grid=192,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05, nms=dict(type='soft_nms', iou_thr=0.5, method='gaussian'), max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/shared_disk/hannah/datasets/guangdong_round2/process/siamese_coco_round2/'#'/shared_disk/zhaoliang/datasets/guangdong_round2/siamese_coco_aug_large/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='ResizeImages', img_scale=(2048, 900), keep_ratio=False),
    dict(type='RandomFlipImages', flip_ratio=0.5),
    dict(type='NormalizeImages', **img_norm_cfg),
    dict(type='PadImages', size_divisor=32),
    dict(type='DefaultFormatBundleImages'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImagesFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 900),
        flip=False,
        transforms=[
            dict(type='ResizeImages', keep_ratio=False),
            dict(type='RandomFlipImages'),
            dict(type='NormalizeImages', **img_norm_cfg),
            dict(type='PadImages', size_divisor=32),
            dict(type='ImagesToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'images/train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'images/val/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1200,
    warmup_ratio=1.0 / 50,
    step=[7, 12])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 16
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/101/grid_rcnn_gn_head_x101_32x4d_fpn_2x'
load_from = None #'/home/zhaoliang/project/batch_model/mmdetection/work_dirs/101/grid_rcnn_gn_head_x101_32x4d_fpn_2x/epoch_1.pth'
resume_from = '/home/zhaoliang/project/batch_model/mmdetection/work_dirs/101/grid_rcnn_gn_head_x101_32x4d_fpn_2x/epoch_5.pth'
workflow = [('train', 1)]
