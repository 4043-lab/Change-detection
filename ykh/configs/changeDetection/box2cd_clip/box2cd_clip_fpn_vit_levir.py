'''
自定义的 以box为弱监督的利用clip进行跨模态训练框架   双分支 后融合  无cross attention
'''

#para
img_scale=(256,256)
lr=0.0001
warmup_iters=500
max_epochs=12*8
lr_step=[9, 11]
# lr_step=[18, 22]
# class_names=('Building','not Building')
# class_names=('Natural Environment','Transportation','Recreation & Sports',
#              'Residential & Buildings','Commercial & Industrial',)
class_names=('Building','Road','Parking lot','Vegetation','Water',)
# class_names=('Natural Environment','Transportation','Recreation Buildings','Sports Buildings',
#              'Residential','Buildings','Commercial Buildings','Industrial Buildings','background objects')
text_len=len(class_names)
num_classes=1

model = dict(
    type='Box2Cd_CLIP_vit',
    pretrained='pretrained/ViT-B-16.pt',
    context_length=5,
    text_head=False,
    class_names=class_names,
    score_thr=0.5,
    backbone=dict(
        type='CLIPVisionTransformer',
        patch_size=16,
        width=768,
        output_dim=512,
        get_embeddings=True,
        drop_path_rate=0.1,
        layers=12,
        input_resolution=640,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    neck=dict(
        type='FPN',
        # in_channels=[256, 512, 1024, 2048+text_len],
        in_channels=[768, 768, 768+text_len, 768],
        out_channels=256,
        start_level=0,
        num_outs=5),
    bbox_head=dict(
        type='Box2cd_CLIP_Head_v2',
        num_classes=num_classes,
        in_channels=256,
        stacked_convs=4,
        seg_feat_channels=256,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        sigma=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cate_down_pos=0,
        loss_cate=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_boxpro=dict(
            type='BoxProjectionLoss',
            loss_weight=3.0),
        loss_levelset=dict(
            type='LevelsetLoss',
            loss_weight=1.0),
        loss_sscr=dict(
            type='SscrLoss',
            loss_weight=1.0
        ),
        loss_similarity=dict(
            type='Similarity_Loss',
            loss_weight=[1.0, 0.0, 0.0]),
        loss_not_inst_weight=[0.0,5],
        ),
    train_cfg = dict(),
    test_cfg = dict(
        nms_pre=500,
        score_thr=0.05,
        mask_thr=0.70,
        filter_thr=0.025,
        kernel='gaussian',  # gaussian/linear
        sigma=2.0,
        max_per_img=100))



# dataset settings
dataset_type = 'LEVIRcdDataset'
data_root = '../../datasets/LEVIR_CD_256/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile_cd'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize_cd', img_scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip_cd', flip_ratio=0.5),
    dict(type='Normalize_cd', **img_norm_cfg),
    dict(type='Pad_cd', size_divisor=32),
    dict(type='DefaultFormatBundle_cd2'),
    dict(type='Collect_cd', keys=['img_1','img_2', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile_cd'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img_1','img_2']),
            dict(type='Collect_cd', keys=['img_1','img_2']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/LEVIR_train.json',
        img_prefix=data_root + 'train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/LEVIR_test.json',
        img_prefix=data_root + 'test',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/LEVIR_test.json',
        img_prefix=data_root + 'test',
        pipeline=test_pipeline))


# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=warmup_iters,
#     warmup_ratio=0.001,
#     step=lr_step)
# optimizer = dict(
#     type='AdamW',
#     lr=lr,
#     weight_decay=0.1,
#     paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)

optimizer = dict(type='AdamW', lr=lr, weight_decay=0.0001,
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.)}))

# optimizer = dict(type='SGD', lr=lr, momentum=0.9, weight_decay=0.0001)

optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))


# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
checkpoint_config = dict(interval=3)
evaluation = dict(interval=1, metric=['segm'])
# runner = dict(type='IterBasedRunner', max_iters=80000)
# checkpoint_config = dict(by_epoch=False, interval=8000)
# evaluation = dict(interval=8000, metric='mIoU')

device_ids = range(4)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/box2cd_clip_fpn_vit_levir_v4'
load_from = None
resume_from = None
workflow = [('train', 1)]
