_base_ = [
    '../_base_/datasets/silkworm11_coco.py',
    '../_base_/default_runtime.py',
]

# -------------------------------------------------
# Dataset Configuration
# -------------------------------------------------
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = '../data/silkworm/'

# Image and heatmap sizes
image_size = [192, 256]   # (w, h)
heatmap_size = [48, 64]

# Data processing pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomBBoxTransform',
         scale_factor=[0.7, 1.3], rotate_factor=40),
    dict(type='TopdownAffine', input_size=image_size),
    dict(type='GenerateTarget',
         encoder=dict(
             type='MSRAHeatmap',
             input_size=image_size,
             heatmap_size=heatmap_size,
             sigma=2)),
    dict(type='PackPoseInputs')
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale', padding=1.25),
    dict(type='TopdownAffine', input_size=image_size),
    dict(type='PackPoseInputs')
]

# Common dataset configuration
_common_cfg = dict(
    type=dataset_type,
    data_mode=data_mode,
    data_root=data_root,
    data_prefix=dict(img='images/'),
    metainfo=dict(from_file='configs/_base_/datasets/silkworm11_coco.py'),
)

# Data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        ann_file='train.json',
        pipeline=train_pipeline,
        **_common_cfg),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        ann_file='val.json',
        pipeline=val_pipeline,
        **_common_cfg),
)

test_dataloader = val_dataloader

# Evaluators
val_evaluator = dict(type='CocoMetric', ann_file=f'{data_root}val.json')
test_evaluator = val_evaluator

# -------------------------------------------------
# Model
# -------------------------------------------------
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[128, 128, 128],
        std=[256, 256, 256],
        bgr_to_rgb=True),

    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK',
                        num_blocks=(4,), num_channels=(64,)),
            stage2=dict(num_modules=1, num_branches=2, block='BASIC',
                        num_blocks=(4, 4), num_channels=(32, 64)),
            stage3=dict(num_modules=4, num_branches=3, block='BASIC',
                        num_blocks=(4, 4, 4), num_channels=(32, 64, 128)),
            stage4=dict(num_modules=3, num_branches=4, block='BASIC',
                        num_blocks=(4, 4, 4, 4),
                        num_channels=(32, 64, 128, 256))),
        
    ),

    head = dict(
        type='HeatmapHead',
        in_channels=32,              # HRNet の出力チャネル
        deconv_out_channels=(),      # 空タプル → deconv 層 0 段
        final_layer=dict(kernel_size=1),
        out_channels=11,             # キーポイント数
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',       # 既定実装
            input_size=[192, 256],       # 画像サイズ (w, h)
            heatmap_size=[48, 64],
            sigma=2)  
    ),

    test_cfg=dict(flip_test=False)      # 左右対称ペアが無いので False
)

load_from = '/home/atmiyashita/python/openmmlab/data/silkworm/pretrain/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'
# -------------------------------------------------
# Optimizer & Schedule
# -------------------------------------------------
train_cfg = dict(max_epochs=210, val_interval=5)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4, weight_decay=0)
)

param_scheduler = [
    # warm-up 500 iters
    dict(type='LinearLR', start_factor=1e-5,
         by_epoch=False, begin=0, end=500),
    # step decay
    dict(type='MultiStepLR',
         by_epoch=True, begin=0, end=210,
         milestones=[170, 200], gamma=0.1),
]

# -------------------------------------------------
# Work-Dir (コマンドライン --work-dir でも可)
# -------------------------------------------------
work_dir = 'work_dirs/hrnet_silkworm11'