# 1. data
dataset_type = "Thumos14Dataset"
data_root = "data/thumos14/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
num_frames = 480
chunk_size = 32
img_shape = (224, 224)
overlap_ratio = 0.25

# keep_ratio can control the amount of GPU memory usage.
keep_ratio = 0.15

feat_downsample = 2
expid = "1.0.0-vswin_b_256x256-12GB"

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=6,
    train=dict(
        typename=dataset_type,
        ann_file=data_root + "annotations/val.json",
        video_prefix=data_root + "frames_15fps_256x256/val",
        pipeline=[
            dict(typename="LoadMetaInfo"),
            dict(typename="LoadAnnotations"),
            dict(typename="Time2Frame"),
            dict(typename="TemporalRandomCrop", num_frames=num_frames, iof_th=0.75),
            dict(typename="LoadFrames", to_float32=True),
            dict(typename="SpatialRandomCrop", crop_size=img_shape),
            dict(
                typename="PhotoMetricDistortion",
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18,
                p=0.5,
            ),
            dict(typename="Rotate", limit=(-45, 45), border_mode="reflect101", p=0.5),
            dict(typename="SpatialRandomFlip", flip_ratio=0.5),
            dict(typename="Normalize", **img_norm_cfg),
            dict(typename="Pad", size=(num_frames, *img_shape)),
            dict(typename="DefaultFormatBundle"),
            dict(
                typename="Collect",
                keys=["imgs", "gt_segments", "gt_labels", "gt_segments_ignore"],
            ),
        ],
    ),
    val=dict(
        typename=dataset_type,
        ann_file=data_root + "annotations/test.json",
        video_prefix=data_root + "frames_15fps_256x256/test",
        pipeline=[
            dict(typename="LoadMetaInfo"),
            dict(typename="Time2Frame"),
            dict(
                typename="OverlapCropAug",
                num_frames=num_frames,
                overlap_ratio=overlap_ratio,
                transforms=[
                    dict(typename="TemporalCrop"),
                    dict(typename="LoadFrames", to_float32=True),
                    dict(typename="SpatialCenterCrop", crop_size=img_shape),
                    dict(typename="Normalize", **img_norm_cfg),
                    dict(typename="Pad", size=(num_frames, *img_shape)),
                    dict(typename="DefaultFormatBundle"),
                    dict(typename="Collect", keys=["imgs"]),
                ],
            ),
        ],
    ),
)

# 2. model
num_classes = 20
strides = [8, 16, 32, 64, 128]
use_sigmoid = True
scales_per_octave = 5
octave_base_scale = 2
num_anchors = scales_per_octave

model = dict(
    typename="MemSingleStageDetector",
    chunk_size=chunk_size,
    backbone=dict(
        typename="ChunkVideoSwin",
        chunk_size=chunk_size,
        do_pooling=True,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=128,
        drop_path_rate=0.2,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=(8, 7, 7),
        patch_norm=True,
        frozen_stages=2,
        use_checkpoint=True,
    ),
    neck=[
        dict(
            typename="SRMSwin",
            srm_cfg=dict(
                in_channels=1024,
                out_channels=512,
                with_transformer=False,
            ),
        ),
        dict(
            typename="Transformer1DRelPos",
            encoder_layer_cfg=dict(
                dim=512,
                num_heads=16,
                max_seq_len=num_frames // strides[0],
                drop_path=0.1,
            ),
            num_layers=3,
        ),
        dict(
            typename="SelfAttnTDM",
            in_channels=512,
            out_channels=512,
            strides=2,
            num_heads=8,
            kernel_sizes=(7, 7, 5, 5),
            stage_layers=(1, 1, 1, 1),
            out_indices=(0, 1, 2, 3, 4),
            out_order="bct",
        ),
        dict(
            typename="FPN",
            in_channels=[512, 512, 512, 512, 512],
            out_channels=256,
            num_outs=5,
            start_level=0,
            conv_cfg=dict(typename="Conv1d"),
            norm_cfg=dict(typename="SyncBN"),
        ),
    ],
    head=dict(
        typename="RetinaHead",
        num_classes=num_classes,
        num_anchors=num_anchors,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        use_sigmoid=use_sigmoid,
        conv_cfg=dict(typename="Conv1d"),
        norm_cfg=dict(typename="SyncBN"),
    ),
)

# 3. engines
meshgrid = dict(
    typename="SegmentAnchorMeshGrid",
    strides=strides,
    base_anchor=dict(
        typename="SegmentBaseAnchor",
        base_sizes=strides,
        octave_base_scale=octave_base_scale,
        scales_per_octave=scales_per_octave,
    ),
)

segment_coder = dict(
    typename="DeltaSegmentCoder", target_means=[0.0, 0.0], target_stds=[1.0, 1.0]
)

train_engine = dict(
    typename="MemBankTrainEngine",
    membank=dict(
        chunk_size=chunk_size,
        keep_ratio=keep_ratio,
        feat_downsample=feat_downsample,
        mode="random",
        mem_bank_meta_file=f"data/tmp/eccv2022/thumos14/memory_mechanism/{expid}/feat_swinb_15fps_256x256_crop224x224/meta_val.json",
        mem_bank_dir=f"data/tmp/eccv2022/thumos14/memory_mechanism/{expid}/feat_swinb_15fps_256x256_crop224x224/val",
    ),
    model=model,
    criterion=dict(
        typename="SegmentAnchorCriterion",
        num_classes=num_classes,
        meshgrid=meshgrid,
        segment_coder=segment_coder,
        reg_decoded_segment=True,
        loss_cls=dict(
            typename="FocalLoss",
            use_sigmoid=use_sigmoid,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_segment=dict(typename="DIoULoss", loss_weight=1.0),
        train_cfg=dict(
            assigner=dict(
                typename="MaxIoUAssigner",
                pos_iou_thr=0.6,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1,
                ignore_wrt_candidates=True,
                iou_calculator=dict(typename="SegmentOverlaps"),
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
    ),
    optimizer=dict(
        typename="SGD",
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0001,
        paramwise_cfg=dict(custom_keys=dict(backbone={"lr_mult": 0.4})),
    ),
)

# 3.2 val engine
val_engine = dict(
    typename="ValEngine",
    model=model,
    meshgrid=meshgrid,
    converter=dict(
        typename="SegmentAnchorConverter",
        num_classes=num_classes,
        segment_coder=segment_coder,
        nms_pre=1000,
        use_sigmoid=use_sigmoid,
    ),
    num_classes=num_classes,
    test_cfg=dict(
        score_thr=0.005, nms=dict(typename="nmw", iou_thr=0.5), max_per_video=1200
    ),
    use_sigmoid=use_sigmoid,
)

# 4. hooks
hooks = [
    dict(typename="OptimizerHook"),
    dict(
        typename="CosineRestartLrSchedulerHook",
        periods=[100] * 12,
        restart_weights=[1] * 12,
        warmup="linear",
        warmup_iters=500,
        warmup_ratio=1e-1,
        min_lr_ratio=1e-2,
    ),
    dict(typename="EvalHook", eval_cfg=dict(mode="anet")),
    dict(typename="SnapshotHook", interval=100),
    dict(typename="LoggerHook", interval=10),
]

# 5. work modes
modes = ["train"]
max_epochs = 1000

# 6. checkpoint
weights = dict(
    filepath="data/pretrained_models/vswin/swin_base_patch244_window877_kinetics400_22k_keysfrom_backbone.pth"
)

# optimizer = dict(filepath='epoch_900_optim.pth')
# meta = dict(filepath='epoch_900_meta.pth')

# 7. misc
seed = 10
dist_params = dict(backend="nccl")
log_level = "INFO"
find_unused_parameters = False

# gpu_mem_fraction = 0.2
