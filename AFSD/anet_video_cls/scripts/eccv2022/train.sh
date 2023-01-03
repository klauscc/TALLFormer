#!/usr/bin/env sh


#==================================
export MASTER_ADDR=localhost
export MASTER_PORT=12441
expid=6.c.ii 
ckpt_path=workspace/eccv2022/$expid
config=AFSD/anet_video_cls/configs/anet_256_mp4.yaml
additional_config=AFSD/anet_video_cls/configs/eccv2022/$expid.py
python3 AFSD/anet_video_cls/train.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --ngpu 4

expid=6.c.ii 
ckpt_path=workspace/eccv2022/$expid
config=AFSD/anet_video_cls/configs/anet_256_mp4.yaml
additional_config=AFSD/anet_video_cls/configs/eccv2022/$expid.py
# epoch=6
for epoch in `seq 11 -2 5`; do
    echo "Testing Epoch: $epoch"
    output_json=4.c.i-epoch_${epoch}-anet_rgb.json
    python3 AFSD/anet_video_cls/test.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=4.c.i-epoch_${epoch}-anet_rgb-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done
#==================================
