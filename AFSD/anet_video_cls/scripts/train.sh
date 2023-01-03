#!/usr/bin/env sh

#==================================
export MASTER_ADDR=localhost
export MASTER_PORT=12441
ckpt_path=workspace/4.c.i
config=AFSD/anet_video_cls/configs/anet.yaml
additional_config=AFSD/anet_video_cls/configs/4.c.i.py
python3 AFSD/anet_video_cls/train.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --ngpu 4

ckpt_path=workspace/4.c.i
config=AFSD/anet_video_cls/configs/anet.yaml
additional_config=AFSD/anet_video_cls/configs/4.c.i.py
# epoch=6
for epoch in `seq 4 5`; do
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

#==================================
export MASTER_ADDR=localhost
export MASTER_PORT=12442
ckpt_path=workspace/4.c.iv
config=AFSD/anet_video_cls/configs/anet.yaml
additional_config=AFSD/anet_video_cls/configs/4.c.iv.py
python3 AFSD/anet_video_cls/train.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --ngpu 4

classifier=cuhk
ckpt_path=workspace/4.c.iv
config=AFSD/anet_video_cls/configs/anet.yaml
additional_config=AFSD/anet_video_cls/configs/4.c.iv.py
for epoch in  6 7 8 9 10 5 11 12 13 14; do
    echo "Testing Epoch: $epoch"
    output_json=4.c.iv-epoch_${epoch}-cuhk-anet_rgb.json
    echo $output_json
    python3 AFSD/anet_video_cls/test.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config \
        --classifier $classifier
    python3 AFSD/anet_video_cls/eval.py output/$output_json \
        --workspace $ckpt_path --epoch ${epoch}_${classifier}
done

expid=4.c.iv
ckpt_path=workspace/${expid}
config=AFSD/anet_video_cls/configs/anet.yaml
additional_config=AFSD/anet_video_cls/configs/${expid}.py
epoch=7
# for epoch in `seq 6 10`; do
    echo "Testing Epoch: $epoch"
    output_json=${expid}-epoch_${epoch}-anet_rgb-ensemble.json
    echo $output_json
    python3 AFSD/anet_video_cls/test_ensemble.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config \
        --ngpu 4
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-epoch_${epoch}-anet_rgb-ensemble-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done
#==================================

#==================================
# training
export MASTER_ADDR=localhost
export MASTER_PORT=12443
expid=4.c.vi
ckpt_path=workspace/$expid
config=AFSD/anet_video_cls/configs/anet_256.yaml
additional_config=AFSD/anet_video_cls/configs/$expid.py
python3 AFSD/anet_video_cls/train.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --ngpu 4

# testing
expid=4.c.vi
ckpt_path=workspace/${expid}
config=AFSD/anet_video_cls/configs/anet_256.yaml
additional_config=AFSD/anet_video_cls/configs/${expid}.py
epoch=7
# for epoch in `seq 5 21`; do
    echo "Testing Epoch: $epoch"
    output_json=${expid}-epoch_${epoch}-anet_rgb.json
    echo $output_json
    python3 AFSD/anet_video_cls/test.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-epoch_${epoch}-anet_rgb-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done
#==================================

#==================================
# training
export MASTER_ADDR=localhost
export MASTER_PORT=12443
expid=4.c.vii
ckpt_path=workspace/$expid
config=AFSD/anet_video_cls/configs/anet_256.yaml
additional_config=AFSD/anet_video_cls/configs/$expid.py
python3 AFSD/anet_video_cls/train.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --ngpu 4

# testing
expid=4.c.vii
ckpt_path=workspace/${expid}
config=AFSD/anet_video_cls/configs/anet_256.yaml
additional_config=AFSD/anet_video_cls/configs/${expid}.py
epoch=7
# for epoch in `seq 5 21`; do
    echo "Testing Epoch: $epoch"
    output_json=${expid}-epoch_${epoch}-anet_rgb.json
    echo $output_json
    python3 AFSD/anet_video_cls/test.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-epoch_${epoch}-anet_rgb-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done
#==================================

#==================================
# training
export MASTER_ADDR=localhost
export MASTER_PORT=12443
expid=5.a.ii
ckpt_path=workspace/$expid
config=AFSD/anet_video_cls/configs/anet.yaml
additional_config=AFSD/anet_video_cls/configs/$expid.py
python3 AFSD/anet_video_cls/train.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --ngpu 4

# testing
expid=5.a.ii
ckpt_path=workspace/${expid}
config=AFSD/anet_video_cls/configs/anet.yaml
additional_config=AFSD/anet_video_cls/configs/${expid}.py
# epoch=7
for epoch in `seq 5 15`; do
    echo "Testing Epoch: $epoch"
    output_json=${expid}-epoch_${epoch}-anet_rgb.json
    echo $output_json
    python3 AFSD/anet_video_cls/test.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-epoch_${epoch}-anet_rgb-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done
#==================================

#==================================
# training
export MASTER_ADDR=localhost
export MASTER_PORT=12443
expid=6.d.i
ckpt_path=workspace/$expid
config=AFSD/anet_video_cls/configs/anet.yaml
additional_config=AFSD/anet_video_cls/configs/$expid.py
python3 AFSD/anet_video_cls/train.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --ngpu 4

# testing
expid=6.d.i
ckpt_path=workspace/${expid}
config=AFSD/anet_video_cls/configs/anet.yaml
additional_config=AFSD/anet_video_cls/configs/${expid}.py
# epoch=7
for epoch in `seq 5 15`; do
    echo "Testing Epoch: $epoch"
    output_json=${expid}-epoch_${epoch}-anet_rgb.json
    echo $output_json
    python3 AFSD/anet_video_cls/test.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-epoch_${epoch}-anet_rgb-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done
#==================================

#==================================
# training
export MASTER_ADDR=localhost
export MASTER_PORT=12443
expid=6.d.ii
ckpt_path=workspace/$expid-run1
config=AFSD/anet_video_cls/configs/anet_256.yaml
additional_config=AFSD/anet_video_cls/configs/${expid}.py
python3 AFSD/anet_video_cls/train.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --ngpu 4

# testing
sleep 3h
expid=6.d.ii
ckpt_path=workspace/${expid}-run1
config=AFSD/anet_video_cls/configs/anet_256_mp4.yaml
additional_config=AFSD/anet_video_cls/configs/${expid}.py
# epoch=6
for epoch in `seq 11 15`; do
    echo "Testing Epoch: $epoch"
    output_json=${expid}-run1-epoch_${epoch}-anet_rgb.json
    echo $output_json
    python3 AFSD/anet_video_cls/test.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-run1-epoch_${epoch}-anet_rgb-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done

# test_ensemble
expid=6.d.ii
ckpt_path=workspace/${expid}-run1
config=AFSD/anet_video_cls/configs/anet_256.yaml
additional_config=AFSD/anet_video_cls/configs/${expid}.py
epoch=10
# for epoch in `seq 6 10`; do
    echo "Testing Epoch: $epoch"
    output_json=${expid}-epoch_${epoch}-anet_rgb-ensemble.json
    echo $output_json
    python3 AFSD/anet_video_cls/test_ensemble.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config \
        --ngpu 3
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-epoch_${epoch}-anet_rgb-ensemble-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done
#==================================
