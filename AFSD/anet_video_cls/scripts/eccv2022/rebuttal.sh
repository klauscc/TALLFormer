#==================================
# exp: r1.b
# training
export MASTER_ADDR=localhost
export MASTER_PORT=12443
expid=r1.b
ckpt_path=workspace/eccv2022/membank/${expid}
config=AFSD/anet_video_cls/configs/anet_256_mp4.yaml
additional_config=AFSD/anet_video_cls/configs/eccv2022/membank/$expid.py
membank_dir=datasets/tmp/activitynet/memory_mechanism/eccv2022/${expid}
mkdir -p $membank_dir
cp -r datasets/activitynet/memory_mechanism/feat_swinb_train_val_npy_256x256 $membank_dir
python3 AFSD/anet_video_cls/membank_flow/train_membank.py $config --lw=1 --cw=1 --piou=0.6 --ssl 0. \
    --checkpoint_path $ckpt_path --addi_config $additional_config \
    --learning_rate 1e-4 \
    --batch_size 1 \
    --ngpu 4

# testing
expid=r1.b
ckpt_path=workspace/eccv2022/membank/${expid}
config=AFSD/anet_video_cls/configs/anet_256_mp4.yaml
additional_config=AFSD/anet_video_cls/configs/eccv2022/membank/$expid.py
# epoch=10
for epoch in `seq 9 -2 3`; do
    echo "Testing Epoch: $epoch"
    output_json=${expid}-epoch_${epoch}-anet_rgb.json
    echo $output_json
    python3 AFSD/anet_video_cls/membank_flow/test.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config \
        --ngpu 4
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-epoch_${epoch}-anet_rgb-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done

# testing top-2
expid=r1.b
ckpt_path=workspace/eccv2022/membank/${expid}
config=AFSD/anet_video_cls/configs/anet_256_mp4.yaml
additional_config=AFSD/anet_video_cls/configs/eccv2022/membank/$expid.py
epoch=5
# for epoch in `seq 9 -2 3`; do
    echo "Testing Epoch: $epoch"
    output_json=${expid}-epoch_${epoch}-anet_rgb-top2.json
    echo $output_json
    python3 AFSD/anet_video_cls/membank_flow/test_top2.py $config  --nms_sigma=0.85 --ngpu=1 \
        --checkpoint_path $ckpt_path/checkpoint-$epoch.ckpt  \
        --output_json=$output_json \
        --addi_config $additional_config \
        --ngpu 4
    for classifier in "builtin" "cuhk" "gt"; do
        output_json=${expid}-epoch_${epoch}-anet_rgb-top2-${classifier}.json
        python3 AFSD/anet_video_cls/eval.py output/$output_json \
            --workspace $ckpt_path --epoch ${epoch}_${classifier}
    done
done
#==================================
