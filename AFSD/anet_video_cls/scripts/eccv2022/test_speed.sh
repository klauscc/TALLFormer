
#ANet I3D-96
python AFSD/anet_video_cls/test_speed.py AFSD/anet_video_cls/configs/anet.yaml \
    --addi_config AFSD/anet_video_cls/configs/4.c.i.py

#ANet Swin-B-224
python AFSD/anet_video_cls/test_speed.py AFSD/anet_video_cls/configs/anet_256.yaml \
    --addi_config AFSD/anet_video_cls/configs/eccv2022/membank/6.c.iv.py

#ANet Swin-T-224
python AFSD/anet_video_cls/test_speed.py AFSD/anet_video_cls/configs/anet_256.yaml \
    --addi_config AFSD/anet_video_cls/configs/4.c.iv.py
