# TALLFormer
This is the offical pyTorch implementation of our ECCV 2022 paper
[TALLFormer: Temporal Action Localization with Long-memory Transformer](https://arxiv.org/abs/2204.01680).

Temporal action localization takes hundreds of frames as input. End-to-end training on this task requires huge GPU memory (>32 GB). This issue becomes even worse with the recent video transformer models, many of which have quadratic memory complexity. To address these issues, we propose TALLFormer, a memory-efficient and end-to-end trainable Temporal Action Localization Transformer with Long-term memory. Our long-term memory mechanism eliminates the need for processing hundreds of redundant video frames during each training iteration, thus, significantly reducing the GPU memory consumption and training time.
TALLFormer outperforms previous state-of-the-arts by a large margin, achieving an average mAP of 59.1% on THUMOS14 and 35.6% on ActivityNet-1.3.

## Installation

Our code is built on [vedatad](https://github.com/Media-Smart/vedatad). Many Thanks! 

### Requirement

- Linux
- pytorch 1.10.1
- Python 3.8.5
- ffmpeg 4.3.11

### Install vedatad

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedatad python=3.8.5 -y
conda activate vedatad
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
# CUDA 10.2
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

c. Clone the vedatad repository.

```shell
git clone https://github.com/klauscc/TALLFormer
cd vedatad
vedatad_root=${PWD}
```

d. Install vedatad.

```shell
pip install -r requirements/build.txt
pip install -v -e .
```

## Data preparation

### THUMOS14

a. Download datasets & Create annotations

Under the root dir of our codebase, download and place data under directory `data/thumos14`.
``` sh
# you can also download it to othe place and use softlink to link it here.
mkdir -p data/thumos14
cd data/thumos14

# download
../../tools/data/thumos14/download.sh

# generate annotations.
cd ../..
python ./tools/data/thumos14/txt2json.py --anno_root data/thumos14/annotations --video_root data/thumos14/videos --mode val
python ./tools/data/thumos14/txt2json.py --anno_root data/thumos14/annotations --video_root data/thumos14/videos --mode test
```

b. Extract frames
Our model use FPS=15 and spatial resolution `256x256`.

``` sh
./tools/data/extract_frames.sh data/thumos14/videos/val data/thumos14/frames_15fps_256x256/val -vf fps=15 -s 256x256 %05d.png
./tools/data/extract_frames.sh data/thumos14/videos/test data/thumos14/frames_15fps_256x256/test -vf fps=15 -s 256x256 %05d.png
```

## Train

### a. Download pretrained weights

Download weights and place into directory `./data/pretrained_weights/vswin`
``` sh
mkdir -p ./data/pretrained_models/vswin

# vswin-B
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_22k.pth -P ./data/pretrained_models/vswin

# vswin-T
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth -P ./data/pretrained_models/vswin

```

Convert the ckpt to the format that our code can accept.
``` sh
# vswin-T
python ./tools/convert_vswin_ckpt.py --src ./data/pretrained_models/vswin/swin_tiny_patch244_window877_kinetics400_1k.pth

# vswin-B
python ./tools/convert_vswin_ckpt.py --src ./data/pretrained_models/vswin/swin_base_patch244_window877_kinetics400_22k.pth

```

### b. Init the memory-bank
We first extract the features of the training set and save it to a file (memory-bank).
During training, our model will load and update this memory-bank.

``` sh
# config can be "swin_b_15fps_256x256", "swin_t_15fps_256x256", "swin_t_15fps_128x128"
python ./tools/data/extract_features.py --config swin_b_15fps_256x256
```
The files will be saved to directory `data/thumos14/memory_mechanism/feat_swinb_15fps_256x256_crop224x224/`.

### c. train and test

``` sh

# if you change the expid, *be sure* to change the `expid` variable in the config file as well.
expid=1.0.0-vswin_b_256x256-12GB

# first copy the memory bank to a tmp directory. Our model will modify it *inplace*.
# be sure the disk for the tmp directory is fast enough. SSD prefered.
mkdir -p data/tmp/eccv2022/thumos14/memory_mechanism/$expid
cp -r data/thumos14/memory_mechanism/feat_swinb_15fps_256x256_crop224x224 data/tmp/eccv2022/thumos14/memory_mechanism/$expid/

workdir=workdir/tallformer/$expid
mkdir -p $workdir
config=configs/trainval/thumos/$expid.py
# train with 4 GPUs.
tools/dist_trainval.sh $config "0,1,2,3" --workdir $workdir

# test
expid=1.0.0-vswin_b_256x256-12GB
workdir=workdir/tallformer/$expid
config=configs/trainval/thumos/$expid.py
for epoch in 600 700 800 900 1000; do
    python tools/test.py $config $workdir/epoch_${epoch}_weights.pth \
        --out $workdir/results_e$epoch-chunk.pkl
done

```

## Credits

Our implementation is inspired by several open-sourced work, including:

- [vedatad](https://github.com/Media-Smart/vedatad)
- [ActionDetection-AFSD](https://github.com/TencentYoutuResearch/ActionDetection-AFSD/tree/master/AFSD)

Many Thanks!

## Citation

If you find this project useful for your research, please use the following BibTeX entry.
```
@article{cheng2022tallformer,
  title={TALLFormer: Temporal Action Localization with Long-memory Transformer},
  author={Cheng, Feng and Bertasius, Gedas},
  journal={arXiv preprint arXiv:2204.01680},
  year={2022}
}
```
