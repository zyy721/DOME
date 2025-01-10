<img src="static/images/favicon.ico" width="3%" align="left">

# DOME: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model
<div align="center">

### [Project Page](https://gusongen.github.io/DOME/) | [Paper](https://arxiv.org/abs/2410.10429v1)

</div>

![](https://gusongen.github.io/DOME/static/videos/cmp/6/output_video.gif)
![](https://gusongen.github.io/DOME/static/videos/cmp/7/output_video.gif)
![](https://gusongen.github.io/DOME/static/videos/cmp/31/output_video.gif)

<img src="static/images/teaser12.png" alt="teaser"/>
Our Occupancy World Model can generate long-duration occupancy forecasts and can be effectively controlled by trajectory conditions.


# 📖 Overview
<img src="static/images/overall_pipeline4.png" alt="overview"/>
Our method consists of two components: <b>(a) Occ-VAE Pipeline</b> encodes occupancy frames into a continuous latent space, enabling efficient data compression. <b>(b)DOME Pipeline</b> learns to predict 4D occupancy based on historical occupancy observations.



## 🗓️ News
- [2025.1.1] We release the code and checkpoints.
- [2024.11.18] Project page is online!

## 🗓️ TODO
- [x] Code release.
- [x] Checkpoint release.


## 🚀 Setup
### clone the repo
```
git clone https://github.com/gusongen/DOME-world-model.git
cd DOME
```

### environment setup
```
conda env create --file environment.yml
```

### data preparation
1. Create soft link from `data/nuscenes` to your_nuscenes_path

2. Prepare the gts semantic occupancy introduced in [Occ3d](https://github.com/Tsinghua-MARS-Lab/Occ3D)

3. Download our generated train/val pickle files and put them in `data/`

    [nuscenes_infos_train_temporal_v3_scene.pkl](https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/)

    [nuscenes_infos_val_temporal_v3_scene.pkl](https://cloud.tsinghua.edu.cn/d/9e231ed16e4a4caca3bd/)

  The dataset should be organized as follows:

```
.
└── data/
    ├── nuscenes            # downloaded from www.nuscenes.org/
    │   ├── lidarseg
    │   ├── maps
    │   ├── samples
    │   ├── sweeps
    │   ├── v1.0-trainval
    │   └── gts             # download from Occ3d
    ├── nuscenes_infos_train_temporal_v3_scene.pkl
    └── nuscenes_infos_val_temporal_v3_scene.pkl
```
### ckpt preparation
Download the pretrained weights from [here](https://drive.google.com/drive/folders/1D1HugOG7JurEqmnQo4XbW_-Ji0chEq-e?usp=sharing) and put them in `ckpts` folder.

## 🏃 Run the code
### (optional) Preprocess resampled data
```
cd resample

python launch.py \
    --dst ../data/resampled_occ \
    --imageset ../data/nuscenes_infos_train_temporal_v3_scene.pkl \
    --data_path ../data/nuscenes
```

### OCC-VAE
```shell
# train 
sh tools/train_vae.sh

# eval
sh tools/eval_vae.sh

# visualize
sh tools/vis_vae.sh
```

### DOME
```shell
# train 
sh tools/train_diffusion.sh 

# eval
sh tools/eval.sh 

# visualize
sh tools/vis_diffusion.sh
```

## 🎫 Acknowledgment
This code draws inspiration from their work. We sincerely appreciate their excellent contribution.
- [OccWorld](https://github.com/wzzheng/OccWorld)
- [Latte](https://github.com/Vchitect/Latte)
- [Vista](https://github.com/OpenDriveLab/Vista.git)
- [PyTorch-VAE](https://github.com/AntixK/PyTorch-VAE)
- [A* serach](https://www.redblobgames.com/pathfinding/a-star/)

## 🖊️ Citation
```
@article{gu2024dome,
  title={Dome: Taming diffusion model into high-fidelity controllable occupancy world model},
  author={Gu, Songen and Yin, Wei and Jin, Bu and Guo, Xiaoyang and Wang, Junming and Li, Haodong and Zhang, Qian and Long, Xiaoxiao},
  journal={arXiv preprint arXiv:2410.10429},
  year={2024}
}
```


