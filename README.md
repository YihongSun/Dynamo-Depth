# Dynamo-Depth: Fixing Unsupervised Depth Estimation for Dynamical Scenes
### [Project Page](https://dynamo-depth.github.io) | [Paper](https://arxiv.org/pdf/2310.18887.pdf) | [Video](https://youtu.be/zTmUaudo_5Q)

Official PyTorch implementation for the NeurIPS 2023 paper: "Dynamo-Depth: Fixing Unsupervised Depth Estimation for Dynamical Scenes".

<a href="#license"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"/></a>  

![](assets/teaser.png)

## Table of Contents
- [Installation](#installation)
- [Quick Demo](#quick-demo)
- [Data Preparation](#data-preparation)
- [Training](#training)
  - [Single-GPU](#-single-gpu-training)
  - [Multi-GPU](#-multi-gpu-training)
- [Evaluation](#evaluation)
  - [Depth](#-depth)
  - [Motion Segmentation](#-motion-segmentation)
  - [Odometry](#-odometry)
  - [Visualization](#%EF%B8%8F-visualization)
- [Citation](#citation)

## Installation
The code is tested with `python=3.7`, `torch==1.12.1+cu102` and `torchvision==0.13.1+cu102` on four RTX 2080 Ti GPUs.
```
git clone --recurse-submodules https://github.com/YihongSun/Dynamo-Depth/
cd Dynamo-Depth/
conda create -n dynamo python=3.7
conda activate dynamo
pip install torch==1.12.1 torchvision==0.13.1
pip install matplotlib wandb opencv-python tqdm gdown scikit-image timm==0.6.13
pip install imageio==2.19.3
pip install imageio-ffmpeg==0.4.7
```

## Quick Demo

Please run [quick-demo.ipynb](quick-demo.ipynb) for a quick example inference (This can be used directly after installations).


## Data Preparation
Please refer to the [preprocessing instructions](prepare_data/README.md) for preparing training data for KITTI, Waymo Open, or nuScenes Dataset.


## Training
Training can be done with a single GPU or multiple GPUs (via `torch.nn.parallel.DistributedDataParallel`)

The following are a set of shared arguments to use with any training method.
- `-n <EXP_NAME>` indicates the name of the experiment.
- `-d <DATASET_NAME>` specifies which dataset (`"waymo"`, `"nuscenes"`, or `"kitti"`) to train on, and the default is `"waymo"`.
- `-l </PATH/TO/MODEL/CKPT>` indicates which model checkpoint to be load before training.
- `--depth_model <MODEL_NAME>` specifies which depth model (`"litemono"` or `"monodepthv2"`) to train, with default `"litemono"`.

### ⏳ Single GPU Training
For instance, to train w/ 1 GPU on Waymo Dataset from scratch:
```
python3 train.py -d waymo -n waymo_example_run 
```

### ⏳ Multi-GPU Training
For instance, to train w/ 4 GPUs on Waymo Dataset
```
python -m torch.distributed.launch --nproc_per_node=4 train.py --cuda_ids 0 1 2 3 -d waymo -n waymo_example_run_parallel 
```
Note: All experiments are ran with 4 RTX 2080 Ti GPUs via `torch.nn.parallel.DistributedDataParallel`. Learning rate and scheduler step size should be adjusted accordingly when training with a single GPU (See [options.py](options.py) for details).


## Evaluation
Scripts for evaluation are found in `eval/`, including [depth](eval/depth.py), [motion segmentation](eval/motion_segmentation.py), [odometry](eval/odometry.py), and [visualization](eval/visualize.py).

The following are a set of shared arguments to use with any of the evaluation scripts above.
- `-l </PATH/TO/MODEL/CKPT>` indicates which model checkpoint to be evaluated.
- `--depth_model <MODEL_NAME>` specifies which depth model (`"litemono"` or `"monodepthv2"`) to use, with default `"litemono"`.
- `-d <DATASET_NAME>` specifies which dataset (`"waymo"`, `"nuscenes"`, or `"kitti"`) to evaluate on, and the default is `"waymo"`.
- `--eval_dir` defines the output directory where the results would be saved, with default `"./outputs"`.

**Note**: To access the trained models for Waymo Open, please fill out the [Google Form](https://forms.gle/nRezg2gr7QDXJGcA9), and [raise an issue](https://github.com/YihongSun/Dynamo-Depth/issues/new) if we don't get back to you in two days. Please note that Waymo open dataset is under strict non-commercial license so we are not allowed to share the model with you if it will used for any profit-oriented activities.

### 📊 Depth
[eval/depth.py](eval/depth.py) evaluates monocular depth estimation, with results saved in `./outputs/<CKPT>_<DATASET>/depth/`.

🔹 To replicate the results reported in the paper (Table 1 and 2), run the following lines. 
```
## === Missing checkpoints will be downloaded automatically === ##

python3 eval/depth.py -l ckpt/W_Dynamo-Depth                                  ## please fill out the form for ckpt!!
python3 eval/depth.py -l ckpt/W_Dynamo-Depth_MD2 --depth_model monodepthv2    ## please fill out the form for ckpt!!
python3 eval/depth.py -l ckpt/N_Dynamo-Depth -d nuscenes
python3 eval/depth.py -l ckpt/N_Dynamo-Depth_MD2 --depth_model monodepthv2 -d nuscenes
python3 eval/depth.py -l ckpt/K_Dynamo-Depth -d kitti
python3 eval/depth.py -l ckpt/K_Dynamo-Depth_MD2 --depth_model monodepthv2 -d kitti
```

|     Model     |   Dataset |  Abs Rel  |   Sq Rel  |    RMSE   |  RMSE log | delta < 1.25 | delta < 1.25<sup>2</sup> | delta < 1.25<sup>3</sup> |
|:-------------------------:|:------:|:---------:|:---------:|:---------:|:---------:|:------------:|:--------------:|:--------------:|
|  [K_Dynamo-Depth_MD2](https://drive.google.com/file/d/1SLQcCQplfAtqeWUD4TQc42aGpevViTGX/view?usp=sharing)  |  KITTI  | 0.120  |  0.864  |  4.850  |  0.195  |  0.858  |  0.956  |  0.982   |
|  [K_Dynamo-Depth](https://drive.google.com/file/d/1b1kwxqUquFbSMU9WLAr6_pIbj1HxoWLJ/view?usp=share_link)(*)  |  KITTI   | 0.112  |  0.768  |  4.528  |  0.184  |  0.874  |  0.961  |  0.984   |
|  [N_Dynamo-Depth_MD2](https://drive.google.com/file/d/1t0Z_2hD0raAi4vDK_VZFXIcwcTFx0elU/view?usp=sharing)  |  nuScenes  |  0.193  |  2.285  |  7.357  |  0.287  |  0.765  |  0.885  |  0.935  |
|  [N_Dynamo-Depth](https://drive.google.com/file/d/1oqQVFyGxo_SxclpinrBlwGSE1gEfVAZY/view?usp=sharing)  |  nuScenes   |  0.179  |  2.118  |  7.050  |  0.271  |  0.787  |  0.896  |  0.940  |
|  W_Dynamo-Depth_MD2(*†)|  Waymo  |  0.137  |  1.502  |  6.582  |  0.194  |  0.833  |  0.951  |  0.982  |
|  W_Dynamo-Depth(*†)|  Waymo   | 0.121  |  1.216  |  6.003  |  0.176  |  0.861  |  0.962  |  0.986   |

(*) Very minor differences compared to the results in the paper. Rest of the checkpoints are consistent with the paper.  
(†) Please refer to the note above for obtaining access to the models trained on Waymo Open Dataset.

🔹 To replicate the results reported in the Appendix (Table 6 and 7), run the following lines.
```
## === Missing checkpoints will be downloaded automatically === ##

python3 eval/depth.py -l ckpt/N_Dynamo-Depth -d nuscenes --split nuscenes_dayclear
python3 eval/depth.py -l ckpt/N_Dynamo-Depth_MD2 --depth_model monodepthv2 -d nuscenes --split nuscenes_dayclear
```
Note that by adding `--split nuscenes_dayclear`, we evaluate on the nuScenes day-clear subset as defined in `splits/nuscenes_dayclear/test_files.txt` instead of the original `splits/nuscenes/test_files.txt`

### 📊 Motion Segmentation
[eval/motion_segmentation.py](eval/motion_segmentation.py) evaluates binary motion segmentation, with results saved in `./outputs/<CKPT>_<DATASET>/mot_seg/`.

🔹 To replicate the results reported in the paper (Figure 4 and 8), run the following line.
```
## === Missing checkpoints will be downloaded automatically === ##

python3 eval/motion_segmentation.py -l ckpt/W_Dynamo-Depth                         ## please fill out the form for ckpt!!
python3 eval/motion_segmentation.py -l ckpt/N_Dynamo-Depth -d nuscenes --split nuscenes_dayclear
```

### 📊 Odometry
[eval/odometry.py](eval/odometry.py) evaluates odometry, with results saved in `./outputs/<CKPT>_<DATASET>/odometry/`.

🔹 To replicate the results reported in the Appendix (Table 8), run the following line.
```
## === Missing checkpoints will be downloaded automatically === ##

python3 eval/odometry.py -l ckpt/W_Dynamo-Depth                                    ## please fill out the form for ckpt!!                                  
python3 eval/odometry.py -l ckpt/W_Dynamo-Depth_MD2 --depth_model monodepthv2      ## please fill out the form for ckpt!!     
python3 eval/odometry.py -l ckpt/N_Dynamo-Depth -d nuscenes --split nuscenes_dayclear
python3 eval/odometry.py -l ckpt/N_Dynamo-Depth_MD2 --depth_model monodepthv2 -d nuscenes --split nuscenes_dayclear
```

### 🖼️ Visualization
[eval/visualize.py](eval/visualize.py) visualize model performances, with results saved  in `./outputs/<CKPT>_<DATASET>/vis/`.

🔹 To generate the _Qualitative Results_ in the [Project Page](https://dynamo-depth.github.io), run the following line.
```
## === Missing checkpoints will be downloaded automatically === ##

python3 eval/visualize.py -l ckpt/W_Dynamo-Depth                                   ## please fill out the form for ckpt!!     
python3 eval/visualize.py -l ckpt/N_Dynamo-Depth -d nuscenes
```


## Citation
If you find our work useful in your research, please consider citing our paper:
```
@inproceedings{sun2023dynamodepth,
  title={Dynamo-Depth: Fixing Unsupervised Depth Estimation for Dynamical Scenes},
  author={Yihong Sun and Bharath Hariharan},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
