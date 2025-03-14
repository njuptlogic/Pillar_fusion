# PillarFusionMultimodal-Fusion-3D-Target-Detection-for-Autonomous-Driving
## OpenPCDet-based 3D target detection for large-scale automated driving with tri-modal (radar,lidar,camera) fusion

<div align="center">
  <img src="illustrations/论文框架图.png" width="700"/>
</div>


> 




## Introduction
![Image 1](illustrations/9_0.png){:style="display:inline-block; width: 500px;"} 
![Image 2](illustrations/9_1.png){:style="display:inline-block; width: 500px;"}


In this paper, we xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
<div align="center">
  <img src="illustrations/pipeline.png" width="800"/>
</div>

## Main results on NUSCENES V1.0-mini
### 3D Object Detection (on NuScenes v1.0-mini)
|  Model  | NDS | mAP |mATE | mASE | mAOE | mAVE| mAAE | Log |
|---------|---------|--------|---------|---------|--------|---------|--------|--------|
|  [Pillar_fusion](https://github.com/njuptlogic/Pillar_fusion/blob/main/tools/cfgs/nuscenes_models/unitr.yaml) | 30.5 | 24.6 | 39.6 | 51.3 | 105.7 | 75.3 | 39.6 | [Log](https://drive.google.com/file/d/10n-eb7vJpXKilV48yHAsywfVaG0Vxg7_/view?usp=drive_link)|
|  [UniTR](https://github.com/Haiyang-W/UniTR/blob/main/tools/cfgs/nuscenes_models/unitr%2Blss.yaml) | 25.8 | 24.1 | 54.1 | 53.4 | 113.4 | 112.1 | 54.7 | [Log](https://drive.google.com/file/d/10n-eb7vJpXKilV48yHAsywfVaG0Vxg7_/view?usp=drive_link)|




### Bev Map Segmentation (on NuScenes validation)
|  Model  | mIoU | Drivable |Ped.Cross.| Walkway |  StopLine  | Carpark |  Divider  |  ckpt | Log |
|---------|----------|--------|--------|--------|--------|---------|--------|---------|--------|


### What's new here?
#### 🔥 Beats previous SOTAs of outdoor multi-modal 3D Object Detection and BEV Segmentation

##### Recommended configuration table
<div align="center">
  <img src="illustrations/8.png" width="700"/>
</div>

##### BEV Map Segmentation
<div align="center">
  <img src="illustrations/9.png" width="700"/>
</div>

#### 🔥 Weight-Sharing among all modalities 
We introduce a modality-agnostic transformer encoder to handle these view-discrepant sensor data for parallel modal-wise representation learning and automatic cross-modal interaction without additional fusion steps.

#### 🔥 Prerequisite for 3D vision foundation models
A weight-shared unified multimodal encoder is a prerequisite for foundation models, especially in the context of 3D perception, unifying information from both images and LiDAR data. This is the first truly multimodal fusion backbone, seamlessly connecting to any 3D detection head.

## Quick Start
we test env both with conda/miniconda/anaconda,and try differnt graphics card (3090 and 4090 are all FINE).
we recommend you train this repo with at leasst 24g graphics memory and 60g general memory for each card

### Installation

```shell
conda create -n pillarfusion python=3.8
conda activate pillarfusion
# Install torch, we only test it in pytorch 1.10
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

or(if you are in CHINA，please install it by the following method)

pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://mirrors.aliyun.com/pytorch-wheels/cu113

git clone https://github.com/njuptlogic/Pillar_fusion.git
cd pillarfusion

# Install extra dependency
pip install -r requirements.txt

# Install nuscenes-devkit
pip install nuscenes-devkit==1.0.5

# Develop
python setup.py develop
```
### The reference dependencies are as follows (conda list results)
You can check your env from [condalist.md](https://drive.google.com/file/d/1ZSDUnbZnMthaENyE6A-VCPRN2hEkx7Dh/view?usp=drive_link)

### Dataset Preparation

* Please download the official [NuScenes 3D object detection dataset](https://www.nuscenes.org/download) and organize the downloaded files as follows: 

Tips:here is the method to download the nuscenes dataset quickly without know much about nuscenes
```shell
cd Pillar_fusion
# the python script in the page can help you download
python download_nuscenes.py

```
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
├── pcdet
├── tools
```

- (optional) To install the Map expansion for bev map segmentation task, please download the files from [Map expansion](https://www.nuscenes.org/download) (Map expansion pack (v1.3)) and copy the files into your nuScenes maps folder, e.g. `/data/nuscenes/v1.0-trainval/maps` as follows:
```
OpenPCDet
├── maps
│   ├── ......
│   ├── boston-seaport.json
│   ├── singapore-onenorth.json
│   ├── singapore-queenstown.json
│   ├── singapore-hollandvillage.json
```

* Generate the data infos by running the following command (it may take several hours): 

```python 
# Create dataset info file, lidar and image gt database
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-trainval \
    --with_cam \
    --with_cam_gt \
    # --share_memory # if use share mem for lidar and image gt sampling (about 24G+143G or 12G+72G)
# share mem will greatly improve your training speed, but need 150G or 75G extra cache mem. 
# NOTE: all the experiments used share memory. Share mem will not affect performance

# you can also test only with mini nuscenes
python -m pcdet.datasets.nuscenes.nuscenes_dataset --func create_nuscenes_infos \
    --cfg_file tools/cfgs/dataset_configs/nuscenes_dataset.yaml \
    --version v1.0-mini \
    --with_cam \
    --with_cam_gt \
```
```shell
if you met this problem
Traceback (most recent call last):
  ...
  File "/lib/python3.8/site-packages/av2/utils/io.py", line 16, in <module>
    import av2.geometry.geometry as geometry_utils
  File "/lib/python3.8/site-packages/av2/geometry/geometry.py", line 11, in <module>
    from av2.utils.typing import NDArrayBool, NDArrayFloat, NDArrayInt
  File "/lib/python3.8/site-packages/av2/utils/typing.py", line 14, in <module>
    NDArrayNumber = np.ndarray[Any, np.dtype[Union[np.integer[Any], np.floating[Any]]]]
TypeError: Type subscription requires python >= 3.9
please turn to  /lib/python3.8/site-packages/av2/utils/typing.py,line 14
change from NDArrayNumber = np.ndarray[Any, np.dtype[Union[np.integer[Any], np.floating[Any]]]]
to
NDArrayNumber = np.ndarray



# when you meet this
# (File "/root/miniconda/envs/xxx/lib/python3.8/site-packages/kornia/geometry/conversions.py",line 556)
if you meet this problem,please try the method below
pip uninstall kornia
pip install kornia==0.6.5
```

* The format of the generated data is as follows:
```
OpenPCDet
├── data
│   ├── nuscenes
│   │   │── v1.0-trainval (or v1.0-mini if you use mini)
│   │   │   │── samples
│   │   │   │── sweeps
│   │   │   │── maps
│   │   │   │── v1.0-trainval  
│   │   │   │── img_gt_database_10sweeps_withvelo
│   │   │   │── gt_database_10sweeps_withvelo
│   │   │   │── nuscenes_10sweeps_withvelo_lidar.npy (optional) # if open share mem
│   │   │   │── nuscenes_10sweeps_withvelo_img.npy (optional) # if open share mem
│   │   │   │── nuscenes_infos_10sweeps_train.pkl  
│   │   │   │── nuscenes_infos_10sweeps_val.pkl
│   │   │   │── nuscenes_dbinfos_10sweeps_withvelo.pkl
├── pcdet
├── tools
```

### Training
## tip:if you want to train the less dataset , please go to the Pillar_fusion/tools/cfgs/nuscenes_models/unitr.yaml,line7

## if you want to use mini nuscenes,turn to tools\cfgs\dataset_configs\nuscenes_dataset.yaml,line 4
VERSION: 'v1.0-trainval' 
change it to 
VERSION: 'v1.0-mini' 
```shell
# alter INTERVAL: 1 to INTERVAL: x (if x is 5,you will get 1/5 dataset to train)
```
Please download pretrained checkpoint from [unitr_pretrain.pth](https://drive.google.com/file/d/1Ly8Gf3DV5ATH8Xw1hRiDgUP9JbVpMPSE/view?usp=sharing) and copy the file under the root folder, eg. `Pillar_fusion/unitr_pretrain.pth`. This file is the weight of pretraining DSVT on Imagenet and Nuimage datasets.

3D object detection:

```shell
# multi-gpu training
## normal
cd tools
bash scripts/dist_train.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr.yaml --sync_bn --pretrained_model ../unitr_pretrain.pth --logger_iter_interval 1000

## add lss
cd tools
bash scripts/dist_train.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr+lss.yaml --sync_bn --pretrained_model ../unitr_pretrain.pth --logger_iter_interval 1000

```
## if you met this problem

bash scripts/dist_train.sh 4 --cfg_file ./cfgs/nuscenes_models/unitr.yaml --sync_bn --pretrained_model ../unitr
_pretrain.pth --logger_iter_interval 1000 
: invalid optionin.sh: line 2: set: -
set: usage: set [-abefhkmnptuvxBCHP] [-o option-name] [--] [arg ...]
scripts/dist_train.sh: line 16: syntax error: unexpected end of file

please try the method before
```shell
sed -i 's/[[:space:]]\+$//' your_script.sh
```
BEV Map Segmentation:

```shell
# multi-gpu training
# note that we don't use image pretrain in BEV Map Segmentation
## normal
cd tools
bash scripts/dist_train.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr_map.yaml --sync_bn --eval_map --logger_iter_interval 1000

## add lss
cd tools
bash scripts/dist_train.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr_map+lss.yaml --sync_bn --eval_map --logger_iter_interval 1000
```

### Testing

3D object detection:

```shell
# multi-gpu testing
## normal
cd tools
bash scripts/dist_test.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr.yaml --ckpt <CHECKPOINT_FILE>

## add LSS
cd tools
bash scripts/dist_test.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr+lss.yaml --ckpt <CHECKPOINT_FILE>
```

BEV Map Segmentation

```shell
# multi-gpu testing
## normal
cd tools
bash scripts/dist_test.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr_map.yaml --ckpt <CHECKPOINT_FILE> --eval_map

## add LSS
cd tools
bash scripts/dist_test.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr_map+lss.yaml --ckpt <CHECKPOINT_FILE> --eval_map
# NOTE: evaluation results will not be logged in *.log, only be printed in the teminal
```

### Cache Testing 
- 🔥If the camera and Lidar parameters of the dataset you are using remain constant, then using our cache mode will not affect performance. You can even cache all mapping calculations during the training phase, which can significantly accelerate your training speed.
- Each sample in Nuscenes will `have some variations in camera parameters`, and during normal inference, we disable the cache mode to ensure result accuracy. However, due to the robustness of our mapping, even in scenarios with camera parameter variations like Nuscenes, the performance will only drop slightly (around 0.4 NDS).
- Cache mode only supports batch_size 1 now, 8x1=8
- Backbone caching will reduce 40% inference latency in our observation.
```shell
# Only for 3D Object Detection
## normal
### cache the mapping computation of multi-modal backbone
cd tools
bash scripts/dist_test.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr_cache.yaml --ckpt <CHECKPOINT_FILE> --batch_size 8

## add LSS
### cache the mapping computation of multi-modal backbone
cd tools
bash scripts/dist_test.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr+LSS_cache.yaml --ckpt <CHECKPOINT_FILE> --batch_size 8

## add LSS
### cache the mapping computation of multi-modal backbone and LSS
cd tools
bash scripts/dist_test.sh 8 --cfg_file ./cfgs/nuscenes_models/unitr+LSS_cache_plus.yaml --ckpt <CHECKPOINT_FILE> --batch_size 8
```
#### Performance of cache testing on NuScenes validation (some variations in camera parameters)
|  Model  | NDS | mAP |mATE | mASE | mAOE | mAVE| mAAE |
|---------|---------|--------|---------|---------|--------|---------|--------|

