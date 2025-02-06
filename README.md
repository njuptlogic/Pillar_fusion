# PillarFusionMultimodal-Fusion-3D-Target-Detection-for-Autonomous-Driving
## 作者人生第一个大型的自动驾驶多模态融合仓库，能力有限，如有不足，还请原谅
## 代码仓库还在完善，完善结束后，会更新完整复现操作


<div align="center">
  <img src="assets/Figure1.png" width="700"/>
</div>


> 




## Introduction

<div align="center">
  <img src="assets/Figure2.png" width="500"/>
</div>

In this paper, we xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
<div align="center">
  <img src="assets/Figure3.png" width="800"/>
</div>

## Main results
### 3D Object Detection (on NuScenes validation)
|  Model  | NDS | mAP |mATE | mASE | mAOE | mAVE| mAAE | ckpt | Log |
|---------|---------|--------|---------|---------|--------|---------|--------|--------|--------|





### Bev Map Segmentation (on NuScenes validation)
|  Model  | mIoU | Drivable |Ped.Cross.| Walkway |  StopLine  | Carpark |  Divider  |  ckpt | Log |
|---------|----------|--------|--------|--------|--------|---------|--------|---------|--------|


### What's new here?
#### 🔥 Beats previous SOTAs of outdoor multi-modal 3D Object Detection and BEV Segmentation

##### 3D Object Detection
<div align="left">
  <img src="assets/Figure4.png" width="700"/>
</div>

##### BEV Map Segmentation
<div align="left">
  <img src="assets/Figure5.png" width="700"/>
</div>

#### 🔥 Weight-Sharing among all modalities 
We introduce a modality-agnostic transformer encoder to handle these view-discrepant sensor data for parallel modal-wise representation learning and automatic cross-modal interaction without additional fusion steps.

#### 🔥 Prerequisite for 3D vision foundation models
A weight-shared unified multimodal encoder is a prerequisite for foundation models, especially in the context of 3D perception, unifying information from both images and LiDAR data. This is the first truly multimodal fusion backbone, seamlessly connecting to any 3D detection head.

## Quick Start
### Installation

```shell
conda create -n pillarfusion python=3.8
# Install torch, we only test it in pytorch 1.10
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/njuptlogic/Pillar_fusion.git
cd pillarfusion

# Install extra dependency
pip install -r requirements.txt

# Install nuscenes-devkit
pip install nuscenes-devkit==1.0.5

# Develop
python setup.py develop
```
### 参考依赖库如下（conda list结果）
| Name                      | Version                  | Build                 | Channel                                                                 |
|---------------------------|--------------------------|-----------------------|-------------------------------------------------------------------------|
| _libgcc_mutex              | 0.1                      | main                  | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| _openmp_mutex              | 5.1                      | 1_gnu                 | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| anyio                     | 4.5.2                    | pypi_0                | pypi                                                                    |
| argcomplete               | 3.5.1                    | pypi_0                | pypi                                                                    |
| argon2-cffi               | 23.1.0                   | pypi_0                | pypi                                                                    |
| argon2-cffi-bindings      | 21.2.0                   | pypi_0                | pypi                                                                    |
| arrow                     | 1.3.0                    | pypi_0                | pypi                                                                    |
| asttokens                 | 2.4.1                    | pypi_0                | pypi                                                                    |
| async-lru                 | 2.0.4                    | pypi_0                | pypi                                                                    |
| attrs                     | 24.2.0                   | pypi_0                | pypi                                                                    |
| av                        | 12.3.0                   | pypi_0                | pypi                                                                    |
| av2                       | 0.2.1                    | pypi_0                | pypi                                                                    |
| babel                     | 2.16.0                   | pypi_0                | pypi                                                                    |
| backcall                  | 0.2.0                    | pypi_0                | pypi                                                                    |
| beautifulsoup4            | 4.12.3                   | pypi_0                | pypi                                                                    |
| bleach                    | 6.1.0                    | pypi_0                | pypi                                                                    |
| ca-certificates           | 2024.9.24                | h06a4308_0            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| cachetools                | 5.5.0                    | pypi_0                | pypi                                                                    |
| ccimport                  | 0.4.4                    | pypi_0                | pypi                                                                    |
| certifi                   | 2024.8.30                | pypi_0                | pypi                                                                    |
| cffi                      | 1.17.1                   | pypi_0                | pypi                                                                    |
| charset-normalizer        | 3.4.0                    | pypi_0                | pypi                                                                    |
| click                     | 8.1.7                    | pypi_0                | pypi                                                                    |
| colorlog                  | 6.9.0                    | pypi_0                | pypi                                                                    |
| comm                      | 0.2.2                    | pypi_0                | pypi                                                                    |
| contourpy                 | 1.1.1                    | pypi_0                | pypi                                                                    |
| cumm-cu113                | 0.4.11                   | pypi_0                | pypi                                                                    |
| cycler                    | 0.12.1                   | pypi_0                | pypi                                                                    |
| cython                    | 3.0.11                   | pypi_0                | pypi                                                                    |
| debugpy                   | 1.8.7                    | pypi_0                | pypi                                                                    |
| decorator                 | 5.1.1                    | pypi_0                | pypi                                                                    |
| defusedxml                | 0.7.1                    | pypi_0                | pypi                                                                    |
| descartes                 | 1.1.0                    | pypi_0                | pypi                                                                    |
| distlib                   | 0.3.9                    | pypi_0                | pypi                                                                    |
| easydict                  | 1.13                     | pypi_0                | pypi                                                                    |
| exceptiongroup            | 1.2.2                    | pypi_0                | pypi                                                                    |
| executing                 | 2.1.0                    | pypi_0                | pypi                                                                    |
| fastjsonschema            | 2.20.0                   | pypi_0                | pypi                                                                    |
| filelock                  | 3.16.1                   | pypi_0                | pypi                                                                    |
| fire                      | 0.7.0                    | pypi_0                | pypi                                                                    |
| fonttools                 | 4.54.1                   | pypi_0                | pypi                                                                    |
| fqdn                      | 1.5.1                    | pypi_0                | pypi                                                                    |
| gpuinfo                   | 1.0.0a7                  | pypi_0                | pypi                                                                    |
| h11                       | 0.14.0                   | pypi_0                | pypi                                                                    |
| httpcore                  | 1.0.6                    | pypi_0                | pypi                                                                    |
| httpx                     | 0.27.2                   | pypi_0                | pypi                                                                    |
| idna                      | 3.10                     | pypi_0                | pypi                                                                    |
| imageio                   | 2.35.1                   | pypi_0                | pypi                                                                    |
| importlib-metadata        | 8.5.0                    | pypi_0                | pypi                                                                    |
| importlib-resources       | 6.4.5                    | pypi_0                | pypi                                                                    |
| ipykernel                 | 6.29.5                   | pypi_0                | pypi                                                                    |
| ipython                   | 8.12.3                   | pypi_0                | pypi                                                                    |
| ipywidgets                | 8.1.5                    | pypi_0                | pypi                                                                    |
| isoduration               | 20.11.0                  | pypi_0                | pypi                                                                    |
| jedi                      | 0.19.1                   | pypi_0                | pypi                                                                    |
| jinja2                    | 3.1.4                    | pypi_0                | pypi                                                                    |
| joblib                    | 1.4.2                    | pypi_0                | pypi                                                                    |
| json5                     | 0.9.25                   | pypi_0                | pypi                                                                    |
| jsonpointer               | 3.0.0                    | pypi_0                | pypi                                                                    |
| jsonschema                | 4.23.0                   | pypi_0                | pypi                                                                    |
| jsonschema-specifications | 2023.12.1                | pypi_0                | pypi                                                                    |
| jupyter                   | 1.1.1                    | pypi_0                | pypi                                                                    |
| jupyter-client            | 8.6.3                    | pypi_0                | pypi                                                                    |
| jupyter-console           | 6.6.3                    | pypi_0                | pypi                                                                    |
| jupyter-core              | 5.7.2                    | pypi_0                | pypi                                                                    |
| jupyter-events            | 0.10.0                   | pypi_0                | pypi                                                                    |
| jupyter-lsp               | 2.2.5                    | pypi_0                | pypi                                                                    |
| jupyter-server            | 2.14.2                   | pypi_0                | pypi                                                                    |
| jupyter-server-terminals  | 0.5.3                    | pypi_0                | pypi                                                                    |
| jupyterlab                | 4.2.5                    | pypi_0                | pypi                                                                    |
| jupyterlab-pygments       | 0.3.0                    | pypi_0                | pypi                                                                    |
| jupyterlab-server         | 2.27.3                   | pypi_0                | pypi                                                                    |
| jupyterlab-widgets        | 3.0.13                   | pypi_0                | pypi                                                                    |
| kiwisolver                | 1.4.7                    | pypi_0                | pypi                                                                    |
| kornia                    | 0.7.3                    | pypi_0                | pypi                                                                    |
| kornia-rs                 | 0.1.7                    | pypi_0                | pypi                                                                    |
| lark                      | 1.2.2                    | pypi_0                | pypi                                                                    |
| lazy-loader               | 0.4                      | pypi_0                | pypi                                                                    |
| ld_impl_linux-64          | 2.40                     | h12ee557_0            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| libffi                    | 3.4.4                    | h6a678d5_1            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| libgcc-ng                 | 11.2.0                   | h1234567_1            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| libgomp                   | 11.2.0                   | h1234567_1            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| libstdcxx-ng              | 11.2.0                   | h1234567_1            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| llvmlite                  | 0.31.0                   | pypi_0                | pypi                                                                    |
| markdown-it-py            | 3.0.0                    | pypi_0                | pypi                                                                    |
| markupsafe                | 2.1.5                    | pypi_0                | pypi                                                                    |
| matplotlib                | 3.7.5                    | pypi_0                | pypi                                                                    |
| matplotlib-inline         | 0.1.7                    | pypi_0                | pypi                                                                    |
| mdurl                     | 0.1.2                    | pypi_0                | pypi                                                                    |
| mistune                   | 3.0.2                    | pypi_0                | pypi                                                                    |
| motmetrics                | 1.4.0                    | pypi_0                | pypi                                                                    |
| nbclient                  | 0.10.0                   | pypi_0                | pypi                                                                    |
| nbconvert                 | 7.16.4                   | pypi_0                | pypi                                                                    |
| nbformat                  | 5.10.4                   | pypi_0                | pypi                                                                    |
| ncurses                   | 6.4                      | h6a678d5_0            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| nest-asyncio              | 1.6.0                    | pypi_0                | pypi                                                                    |
| networkx                  | 3.1                      | pypi_0                | pypi                                                                    |
| ninja                     | 1.11.1.1                 | pypi_0                | pypi                                                                    |
| notebook                  | 7.2.2                    | pypi_0                | pypi                                                                    |
| notebook-shim             | 0.2.4                    | pypi_0                | pypi                                                                    |
| nox                       | 2024.10.9                | pypi_0                | pypi                                                                    |
| numba                     | 0.48.0                   | pypi_0                | pypi                                                                    |
| numpy                     | 1.23.3                   | pypi_0                | pypi                                                                    |
| nuscenes-devkit           | 1.0.5                    | pypi_0                | pypi                                                                    |
| opencv-python             | 4.10.0.84                | pypi_0                | pypi                                                                    |
| openssl                   | 3.0.15                   | h5eee18b_0            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| overrides                 | 7.7.0                    | pypi_0                | pypi                                                                    |
| packaging                 | 24.1                     | pypi_0                | pypi                                                                    |
| pandas                    | 2.0.3                    | pypi_0                | pypi                                                                    |
| pandocfilters             | 1.5.1                    | pypi_0                | pypi                                                                    |
| parso                     | 0.8.4                    | pypi_0                | pypi                                                                    |
| pccm                      | 0.4.16                   | pypi_0                | pypi                                                                    |
| pcdet                     | 0.6.0+81c323e            | dev_0                 | <develop>                                                              |
| pexpect                   | 4.9.0                    | pypi_0                | pypi                                                                    |
| pickleshare               | 0.7.5                    | pypi_0                | pypi                                                                    |
| pillow                    | 10.4.0                   | pypi_0                | pypi                                                                    |
| pip                       | 24.2                     | py38h06a4308_0        | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| pkgutil-resolve-name      | 1.3.10                   | pypi_0                | pypi                                                                    |
| platformdirs              | 4.3.6                    | pypi_0                | pypi                                                                    |
| portalocker               | 2.10.1                   | pypi_0                | pypi                                                                    |
| prometheus-client         | 0.21.0                   | pypi_0                | pypi                                                                    |
| prompt-toolkit            | 3.0.48                   | pypi_0                | pypi                                                                    |
| protobuf                  | 5.28.3                   | pypi_0                | pypi                                                                    |
| psutil                    | 6.1.0                    | pypi_0                | pypi                                                                    |
| ptyprocess                | 0.7.0                    | pypi_0                | pypi                                                                    |
| pure-eval                 | 0.2.3                    | pypi_0                | pypi                                                                    |
| pyarrow                   | 17.0.0                   | pypi_0                | pypi                                                                    |
| pybind11                  | 2.13.6                   | pypi_0                | pypi                                                                    |
| pycocotools               | 2.0                      | pypi_0                | pypi                                                                    |
| pycparser                 | 2.22                     | pypi_0                | pypi                                                                    |
| pygments                  | 2.18.0                   | pypi_0                | pypi                                                                    |
| pyparsing                 | 3.1.4                    | pypi_0                | pypi                                                                    |
| pyproj                    | 3.5.0                    | pypi_0                | pypi                                                                    |
| pyquaternion              | 0.9.9                    | pypi_0                | pypi                                                                    |
| python                    | 3.8.20                   | he870216_0            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| python-dateutil           | 2.9.0.post0              | pypi_0                | pypi                                                                    |
| python-json-logger        | 2.0.7                    | pypi_0                | pypi                                                                    |
| pytz                      | 2024.2                   | pypi_0                | pypi                                                                    |
| pywavelets                | 1.4.1                    | pypi_0                | pypi                                                                    |
| pyyaml                    | 6.0.2                    | pypi_0                | pypi                                                                    |
| pyzmq                     | 26.2.0                   | pypi_0                | pypi                                                                    |
| readline                  | 8.2                      | h5eee18b_0            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| referencing               | 0.35.1                   | pypi_0                | pypi                                                                    |
| requests                  | 2.32.3                   | pypi_0                | pypi                                                                    |
| rfc3339-validator         | 0.1.4                    | pypi_0                | pypi                                                                    |
| rfc3986-validator         | 0.1.1                    | pypi_0                | pypi                                                                    |
| rich                      | 13.9.3                   | pypi_0                | pypi                                                                    |
| rpds-py                   | 0.20.0                   | pypi_0                | pypi                                                                    |
| scikit-image              | 0.21.0                   | pypi_0                | pypi                                                                    |
| scikit-learn              | 1.3.2                    | pypi_0                | pypi                                                                    |
| scipy                     | 1.10.1                   | pypi_0                | pypi                                                                    |
| send2trash                | 1.8.3                    | pypi_0                | pypi                                                                    |
| setuptools                | 75.1.0                   | py38h06a4308_0        | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| shapely                   | 1.8.4                    | pypi_0                | pypi                                                                    |
| sharedarray               | 3.0.0                    | pypi_0                | pypi                                                                    |
| six                       | 1.16.0                   | pypi_0                | pypi                                                                    |
| sniffio                   | 1.3.1                    | pypi_0                | pypi                                                                    |
| soupsieve                 | 2.6                      | pypi_0                | pypi                                                                    |
| spconv-cu113              | 2.3.6                    | pypi_0                | pypi                                                                    |
| sqlite                    | 3.45.3                   | h5eee18b_0            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| stack-data                | 0.6.3                    | pypi_0                | pypi                                                                    |
| tensorboardx              | 2.6.2.2                  | pypi_0                | pypi                                                                    |
| termcolor                 | 2.4.0                    | pypi_0                | pypi                                                                    |
| terminado                 | 0.18.1                   | pypi_0                | pypi                                                                    |
| threadpoolctl             | 3.5.0                    | pypi_0                | pypi                                                                    |
| tifffile                  | 2023.7.10                | pypi_0                | pypi                                                                    |
| tinycss2                  | 1.4.0                    | pypi_0                | pypi                                                                    |
| tk                        | 8.6.14                   | h39e8969_0            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| tomli                     | 2.0.2                    | pypi_0                | pypi                                                                    |
| torch                     | 1.10.1+cu113             | pypi_0                | pypi                                                                    |
| torch-scatter             | 2.1.2                    | pypi_0                | pypi                                                                    |
| torchvision               | 0.11.2+cu113             | pypi_0                | pypi                                                                    |
| tornado                   | 6.4.1                    | pypi_0                | pypi                                                                    |
| tqdm                      | 4.66.6                   | pypi_0                | pypi                                                                    |
| traitlets                 | 5.14.3                   | pypi_0                | pypi                                                                    |
| types-python-dateutil     | 2.9.0.20241003           | pypi_0                | pypi                                                                    |
| typing-extensions         | 4.12.2                   | pypi_0                | pypi                                                                    |
| tzdata                    | 2024.2                   | pypi_0                | pypi                                                                    |
| uri-template              | 1.3.0                    | pypi_0                | pypi                                                                    |
| urllib3                   | 2.2.3                    | pypi_0                | pypi                                                                    |
| virtualenv                | 20.27.1                  | pypi_0                | pypi                                                                    |
| wcwidth                   | 0.2.13                   | pypi_0                | pypi                                                                    |
| webcolors                 | 24.8.0                   | pypi_0                | pypi                                                                    |
| webencodings              | 0.5.1                    | pypi_0                | pypi                                                                    |
| websocket-client          | 1.8.0                    | pypi_0                | pypi                                                                    |
| wheel                     | 0.44.0                   | py38h06a4308_0        | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| widgetsnbextension        | 4.0.13                   | pypi_0                | pypi                                                                    |
| xmltodict                 | 0.14.2                   | pypi_0                | pypi                                                                    |
| xz                        | 5.4.6                    | h5eee18b_1            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |
| zipp                      | 3.20.2                   | pypi_0                | pypi                                                                    |
| zlib                      | 1.2.13                   | h5eee18b_1            | https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main                 |


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

