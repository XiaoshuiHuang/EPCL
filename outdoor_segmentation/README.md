# EPCL-OUTDOOR-SEGMENTATION

## 1. Requirements
Code has been tested with Ubuntu 20.04, GCC 9.4.0, Python 3.7.16, PyTorch 1.10.0, CUDA 11.3 and RTX 3090.
***
First, it is recommended to create a new environment and install PyTorch and torchvision. Next, please use the following command for installation.

```
# Install sparsehash
cd package
unzip sparsehash.zip
unzip torchsparse.zip
mv sparsehash-master/ sparsehash/
cd sparsehash/
./configure
make
make install

# Compile torchsparse
cd ..
pip install ./torchsparse

pip install -r requirements.txt

pip uninstall setuptools
pip install setuptools==59.5.0

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

In addition, you need to install CLIP according to [CLIP](https://github.com/openai/CLIP).

## 2. Datasets

To install the [SemanticKITTI](http://semantic-kitti.org/index) dataset, download the data, annotations, and other files from http://semantic-kitti.org/dataset. Unpack the compressed file(s) into `./data_root/semantickitti` and re-organize the data structure. Finally remember to modify the DATA_PATH in `./tools/cfgs/voxel/semantic_kitti/EPCL.yaml`. Your folder structure should end up looking like this:
```
└── SemanticKitti  
    └── dataset
        ├── velodyne <- contains the .bin files; a .bin file contains the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 21
        ├── labels   <- contains the .label files; a .label file contains the labels of the points in a point cloud
        │    └── 00
        │    └── ···
        │    └── 10
        ├── calib
        │    └── 00
        │    └── ···
        │    └── 21
        └── semantic-kitti.yaml
```

## 3. Pretrain Model
|  Task | Dataset | mIoU(val.)| Download |      
|  ----- | ----- | -----|  -----|
|  Outdoor segmentation | SemanticKITTI | 72.4 |[here](https://drive.google.com/file/d/1ZyY7pVeJeHflkb3ox4fZxhWEYbwuhCwS/view?usp=drive_link) |

## 4. Usage
Training on SemanticKITTI, run:
```
bash scripts/train.sh
```
Testing  on the val set of SemanticKITTI dataset, run:
```
bash scripts/val.sh
```

## Acknowledgements

The code for this task is built upon [OpenPCSeg](https://github.com/PJLab-ADG/OpenPCSeg).
