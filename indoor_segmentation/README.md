# EPCL-INDOOR-SEGMENTATION

## 1. Requirements
Code has been tested with Ubuntu 20.04, GCC 9.4.0, Python 3.9.12, PyTorch 1.11.0, CUDA 11.3 and RTX 3090.
***
First, it is recommended to create a new environment and install PyTorch and torchvision. Next, please use the following command for installation.

```
pip install -r requirements.txt

# Install pointops2
cd lib/pointops2/
python setup.py install

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

In addition, you need to install CLIP according to [CLIP](https://github.com/openai/CLIP).

## 2. Datasets

We use s3dis in this work. You can download the data from [PAConv](https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg#dataset). After you download the dataset, remember to modify the DATA_PATH in `script/train.sh`.

**Resulting dataset structure**

```
${data_dir}
├── s3dis
    ├── list
    │   ├── s3dis_names.txt
    │   ├── val5.txt
    │   └── ...
    ├── trainval
    │   ├── 00001071.h5
    │   └── ...
    └── trainval_fullarea
        ├── Area_1_conferenceRoom_1.npy
        └── ...
```
## 3. Pretrain Model
|  Task | Dataset | mAcc | mIoU | Download |      
|  ----- | ----- | -----|  -----| -----|
|  Indoor segmentation | S3DIS (Area5) | 77.8 | 71.5 |[here](https://drive.google.com/file/d/1PVyLlS4R1fUXs9HbLuN7OAMoMUlGXzTg/view?usp=drive_link) |

## 4. Usage
Training on S3DIS, run:
```
cd scripts
bash train.sh
```
Testing on S3DIS, run:
```
cd scripts
bash test.sh
```

## Acknowledgements

The code for this task is built upon [PointMixer](https://github.com/LifeBeyondExpectations/ECCV22-PointMixer).
