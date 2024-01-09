# EPCL-DETECTION

## 1. Requirements

Code has been tested with Ubuntu 20.04, GCC 9.4.0, Python 3.8.18, PyTorch 1.9.1, CUDA 11.1 and RTX 3090.
***
First, it is recommended to create a new environment and install PyTorch and torchvision. Next, please use the following command for installation.

```
pip install -r requirements.txt

# install spconv
pip install spconv-cu113

python setup.py develop
# if you meet some pakage not matched errors, just pip install them individually before install pcdet

# rotate iou ops
cd pcdet/ops/rotated_iou/cuda_op
python setup.py install

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

In addition, you need to install CLIP and Minkowski Engine according to [CLIP](https://github.com/openai/CLIP) and [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine).

## 2. Datasets

We use ScanNetV2 in this work. You can download the processed data from [CAGroup3D repo](https://github.com/Haiyang-W/CAGroup3D). After you download the dataset, remember to modify the DATA_PATH in `tools/cfgs/dataset_configs/scannet_dataset.yaml`.

## 3. Pretrain Model
|  Task | Dataset | mAP@0.25 | mAP@0.50| Download |      
|  ----- | ----- | -----|  -----| -----|
|  Detection | ScanNetV2 | 73.7 | 61.1 |[here](https://drive.google.com/file/d/1dGM8cDjXUQ_8j0nRNjTEUyvggswPon0D/view?usp=drive_link) |

## 4. Usage
Training on ScanNetV2, and we set num_gpus x batch_size to 4x4, run:
```
cd scripts
bash train.sh
```
Testing on ScanNetV2, run:
```
cd scripts
bash test.sh
```

## Acknowledgements

The code for this task is built upon [CAGroup3D](https://github.com/Haiyang-W/CAGroup3D).
