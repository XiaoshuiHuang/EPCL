# EPCL-CLASSIFICATION

## 1. Requirements

Code has been tested with Ubuntu 20.04, GCC 9.4.0, Python 3.8.18, PyTorch 1.11.0, CUDA 11.3 and RTX 3090.
***
First, it is recommended to create a new environment and install PyTorch and torchvision. Next, please use the following command for installation.

```
pip install -r requirements.txt

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```
In addition, you need to install CLIP according to [CLIP](https://github.com/openai/CLIP).

## 2. Datasets

We use ModelNet40 in this work. 
```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  ├── modelnet40_test_8192pts_fps.dat
```
Download: You can download the processed data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or download from the [official website](https://modelnet.cs.princeton.edu/#) and process it by yourself.

## 3. Pretrain Models
|  Task | Dataset | Acc. | Download |      
|  ----- | ----- | -----|  -----|
|  Classification | ModelNet40 | 92.9% | [here](https://drive.google.com/file/d/1qcYKq-ZoQJ4JHhWYCLhmBCiqigzaY_DO/view?usp=drive_link) |

## 4. Usage
Train on ModelNet40, run:
```
python main.py --config cfgs/modelnet.yaml --finetune_model
```
Voting on ModelNet40, run:
```
python main.py --test --config cfgs/modelnet.yaml --ckpts checkpoints/best_model.pth
```

## Acknowledgements

The code for this task is built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE). 

