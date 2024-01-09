#!/bin/bash

cd <ROOT>/EPCL/indoor_segmentation/script

### Path
SCANNET_TRAIN=/root/dataset/deepmvs/train
SCANNET_TEST=/root/dataset/deepmvs/test
SCANNET_SEMSEG=/root/dataset/scannet_semseg
SHAPENET=/root/dataset/shapenet_part/shapenetcore_partanno_segmentation_benchmark_v0/
SHAPNETCORE=/root/dataset/shapenetcore/ShapeNetCore.v2/
S3DIS=<ROOT>/EPCL/indoor_segmentation/s3dis # your dataset path
SAVEROOT="<ROOT>/EPCL/indoor_segmentation/output"

### Setup 
MYSHELL="train.sh"
DATE_TIME=`date +"%Y-%m-%d"`
NEPTUNE_PROJ="<YOUR NEPTUNE PROJ>"
COMPUTER="S3DIS-EPCL-00"
export MASTER_ADDR='localhost'
export NODE_RANK=0
export CUDA_VISIBLE_DEVICES=1

### Params
WORKERS=4
NUM_GPUS=1
NUM_TRAIN_BATCH=2
NUM_VAL_BATCH=2
NUM_TEST_BATCH=2


ARCH="epcl"
DATASET="loader_s3dis"
INTRALAYER="PointMixerIntraSetLayer"
INTERLAYER="PointMixerInterSetLayer"
TRANSDOWN="SymmetricTransitionDownBlock"
TRANSUP="SymmetricTransitionUpBlock"

MYCHECKPOINT="${SAVEROOT}/${DATE_TIME}"

if [ ! -d "$MYCHECKPOINT" ]; then
  mkdir -p $MYCHECKPOINT
fi
cp -a $MYSHELL $MYCHECKPOINT
cd ../

### TRAIN
python train_pl.py \
  --MYCHECKPOINT $MYCHECKPOINT --computer $COMPUTER --shell $MYSHELL \
  --MASTER_ADDR $MASTER_ADDR \
  --train_worker $WORKERS --val_worker $WORKERS \
  --NUM_GPUS $NUM_GPUS  \
  --train_batch $NUM_TRAIN_BATCH  \
  --val_batch $NUM_VAL_BATCH  \
  --test_batch $NUM_TEST_BATCH \
  \
  --scannet_train_root $SCANNET_TRAIN  --scannet_test_root $SCANNET_TEST \
  --scannet_semgseg_root $SCANNET_SEMSEG \
  --shapenet_root $SHAPENET  --shapenetcore_root $SHAPNETCORE \
  --s3dis_root $S3DIS \
  \
  --neptune_proj $NEPTUNE_PROJ \
  --epochs 80  --CHECKPOINT_PERIOD 1  --lr 0.1 \
  --dataset $DATASET  --optim 'SGD' \
  \
  --model 'net_epcl' --arch $ARCH  \
  --intraLayer $INTRALAYER  --interLayer $INTERLAYER \
  --transdown  $TRANSDOWN --transup $TRANSUP \
  --nsample 8 16 16 16 16  --drop_rate 0.1  --fea_dim 6  --classes 13 \
  \
  --voxel_size 0.04  --eval_voxel_max 800000  --test_batch 1  --cudnn_benchmark False