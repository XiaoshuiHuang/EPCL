cd <YOUR_ROOT_DIR>/EPCL/detection

cfg_file=./tools/cfgs/scannet_models/EPCL.yaml
ckpt=./checkpoints/checkpoint_epoch_12.pth
python ./tools/test.py --cfg_file ${cfg_file} --ckpt ${ckpt} \