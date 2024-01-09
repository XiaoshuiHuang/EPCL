cd <YOUR_ROOT_DIR>/EPCL/detection

set -x

export CUDA_VISIBLE_DEVICES=0,1,2,3

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

num_gpus=4
extra_tag=EPCL
cfg_file=./tools/cfgs/scannet_models/${extra_tag}.yaml

python -m torch.distributed.launch \
--nproc_per_node ${num_gpus} \
--rdzv_endpoint localhost:${PORT} \
./tools/train.py \
--launcher pytorch \
--cfg_file  ${cfg_file} \
--ckpt_save_interval 1 \
--extra_tag $extra_tag \
--num_epochs_to_eval 12 \
--fix_random_seed \
