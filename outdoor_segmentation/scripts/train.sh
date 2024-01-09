cd <ROOT>/EPCL/outdoor_segmeantation

export CUDA_VISIBLE_DEVICES=0,1,2,3

NGPUS=4
cfg_name=EPCL
extra_tag=train_${cfg_name}
cfg_file=tools/cfgs/voxel/semantic_kitti/${cfg_name}.yaml

set -x

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

python -m torch.distributed.launch \
--nproc_per_node=${NGPUS} \
--rdzv_endpoint=localhost:${PORT} \
train.py \
--launcher pytorch \
--cfg_file ${cfg_file} \
--extra_tag ${extra_tag} \

echo 'dist_train finished!'


