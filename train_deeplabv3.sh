#!/usr/bin/env bash

# train
# export NGPUS=2
# export CUDA_VISIBLE_DEVICES="1, 2"
# export OMP_NUM_THREADS=1
# export PYTHONPATH=./:$PYTHONPATH
# python -m torch.distributed.launch --nproc_per_node=$NGPUS \
#     scripts/train.py \
#     --backbone=resnet50 \
#     --dataset=pascal_voc \
#     --dataset_root=/home/zhfeing/datasets/voc \
#     --pretrained_dir=/home/zhfeing/model-zoo \
#     --batch-size=4 \
#     --epochs=1 \
#     --save-dir=./checkpoints/ \
#     --log-dir=./logs/

export NGPUS=1
export CUDA_VISIBLE_DEVICES="1"
export OMP_NUM_THREADS=1
export PYTHONPATH=./:$PYTHONPATH
python -m torch.distributed.launch --nproc_per_node=$NGPUS \
    scripts/train.py \
    --backbone=resnet50 \
    --dataset=pascal_voc \
    --dataset_root=/home/zhfeing/datasets/voc \
    --pretrained_dir=/home/zhfeing/model-zoo \
    --batch-size=20 \
    --epochs=50 \
    --save-dir=./checkpoints/ \
    --log-dir=./logs/ \
    --log-iter=3
echo Done
