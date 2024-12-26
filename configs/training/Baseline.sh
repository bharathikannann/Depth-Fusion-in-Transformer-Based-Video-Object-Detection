#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=/output/baseline/
COCO_PATH = /coco/images/ # Base path to COCO images folder
RESUME_PATH = model_weights/pretrained_models/r50_single.pth # Path to the pretrained model
mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet50 \
    --epochs 20 \
    --num_feature_levels 1 \
    --num_queries 300 \
    --batch_size 4 \
    --dilation \
    --with_box_refine \
    --dataset_file vid_single \
    --output_dir ${EXP_DIR} \
    --coco_path ${COCO_PATH} \
    --num_classes 3 \
    --dropout 0.2 \
    --lr 1e-4 \
    --weight_decay 2e-5 \
    --coco_pretrain \
    --del_class_weights \
    --resume ${RESUME_PATH} | tee ${EXP_DIR}/logs/baseline_output.txt
