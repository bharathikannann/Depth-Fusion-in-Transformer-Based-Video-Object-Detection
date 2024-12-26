#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=output/backbone_crossfusion/
COCO_PATH = /coco/images/ # Bae path to COCO images folder
RESUME_PATH = /../r50_single.pth

mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet50 \
    --epochs 10 \
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
    --use_depth \
    --fusion_type Backbone_CrossFusion \
    --coco_pretrain \
    --resume ${RESUME_PATH} | tee ${EXP_DIR}/logs/output_backbone_crossfusion.txt
