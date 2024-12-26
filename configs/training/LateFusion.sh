#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=output/latefusion/
COCO_PATH = /coco/images/ # Bae path to COCO images folder
RESUME_PATH = /r50_single.pth
DFORMER_WEIGHTS=/DFormer_Base.pth.tar

mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet50 \
    --epochs 20 \
    --num_feature_levels 1 \
    --num_queries 300 \
    --batch_size 6 \
    --dilation \
    --with_box_refine \
    --dataset_file vid_single \
    --output_dir ${EXP_DIR} \
    --coco_path ${COCO_PATH} \
    --num_classes 3 \
    --dformer_backbone \
    --dropout 0.2 \
    --lr 1e-5 \
    --weight_decay 2e-5 \
    --use_depth \
    --fusion_type LateFusion \
    --dformer_weights ${DFORMER_WEIGHTS} \
    --coco_pretrain \
    --del_class_weights \
    --resume ${RESUME_PATH} | tee ${EXP_DIR}/logs/latefusion_output.txt