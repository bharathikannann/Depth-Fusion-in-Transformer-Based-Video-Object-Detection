#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=/output/TransVOD++/
COCO_PATH=/coco/images/
TRANSVOD_TEMPORAL_WEIGHTS=../pretrained_models/r50_multi_plusplus.pth
SPATIAL_WEIGHTS=../finetuned_models/LateFusion/checkpoint.pth
RESUME_PATH=..output/baseline/checkpoint.pth
DFORMER_WEIGHTS=/DFormer_Base.pth.tar

mkdir ${EXP_DIR}
PY_ARGS=${@:1}
python -u main_multi.py \
    --backbone resnet50 \
    --num_ref_frames 4 \
    --epochs 7 \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --with_box_refine \
    --dataset_file vid_multi_plusplus \
    --output_dir ${EXP_DIR} \
    --spatial_weights ${SPATIAL_WEIGHTS} \
    --transvod_temporal_weights ${TRANSVOD_TEMPORAL_WEIGHTS} \
    --coco_path ${COCO_PATH} \
    --num_classes 3 \
    --dropout 0.2 \
    --lr 1e-4 \
    --weight_decay 2e-5 \
    --dformer_backbone \
    --use_depth \
    --fusion_type LateFusion \
    --dformer_weights ${DFORMER_WEIGHTS} \
    --resume ${RESUME_PATH} | tee ${EXP_DIR}/logs/output.txt