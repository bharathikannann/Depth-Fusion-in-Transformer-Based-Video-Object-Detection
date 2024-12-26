# Depth Fusion in Transformer-Based Video Object Detection

## Abstract

Object detection is a fundamental task in computer vision, with many algorithms developed for this purpose. Most methods rely only on RGB data. However, in challenging scenarios where objects share similar colors with their backgrounds, are partially occluded, or are under varying lighting conditions, depth data can provide valuable additional information. Integrating depth into object detection models offers a promising solution to these challenges. Depth information has been primarily researched and applied in salient object detection and semantic segmentation, where it has demonstrated significant advantages. These methods have been extended to object detection, but many techniques often use simple fusion strategies like 4-channel input and feature concatenation. Few methods in salient object detection and segmentation have effectively utilized cross-attention mechanisms to fuse RGB and depth data. Building on the proven effectiveness of deformable attention for object detection, this research adapts the cross-attention-based fusion mechanism for object detection by using deformable attention. This study investigates the integration of depth information into transformer-based object detection models, specifically focusing on the Deformable DETR framework for both single-frame and video-based scenarios. We introduce a plug-and-play fusion module based on deformable cross-attention to effectively combine RGB and depth data. Various depth fusion strategies are implemented using this fusion module, including Late Fusion, Backbone Cross Fusion, and Encoder Cross Fusion. Experimental evaluations are performed on a proprietary RGB-D video dataset. Results show that incorporating depth data enhances single-frame object detection performance, particularly in challenging conditions where objects share similar colors with the background. The Late Fusion method showed improved performance in metrics such as mean Average Precision (mAP) and F1 scores. However, when extending these techniques to video object detection models like TransVOD and TransVOD++, the performance is still limited, not necessarily due to the extension of fusion techniques, but because the base video object detection models themselves showed less performance. This study also provides a foundation for future research into depth integration in transformer-based detection models.

## Main Results

This repository contains code for three implementations of depth fusion: Late Fusion, Backbone Cross Fusion, and Encoder Cross Fusion. Refer to the thesis for more details about the architecture.

### Performance Comparison of All Models on Hand Pose and Hand Blur Datasets

| Model                             | Hand Pose (mAP 0.5) | Hand Pose (mAP 0.5-0.95) | Hand Pose (F1 Score) | Hand Blur (mAP 0.5) | Hand Blur (mAP 0.5-0.95) | Hand Blur (F1 Score) |
|-----------------------------------|---------------------|--------------------------|----------------------|---------------------|--------------------------|----------------------|
| **Baseline Models**               |                     |                          |                      |                     |                          |                      |
| Baseline RGB (Deformable DETR)    | 93.85               | 67.58                    | 94.7                 | 93.08               | 72.46                    | 94.8                 |
| **Fusion Models**                 |                     |                          |                      |                     |                          |                      |
| Baseline + Late Fusion            | 94.78               | 67.85                    | 95.1                 | 94.07               | 69.22                    | 95.8                 |
| Baseline + Backbone Cross Fusion  | 92.52               | 67.06                    | 95.0                 | 92.91               | 73.12                    | 95.2                 |
| Baseline + Encoder Cross Fusion   | 93.35               | 66.88                    | 94.4                 | 94.27               | 72.67                    | 95.9                 |
| **Video Object Detection Models** |                     |                          |                      |                     |                          |                      |
| TransVOD RGBV                     | 90.38               | 62.55                    | 94.8                 | 90.05               | 66.02                    | 95.0                 |
| TransVOD_wo-TTE                   | 92.04               | 65.06                    | 94.8                 | 92.56               | 71.67                    | 94.9                 |
| TransVOD++ RGBV                   | 93.34               | 66.72                    | 94.9                 | 92.79               | 70.68                    | 95.1                 |
| **TransVOD++ with Fusion Models** |                     |                          |                      |                     |                          |                      |
| TransVOD++ with Late Fusion       | 93.48               | 66.72                    | 94.4                 | 94.10               | 70.68                    | 95.8                 |
| TransVOD++ with Backbone Cross Fusion | 93.55           | 66.53                    | 94.3                 | 92.86               | 70.88                    | 95.3                 |
| TransVOD++ with Encoder Cross Fusion  | 91.68           | 65.91                    | 93.2                 | 93.98               | 73.88                    | 96.0                 |

- *Baseline is the Deformable DETR.*
- *Hand Pose and Hand Blur are proprietary datasets*

## Installation

This repository is built on top of:

- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [TransVOD](https://github.com/SJTU-LuHe/TransVOD)
- [TransVOD++](https://github.com/qianyuzqy/TransVOD_plusplus)
- [DFormer](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox)

Please refer to the official repositories if any errors occur during installation.

### Requirements

- **Operating System:** Linux
- **CUDA:** Version >= 9.2
- **GCC:** Version >= 5.4
- **Python:** Version >= 3.7

Anaconda to create a conda environment is recommended:

```bash
conda create -n transvod python=3.7 pip
```
Then, activate the environment:

```bash
conda activate transvod
```

Install PyTorch and torchvision (official docs - [link](https://pytorch.org/get-started/locally/))

```bash
conda install pytorch=1.13.1 torchvision=0.14.1 cudatoolkit=12.2 -c pytorch
```
For any issues, please refer to the official documentation. While the code may support newer versions, there can be compatibility problems. Therefore, it is recommended to verify your setup. If you encounter any errors, use the versions specified.

For TransVOD models, install mmcv and mmdet (refer to the official documentation for installation): The ROI crop uses mmcv library and it is only used in TransVOD++

```bash
pip install mmcv-full==1.7.0 mmdet==3.3.0
```

- [MMDetection Installation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation)
- [MMCV Installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

Install other requirements:

```bash
pip install -r requirements.txt
```

For any compatibility issues, refer to `supporting_files/detailed_requirements.txt` to see the exact versions used during this research.

### Compiling CUDA Operators

Compile the Deformable Attention CUDA operators:

```bash
cd ./models/ops
sh ./make.sh
```

Alternatively, run:

```bash
python setup.py build develop --user
```

For any issues in compilation, check:
- [Deformable DETR Issues](https://github.com/fundamentalvision/Deformable-DETR/issues)
- [TransVOD Issues](https://github.com/SJTU-LuHe/TransVOD/issues)

### Test the Installation

```bash
python test.py
```

You should see all checks passing.

## Pretrained Weights

Download the pretrained weights and place them in the appropriate directories:

- Deformable DETR Spatial Model Weights from TransVOD official [repo](https://github.com/SJTU-LuHe/TransVOD) (Checkpoint 0009): [Download Link](https://drive.google.com/file/d/1YOtR1GfEja_rSw4LpyPd_ogXEnYs3jib/view?usp=drive_link)
- TransVOD Models from official [repo](https://github.com/SJTU-LuHe/TransVOD): [Download Link](https://drive.google.com/file/d/191nEtAKRYCSdzNYo1a4JVmXPxVUFxBfN/view?usp=drive_link)
- DFormer Depth Backbone Weights from DFormer [repo](https://github.com/VCIP-RGBD/DFormer): [Download Link](https://drive.google.com/file/d/1-ZsxErdrwrg20NjlDkobbZuBIp-fsggK/view?usp=drive_link)
  - *Note*: Downlaod the whole DFomer model, When it is linked in this repo, it considers only the depth backbone

## Datasets

- Proprietary Dataset: Used for training the models. 
- OpenImages Dataset: This is the only opensource dataset used during training, and only the hand instaces is used, dataset available [here](https://storage.googleapis.com/openimages/web/index.html).

### Dataset Preparation

For training on custom datasets, follow the common format:

1. Create a folder and place your images inside `images/`.
2. The folder structure should be:

   ```
   dataset_folder/
     images/
       ... (subfolders and images)
     depth_pred/ (only for fusion models)
       ... (subfolders and depth maps)
     labels/
       coco/
         annotations.json
   ```

3. Create a COCO-format JSON file for your dataset.
4. Modify the path in `datasets/vid_single.py` and `datasets/vid_multi.py` within the build function to link to your files.
5. Adjust the number of classes using the `--num_classes` argument.
6. For multiple classes, change `--num_classes` to `actual_classes + 1`, use `--del_class_weights` to delete the class weights and train from scratch, and change the classification loss to `sigmoid_focal_loss` in the `set_criterion` class. This repo uses `modified_sigmoid_focal_loss` which is based on focal loss for single hand class.
7. If your labels are in YOLO format, use `data_conversion_tools/convert_to_coco.py` to convert them to COCO JSON format.
8. For generating depth data for your dataset, use the code in `depth_tools/rgb2d.py` and `rgb2d_playground` to obtain depth maps using the depth estimation model.

## Training

To train the model using multiple GPUs (e.g., 2 GPUs):

```bash
GPUS_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 ./tools/run_dist_launch.sh 2 configs/training/Baseline.sh
```

Experiments are monitored using Weights & Biases (wandb). If you don't want to use wandb, disable it by adding `--no_wandb` to the training arguments in the config.

Edit the parameters in the config files for all models under `configs/training/` to link to the paths of your dataset and verify other parameters.

To train on a single GPU, use the parameters from the .sh file directly in the command line, setting the device to the selected GPU device number.

Refer to the `configs/training/` folder to train all the models, including fusion models and video models.

## Inference

Use the `inference.py` script for doing inference for all models. Refer to `inference_playground` to experiment with inference on single images.

Example for the Baseline Model:

```bash
python inference.py \
  --resume model_weights/finetuned_models/baseline/checkpoint.pth \
  --inference_coco_path your_dataset_folder/labels/coco/train.json \
  --img_path your_dataset_folder \
  --dataset_file vid_single \
  --dilation \
  --with_box_refine \
  --keep_prob 0.01 \
  --num_classes 3 \
  --num_feature_levels 1 \
  --device cuda:0 \
  --save_txt \
  --output_dir inference_outputs
```

For Fusion Models:

```bash
python inference.py \
  --resume model_weights/finetuned_models/latefusion/checkpoint_v1.pth \
  --inference_coco_path your_dataset_folder/labels/coco/train.json \
  --img_path your_dataset_folder \
  --dataset_file vid_single \
  --dilation \
  --with_box_refine \
  --keep_prob 0.01 \
  --num_classes 3 \
  --num_feature_levels 1 \
  --device cuda:0 \
  --save_txt \
  --output_dir inference_outputs \
  --use_depth \
  --fusion_type LateFusion \
  --dformer_backbone
```

For Video Models:


This script requires the coco file to work, which means it required the original dataset, The propretiary dataset is not included in this repo. It can be linked and can be used for inference. This code is just the template. 

```
python inference.py \
  --resume model_weights/finetuned_models/transvod/checkpoint.pth \
  --inference_coco_path your_dataset_folder/labels/coco/train.json \
  --img_path your_dataset_folder \
  --dataset_file vid_single \
  --dilation \
  --with_box_refine \
  --keep_prob 0.01 \
  --num_classes 3 \
  --num_feature_levels 1 \
  --device cuda:0 \
  --save_txt \
  --output_dir inference_outputs \
  --num_ref_frames 4
```

For Video with Fusion models

```
python inference.py \
  --resume model_weights/finetuned_models/latefusion/checkpoint_v1.pth \
  --spaial_weights model_weights/finetuned_models/latefusion/checkpoint_v1.pth \
  --inference_coco_path your_dataset_folder/labels/coco/train.json \
  --img_path your_dataset_folder \
  --dataset_file vid_single \
  --dilation \
  --with_box_refine \
  --keep_prob 0.01 \
  --num_classes 3 \
  --num_feature_levels 1 \
  --device cuda:0 \
  --save_txt \
  --output_dir inference_outputs \
  --num_ref_frames 4
  --use_depth \
  --fusion_type LateFusion \
  --dformer_backbone
```

- Include the fusion model weights separately and --resume takes the temporal weights from TransVOD++

The inference will be done and the labels will be saved in YOLO format. For each image, there will be a single .txt file with each line as `class center_x center_y width height`.

## Evaluation

To evaluate the models, once the inference is done and the labels are saved, you can use any evaluation tool that supports the YOLO output format to get the mAP.

This repository includes a benchmark tool that does evaluation for the models. Go to `benchmark_tool` and refer to the README file to see how to evaluate the models once the inference is done.

## Fusion Strategies

Here is a short intro about the there fusion methodologies: Late Fusion, Backbone Cross Fusion, and Encoder Cross Fusion. All three techniques use a similar fusion block but are applied at different stages within the model architecture.

1. **Late Fusion**: Integration of RGB and depth features is performed after both the backbone networks have completed their forward passes.
2. **Backbone Cross Fusion**: RGB and depth features share information during the forward pass through the backbone. Outputs from RGB and depth backbones are taken at every layer, fused together, and merged back to the respective layer.
3. **Encoder Cross Fusion**: Applies the cross-fusion mechanism within every encoder stages of the network.

---
This repository is part of the thesis by Bharathi Kannan Nithyanantham, submitted to the University of Siegen under Prof. Michael Moeller, Department of Computer Vision, and supervised by Dr. Junli Tao and Jan Philipp Schneider. This thesis is supported by Virtual Retail GmbH; and provided essential resources, including access to GPUs for model training and the dataset for this research.