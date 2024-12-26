# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import argparse
import datetime
import json
import random
import time
import psutil
import yaml
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets

import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
import wandb
from collections import Counter, defaultdict


def get_args_parser():
    """
    Parses and returns the command-line arguments for the Deformable DETR detector.
    """
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    
    # Learning rate parameters
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+',
                        help='Names of backbone layers to apply different learning rates')
    parser.add_argument('--lr_backbone', default=2e-5, type=float, help='Learning rate for backbone layers')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+',
                        help='Names of linear projection layers with different learning rates')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float,
                        help='Multiplier for learning rate of linear projection layers')
    parser.add_argument('--lr_depth_encoder', default=10, type=float,
                        help='Multiplier for learning rate of depth encoder layers')
    
    # Training parameters
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
    parser.add_argument('--epochs', default=15, type=int, help='Number of training epochs')
    parser.add_argument('--lr_drop', default=5, type=int, help='Epoch to drop learning rate')
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+',
                        help='Specific epochs to drop learning rate')
    parser.add_argument('--clip_max_norm', default=0.5, type=float,
                        help='Maximum norm for gradient clipping')
    
    # Model configuration
    parser.add_argument('--num_ref_frames', default=3, type=int, help='Number of reference frames')
    parser.add_argument('--sgd', action='store_true', help='Use SGD optimizer if set')
    
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true',
                        help='Enable box refinement')
    parser.add_argument('--two_stage', default=False, action='store_true',
                        help='Enable two-stage detection')
    
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="Use dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding for image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="Scale factor for positional embeddings")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='Number of feature levels in the backbone')
    
    # Transformer parameters
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Feedforward network dimension in transformer")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Dimension of the transformer embeddings")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout rate in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads in the transformer")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots in the transformer")
    parser.add_argument('--dec_n_points', default=4, type=int, help='Number of points in decoder attention')
    parser.add_argument('--enc_n_points', default=4, type=int, help='Number of points in encoder attention')
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int,
                        help='Number of temporal decoder layers')
    parser.add_argument('--interval1', default=20, type=int, help='First interval parameter')
    parser.add_argument('--interval2', default=60, type=int, help='Second interval parameter')
    
    parser.add_argument("--fixed_pretrained_model", default=False, action='store_true',
                        help='Use fixed pretrained model if set')
    
    # Segmentation parameters
    parser.add_argument('--masks', action='store_true',
                        help="Enable training of the segmentation head")
    
    # Loss parameters
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disable auxiliary decoding losses")
    
    # Matcher parameters for loss computation
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="GIoU box coefficient in the matching cost")
    
    # Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float, help='Coefficient for mask loss')
    parser.add_argument('--dice_loss_coef', default=1, type=float, help='Coefficient for dice loss')
    parser.add_argument('--cls_loss_coef', default=2, type=float, help='Coefficient for classification loss')
    parser.add_argument('--bbox_loss_coef', default=5, type=float, help='Coefficient for bounding box loss')
    parser.add_argument('--giou_loss_coef', default=2, type=float, help='Coefficient for GIoU loss')
    parser.add_argument('--focal_alpha', default=0.25, type=float, help='Alpha parameter for focal loss')
    
    # Dataset parameters
    parser.add_argument('--dataset_file', default='vid_multi',
                        help='Dataset type to use (e.g., vid_multi, vid_single)')
    parser.add_argument('--coco_path', default='./data/coco', type=str,
                        help='Path to COCO dataset')
    parser.add_argument('--vid_path', default='./data/vid', type=str,
                        help='Path to VID dataset')
    parser.add_argument('--coco_pretrain', default=False, action='store_true',
                        help='Use COCO pretraining if set')
    parser.add_argument('--coco_panoptic_path', type=str, help='Path to COCO panoptic dataset')
    parser.add_argument('--remove_difficult', action='store_true',
                        help='Remove difficult samples from the dataset')
    
    # Output and device configurations
    parser.add_argument('--output_dir', default='',
                        help='Directory to save outputs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training/testing (e.g., cuda, cpu)')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--resume', default='', help='Path to resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch number')
    parser.add_argument('--eval', action='store_true', help='Evaluate model on validation set if set')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of data loader workers')
    parser.add_argument('--cache_mode', default=False, action='store_true',
                        help='Cache images in memory if set')
    
    # Class and weight configurations
    parser.add_argument('--num_classes', default=31, type=int,
                        help='Number of object classes')
    parser.add_argument('--del_class_weights', default=False, action='store_true',
                        help='Retrain class weights if set')
    parser.add_argument('--no_wandb', default=False, action='store_true',
                        help='Disable Weights & Biases logging if set')
    parser.add_argument('--dformer_weights', default=None, type=str,
                        help='Path to the DFormer weights')
    parser.add_argument('--dformer_backbone', default=False, action='store_true',
                        help='Use DFormer backbone if set')
    
    # Depth-related parameters
    parser.add_argument('--dpth_n_points', default=4, type=int,
                        help='Number of points in the depth transformer')
    parser.add_argument('--use_depth', default=False, action='store_true',
                        help='Use depth images if set')
    parser.add_argument('--fusion_type', default="Baseline", type=str, help="Type of Fusion method used", 
                        choices=["Baseline", "LateFusion","Backbone_CrossFusion", "Encoder_CrossFusion"])
    
    # This is an argument internally used during research purposes #TODO: Remove this and only use --fusion-type
    # Internally, Baseline -> Baseline_rgb, LateFusion -> DepthDeform_latefusion_dformer
    #             Backbone_CrossFusion -> DepthDeform_dformer_crossfusion, Encoder_CrossFusion -> DepthDeform_encoder_cf_dformer
    parser.add_argument('--depth_type', default="Baseline_rgb", type=str,
                        choices=["Baseline_rgb", "Baseline_rgbd", "Baseline_rgb_d_concat",
                                 "DepthDeform_latefusion", "DepthDeform_crossfusion",
                                 "DepthDeform_crossfusion_2way", "DepthDeform_crossfusion_2way_concat",
                                 "DepthDeform_latefusion_2way", "DepthDeform_latefusion_2way_concat",
                                 "DepthDeform_latefusion_noresidual", "DepthDeform_latefusion_dformer",
                                 "DepthDeform_encoder_cf_dformer", "DepthDeform_dformer_crossfusion",
                                 "DepthDeform_dformer_crossfusion_2way"],
                        help='Type of depth model to use')
    
    return parser

def main(args):
    """
    Main function to train and evaluate the Deformable DETR model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    print("Dataset File:", args.dataset_file)
    
    if args.dataset_file == "vid_single":
        from engine_single import evaluate, train_one_epoch
        import util.misc as utils
    else:
        from engine_multi import evaluate, train_one_epoch
        import util.misc_multi as utils
    
    # Enable DFormer backbone if DFormer weights are provided
    if args.dformer_weights:
        args.dformer_backbone = True
        
    # Set the device for training
    device = torch.device(args.device)
    
    # Initialize distributed training mode
    utils.init_distributed_mode(args)
    print("Git commit:", utils.get_sha())
    
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)
    
    # Fix the random seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    # Prepare model for distributed training if necessary
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters)

    # Build training and validation datasets
    dataset_train = build_dataset(image_set='train_joint', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn, num_workers=args.num_workers,
        pin_memory=True
    )
    
    data_loader_val = DataLoader(
        dataset_val, args.batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=utils.collate_fn,
        num_workers=args.num_workers, pin_memory=True
    )

    # Helper function to match parameter names with keywords
    def match_name_keywords(n, name_keywords):
        """
        Checks if any of the keywords are present in the parameter name.

        Args:
            n (str): Parameter name.
            name_keywords (list of str): List of keywords to match.

        Returns:
            bool: True if any keyword is found in the name, False otherwise.
        """
        return any(b in n for b in name_keywords)

    # Print all parameter names for debugging
    for n, p in model_without_ddp.named_parameters():
        print(n)
        
    def is_depthencoder(param_name, include_keywords=['transformer.depth_encoder_layer'], exclude_keywords=['']):
        """
        Determines if a parameter belongs to the depth encoder based on inclusion keywords.

        Args:
            param_name (str): The name of the parameter.
            include_keywords (list of str): Keywords to include.
            exclude_keywords (list of str): Keywords to exclude.

        Returns:
            bool: True if parameter is part of the depth encoder, False otherwise.
        """
        include_match = any(keyword in param_name for keyword in include_keywords)
        return include_match

    # Initialize parameter groups for the optimizer
    param_dicts = []
    
    # Configure parameter groups based on depth type
    if "latefusion" in args.depth_type:
        for name, param in model_without_ddp.named_parameters():
            if 'backbone.0.body' in name and not 'depth' in name:
                param.requires_grad = False
            else: 
                param.requires_grad = True
        param_dicts = [
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For all other layers
                    if not match_name_keywords(n, ['backbone']) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad and not is_depthencoder(n)],
            "lr": args.lr,
        },
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For depth backbone
                    if match_name_keywords(n, ['depth_backbone']) and p.requires_grad and not is_depthencoder(n)],
            "lr": args.lr,
        },
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For Linear Projection layers
                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad and not is_depthencoder(n)],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and is_depthencoder(n) and not match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr * 10,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and is_depthencoder(n) and match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr,
        }
    ]
    elif "crossfusion" in args.depth_type:
        for name, param in model_without_ddp.named_parameters():
            if 'backbone.0.body' in name:
                param.requires_grad = True
            else: 
                param.requires_grad = True
        param_dicts = [
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For all other layers
                    if match_name_keywords(n, ['transformer', 'class_embed', 'query_embed','input_proj']) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad and not is_depthencoder(n, ['d2r_fusion', 'r2d_fusion', 'rgb_proj', 'd_proj'])],
            "lr": args.lr,
        },
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For depth backbone
                    if match_name_keywords(n, ['backbone']) and p.requires_grad and not is_depthencoder(n, ['d2r_fusion', 'r2d_fusion', 'rgb_proj', 'd_proj'])],
            "lr": args.lr,
        },
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For Linear Projection layers
                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad and not is_depthencoder(n, ['d2r_fusion', 'r2d_fusion', 'rgb_proj', 'd_proj'])],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and is_depthencoder(n, ['d2r_fusion', 'r2d_fusion', 'rgb_proj', 'd_proj']) and not match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr * 10,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and is_depthencoder(n, ['d2r_fusion', 'r2d_fusion', 'rgb_proj', 'd_proj']) and match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr,
        }
    ]
    elif "encoder_cf" in args.depth_type:
        for name, param in model_without_ddp.named_parameters():
            if 'backbone.0.body' in name:
                param.requires_grad = False
            else: 
                param.requires_grad = True
        param_dicts = [
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For all other layers
                    if not match_name_keywords(n, ['backbone']) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad and not is_depthencoder(n, ['encoder.fusion_layers'])],
            "lr": args.lr,
        },
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For depth backbone
                    if match_name_keywords(n, ['depth_backbone']) and p.requires_grad and not is_depthencoder(n, ['encoder.fusion_layers'])],
            "lr": args.lr,
        },
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For Linear Projection layers
                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad and not is_depthencoder(n, ['encoder.fusion_layers'])],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and is_depthencoder(n, ['encoder.fusion_layers']) and not match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr * 10,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if p.requires_grad and is_depthencoder(n, ['encoder.fusion_layers']) and match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr,
        }
    ]
    else:
        param_dicts = [
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For all other layers
                    if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For backbone
                    if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr * 0.1,
        },
        {
            "params":[p for n, p in model_without_ddp.named_parameters() # For Linear Projection layers
                    if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
        ]
    
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        
    param_names = {id(p): n for n, p in model_without_ddp.named_parameters()}
    
    # Initialize learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    # Alternative scheduler options:
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    # Load pretrained weights if provided
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # Setup output directory
    output_dir = Path(args.output_dir)
    if args.resume:
        # Resume from a checkpoint
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            print("Number of classes:", args.num_classes)
            
            if not args.eval: # During training
                if args.coco_pretrain: 
                    if args.del_class_weights: # delete class weights if needed
                        if args.num_classes != 31: # delete class weights only if the number of classes is not 31
                            print("Deleting class weights")
                            keys = list(checkpoint['model'].keys())
                            for i in keys:
                                if 'class_embed' in i: 
                                    del checkpoint["model"][i]
                        else:
                            print("Keeping all the original weights.")
                    
                    # Used in earlier versions of the code, but not needed anymore
                    # if args.use_depth and args.depth_type == "Baseline_rgbd":
                    #     # Modify the first convolutional layer to accept an additional depth channel
                    #     if "backbone.0.body.conv1.weight" in checkpoint["model"]:
                    #         conv1_weights = checkpoint["model"]["backbone.0.body.conv1.weight"]
                    #         mean_weights = conv1_weights.mean(dim=1, keepdim=True)  # Mean over input channels
                    #         new_conv1_weights = torch.cat((conv1_weights, mean_weights), dim=1)  # Add as new channel
                    #         checkpoint["model"]["backbone.0.body.conv1.weight"] = new_conv1_weights
                    #     else:
                    #         print("backbone.0.body.conv1.weight not found in the checkpoint.")
                    
                    # elif args.use_depth and args.depth_type == "Baseline_rgb_d_concat":
                    #     # Concatenate depth channels for the backbone
                    #     for key, value in list(checkpoint["model"].items()):
                    #         if key.startswith("backbone"):
                    #             depth_key = key.replace("backbone.0", "backbone.1")
                    #             checkpoint["model"][depth_key] = value.clone()
                    
                    # Load the modified checkpoint into the model
                    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
                        checkpoint['model'], strict=False
                    )
            else:  # During evaluation               
                # Load the checkpoint normally if not resuming
                missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
                    checkpoint['model'], strict=False
                )
        
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys:', missing_keys)
        if len(unexpected_keys) > 0:
            print('Unexpected Keys:', unexpected_keys)
    
    # Ensure all model parameters are included in parameter groups
    total_params_in_model = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    total_params_in_dicts = sum(p.numel() for pd in param_dicts for p in pd["params"])
    if total_params_in_model == total_params_in_dicts:
        print("All parameters are included in param_dicts.")
    else:
        raise ValueError("Some parameters are not included in param_dicts.")
    
    # Optional: Resume optimizer and scheduler states if available
    # (Commented out as functionality remains unchanged)
    # if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #     import copy
    #     p_groups = copy.deepcopy(optimizer.param_groups)
    #     # optimizer.load_state_dict(checkpoint['optimizer']) # Avoid loading optimizer state if parameters have changed
    #     for pg, pg_old in zip(optimizer.param_groups, p_groups):
    #         pg['lr'] = pg_old['lr']
    #         pg['initial_lr'] = pg_old['initial_lr']
    #     # print(optimizer.param_groups)
    #     lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #     # Hack to override LR drop if needed
    #     args.override_resumed_lr_drop = True
    #     if args.override_resumed_lr_drop:
    #         print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
    #         lr_scheduler.step_size = args.lr_drop
    #         lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
    #     lr_scheduler.step(lr_scheduler.last_epoch)
    #     args.start_epoch = checkpoint['epoch'] + 1
    
    # Optional: Perform evaluation if needed
    # test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
    #                                       data_loader_val, base_ds, device, args.output_dir)
    # if args.output_dir:
    #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
    # return
    
    # Initialize Weights & Biases for experiment tracking if not disabled
    if utils.is_main_process() and not args.no_wandb:
        wandb.init(project="transvod-single", config=args) # Initialize W&B run and log hyperparameters
        wandb.config.update(args)
        wandb.watch(model, log="all")

    print("Start training")
    start_time = time.time()
    print("CUDA is available:", torch.cuda.is_available())
    print("Using Depth:", args.use_depth)
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        
        # Step the learning rate scheduler
        lr_scheduler.step()
        print('Output directory:', args.output_dir)
        
        # Save checkpoints if an output directory is specified
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        
        # Optional: Perform evaluation and log to W&B (currently commented out)
        # if epoch == 0 or epoch == args.epochs - 1 or (epoch + 1) % 10 == 0:
        #     test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
        #                                         data_loader_val, base_ds, device, args.output_dir)
        #     wandb.log({"val_loss": test_stats, "epoch": epoch})
        # if args.output_dir:
        #     utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        
        name_to_lr = {}
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            for param in param_group['params']:
                # Assuming each parameter is unique and can be identified by its object id
                param_name = next(name for name, p in model_without_ddp.named_parameters() if id(p) == id(param))
                name_to_lr[param_name] = lr
    
        # Print learning rate information for each parameter
        for name, param in model_without_ddp.named_parameters():
            lr = name_to_lr.get(name, 0)  # Default to 0 if name is not in the mapping
            print(f"Name: {name}, Shape: {param.shape}, Requires gradient: {param.requires_grad}, LR: {lr}")
        
        # Verify all parameters are included in parameter groups (comment out if you want to skip this check)
        total_params_in_model = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        total_params_in_dicts = sum(p.numel() for pd in param_dicts for p in pd["params"])
        if total_params_in_model == total_params_in_dicts:
            print("All parameters are included in param_dicts.")
        else:
            raise ValueError("Some parameters are not included in param_dicts.")
        
        # Prepare log statistics
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # Log training statistics to Weights & Biases if enabled
        if utils.is_main_process() and not args.no_wandb:
            wandb.log({"train_loss": train_stats, "epoch": epoch})
            # Uncomment below to log CPU and GPU usage
            # wandb.log({"cpu_usage": psutil.cpu_percent(), "gpu_usage": torch.cuda.memory_allocated()})
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time:', total_time_str)
    
    if utils.is_main_process() and not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Deformable DETR training and evaluation script',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        args_dict = vars(args)
        with open(Path(args.output_dir) / 'args.yaml', 'w') as f:
            yaml.dump(args_dict, f)
    
    main(args)
