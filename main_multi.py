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
from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets

import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
import wandb

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    
    # Learning rate and optimization parameters
    parser.add_argument('--lr', default=1e-4, type=float, help='Base learning rate')
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+',
                        help='Names of backbone parameters for different learning rates')
    parser.add_argument('--lr_backbone', default=2e-5, type=float, help='Learning rate for backbone')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+',
                        help='Names of linear projection parameters for different learning rates')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float,
                        help='Multiplier for linear projection learning rate')
    parser.add_argument('--batch_size', default=2, type=int, help='Batch size per training step')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for optimizer')
    parser.add_argument('--epochs', default=10, type=int, help='Total number of training epochs')
    parser.add_argument('--lr_drop', default=5, type=int, help='Epoch at which to drop learning rate')
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+',
                        help='Specific epochs at which to drop learning rate')
    parser.add_argument('--clip_max_norm', default=1.0, type=float,
                        help='Maximum norm for gradient clipping')
    
    # Temporal parameters
    parser.add_argument('--num_ref_frames', default=3, type=int, help='Number of reference frames')
    
    # Optimizer choice
    parser.add_argument('--sgd', action='store_true', help='Use SGD optimizer if set')
    
    # Deformable DETR variants
    parser.add_argument('--with_box_refine', default=False, action='store_true',
                        help='Enable box refinement in the model')
    parser.add_argument('--two_stage', default=False, action='store_true',
                        help='Enable two-stage processing')
    
    # Model configuration
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # Backbone parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="Replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="Scale for positional embedding")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='Number of feature levels')
    
    # Transformer parameters
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Dimension of the transformer embeddings")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout rate applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads in the transformer")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int, help='Number of points in decoder')
    parser.add_argument('--enc_n_points', default=4, type=int, help='Number of points in encoder')
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int,
                        help='Number of temporal decoder layers')
    parser.add_argument('--interval1', default=20, type=int, help='First interval parameter')
    parser.add_argument('--interval2', default=60, type=int, help='Second interval parameter')
    parser.add_argument("--fixed_pretrained_model", default=False, action='store_true',
                        help='Use fixed pretrained model if set')
    
    # Segmentation parameters
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    
    # Loss configuration
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disable auxiliary decoding losses")
    
    # Matcher parameters
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
    parser.add_argument('--bbox_loss_coef', default=5, type=float, help='Coefficient for bbox loss')
    parser.add_argument('--giou_loss_coef', default=2, type=float, help='Coefficient for GIoU loss')
    parser.add_argument('--focal_alpha', default=0.25, type=float, help='Alpha parameter for focal loss')
    
    # Dataset parameters
    parser.add_argument('--dataset_file', default='vid_multi', help='Dataset to use')
    parser.add_argument('--coco_path', default='./data/coco', type=str, help='Path to COCO dataset')
    parser.add_argument('--vid_path', default='./data/vid', type=str, help='Path to VID dataset')
    parser.add_argument('--coco_pretrain', default=False, action='store_true',
                        help='Use COCO pretrained weights if set')
    parser.add_argument('--coco_panoptic_path', type=str, help='Path to COCO panoptic annotations')
    parser.add_argument('--remove_difficult', action='store_true',
                        help='Remove difficult samples if set')
    
    # Output and device configuration
    parser.add_argument('--output_dir', default='', help='Directory to save outputs')
    parser.add_argument('--device', default='cuda', help='Device to use for training/testing')
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--resume', default='', help='Path to resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Starting epoch number')
    parser.add_argument('--eval', action='store_true', help='Evaluate model if set')
    parser.add_argument('--num_workers', default=1, type=int, help='Number of data loading workers')
    parser.add_argument('--cache_mode', default=False, action='store_true',
                        help='Cache images in memory if set')
    
    # Additional parameters
    parser.add_argument('--num_classes', default=31, type=int, help='Number of classes')
    parser.add_argument('--no_wandb', default=False, action='store_true', help='Disable Weights & Biases logging if set')
    parser.add_argument('--dformer_weights', default=None, type=str, help='Path to the DFormer weights')
    parser.add_argument('--dformer_backbone', default=False, action='store_true',
                        help='Use DFormer backbone if set')
    parser.add_argument('--dpth_n_points', default=4, type=int, help='Number of points in the depth transformer')
    
    # Depth-related parameters
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
    parser.add_argument('--transvod_temporal_weights', default='', type=str,
                        help='Path to the temporal TransVOD weights')
    parser.add_argument('--spatial_weights', default='', type=str,
                        help='Path to the spatial TransVOD weights')
    
    return parser

def main(args):
    print(f"Dataset File: {args.dataset_file} -------------------")
    
    if args.dataset_file == "vid_single":
        from engine_single import evaluate, train_one_epoch
        import util.misc as utils
    else:
        from engine_multi import evaluate, train_one_epoch
        import util.misc_multi as utils
    
    print(f"Using dataset: {args.dataset_file}")
    device = torch.device(args.device)
    
    utils.init_distributed_mode(args)
    print(f"Git SHA:\n  {utils.get_sha()}\n")
    
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    
    print(args)
    
    # Fix the random seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Uncomment the following lines if deterministic behavior is required
    # torch.cuda.manual_seed_all(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    
    # Build the model, criterion, and postprocessors
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of trainable parameters: {n_parameters}')
    
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
    
    custom_collate_fn = partial(utils.collate_fn, use_depth=args.use_depth)
    
    data_loader_train = DataLoader(
        dataset_train, batch_sampler=batch_sampler_train,
        collate_fn=custom_collate_fn, num_workers=args.num_workers,
        pin_memory=True
    )
    
    data_loader_val = DataLoader(
        dataset_val, args.batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=custom_collate_fn, num_workers=args.num_workers,
        pin_memory=True
    )
    
    def match_name_keywords(name, keywords):
        """
        Check if any of the keywords are present in the parameter name.

        Args:
            name (str): Parameter name.
            keywords (list): List of keywords to match.

        Returns:
            bool: True if any keyword matches, False otherwise.
        """
        return any(keyword in name for keyword in keywords)
    
    def is_backbone_excluding_depth(param_name, include_keywords=['backbone'], exclude_keywords=['depth_backbone']):
        """
        Determine if a parameter belongs to the backbone excluding depth backbone.

        Args:
            param_name (str): Parameter name.
            include_keywords (list): Keywords to include.
            exclude_keywords (list): Keywords to exclude.

        Returns:
            bool: True if parameter matches criteria, False otherwise.
        """
        include_match = any(keyword in param_name for keyword in include_keywords)
        exclude_match = any(keyword in param_name for keyword in exclude_keywords)
        return include_match and not exclude_match
    
    for name, _ in model_without_ddp.named_parameters():
        print(name)
    
    # Train all parameters, and linear projections should have 0.1x learning rate
    param_dicts = [
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if p.requires_grad and not match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if p.requires_grad and match_name_keywords(n, args.lr_linear_proj_names)],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    
    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )
    
    print(f"Using CosineAnnealingLR scheduler with {args.epochs} epochs and eta_min of 0.00001")
    
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=0.00001
    )
    # Alternative scheduler (commented out)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=args.epochs // 2, T_mult=1, eta_min=0.000001
    # )
    
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
    
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    
    output_dir = Path(args.output_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # Load temporal TransVOD weights for multi-frame datasets
            if args.transvod_temporal_weights and args.dataset_file in ["vid_multi", "vid_multi_plusplus"]:
                print(f"Loading temporal weights from {args.transvod_temporal_weights}")
                transvod_checkpoint = torch.load(args.transvod_temporal_weights, map_location='cpu')
                
                # Select keys based on dataset type
                if args.dataset_file == "vid_multi":
                    relevant_keys = ['temporal_query', 'temporal_decoder', 'temp_bbox_embed']
                elif args.dataset_file == "vid_multi_plusplus":
                    relevant_keys = ['temporal_query', 'dynamic_layer', 'temporal_decoder', 'temp_bbox_embed']
                
                # Update checkpoint with temporal weights
                for k, v in transvod_checkpoint['model'].items():
                    if any(key in k for key in relevant_keys):
                        print(f'Moving {k} weights from TransVOD checkpoint to current model')
                        checkpoint['model'][k] = v
            
            # Load spatial TransVOD weights if specified
            if args.spatial_weights:
                print(f"Loading spatial weights from {args.spatial_weights}")
                spatial_checkpoint = torch.load(args.spatial_weights, map_location='cpu')
                for k, v in spatial_checkpoint['model'].items():
                    print(f'Moving {k} weights from spatial checkpoint to current model')
                    checkpoint['model'][k] = v
        
        if args.eval:
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        else:
            tmp_dict = model_without_ddp.state_dict().copy()
            
            if args.coco_pretrain:
                # For single baseline, exclude 'class_embed' parameters
                for k, v in checkpoint['model'].items():
                    if 'class_embed' not in k:
                        tmp_dict[k] = v
                    else:
                        print(f'Excluding parameter: {k}')
            else:
                tmp_dict = checkpoint['model']
                
                for name, param in model_without_ddp.named_parameters():
                    if 'temp' in name or 'dynamic' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(tmp_dict, strict=False)
        
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        
        if missing_keys:
            print(f'Missing Keys: {missing_keys}')
        if unexpected_keys:
            print(f'Unexpected Keys: {unexpected_keys}')
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Trainable Parameter - Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    
    # Initialize Weights & Biases (WandB) for experiment tracking if enabled
    if utils.is_main_process():
        if not args.no_wandb:
            wandb.init(project="transvod-multi", config=vars(args))
            wandb.watch(model, log="all")
    
    print("Start training")
    start_time = time.time()
    
    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )
        
        lr_scheduler.step()
        
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
        
        # Optional: Evaluation after each epoch (commented out)
        # test_stats, coco_evaluator = evaluate(
        #    model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        # )
        
        # Optional: Log training statistics to TensorBoard (commented out)
        # for k, v in train_stats.items():
        #     writer.add_scalar(f'Train/{k}', v, epoch)
        
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'n_parameters': n_parameters
        }
        
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # Log to WandB if enabled
        if utils.is_main_process() and not args.no_wandb:
            wandb.log({"train_loss": train_stats, "epoch": epoch})
    
    # Calculate and print total training time
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time: {total_time_str}')
    
    if utils.is_main_process() and not args.no_wandb:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script',
                                     parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
