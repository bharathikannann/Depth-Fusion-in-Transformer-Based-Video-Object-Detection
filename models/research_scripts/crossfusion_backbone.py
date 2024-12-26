# ------------------------------------------------------------------------
# Backbone CrossFusion module:
# RGB Backbone: ResNet 50
# Depth Backbone: ResNet 18
# Fusion: Deformable Cross Attention with input and output projection layers
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
from itertools import chain

from util.misc import NestedTensor, is_main_process

from models.ops.modules import MSDeformAttn
from ..position_encoding import build_position_encoding
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    
class FusionBackboneBase(nn.Module):
    def __init__(self, rgb_name: str, d_name:str, 
                 rgb_backbone: nn.Module, depth_backbone:nn.Module, position_embedding:nn.Module, 
                 train_backbone: bool, return_interm_layers: bool, 
                 fusion_mode: str, fusion_layers: List[int], 
                 d_model: int, bidirectional: bool,
                 dim_feedforward=1024, dropout=0.1, activation="relu", n_head=8, 
                 fusion_levels=1, fusion_n_points=4):
        super().__init__()
        
        assert rgb_name in ['resnet50'] and d_name in ['resnet18'], "Fusion Backbone not implemented"
        if not train_backbone:
            for param in chain(rgb_backbone.parameters(), depth_backbone.parameters()):
                param.requires_grad = False
        
        self.name = self.rgb_name = rgb_name
        self.d_name = d_name
        self.body = rgb_backbone
        self.d_body = depth_backbone
        self.position_embedding = position_embedding
        self.fusion_mode = fusion_mode
        self.fusion_layers = fusion_layers
        self.return_interm_layers = return_interm_layers
        self.d_model = d_model
        self.bidirectional = bidirectional
        
        self.model_strides = {"resnet18": [2, 8, 16, 32], "resnet50": [2, 4, 16, 32]}
        self.model_num_channels = {"resnet18": [64, 128, 256, 512], "resnet50": [256, 512, 1024, 2048]}
        self.return_layer_no = [2, 3, 4] if return_interm_layers else [4]
        self.strides = [8, 16, 32] if return_interm_layers else [32]
        self.return_layers = {f"layer{i}": str(index) for index, i in enumerate(self.return_layer_no)}
        
        if 0 in self.fusion_layers:
            # projection layers for the 1st fusion module
            self.input_rgb_proj1 = nn.Sequential(
                nn.Conv2d(self.model_num_channels[rgb_name][0], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model)
            )
            self.output_rgb_proj1 = nn.Sequential(
                nn.Conv2d(d_model, self.model_num_channels[rgb_name][0], kernel_size=1),
                nn.GroupNorm(32, self.model_num_channels[rgb_name][0])
            )
            self.input_d_proj1 = nn.Sequential(
                nn.Conv2d(self.model_num_channels[d_name][0], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model)
            )
            self.output_d_proj1 = nn.Sequential(
                nn.Conv2d(d_model, self.model_num_channels[d_name][0], kernel_size=1),
                nn.GroupNorm(32, self.model_num_channels[d_name][0])
            )
        
        if 1 in self.fusion_layers:
            # projection layers for the 2nd fusion module
            self.input_rgb_proj2 = nn.Sequential(
                nn.Conv2d(self.model_num_channels[rgb_name][1], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model)
            )
            self.output_rgb_proj2 = nn.Sequential(
                nn.Conv2d(d_model, self.model_num_channels[rgb_name][1], kernel_size=1),
                nn.GroupNorm(32, self.model_num_channels[rgb_name][1])
            )
            self.input_d_proj2 = nn.Sequential(
                nn.Conv2d(self.model_num_channels[d_name][1], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model)
            )
            self.output_d_proj2 = nn.Sequential(
                nn.Conv2d(d_model, self.model_num_channels[d_name][1], kernel_size=1),
                nn.GroupNorm(32, self.model_num_channels[d_name][1])
            )
        
        if 2 in self.fusion_layers:
            # projection layers for the 3rd fusion module
            self.input_rgb_proj3 = nn.Sequential(
                nn.Conv2d(self.model_num_channels[rgb_name][2], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model)
            )
            self.output_rgb_proj3 = nn.Sequential(
                nn.Conv2d(d_model, self.model_num_channels[rgb_name][2], kernel_size=1),
                nn.GroupNorm(32, self.model_num_channels[rgb_name][2])
            )
            self.input_d_proj3 = nn.Sequential(
                nn.Conv2d(self.model_num_channels[d_name][2], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model)
            )
            self.output_d_proj3 = nn.Sequential(
                nn.Conv2d(d_model, self.model_num_channels[d_name][2], kernel_size=1),
                nn.GroupNorm(32, self.model_num_channels[d_name][2])
            )
        
        if 3 in self.fusion_layers:
            # projection layers for the 4th fusion module
            self.input_rgb_proj4 = nn.Sequential(
                nn.Conv2d(self.model_num_channels[rgb_name][3], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model)
            )
            self.output_rgb_proj4 = nn.Sequential(
                nn.Conv2d(d_model, self.model_num_channels[rgb_name][3], kernel_size=1),
                nn.GroupNorm(32, self.model_num_channels[rgb_name][3])
            )
            self.input_d_proj4 = nn.Sequential(
                nn.Conv2d(self.model_num_channels[d_name][3], d_model, kernel_size=1),
                nn.GroupNorm(32, d_model)
            )
            self.output_d_proj4 = nn.Sequential(
                nn.Conv2d(d_model, self.model_num_channels[d_name][3], kernel_size=1),
                nn.GroupNorm(32, self.model_num_channels[d_name][3])
            )
        
        # Depth to RGB Fusion layers
        if 0 in self.fusion_layers:
            self.d2r_fusion1 = DepthDeformableTransformerEncoderLayer(self.d_model, dim_feedforward, dropout, 
                                                                  activation, fusion_levels, n_head, fusion_n_points) 
        if 1 in self.fusion_layers:
            self.d2r_fusion2 = DepthDeformableTransformerEncoderLayer(self.d_model, dim_feedforward, dropout, 
                                                              activation, fusion_levels, n_head, fusion_n_points)
        if 2 in self.fusion_layers:
            self.d2r_fusion3 = DepthDeformableTransformerEncoderLayer(self.d_model, dim_feedforward, dropout, 
                                                                  activation, fusion_levels, n_head, fusion_n_points)
        if 3 in self.fusion_layers:
            self.d2r_fusion4 = DepthDeformableTransformerEncoderLayer(self.d_model, dim_feedforward, dropout, 
                                                                  activation, fusion_levels, n_head, fusion_n_points)
        
        # RGB to Depth Fusion layers
        if self.bidirectional:
            if 0 in self.fusion_layers:
                self.r2d_fusion1 = DepthDeformableTransformerEncoderLayer(self.d_model, dim_feedforward, dropout, 
                                                                    activation, fusion_levels, n_head, fusion_n_points) 
            if 1 in self.fusion_layers:
                self.r2d_fusion2 = DepthDeformableTransformerEncoderLayer(self.d_model, dim_feedforward, dropout, 
                                                                    activation, fusion_levels, n_head, fusion_n_points)
            if 2 in self.fusion_layers:
                self.r2d_fusion3 = DepthDeformableTransformerEncoderLayer(self.d_model, dim_feedforward, dropout, 
                                                                    activation, fusion_levels, n_head, fusion_n_points)
            if 3 in self.fusion_layers:
                self.r2d_fusion4 = DepthDeformableTransformerEncoderLayer(self.d_model, dim_feedforward, dropout, 
                                                                    activation, fusion_levels, n_head, fusion_n_points)
                
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        # 3. Projection Layers (input/output, RGB/depth) 
        l1 = [self.input_rgb_proj1, self.output_rgb_proj1, self.input_d_proj1, self.output_d_proj1] if 0 in self.fusion_layers else []
        l2 = [self.input_rgb_proj2, self.output_rgb_proj2, self.input_d_proj2, self.output_d_proj2] if 1 in self.fusion_layers else []
        l3 = [self.input_rgb_proj3, self.output_rgb_proj3, self.input_d_proj3, self.output_d_proj3] if 2 in self.fusion_layers else []
        l4 = [self.input_rgb_proj4, self.output_rgb_proj4, self.input_d_proj4, self.output_d_proj4] if 3 in self.fusion_layers else []
        for layer in chain(l1, l2, l3, l4):
            if layer is not None:
                for sublayer in layer:
                    if isinstance(sublayer, nn.Conv2d):
                        nn.init.xavier_uniform_(sublayer.weight, gain=1)
                        nn.init.constant_(sublayer.bias, 0)

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device): 
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points  
    
    @staticmethod
    def fuse_layers(src, target, pos_src, mask_src, mask_target, fusion_layer):
        """
        Args:
            src, target: source features of the source and target
            pos_src, pos_target: positional encodings of the source and target
            mask_src, mask_target: masks of the source and target
            fusion_layer: fusion layer to be used
        """
        src_shape = src.shape
        target_shape = target.shape
        src_flatten= src.flatten(2).transpose(1, 2)
        target_flatten = target.flatten(2).transpose(1, 2)
        pos_src_flatten= pos_src.flatten(2).transpose(1, 2)
        mask_src_flatten= mask_src.flatten(1)
        mask_target_flatten = mask_target.flatten(1)
        
        spatial_shapes_src = [(src_shape[-2], src_shape[-1])]
        spatial_shapes_target = [(target_shape[-2], target_shape[-1])]
        spatial_shapes_src = torch.as_tensor(spatial_shapes_src, dtype=torch.long, device=src.device)
        spatial_shapes_target = torch.as_tensor(spatial_shapes_target, dtype=torch.long, device=target.device)
        lvl_start_index_target = torch.cat((spatial_shapes_target.new_zeros((1,)), spatial_shapes_target.prod(1).cumsum(0)[:-1]))
        
        valid_ratios_src = torch.stack([FusionBackboneBase.get_valid_ratio(m) for m in [mask_src]], 1)
        reference_points_target = FusionBackboneBase.get_reference_points(spatial_shapes_src, valid_ratios_src, target.device)
        
        fused_src = fusion_layer(src_flatten, pos_src_flatten, spatial_shapes_src, reference_points_target, target_flatten, spatial_shapes_target, lvl_start_index_target, mask_src_flatten, mask_target_flatten)
        fused_src = fused_src.transpose(1, 2).view(src.shape)
        return fused_src

    def forward(self, tensor_list: NestedTensor):
        # assert tensor_list.tensors.shape[1] == 4, "Input tensor should have 4 channels for FusionBackboneBase"

        samples_rgb_tensor = tensor_list.tensors[:, :3, :, :]  # 3
        samples_depth_tensor = tensor_list.tensors[:, 3:4, :, :]  # 1
        x_rgb = samples_rgb_tensor
        x_d = samples_depth_tensor
        rgb_out: Dict[str, NestedTensor] = {}
        d_out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None, "Mask should not be None"

        # 1st conv layer, H/4, w/4 (1,3, 566, 999)
        x_rgb = self.body.conv1(x_rgb)  # 3 -> 64 
        x_rgb = self.body.bn1(x_rgb)  # 64 -> 64 
        x_rgb = self.body.relu(x_rgb)  # 64 -> 64 
        x_rgb = self.body.maxpool(x_rgb)  # 64 -> 64 
        
        x_d = self.d_body.conv1(x_d)  # 1 -> 64, 1,3, 566, 999
        x_d = self.d_body.bn1(x_d)  # 64 -> 64 
        x_d = self.d_body.relu(x_d)  # 64 -> 64 
        x_d = self.d_body.maxpool(x_d)  # 64 -> 64 

        x_rgb = self.body.layer1(x_rgb)  # 64 -> 256, (1, 256, 142, 250)
        x_d = self.d_body.layer1(x_d)  # 64 -> 64, (1, 64, 142, 250)
        
        mask_rgb_l1 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        mask_d_l1 = F.interpolate(m[None].float(), size=x_d.shape[-2:]).to(torch.bool)[0]
        if 0 in self.fusion_layers:
            src_rgb_l1 = self.input_rgb_proj1(x_rgb)  # 256 -> 256 
            src_d_l1 = self.input_d_proj1(x_d)  # 64 -> 256 
            pos_rgb_l1 = self.position_embedding(NestedTensor(src_rgb_l1, mask_rgb_l1))  # 256 -> 256 
            pos_depth_l1 = self.position_embedding(NestedTensor(src_d_l1, mask_d_l1))  # 256 -> 256 
            
            fused_rgb_l1 = self.fuse_layers(src_rgb_l1, src_d_l1, pos_rgb_l1, mask_rgb_l1, mask_d_l1, self.d2r_fusion1)  # 256 -> 256
            fused_rgb_l1 = self.output_rgb_proj1(fused_rgb_l1)  # 256 -> 256 
            x_rgb = x_rgb + fused_rgb_l1  # 256 -> 256 
            
            if self.bidirectional:
                fused_d_l1 = self.fuse_layers(src_d_l1, src_rgb_l1, pos_depth_l1, mask_d_l1, mask_rgb_l1, self.r2d_fusion1)  # 256 -> 256
                fused_d_l1 = self.output_d_proj1(fused_d_l1)  # 256 -> 64 
                x_d = x_d + fused_d_l1  # 64 -> 64 
            
        if self.return_interm_layers and "0" in self.return_layers.values():
            rgb_out["0"] = NestedTensor(x_rgb, mask_rgb_l1)
            d_out["0"] = NestedTensor(x_d, mask_d_l1)
                                
        x_rgb = self.body.layer2(x_rgb)  # 256 -> 512 (1, 512, 71, 125)
        x_d = self.d_body.layer2(x_d)  # 64 -> 128 (1, 128, 71, 125)
        
        mask_rgb_l2 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        mask_d_l2 = F.interpolate(m[None].float(), size=x_d.shape[-2:]).to(torch.bool)[0]
        if 1 in self.fusion_layers:
            src_rgb_l2 = self.input_rgb_proj2(x_rgb)  # 512 -> 256 
            src_d_l2 = self.input_d_proj2(x_d)  # 128 -> 256 
            pos_rgb_l2 = self.position_embedding(NestedTensor(src_rgb_l2, mask_rgb_l2))  # 256 -> 256 
            pos_depth_l2 = self.position_embedding(NestedTensor(src_d_l2, mask_d_l2))  # 256 -> 256 
            
            fused_rgb_l2 = self.fuse_layers(src_rgb_l2, src_d_l2, pos_rgb_l2, mask_rgb_l2, mask_d_l2, self.d2r_fusion2)  # 256 -> 256
            fused_rgb_l2 = self.output_rgb_proj2(fused_rgb_l2)  # 256 -> 512 
            x_rgb = x_rgb + fused_rgb_l2  # 512 -> 512 
            
            if self.bidirectional:
                fused_d_l2 = self.fuse_layers(src_d_l2, src_rgb_l2, pos_depth_l2, mask_d_l2, mask_rgb_l2, self.r2d_fusion2)  # 256 -> 256
                fused_d_l2 = self.output_d_proj2(fused_d_l2)  # 256 -> 128 
                x_d = x_d + fused_d_l2  # 128 -> 128 
        
        if self.return_interm_layers and "1" in self.return_layers.values():
            rgb_out["1"] = NestedTensor(x_rgb, mask_rgb_l2)
            d_out["1"] = NestedTensor(x_d, mask_d_l2)

        x_rgb = self.body.layer3(x_rgb)  # 512 -> 1024 (1, 1024, 36, 63)
        x_d = self.d_body.layer3(x_d)  # 128 -> 256 (1, 256, 36, 63)
        
        mask_rgb_l3 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        mask_d_l3 = F.interpolate(m[None].float(), size=x_d.shape[-2:]).to(torch.bool)[0]
        if 2 in self.fusion_layers:
            src_rgb_l3 = self.input_rgb_proj3(x_rgb)  # 1024 -> 256 
            src_d_l3 = self.input_d_proj3(x_d)  # 256 -> 256 
            pos_rgb_l3 = self.position_embedding(NestedTensor(src_rgb_l3, mask_rgb_l3))  # 256 -> 256 
            pos_depth_l3 = self.position_embedding(NestedTensor(src_d_l3, mask_d_l3))  # 256 -> 256 
            
            fused_rgb_l3 = self.fuse_layers(src_rgb_l3, src_d_l3, pos_rgb_l3, mask_rgb_l3, mask_d_l3, self.d2r_fusion3)  # 256 -> 256 
            fused_rgb_l3 = self.output_rgb_proj3(fused_rgb_l3)  # 256 -> 1024 
            x_rgb = x_rgb + fused_rgb_l3  # 1024 -> 1024 
            
            if self.bidirectional:
                fused_d_l3 = self.fuse_layers(src_d_l3, src_rgb_l3, pos_depth_l3, mask_d_l3, mask_rgb_l3, self.r2d_fusion3)  # 256 -> 256
                fused_d_l3 = self.output_d_proj3(fused_d_l3)  # 256 -> 256 
                x_d = x_d + fused_d_l3  # 256 -> 256 
        
        if self.return_interm_layers and "2" in self.return_layers.values():
            rgb_out["2"] = NestedTensor(x_rgb, mask_rgb_l3)
            d_out["2"] = NestedTensor(x_d, mask_d_l3)

        x_rgb = self.body.layer4(x_rgb)  # 1024 -> 2048 (1, 2048, 36, 63)
        x_d = self.d_body.layer4(x_d)  # 256 -> 512 (1, 512, 18, 32)
        
        mask_rgb_l4 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        mask_d_l4 = F.interpolate(m[None].float(), size=x_d.shape[-2:]).to(torch.bool)[0]
        if 3 in self.fusion_layers:
            src_rgb_l4 = self.input_rgb_proj4(x_rgb)  # 2048 -> 256 
            src_d_l4 = self.input_d_proj4(x_d)  # 512 -> 256 
            pos_rgb_l4 = self.position_embedding(NestedTensor(src_rgb_l4, mask_rgb_l4))  # 256 -> 256 
            pos_depth_l4 = self.position_embedding(NestedTensor(src_d_l4, mask_d_l4))  # 256 -> 256 
            
            fused_rgb_l4 = self.fuse_layers(src_rgb_l4, src_d_l4, pos_rgb_l4, mask_rgb_l4, mask_d_l4, self.d2r_fusion4)  # 256 -> 256
            fused_rgb_l4 = self.output_rgb_proj4(fused_rgb_l4)  # 256 -> 2048 
            x_rgb = x_rgb + fused_rgb_l4  # 2048 -> 2048 
            
            if self.bidirectional:
                fused_d_l4 = self.fuse_layers(src_d_l4, src_rgb_l4, pos_depth_l4, mask_d_l4, mask_rgb_l4, self.r2d_fusion4)  # 256 -> 256
                fused_d_l4 = self.output_d_proj4(fused_d_l4)  # 256 -> 512 
                x_d = x_d + fused_d_l4  # 512 -> 512 
        
        if self.return_interm_layers and "3" in self.return_layers.values():
            rgb_out["3"] = NestedTensor(x_rgb, mask_rgb_l4)
            d_out["3"] = NestedTensor(x_d, mask_d_l4)
        else:
            rgb_out["0"] = NestedTensor(x_rgb, mask_rgb_l4)
            d_out["0"] = NestedTensor(x_d, mask_d_l4)

        return rgb_out, d_out

class FusionBackbone(FusionBackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, rgb_name: str,
                 d_name: str,
                 train_backbone: bool,
                 position_embedding: nn.Module,
                 return_interm_layers: bool,
                 dilation: bool,
                 depth_type: str,
                 fusion_layers: List[int],
                 d_model: int,
                 bidirectional: bool):
        
        norm_layer = FrozenBatchNorm2d
        rgb_backbone = getattr(torchvision.models, rgb_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer)
        if d_name in ['resnet18', 'resnet34']:
            depth_backbone = getattr(torchvision.models, d_name)(
                pretrained=is_main_process(), norm_layer=norm_layer)
            
            # take the mean of first 3 channels for the depth backbone
            original_conv1 = depth_backbone.conv1
            new_conv1 = torch.nn.Conv2d(1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size, 
                                            stride=original_conv1.stride, padding=original_conv1.padding, bias=False)        
            with torch.no_grad():
                new_conv1.weight[:, 0] = torch.mean(original_conv1.weight, dim=1) # 1st channel is the mean of the first 3 channels
            depth_backbone.conv1 = new_conv1
            
        else:
            raise NotImplementedError(f"Backbone {d_name} not implemented")
        
        super().__init__(rgb_name, d_name, rgb_backbone, depth_backbone, position_embedding, train_backbone, return_interm_layers, fusion_mode=depth_type, fusion_layers=fusion_layers, d_model=d_model, bidirectional=bidirectional)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.model_num_channels
        self.name = self.rgb_name = backbone.name
        self.d_name = backbone.d_name

    def forward(self, tensor_list: NestedTensor):
        xs, xd = self[0](tensor_list)
        out: List[NestedTensor] = []
        d_out: List[NestedTensor] = []
        pos = []
        d_pos = []
        for name, x in sorted(xs.items()):
            out.append(x)
        for name, x in sorted(xd.items()):
            d_out.append(x)
        
        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        
        for x in d_out:
            d_pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos, d_out, d_pos
    
class DepthDeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model = 256, d_ffn=1024, dropout=0.1, 
                 activation='relu', n_depth_levels = 1, n_heads = 8, dpth_n_points=4):
        super().__init__()

        # cross attention 
        self.cross_attn = MSDeformAttn(d_model, n_depth_levels, n_heads, dpth_n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        # self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.self_attn = MSDeformAttn(d_model, n_depth_levels, n_heads, dpth_n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, tgt_spatial_shapes, reference_points, src, src_spatial_shapes, frame_start_index, tgt_padding_mask=None, src_padding_mask=None):
        # self attention
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), key_padding_mask=tgt_padding_mask)[0].transpose(0, 1)
        tgt2 = self.self_attn(self.with_pos_embed(tgt, query_pos), reference_points, tgt, tgt_spatial_shapes, frame_start_index, tgt_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
    
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, frame_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    

def build_fusion_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    d_model = 256
    d_name = 'resnet18'
    fusion_layers = [2,3] if "crossfusion" in args.depth_type else [3]
    bidirectional = True if "crossfusion_2way" in args.depth_type else False
    print(f"Building Fusion Backbone with {args.backbone} and {d_name} backbone")
    print("Fusion Layers: ", fusion_layers)
    print("Bidirectional: ", bidirectional)
    
    backbone = FusionBackbone(args.backbone, d_name, train_backbone, position_embedding, return_interm_layers, args.dilation, args.depth_type, fusion_layers, d_model, bidirectional)
    model = Joiner(backbone, position_embedding)  
    return model  
