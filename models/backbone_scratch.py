# ------------------------------------------------------------------------
# Resnet Backbone forward pass, implemented from scratch.
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
from .position_encoding import build_position_encoding


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
    
        assert rgb_name in ['resnet50'], f"Backbone {rgb_name} not supported"
        if not train_backbone:
            for param in rgb_backbone.parameters():
                param.requires_grad = False
        
        self.name = self.rgb_name = rgb_name
        self.d_name = d_name
        self.body = rgb_backbone
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

    def forward(self, tensor_list: NestedTensor):
        # assert tensor_list.tensors.shape[1] == 4, "Input tensor should have 4 channels for FusionBackboneBase"

        samples_rgb_tensor = tensor_list.tensors[:, :3, :, :]  # 3
        x_rgb = samples_rgb_tensor
        rgb_out: Dict[str, NestedTensor] = {}
        m = tensor_list.mask
        assert m is not None, "Mask should not be None"

        # 1st conv layer, H/4, w/4 (1,3, 566, 999)
        x_rgb = self.body.conv1(x_rgb)  # 3 -> 64 
        x_rgb = self.body.bn1(x_rgb)  # 64 -> 64 
        x_rgb = self.body.relu(x_rgb)  # 64 -> 64 
        x_rgb = self.body.maxpool(x_rgb)  # 64 -> 64 
        
        x_rgb = self.body.layer1(x_rgb)  # 64 -> 256, (1, 256, 142, 250)
        
        mask_rgb_l1 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        if self.return_interm_layers and "0" in self.return_layers.values():
            rgb_out["0"] = NestedTensor(x_rgb, mask_rgb_l1)
                                
        x_rgb = self.body.layer2(x_rgb)  # 256 -> 512 
        
        mask_rgb_l2 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        if self.return_interm_layers and "1" in self.return_layers.values():
            rgb_out["1"] = NestedTensor(x_rgb, mask_rgb_l2)

        x_rgb = self.body.layer3(x_rgb)  # 512 -> 1024 
        mask_rgb_l3 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        if self.return_interm_layers and "2" in self.return_layers.values():
            rgb_out["2"] = NestedTensor(x_rgb, mask_rgb_l3)
        
        x_rgb = self.body.layer4(x_rgb)  # 1024 -> 2048 
        mask_rgb_l4 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        if self.return_interm_layers and "3" in self.return_layers.values():
            rgb_out["3"] = NestedTensor(x_rgb, mask_rgb_l4)
        else:
            rgb_out["0"] = NestedTensor(x_rgb, mask_rgb_l4)

        return rgb_out, None

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
        
        depth_backbone = None
        
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
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)
        
        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos, None, None
    

def build_backbone_fromscratch(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)
    d_model = 256
    d_name = 'resnet18'
    fusion_layers = [1,2,3] if "crossfusion" in args.depth_type else [3]
    bidirectional = True if "2way" in args.depth_type else False
    
    backbone = FusionBackbone(args.backbone, d_name, train_backbone, position_embedding, return_interm_layers, args.dilation, args.depth_type, fusion_layers, d_model, bidirectional)
    model = Joiner(backbone, position_embedding)  
    return model  
