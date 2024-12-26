# Backbone CrossFusion Module Configuration:
# RGB Backbone: ResNet 50
# Depth Backbone: Dformer Depth Backbone
# Fusion Mechanism: Deformable Cross Attention with projection layers
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import os
from itertools import chain
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.nn.init import xavier_uniform_, constant_
from torchvision.models._utils import IntermediateLayerGetter

from models.ops.modules import MSDeformAttn
from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


def _get_activation_fn(activation):
    """Return activation function based on the string name."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"Unsupported activation: {activation}")


class FrozenBatchNorm2d(nn.Module):
    """BatchNorm2d with fixed batch statistics and affine parameters."""

    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        # Register buffers for weights and biases
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs
    ):
        # Remove 'num_batches_tracked' if present
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # Reshape parameters for broadcasting
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        # Compute scale and bias
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class DownsamplePath(nn.Module):
    """Path for downsampling depth input."""

    def __init__(self, in_channels, dims, train_backbone=True, freeze_batchnorm=False):
        super(DownsamplePath, self).__init__()
        self.downsample_layers_e = nn.ModuleList()

        # Initial stem layer with two convolutional blocks
        stem_e = nn.Sequential(
            nn.Conv2d(in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            self._get_bn_layer(dims[0] // 2, freeze_batchnorm),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            self._get_bn_layer(dims[0], freeze_batchnorm),
        )
        self.downsample_layers_e.append(stem_e)

        # Additional downsampling layers
        for i in range(len(dims) - 1):
            downsample_layer = nn.Sequential(
                self._get_bn_layer(dims[i], freeze_batchnorm),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
            )
            self.downsample_layers_e.append(downsample_layer)

        # Freeze backbone if required
        if not train_backbone:
            for param in self.downsample_layers_e.parameters():
                param.requires_grad = False

    def _get_bn_layer(self, num_features, freeze_batchnorm):
        """Return a BatchNorm layer."""
        bn = nn.BatchNorm2d(num_features)
        return bn

    def forward(self, x):
        """Apply downsampling layers sequentially."""
        for layer in self.downsample_layers_e:
            x = layer(x)
        return x


class DepthDeformableTransformerEncoderLayer(nn.Module):
    """Transformer encoder layer for depth fusion."""

    def __init__(
        self, d_model=256, d_ffn=1024, dropout=0.1,
        activation='relu', n_depth_levels=1, n_heads=8,
        dpth_n_points=4, depth_self_attn=True
    ):
        super(DepthDeformableTransformerEncoderLayer, self).__init__()
        self.depth_self_attn = depth_self_attn

        # Cross attention layer
        self.cross_attn = MSDeformAttn(d_model, n_depth_levels, n_heads, dpth_n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = _get_activation_fn(activation)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # Scale adaptation layers
        self.depth_scale_adapt = nn.Linear(d_model, d_model)
        self.norm_depth_scale = nn.LayerNorm(d_model)
        self.cross_scale_adapt = nn.Linear(d_model, d_model)

    def with_pos_embed(self, tensor, pos):
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Forward pass through the feed-forward network."""
        tgt2 = self.activation(self.linear1(tgt))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self, tgt, query_pos, src_pos, tgt_spatial_shapes,
        reference_points, depth_reference_points, src,
        src_spatial_shapes, frame_start_index,
        tgt_padding_mask=None, src_padding_mask=None
    ):
        # Adapt depth features
        src = self.depth_scale_adapt(src)
        src = self.norm_depth_scale(src)

        # Apply cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src, src_spatial_shapes,
            frame_start_index, src_padding_mask
        )
        tgt2 = self.cross_scale_adapt(tgt2)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Apply feed-forward network
        tgt = self.forward_ffn(tgt)
        return tgt


class FusionBackboneBase(nn.Module):
    """Base class for Fusion Backbone with RGB and Depth backbones."""

    def __init__(
        self, rgb_name: str, d_name: str,
        rgb_backbone: nn.Module, depth_backbone: nn.Module,
        position_embedding: nn.Module, train_backbone: bool,
        return_interm_layers: bool, fusion_mode: str,
        fusion_layers: List[int], d_model: int, bidirectional: bool,
        dim_feedforward=1024, dropout=0.1, activation="relu",
        n_head=8, fusion_levels=1, fusion_n_points=4,
        depth_pretrained_path=None, eval=False
    ):
        super(FusionBackboneBase, self).__init__()
        assert rgb_name in ['resnet50'] and d_name in ['dformer'], "Fusion Backbone not implemented"

        # Freeze backbone parameters if not training
        if not train_backbone:
            for param in chain(rgb_backbone.parameters(), depth_backbone.parameters()):
                param.requires_grad = False

        # Initialize attributes
        self.rgb_name = rgb_name
        self.d_name = d_name
        self.body = rgb_backbone
        self.d_body = depth_backbone
        self.position_embedding = position_embedding
        self.fusion_mode = fusion_mode
        self.fusion_layers = fusion_layers
        self.return_interm_layers = return_interm_layers
        self.d_model = d_model
        self.bidirectional = bidirectional
        self.depth_self_attn = True

        # Define strides and channels for models
        self.model_strides = {
            "resnet18": [2, 8, 16, 32],
            "resnet50": [2, 4, 16, 32],
            "dformer": [2**(i+2) for i in range(3)]
        }
        self.model_num_channels = {
            "resnet18": [64, 128, 256, 512],
            "resnet50": [256, 512, 1024, 2048],
            'dformer': [32, 64, 128, 256]
        }
        self.return_layer_no = [3, 4] if return_interm_layers else [4]
        self.strides = [8, 16, 32] if return_interm_layers else [32]
        self.depth_strides = [2**(i+2) for i in range(3)]
        self.depth_num_channels = [32, 64, 128, 256]
        self.return_layers = {f"layer{i}": str(index) for index, i in enumerate(self.return_layer_no)}

        # Load pretrained depth weights if provided
        if depth_pretrained_path and not eval:
            self.load_pretrained_weights(self.d_body, depth_pretrained_path)
            print("DFormer backbone initialized")

        # Ensure unsupported fusion layers are not used
        assert not any(layer in self.fusion_layers for layer in (0, 1)), \
            "Fusion layers 0 and 1 not supported as dformer has only 2 levels"

        # Initialize projection and fusion layers based on fusion_layers
        for layer in self.fusion_layers:
            if layer in [2, 3, 4]:
                rgb_channel = self.model_num_channels[rgb_name][layer - 2]
                depth_channel = self.model_num_channels[d_name][layer - 2]
                norm_rgb = 32
                norm_depth = {2: 4, 3: 8, 4: 16}[layer]
                self._init_fusion_layer(layer, rgb_channel, depth_channel, norm_rgb, norm_depth, d_model)

                # Initialize Depth to RGB fusion layer
                setattr(self, f'd2r_fusion{layer}', DepthDeformableTransformerEncoderLayer(
                    d_model, dim_feedforward, dropout, activation,
                    fusion_levels, n_head, fusion_n_points, depth_self_attn=self.depth_self_attn
                ))

                # Initialize RGB to Depth fusion layer if bidirectional
                if self.bidirectional:
                    setattr(self, f'r2d_fusion{layer}', DepthDeformableTransformerEncoderLayer(
                        d_model, dim_feedforward, dropout, activation,
                        fusion_levels, n_head, fusion_n_points, depth_self_attn=self.depth_self_attn
                    ))

        self._reset_parameters()

    def _init_fusion_layer(self, layer, rgb_channel, depth_channel, norm_rgb, norm_depth, d_model):
        """Initialize input and output projection layers for a fusion layer."""
        setattr(self, f'input_rgb_proj{layer}', nn.Sequential(
            nn.Conv2d(rgb_channel, d_model, kernel_size=1),
            nn.GroupNorm(32, d_model)
        ))
        setattr(self, f'output_rgb_proj{layer}', nn.Sequential(
            nn.Conv2d(d_model, rgb_channel, kernel_size=1),
            nn.GroupNorm(32, rgb_channel)
        ))
        setattr(self, f'input_d_proj{layer}', nn.Sequential(
            nn.Conv2d(depth_channel, d_model, kernel_size=1),
            nn.GroupNorm(norm_depth, d_model)
        ))
        setattr(self, f'output_d_proj{layer}', nn.Sequential(
            nn.Conv2d(d_model, depth_channel, kernel_size=1),
            nn.GroupNorm(norm_depth, depth_channel)
        ))

    def _reset_parameters(self):
        """Initialize parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # Initialize projection layers
        for layer_num in self.fusion_layers:
            input_proj = getattr(self, f'input_rgb_proj{layer_num}', None)
            if input_proj:
                for sublayer in input_proj:
                    if isinstance(sublayer, nn.Conv2d):
                        nn.init.xavier_uniform_(sublayer.weight, gain=1)
                        nn.init.constant_(sublayer.bias, 0)
            output_proj = getattr(self, f'output_rgb_proj{layer_num}', None)
            if output_proj:
                for sublayer in output_proj:
                    if isinstance(sublayer, nn.Conv2d):
                        nn.init.xavier_uniform_(sublayer.weight, gain=1)
                        nn.init.constant_(sublayer.bias, 0)
            input_d_proj = getattr(self, f'input_d_proj{layer_num}', None)
            if input_d_proj:
                for sublayer in input_d_proj:
                    if isinstance(sublayer, nn.Conv2d):
                        nn.init.xavier_uniform_(sublayer.weight, gain=1)
                        nn.init.constant_(sublayer.bias, 0)
            output_d_proj = getattr(self, f'output_d_proj{layer_num}', None)
            if output_d_proj:
                for sublayer in output_d_proj:
                    if isinstance(sublayer, nn.Conv2d):
                        nn.init.xavier_uniform_(sublayer.weight, gain=1)
                        nn.init.constant_(sublayer.bias, 0)

    def load_pretrained_weights(self, model, pretrained_weights_path, prefix="downsample_layers_e"):
        """Load pretrained weights for the depth backbone."""
        if not os.path.exists(pretrained_weights_path):
            print(f"Invalid path: {pretrained_weights_path}")
            return
        dformer_weights = torch.load(pretrained_weights_path)['state_dict']
        print(f"Loading pretrained weights from {pretrained_weights_path}")
        loaded_layers = set()
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                for dformer_name, dformer_param in dformer_weights.items():
                    if prefix in dformer_name and name in dformer_name:
                        if 'running' not in dformer_name and 'num_batches_tracked' not in dformer_name:
                            if module.weight.shape == dformer_param.shape:
                                print(f"Loading: {dformer_name} into {name}")
                                module.weight.data = dformer_param.data.clone()
                                loaded_layers.add(name)
                            if module.bias is not None and "bias" in dformer_name:
                                print(f"Loading: {dformer_name} into {name}")
                                module.bias.data = dformer_param.data.clone()
                                loaded_layers.add(name)

        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                if name not in loaded_layers:
                    print(f"Warning: Layer {name} was not loaded with pretrained weights.")

    @staticmethod
    def get_valid_ratio(mask):
        """Compute valid ratios of the mask."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], dim=1)
        valid_W = torch.sum(~mask[:, 0, :], dim=1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Generate reference points for attention."""
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, dim=1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @staticmethod
    def fuse_layers(src, target, pos_src, pos_target, mask_src, mask_target, fusion_layer):
        """Fuse source and target layers using the fusion_layer."""
        src_shape = src.shape
        target_shape = target.shape

        # Flatten spatial dimensions
        src_flatten = src.flatten(2).transpose(1, 2)
        target_flatten = target.flatten(2).transpose(1, 2)
        pos_src_flatten = pos_src.flatten(2).transpose(1, 2)
        pos_target_flatten = pos_target.flatten(2).transpose(1, 2)
        mask_src_flatten = mask_src.flatten(1)
        mask_target_flatten = mask_target.flatten(1)

        # Define spatial shapes
        spatial_shapes_src = [(src_shape[-2], src_shape[-1])]
        spatial_shapes_target = [(target_shape[-2], target_shape[-1])]
        spatial_shapes_src = torch.as_tensor(spatial_shapes_src, dtype=torch.long, device=src.device)
        spatial_shapes_target = torch.as_tensor(spatial_shapes_target, dtype=torch.long, device=target.device)
        lvl_start_index_target = torch.cat(
            (spatial_shapes_target.new_zeros((1,)), spatial_shapes_target.prod(1).cumsum(0)[:-1])
        )

        # Compute valid ratios and reference points
        valid_ratios_src = torch.stack([FusionBackboneBase.get_valid_ratio(m) for m in [mask_src]], dim=1)
        valid_ratios_target = torch.stack([FusionBackboneBase.get_valid_ratio(m) for m in [mask_target]], dim=1)
        reference_points_target = FusionBackboneBase.get_reference_points(
            spatial_shapes_src, valid_ratios_target, target.device
        )
        reference_points_src = FusionBackboneBase.get_reference_points(
            spatial_shapes_target, valid_ratios_src, src.device
        )

        # Apply fusion layer
        fused_src = fusion_layer(
            src_flatten, pos_src_flatten, pos_target_flatten,
            spatial_shapes_src, reference_points_target,
            reference_points_src, target_flatten, spatial_shapes_target,
            lvl_start_index_target, mask_src_flatten, mask_target_flatten
        )
        fused_src = fused_src.transpose(1, 2).view(src.shape)
        return fused_src

    def forward(self, tensor_list: NestedTensor):
        """Forward pass for FusionBackboneBase."""
        # Split RGB and Depth tensors
        samples_rgb_tensor = tensor_list.tensors[:, :3, :, :]
        samples_depth_tensor = tensor_list.tensors[:, 3:4, :, :]

        # Process RGB through backbone
        x_rgb = self.body.conv1(samples_rgb_tensor)
        x_rgb = self.body.bn1(x_rgb)
        x_rgb = self.body.relu(x_rgb)
        x_rgb = self.body.maxpool(x_rgb)

        x_rgb = self.body.layer1(x_rgb)
        x_rgb = self.body.layer2(x_rgb)

        # Process Depth through downsample path
        x_d = self.d_body.downsample_layers_e[0](samples_depth_tensor)

        # Prepare masks
        m = tensor_list.mask
        assert m is not None, "Mask should not be None"
        mask_rgb_l2 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        mask_d_l2 = F.interpolate(m[None].float(), size=x_d.shape[-2:]).to(torch.bool)[0]

        rgb_out = {}
        d_out = {}

        # Fuse layer 2 if enabled
        if 2 in self.fusion_layers:
            # Project RGB and Depth features
            src_rgb_l2 = self.input_rgb_proj2(x_rgb)
            src_d_l2 = self.input_d_proj2(x_d)

            # Compute positional embeddings
            pos_rgb_l2 = self.position_embedding(NestedTensor(src_rgb_l2, mask_rgb_l2))
            pos_depth_l2 = self.position_embedding(NestedTensor(src_d_l2, mask_d_l2))

            # Fuse RGB with Depth
            fused_rgb_l2 = self.fuse_layers(
                src_rgb_l2, src_d_l2, pos_rgb_l2, pos_depth_l2,
                mask_rgb_l2, mask_d_l2, self.d2r_fusion2
            )
            fused_rgb_l2 = self.output_rgb_proj2(fused_rgb_l2)
            x_rgb = x_rgb + fused_rgb_l2

            # Optionally fuse Depth with RGB if bidirectional
            if self.bidirectional:
                fused_d_l2 = self.fuse_layers(
                    src_d_l2, src_rgb_l2, pos_depth_l2, pos_rgb_l2,
                    mask_d_l2, mask_rgb_l2, self.r2d_fusion2
                )
                fused_d_l2 = self.output_d_proj2(fused_d_l2)
                x_d = x_d + fused_d_l2

        # Continue processing RGB and Depth
        x_rgb = self.body.layer3(x_rgb)
        x_d = self.d_body.downsample_layers_e[1](x_d)

        # Prepare masks for layer 3
        mask_rgb_l3 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        mask_d_l3 = F.interpolate(m[None].float(), size=x_d.shape[-2:]).to(torch.bool)[0]

        # Fuse layer 3 if enabled
        if 3 in self.fusion_layers:
            src_rgb_l3 = self.input_rgb_proj3(x_rgb)
            src_d_l3 = self.input_d_proj3(x_d)

            pos_rgb_l3 = self.position_embedding(NestedTensor(src_rgb_l3, mask_rgb_l3))
            pos_depth_l3 = self.position_embedding(NestedTensor(src_d_l3, mask_d_l3))

            fused_rgb_l3 = self.fuse_layers(
                src_rgb_l3, src_d_l3, pos_rgb_l3, pos_depth_l3,
                mask_rgb_l3, mask_d_l3, self.d2r_fusion3
            )
            fused_rgb_l3 = self.output_rgb_proj3(fused_rgb_l3)
            x_rgb = x_rgb + fused_rgb_l3

            # Optionally fuse Depth with RGB if bidirectional
            if self.bidirectional:
                fused_d_l3 = self.fuse_layers(
                    src_d_l3, src_rgb_l3, pos_depth_l3, pos_rgb_l3,
                    mask_d_l3, mask_rgb_l3, self.r2d_fusion3
                )
                fused_d_l3 = self.output_d_proj3(fused_d_l3)
                x_d = x_d + fused_d_l3

        # Save intermediate layers if required
        if self.return_interm_layers and "2" in self.return_layers.values():
            rgb_out["2"] = NestedTensor(x_rgb, mask_rgb_l3)
            d_out["2"] = NestedTensor(x_d, mask_d_l3)

        # Final layers of RGB and Depth backbones
        x_rgb = self.body.layer4(x_rgb)
        x_d = self.d_body.downsample_layers_e[2](x_d)

        # Prepare masks for layer 4
        mask_rgb_l4 = F.interpolate(m[None].float(), size=x_rgb.shape[-2:]).to(torch.bool)[0]
        mask_d_l4 = F.interpolate(m[None].float(), size=x_d.shape[-2:]).to(torch.bool)[0]

        # Fuse layer 4 if enabled
        if 4 in self.fusion_layers:
            src_rgb_l4 = self.input_rgb_proj4(x_rgb)
            src_d_l4 = self.input_d_proj4(x_d)

            pos_rgb_l4 = self.position_embedding(NestedTensor(src_rgb_l4, mask_rgb_l4))
            pos_depth_l4 = self.position_embedding(NestedTensor(src_d_l4, mask_d_l4))

            fused_rgb_l4 = self.fuse_layers(
                src_rgb_l4, src_d_l4, pos_rgb_l4, pos_depth_l4,
                mask_rgb_l4, mask_d_l4, self.d2r_fusion4
            )
            fused_rgb_l4 = self.output_rgb_proj4(fused_rgb_l4)
            x_rgb = x_rgb + fused_rgb_l4

            # Optionally fuse Depth with RGB if bidirectional
            if self.bidirectional:
                fused_d_l4 = self.fuse_layers(
                    src_d_l4, src_rgb_l4, pos_depth_l4, pos_rgb_l4,
                    mask_d_l4, mask_rgb_l4, self.r2d_fusion4
                )
                fused_d_l4 = self.output_d_proj4(fused_d_l4)
                x_d = x_d + fused_d_l4

        # Save final intermediate layers if required
        if self.return_interm_layers and "3" in self.return_layers.values():
            rgb_out["3"] = NestedTensor(x_rgb, mask_rgb_l4)
            d_out["3"] = NestedTensor(x_d, mask_d_l4)
        else:
            rgb_out["0"] = NestedTensor(x_rgb, mask_rgb_l4)
            d_out["0"] = NestedTensor(x_d, mask_d_l4)

        return rgb_out, d_out


class FusionBackbone(FusionBackboneBase):
    """Fusion Backbone using ResNet with frozen BatchNorm."""

    def __init__(
        self, rgb_name: str, d_name: str, train_backbone: bool,
        position_embedding: nn.Module, return_interm_layers: bool,
        dilation: bool, depth_type: str, fusion_layers: List[int],
        d_model: int, bidirectional: bool,
        depth_pretrained_path=None, eval=False
    ):
        # Use FrozenBatchNorm2d for normalization
        norm_layer = FrozenBatchNorm2d

        # Initialize RGB backbone (ResNet)
        rgb_backbone = getattr(torchvision.models, rgb_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=norm_layer
        )

        # Initialize Depth backbone based on d_name
        if d_name in ['resnet18', 'resnet34']:
            depth_backbone = getattr(torchvision.models, d_name)(
                pretrained=is_main_process(), norm_layer=norm_layer
            )
        elif d_name == 'dformer':
            depth_backbone = DownsamplePath(
                in_channels=1, dims=[32, 64, 128, 256],
                train_backbone=True, freeze_batchnorm=False
            )
        else:
            raise NotImplementedError(f"Backbone {d_name} not implemented")

        # Initialize the base FusionBackbone
        super(FusionBackbone, self).__init__(
            rgb_name, d_name, rgb_backbone, depth_backbone,
            position_embedding, train_backbone, return_interm_layers,
            fusion_mode=depth_type, fusion_layers=fusion_layers,
            d_model=d_model, bidirectional=bidirectional,
            depth_pretrained_path=depth_pretrained_path, eval=eval
        )

        # Adjust strides if dilation is used
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    """Combine backbone and position embedding into a single module."""

    def __init__(self, backbone, position_embedding):
        super(Joiner, self).__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.model_num_channels
        self.rgb_name = backbone.rgb_name
        self.d_name = backbone.d_name

    def forward(self, tensor_list: NestedTensor):
        """Forward pass combining backbone and position embedding."""
        xs, xd = self[0](tensor_list)
        out = [x for name, x in sorted(xs.items())]
        d_out = [x for name, x in sorted(xd.items())]

        # Compute positional embeddings for RGB and Depth
        pos = [self[1](x).to(x.tensors.dtype) for x in out]
        d_pos = [self[1](x).to(x.tensors.dtype) for x in d_out]

        return out, pos, d_out, d_pos


def build_dformer_fusion_backbone(args):
    """Build the Fusion Backbone model with DFormer."""
    # Create position encoding
    position_embedding = build_position_encoding(args)

    # Determine if backbone should be trained
    train_backbone = args.lr_backbone > 0

    # Check if intermediate layers should be returned
    return_interm_layers = args.masks or (args.num_feature_levels > 1)

    # Define model parameters
    d_model = 256
    d_name = 'dformer'
    fusion_layers = [2, 3, 4] if "crossfusion" in args.depth_type else [4]
    bidirectional = "crossfusion_2way" in args.depth_type

    print(f"Building Fusion Backbone with {args.backbone} and {d_name} backbone")
    print("Fusion Layers:", fusion_layers)
    print("Bidirectional:", bidirectional)

    # Get pretrained depth weights path if available
    depth_pretrained_path = getattr(args, 'dformer_weights', False)
    eval_mode = False

    # Initialize FusionBackbone
    backbone = FusionBackbone(
        rgb_name=args.backbone, d_name=d_name, train_backbone=train_backbone,
        position_embedding=position_embedding, return_interm_layers=return_interm_layers,
        dilation=args.dilation, depth_type=args.depth_type, fusion_layers=fusion_layers,
        d_model=d_model, bidirectional=bidirectional,
        depth_pretrained_path=depth_pretrained_path, eval=eval_mode
    )

    # Combine backbone and position embedding using Joiner
    model = Joiner(backbone, position_embedding)
    return model