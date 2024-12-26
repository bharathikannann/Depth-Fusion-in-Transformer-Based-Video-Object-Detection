# Resnet 18 depth backbone
# not used in the final model, this is just a research code
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from ..position_encoding import build_position_encoding


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


class DepthBackboneBase(nn.Module):

    def __init__(self, name: str, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        if not train_backbone:
            for param in backbone.parameters():
                param.requires_grad = False
        
        if return_interm_layers:
            if name in ['resnet18', 'resnet34']:
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [128, 256, 512]
            else:
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.strides = [8, 16, 32]
                self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer3': "0"}
            self.strides = [16]
            self.num_channels = [256] if name in ['resnet18', 'resnet34'] else [2048]
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
    
# # Custom forward pass for the DepthBackboneBase class
# class DepthBackboneBaseCustom(nn.Module):
    
#     def __init__(self, name: str, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
#         super().__init__()
#         if not train_backbone:
#             for param in backbone.parameters():
#                 param.requires_grad = False

#         self.backbone = backbone
#         self.return_interm_layers = return_interm_layers
        
#         if return_interm_layers:
#             self.return_layers = {"layer1":"0", "layer2": "1", "layer3": "2", "layer4": "3"}
#             if name in ['resnet18', 'resnet34']:
#                 self.strides = [2, 8, 16, 32]
#                 self.num_channels = [64, 128, 256, 512]
#             else:
#                 self.strides = [2, 8, 16, 32]
#                 self.num_channels = [64, 512, 1024, 2048]
#         else:
#             self.return_layers = {'layer3': "0"}
#             if name in ['resnet18', 'resnet34']:
#                 self.strides = [16]
#                 self.num_channels = [256]
#             else:
#                 self.strides = [32]
#                 self.num_channels = [2048]            

#     def forward(self, tensor_list: NestedTensor):
#         x = tensor_list.tensors
#         out: Dict[str, NestedTensor] = {}

#         x = self.backbone.conv1(x)
#         x = self.backbone.bn1(x)
#         x = self.backbone.relu(x)
#         x = self.backbone.maxpool(x)

#         x = self.backbone.layer1(x)
#         if self.return_interm_layers or "0" in self.return_layers.values():
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             out["0"] = NestedTensor(x, mask)

#         x = self.backbone.layer2(x)
#         if self.return_interm_layersor or "1" in self.return_layers.values():
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             out["1"] = NestedTensor(x, mask)

#         x = self.backbone.layer3(x)
#         if self.return_interm_layers or "2" in self.return_layers.values():
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             out["2"] = NestedTensor(x, mask)

#         x = self.backbone.layer4(x)
#         if self.return_interm_layers or "3" in self.return_layers.values():
#             m = tensor_list.mask
#             assert m is not None
#             mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
#             key = "3" if self.return_interm_layers else "0"
#             out[key] = NestedTensor(x, mask)

#         return out
    

class DepthBackbone(DepthBackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # norm_layer = FrozenBatchNorm2d
        norm_layer = nn.BatchNorm2d
        depth_backbone= getattr(torchvision.models, name)(
            pretrained=is_main_process(), norm_layer=norm_layer)
        
        original_conv1 = depth_backbone.conv1
        new_conv1 = torch.nn.Conv2d(1, original_conv1.out_channels, kernel_size=original_conv1.kernel_size, 
                                        stride=original_conv1.stride, padding=original_conv1.padding, bias=False)        
        with torch.no_grad():
            new_conv1.weight[:, 0] = torch.mean(original_conv1.weight, dim=1) # 1st channel is the mean of the first 3 channels
        depth_backbone.conv1 = new_conv1
        
        super().__init__(name, depth_backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_depth_backbone(args):
    position_embedding = build_position_encoding(args)
    train_depth_backbone = True # Always train the depth backbone
    return_interm_layers = False  # Always return intermediate layers
    depth_backbone_name = 'resnet18'
    dilation = 1
    
    # For all model types
    depth_backbone = DepthBackbone(depth_backbone_name, train_depth_backbone, return_interm_layers, dilation)
    model = Joiner(depth_backbone, position_embedding)  
    return model  