# DFormer depth backbone implementation
# ---------------------------------------------------------
# Adapted from  DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation
# https://github.com/VCIP-RGBD/DFormer

import os
from typing import Dict, List

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding


class DownsamplePath(nn.Module):
    """
    DownsamplePath module for the DFormer backbone.
    """
    def __init__(self, in_channels: int, dims: List[int], train_backbone: bool = True, freeze_batchnorm: bool = False):
        """
        Args:
            in_channels (int): Number of input channels.
            dims (List[int]): List of dimensions for each downsampling layer.
            train_backbone (bool): Flag to indicate if the backbone should be trained.
            freeze_batchnorm (bool): If True, freezes the BatchNorm layers.
        """
        super().__init__()
        self.downsample_layers_e = nn.ModuleList()

        # Initial stem layer with two convolutional layers and GELU activation
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            self._get_bn_layer(dims[0] // 2, freeze_batchnorm),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            self._get_bn_layer(dims[0], freeze_batchnorm),
        )
        self.downsample_layers_e.append(stem)

        # Additional downsampling layers based on the provided dimensions
        for i in range(len(dims) - 1):
            stride = 2  # Stride of 2 for downsampling
            downsample_layer = nn.Sequential(
                self._get_bn_layer(dims[i], freeze_batchnorm),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=stride, padding=1),
            )
            self.downsample_layers_e.append(downsample_layer)

        # Freeze backbone parameters if training is not required
        if not train_backbone:
            for param in self.downsample_layers_e.parameters():
                param.requires_grad = False

    def _get_bn_layer(self, num_features: int, freeze_batchnorm: bool) -> nn.BatchNorm2d:
        """
        Args:
            num_features (int): Number of features for BatchNorm.
            freeze_batchnorm (bool): If True, freezes the BatchNorm parameters.

        Returns:
            nn.BatchNorm2d: Configured BatchNorm layer.
        """
        bn = nn.BatchNorm2d(num_features)
        if freeze_batchnorm:
            bn.eval()  # Set BatchNorm to evaluation mode
            for param in bn.parameters():
                param.requires_grad = False  # Freeze BatchNorm parameters
        return bn


class DFormerBackbone(nn.Module):
    """
    DFormerBackbone serves as the depth feature extractor using the DFormer architecture.
    """
    def __init__(
        self,
        dims: List[int] = (32, 64, 128, 256),
        train_backbone: bool = True,
        return_interm_layers: bool = False,
        pretrained_path: str = None,
        freeze_batchnorm: bool = True,
        eval_mode: bool = False
    ):
        """
        Args:
            dims (List[int]): Dimensions for each downsampling layer.
            train_backbone (bool): Flag to indicate if the backbone should be trained.
            return_interm_layers (bool): If True, returns intermediate feature maps.
            pretrained_path (str): Path to the pretrained weights.
            freeze_batchnorm (bool): If True, freezes BatchNorm layers.
            eval_mode (bool): If True, sets the backbone to evaluation mode.
        """
        super().__init__()

        self.return_interm_layers = return_interm_layers
        # Initialize the DownsamplePath with single-channel input
        self.depth_backbone = DownsamplePath(
            in_channels=1,
            dims=dims,
            train_backbone=train_backbone,
            freeze_batchnorm=freeze_batchnorm
        )
        self._reset_parameters()

        # Calculate strides and number of channels based on dimensions
        self.strides = [2 ** (i + 2) for i in range(len(dims) - 1)]
        self.num_channels = dims[1:]

        # Load pretrained weights if a path is provided and not in evaluation mode
        if pretrained_path and not eval_mode:
            self.load_pretrained_weights(self.depth_backbone, pretrained_path)
            print("DFormer backbone initialized with pretrained weights.")

    def _reset_parameters(self):
        """
        Initializes weights of convolutional and BatchNorm layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)  # Xavier initialization for weights
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # Zero initialization for biases
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # Initialize BatchNorm weights to 1
                nn.init.constant_(m.bias, 0)    # Initialize BatchNorm biases to 0

    def forward(self, tensor_list: NestedTensor) -> Dict[str, NestedTensor]:
        """
        Args:
            tensor_list (NestedTensor): Input tensor with mask.

        Returns:
            Dict[str, NestedTensor]: Dictionary of feature maps with corresponding masks.
        """
        x = tensor_list.tensors  # Extract tensor from NestedTensor
        out: Dict[str, NestedTensor] = {}

        # Pass input through each downsampling layer except the last one
        for i, layer in enumerate(self.depth_backbone.downsample_layers_e[:-1]):
            x = layer(x)

            if self.return_interm_layers:
                # Resize and apply mask to the current feature map
                m = tensor_list.mask
                assert m is not None, "Input mask is None."
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                out[str(i)] = NestedTensor(x, mask)

        if not self.return_interm_layers:
            # If intermediate layers are not returned, provide only the last feature map
            m = tensor_list.mask
            assert m is not None, "Input mask is None."
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out["0"] = NestedTensor(x, mask)

        return out

    def load_pretrained_weights(self, model: nn.Module, pretrained_weights_path: str, prefix: str = "downsample_layers_e"):
        """
        Args:
            model (nn.Module): The model to load weights into.
            pretrained_weights_path (str): Path to the pretrained weights file.
            prefix (str): Prefix to match the layer names in the state dict.
        """
        # Verify if the pretrained weights path exists
        if not os.path.exists(pretrained_weights_path):
            print(f"Invalid path for pretrained weights: {pretrained_weights_path}")
            return

        # Load the state dictionary from the pretrained weights
        dformer_weights = torch.load(pretrained_weights_path)['state_dict']
        print(f"Loading pretrained weights from {pretrained_weights_path}")

        # Iterate through model's named modules to load matching weights
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                for dformer_name, dformer_param in dformer_weights.items():
                    if prefix in dformer_name and name in dformer_name:
                        # Exclude certain BatchNorm parameters
                        if 'running_mean' not in dformer_name and 'running_var' not in dformer_name and 'num_batches_tracked' not in dformer_name:
                            if module.weight.shape == dformer_param.shape:
                                print(f"Loading weight: {dformer_name} into {name}.weight")
                                module.weight.data = dformer_param.data.clone()
                            if module.bias is not None and "bias" in dformer_name:
                                print(f"Loading bias: {dformer_name} into {name}.bias")
                                module.bias.data = dformer_param.data.clone()

                        # Uncomment below to load BatchNorm running statistics, not used in this project
                        # if isinstance(module, nn.BatchNorm2d):
                        #     if "running_mean" in dformer_name:
                        #         module.running_mean.data = dformer_param.data.clone()
                        #     if "running_var" in dformer_name:
                        #         module.running_var.data = dformer_param.data.clone()
                        #     if "num_batches_tracked" in dformer_name:
                        #         module.num_batches_tracked.data = dformer_param.data.clone()


class Joiner(nn.Sequential):
    """
    Joiner module combines the backbone with position encoding.
    """
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module):
        """
        Args:
            backbone (nn.Module): The backbone module for feature extraction.
            position_embedding (nn.Module): The module for position encoding.
        """
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list (NestedTensor): Input tensor with mask.

        Returns:
            Tuple[List[NestedTensor], List[torch.Tensor]]: 
                - List of feature maps.
                - List of corresponding position encodings.
        """
        xs = self[0](tensor_list)  # Pass through backbone
        out: List[NestedTensor] = []
        pos: List[torch.Tensor] = []

        # Collect feature maps from backbone
        for name, x in sorted(xs.items()):
            out.append(x)

        # Generate position encodings for each feature map
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_dformer_backbone(args) -> Joiner:
    """
    Args:
        args: Configuration arguments containing necessary parameters.

    Returns:
        Joiner: Combined backbone and position encoding module.
    """
    # Initialize position encoding based on provided arguments
    position_embedding = build_position_encoding(args)

    # Configuration parameters for the backbone
    train_depth_backbone = True          # Always train the depth backbone
    return_interm_layers = False         # Control intermediate layer outputs
    dims = (32, 64, 128, 256)             # Dimensions for downsampling layers

    try:
        pretrained_path = args.dformer_weights
    except AttributeError:
        pretrained_path = None  # Set to None if not provided

    freeze_batchnorm = False             # Do not freeze BatchNorm layers
    eval_mode = False                    # Not in evaluation mode

    # Instantiate the DFormer backbone
    depth_backbone = DFormerBackbone(
        dims=dims,
        train_backbone=train_depth_backbone,
        return_interm_layers=return_interm_layers,
        pretrained_path=pretrained_path,
        freeze_batchnorm=freeze_batchnorm,
        eval_mode=eval_mode
    )

    # Combine backbone with position encoding
    model = Joiner(depth_backbone, position_embedding)
    return model
