# Taken from https://github.com/SJTU-LuHe/TransVOD, and adapted for depth fusion
# Supports LateFusion, Backbone CrossFusion, and Encoder CrossFusion, 
# with all the fusion models using the Dformer depth backbone
# ------------------------------------------------------------------------
# Modified by Qianyu Zhou and Lu He
# ------------------------------------------------------------------------
# TransVOD++
# Copyright (c) 2022 Shanghai Jiao Tong University. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Deformable DETR
# Copyright (c) SenseTime. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Deformable DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math

from util import box_ops
from util.misc_multi import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .research_scripts.depth_backbone import build_depth_backbone
from .backbone_scratch import build_backbone_fromscratch
from .research_scripts.crossfusion_backbone import build_fusion_backbone
from .dformer_backbone import build_dformer_backbone
from .dformer_crossfusion_backbone import build_dformer_fusion_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss, modified_sigmoid_focal_loss)
from .deformable_transformer_multi_plusplus import build_deforamble_transformer
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """
    def __init__(self, backbone, depth_backbone, transformer, num_classes, num_queries, num_feature_levels, 
                 num_ref_frames = 3, aux_loss=True, with_box_refine=False, two_stage=False, use_depth = False, depth_type=''):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        # Configuration Flags
        self.use_depth = use_depth
        self.depth_type = depth_type
        self.crossfusion_features_concat = 'crossfusion_2way_concat' in depth_type

        if self.crossfusion_features_concat:
            print("Using Concatenated RGB and Depth features in crossfusion_2way_concat mode")

        self.num_queries = num_queries
        self.num_ref_frames = num_ref_frames
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        # Transformer and Embedding Setup
        self.transformer = transformer
        hidden_dim = transformer.d_model

        # Classification and Bounding Box Embeddings
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.temp_class_embed = nn.Linear(hidden_dim, num_classes)
        self.temp_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Initialize Classification Embeddings with Prior Probability Bias
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        self.temp_class_embed.bias.data = torch.ones(num_classes) * bias_value

        # Initialize Bounding Box Embeddings
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.temp_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.temp_bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.temp_bbox_embed.layers[-1].bias.data[2:], -2.0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)

        # Feature Channels Configuration based on Backbone
        self.num_feature_levels = num_feature_levels
        self.num_channels = {
            "resnet18": [64, 128, 256, 512],
            "resnet50": [64, 512, 1024, 2048],
            "dformer": [32, 64, 128]
        }

        # Query Embeddings for Non-Two-Stage Models
        if not self.two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        # Input Projection Layers for Backbone Features
        self.input_proj = nn.ModuleList()
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            for i in range(num_backbone_outs):
                in_channels = self.num_channels[backbone.name][i]
                proj = nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )
                self.input_proj.append(proj)
            for _ in range(num_feature_levels - num_backbone_outs):
                proj = nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                )
                self.input_proj.append(proj)
        else:
            proj = nn.Sequential(
                nn.Conv2d(self.num_channels[backbone.name][-1], hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )
            self.input_proj.append(proj)

        # Depth Feature Projections if Depth is Used
        if self.use_depth:
            if self.crossfusion_features_concat:
                # Projection for Depth Features
                self.input_proj_depth = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(self.num_channels[backbone.d_name][-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ])
                # Projection after Concatenation
                self.concat_input_proj = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ])
            elif 'dformer' in self.depth_type and ('latefusion' in self.depth_type or "encoder_cf" in self.depth_type):
                self.input_proj_depth = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(128, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ])

        # Initialize Backbones
        self.backbone = backbone
        self.depth_backbone = depth_backbone

        # Initialize Input Projection Weights
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        if self.use_depth and self.crossfusion_features_concat:
            for proj in self.input_proj_depth:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)
            for proj in self.concat_input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)

        if self.use_depth and hasattr(self, 'input_proj_depth') and ('dformer' in self.depth_type or 
                                                                         'latefusion' in self.depth_type or 
                                                                         "encoder_cf" in self.depth_type):
            for proj in self.input_proj_depth:
                nn.init.xavier_uniform_(proj[0].weight, gain=1)
                nn.init.constant_(proj[0].bias, 0)

        # Clone Temporary Embeddings for Multiple Decoder Layers
        self.temp_class_embed_list = nn.ModuleList([copy.deepcopy(self.temp_class_embed) for _ in range(3)])
        self.temp_bbox_embed_list = nn.ModuleList([copy.deepcopy(self.temp_bbox_embed) for _ in range(3)])

        # Setup for Two-Stage and Box Refinement Configurations
        num_pred = (transformer.decoder.num_layers + 1) if two_stage else transformer.decoder.num_layers

        if with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
    
    @staticmethod     
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
            
        bs, c, h, w = samples.tensors.shape # torch.Size([5, 3, 562, 999])
        imgs_whwh_shape = (w, h, w, h)
        
        samples_rgb = None
        samples_depth = None
        # Split RGB and Depth tensors
        if samples.tensors.shape[1] == 4:
            assert self.use_depth, "Input tensors have 4 channels but use_depth is not set to True"
        if self.use_depth:
            assert samples.tensors.shape[1] == 4, "Input tensors do not have 4 channels but use_depth is set to True"
        
        if samples.tensors.shape[1] == 4 and self.use_depth:
            samples_rgb_tensor = samples.tensors[:, :3, :, :]
            samples_depth_tensor = samples.tensors[:, 3:4, :, :]
            samples_rgb = NestedTensor(samples_rgb_tensor, samples.mask)
            samples_depth = NestedTensor(samples_depth_tensor, samples.mask)
        else:
            samples_rgb = samples
        
        features = None
        pos = None
        depth_features = None 
        depth_pos = None
        # Cross Fusion:
        if "crossfusion" in self.depth_type:
            assert samples is not None, "RGB samples are None"
            features, pos, depth_features, depth_pos = self.backbone(samples)
        elif "latefusion" in self.depth_type or "encoder_cf" in self.depth_type:
            assert samples_rgb is not None, "RGB samples are None"
            assert samples_depth is not None, "Depth samples are None"
            features, pos, _, _ = self.backbone(samples_rgb)
            depth_features, depth_pos = self.depth_backbone(samples_depth)
        else:
            assert samples_rgb is not None, "RGB samples are None"
            features, pos, _, _ = self.backbone(samples_rgb)   
                     
        srcs = []
        masks = []
        depth_srcs = []
        depth_masks = []
        rgbd_query = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)
        
        # Get depth features
        if samples_depth is not None and "latefusion" in self.depth_type or "encoder_cf" in self.depth_type:
            for l, feat in enumerate(depth_features):
                src, mask = feat.decompose()
                if "dformer" in self.depth_type:
                    depth_srcs.append(self.input_proj_depth[l](src))
                else:
                    depth_srcs.append(src)
                depth_masks.append(mask)
                assert mask is not None
                
        if self.crossfusion_features_concat:
            for l, feat in enumerate(depth_features):
                src, mask = feat.decompose()
                depth_srcs.append(self.input_proj_depth[l](src))
                depth_masks.append(mask)
                assert mask is not None
            
            for i, (rgb, rgb_pos, depth, depth_pos) in enumerate(zip(srcs, pos, depth_srcs, depth_pos)):
                if rgb.shape[2:] != depth.shape[2:]:
                    depth_srcs[i] = F.interpolate(depth, size=rgb.shape[2:], mode='bilinear', align_corners=False)
                    depth_pos[i] = F.interpolate(depth_pos, size=rgb.shape[2:], mode='bilinear', align_corners=False)

                assert rgb.shape == depth_srcs[i].shape, f"Shape mismatch: {rgb.shape} != {depth_srcs[i].shape}" 
                rgb_with_pos = self.with_pos_embed(rgb, rgb_pos) 
                depth_with_pos = self.with_pos_embed(depth_srcs[i], depth_pos[i]) 
                final_src = torch.cat([rgb_with_pos, depth_with_pos], dim=1)
                rgbd_query.append(self.concat_input_proj[0](final_src)) 

        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact, final_hs, final_references_out, out = self.transformer(srcs, masks, pos, depth_srcs, depth_masks, depth_pos, imgs_whwh_shape, query_embeds, self.class_embed[-1], self.bbox_embed[-1], self.temp_class_embed_list, self.temp_bbox_embed_list, rgbd_query)
        

        outputs_classes = []
        outputs_coords = []

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
     
        if final_hs is not None:
            reference = inverse_sigmoid(final_references_out)
            output_class = self.temp_class_embed_list[2](final_hs)
            tmp = self.temp_bbox_embed_list[2](final_hs)
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            output_coord = tmp.sigmoid()
            out["pred_logits"] = output_class
            out["pred_boxes"] = output_coord 
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:], outputs_coord[:])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, losses, focal_alpha=0.25):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = modified_sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list([t["masks"] for t in targets]).decompose()
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": modified_sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs["pred_logits"])).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss == 'masks':
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        return losses



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = args.num_classes if args.num_classes != 31 else 31 # change the default value of num_classes
    device = torch.device(args.device)

    # Swin Backbone currently not supported
    # if 'swin' in args.backbone:
    #     print('yes')
    #     from .swin_transformer import build_swin_backbone
    #     backbone = build_swin_backbone(args) 
    # else:
    # backbone = build_backbone(args)
    # backbone = build_backbone(args)
    
    depth_backbone = None # Default to None
    if args.dformer_weights:
        args.dformer_backbone = True
        
    # Change from --fusion_type argument to --depth_type argument
    # Reason: Used depth type throught out the code, while research. Need to be cleaned up
    # TODO: Clean up the code to remove depth type and use only fusion_type argument
    if args.fusion_type == "Baseline":
        args.depth_type = "Baseline_rgb"
        print("Using only RGB images")
    elif args.fusion_type == "LateFusion":
        args.depth_type = "DepthDeform_latefusion_dformer"
        print("Using LateFusion for depth integration")
    elif args.fusion_type == "Backbone_CrossFusion":
        args.depth_type = "DepthDeform_dformer_crossfusion"
        print("Using Backbone CrossFusion for depth integration")
    elif args.fusion_type == "Encoder_CrossFusion":
        args.depth_type = "DepthDeform_encoder_cf_dformer"
        print("Using Encoder CrossFusion for depth integration")
    else:
        raise NotImplementedError("Fusion type not implemented.")
    
    #CrossFusion refers to Backbone Crossfusion, Encoder_cf is Encoder CrossFusion and then LateFusion
    if "crossfusion" in args.depth_type:
        if "dformer" in args.depth_type:
            depth_backbone = build_dformer_fusion_backbone(args)
        else:
            depth_backbone = build_depth_backbone(args)
        backbone = build_backbone_fromscratch(args)

    elif any(fusion_type in args.depth_type for fusion_type in ["DepthDeform_latefusion", "DepthDeform_encoder_cf"]):
        backbone = build_backbone_fromscratch(args)
        if args.dformer_weights or args.dformer_backbone:
            depth_backbone = build_dformer_backbone(args)
        else:
            depth_backbone = build_depth_backbone(args)

    else:
        backbone = build_backbone_fromscratch(args)

    transformer = build_deforamble_transformer(args)
    model = DeformableDETR(
        backbone,
        depth_backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        num_ref_frames=args.num_ref_frames,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        use_depth=args.use_depth,
        depth_type=args.depth_type
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    matcher = build_matcher(args)
    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(num_classes, matcher, weight_dict, losses, focal_alpha=args.focal_alpha)
    criterion.to(device)
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors