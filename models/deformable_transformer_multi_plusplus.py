# TransVOD++ main class
# Taken from https://github.com/SJTU-LuHe/TransVOD, and adapted for depth fusion
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

import copy
from typing import Optional, List, Union
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from util.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn
from mmcv import ops
from util import box_ops
# from torchvision.ops import roi_align as RoIAlign

# from mmdet.core import bbox2result, bbox2roi, bbox_xyxy_to_cxcywh
# from mmdet.core.bbox.samplers import PseudoSampler
from models.sparse_roi_head.head import RCNNHead 

def get_box_tensor(boxes) -> Tensor:
    """Get tensor data from box type boxes.

    Args:
        boxes (Tensor or BaseBoxes): boxes with type of tensor or box type.
            If its type is a tensor, the boxes will be directly returned.
            If its type is a box type, the `boxes.tensor` will be returned.

    Returns:
        Tensor: boxes tensor.
    """
    if hasattr(boxes, 'tensor'):
        boxes = boxes.tensor
    return boxes

def bbox2roi(bbox_list) -> Tensor:
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (List[Union[Tensor, :obj:`BaseBoxes`]): a list of bboxes
            corresponding to a batch of images.

    Returns:
        Tensor: shape (n, box_dim + 1), where ``box_dim`` depends on the
        different box types. For example, If the box type in ``bbox_list``
        is HorizontalBoxes, the output shape is (n, 5). Each row of data
        indicates [batch_ind, x1, y1, x2, y2].
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        bboxes = get_box_tensor(bboxes)
        img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
        rois = torch.cat([img_inds, bboxes], dim=-1)
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois

class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4,  enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300, num_query=300, n_temporal_decoder_layers = 1,
                 num_ref_frames = 3, fixed_pretrained_model = False, args=None,
                 use_depth = False, depth_type='', dpth_feature_levels=1, dpth_n_points=4):
        super().__init__()

        self.use_depth = use_depth
        self.depth_type = depth_type
        self.residual_fusion = False if "noresidual" in self.depth_type else True
        self.rgbd_query = True if "concat" in self.depth_type else False
        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.num_ref_frames = num_ref_frames
        self.two_stage_num_proposals = two_stage_num_proposals
        self.fixed_pretrained_model = fixed_pretrained_model
        self.n_temporal_query_layers = 3
        self.num_query = num_query
        self.depth_self_attn = True
        self.late_fusion_layers = 1
        self.adaptation_layers = True
        self.gate = True
        self.encoder_cross_fusion = True

        # Encoder Cross Fusion Method
        if "encoder_cf" in self.depth_type:
            self.num_depth_encoder_layers = 4
            self.num_enc_fusion_layers = 4
            self.enc_fusion_layers_order = [0, 1, 2, 3]
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
            fusion_encoder_layer = DeformableTransformerFusionLayerV2(d_model, dim_feedforward,
                                                            dropout, activation,
                                                           num_feature_levels, nhead, enc_n_points)
            self.encoder = RGBDDeformableTransformerEncoderV2(encoder_layer, fusion_encoder_layer, num_encoder_layers, 
                                                            self.num_depth_encoder_layers, self.num_enc_fusion_layers, self.enc_fusion_layers_order)
            
        else:
            encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            num_feature_levels, nhead, enc_n_points)
            self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        
        # LateFusion Method
        if "latefusion" in self.depth_type:
            self.depth_encoder_layer = DepthDeformableTransformerEncoderLayer(d_model, dim_feedforward,
                                                            dropout, activation,
                                                            dpth_feature_levels, nhead, dpth_n_points, self.depth_self_attn, self.gate, self.adaptation_layers)
            
        # Define the roi_layers
        layer_cfg = dict(type='RoIAlign', output_size=7, sampling_ratio=2)
        layer_type = layer_cfg.pop("type")
        layer_cls = getattr(ops, layer_type)
        self.temporal_roi_layers1= nn.ModuleList([layer_cls(spatial_scale=1 / s, **layer_cfg) for s in [32]])
        
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
                                                          
        self.cfg = {"MODEL": {"SparseRCNN": {"NHEADS":8, "DROPOUT":0.0, "DIM_FEEDFORWARD":2048,  "ACTIVATION": 'relu', "HIDDEN_DIM": self.d_model,
                         "NUM_CLS":1, "NUM_REG":3, "NUM_HEADS":6, "NUM_DYNAMIC":2, "DIM_DYNAMIC":64
                        },
                     "ROI_BOX_HEAD":{
                        'POOLER_RESOLUTION': 7,
                     }
                    },
              }
    
        # TQE Module
        self.temporal_query_layer1 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
        self.temporal_query_layer2 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
        self.temporal_query_layer3 = TemporalQueryEncoderLayer(d_model, dim_feedforward, dropout, activation, nhead)
        
        num_classes = 3

        # QRF Module
        self.dynamic_layer_for_current_query1 = RCNNHead(self.cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.dynamic_layer_for_current_query2 = RCNNHead(self.cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)
        self.dynamic_layer_for_current_query3 = RCNNHead(self.cfg, d_model, num_classes, dim_feedforward, nhead, dropout, activation)

        # TDTD Module
        self.temporal_decoder1 = TemporalDeformableTransformerDecoder(decoder_layer, n_temporal_decoder_layers, False)
        self.temporal_decoder2= TemporalDeformableTransformerDecoder(decoder_layer, n_temporal_decoder_layers, False)
        self.temporal_decoder3 = TemporalDeformableTransformerDecoder(decoder_layer, n_temporal_decoder_layers, False)

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
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

    def forward(self, srcs, masks, pos_embeds, depth_srcs, depth_masks, depth_pos_embeds, imgs_whwh_shape, query_embed=None, class_embed = None, cur_bbox_embed = None,  temp_class_embed_list = None, temp_bbox_embed_list = None, rgbd_query=[]):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        h, w = 0,0
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            feats_whwh = (w, h, w, h)
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1) 
        # print("src_flatten", src_flatten.shape)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        #print("lvl_pos_embed_flatten", lvl_pos_embed_flatten.shape)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        # feats_whwh = torch.as_tensor(feats_whwh, dtype = torch.long, device=src_flatten.device)
        # feats_whwh = feats_whwh.repeat(1, 300, 1)
        imgs_whwh_shape = torch.as_tensor(imgs_whwh_shape, dtype = torch.long, device=src_flatten.device)
        imgs_whwh_shape = imgs_whwh_shape.repeat(1, self.num_query, 1)

        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        # print("valid_ratios", valid_ratios.shape)

        # Depth Integration via latefusion: cross attention between RGB and Depth
        if "latefusion" in self.depth_type and self.use_depth:
            assert depth_srcs is not None and depth_masks is not None and depth_pos_embeds is not None, "Depth information is required for Deformable DETR with depth"
            assert len(depth_srcs) == len(depth_masks) == len(depth_pos_embeds), "The number of depth sources, masks and pos_embeds should be the same"
            depth_src_flatten = []
            depth_mask_flatten = []
            depth_lvl_pos_embed_flatten = []
            depth_spatial_shapes = []
            for lvl, (depth_src, depth_mask, depth_pos_embed) in enumerate(zip(depth_srcs, depth_masks, depth_pos_embeds)):
                bs, c, h, w = depth_src.shape
                depth_spatial_shape = (h, w)
                depth_spatial_shapes.append(depth_spatial_shape)
                depth_src = depth_src.flatten(2).transpose(1, 2)
                depth_mask = depth_mask.flatten(1)
                depth_pos_embed = depth_pos_embed.flatten(2).transpose(1, 2)
                depth_lvl_pos_embed = depth_pos_embed #+ self.level_embed[lvl].view(1, 1, -1), currently no multiscale for depth
                depth_lvl_pos_embed_flatten.append(depth_lvl_pos_embed)
                depth_src_flatten.append(depth_src)
                depth_mask_flatten.append(depth_mask)
            depth_src_flatten = torch.cat(depth_src_flatten, 1)
            depth_mask_flatten = torch.cat(depth_mask_flatten, 1)
            depth_lvl_pos_embed_flatten = torch.cat(depth_lvl_pos_embed_flatten, 1)
            depth_spatial_shapes = torch.as_tensor(depth_spatial_shapes, dtype=torch.long, device=depth_src_flatten.device)    
            depth_lvl_start_index = torch.cat((depth_spatial_shapes.new_zeros((1, )), depth_spatial_shapes.prod(1).cumsum(0)[:-1]))
            depth_valid_ratios = torch.stack([self.get_valid_ratio(m) for m in depth_masks], 1)
            
            # Get reference points for depth
            rgb_reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)
            depth_reference_points = self.get_reference_points(depth_spatial_shapes, depth_valid_ratios, device=depth_src_flatten.device)
            # Deformable cross attention between RGB and Depth (RGB single scale or multi scale, Depth single scale)
            fused_src_flatten = self.depth_encoder_layer(src_flatten, lvl_pos_embed_flatten, depth_lvl_pos_embed_flatten, spatial_shapes, rgb_reference_points, depth_reference_points, depth_src_flatten, depth_spatial_shapes, depth_lvl_start_index, mask_flatten, depth_mask_flatten)
            src_flatten = src_flatten + fused_src_flatten
        
        # Cross fusion encoder
        if "encoder_cf" in self.depth_type and self.use_depth:
            assert depth_srcs is not None and depth_masks is not None and depth_pos_embeds is not None, "Depth information is required for Deformable DETR with depth"
            assert len(depth_srcs) == len(depth_masks) == len(depth_pos_embeds), "The number of depth sources, masks and pos_embeds should be the same"
            depth_src_flatten = []
            depth_mask_flatten = []
            depth_lvl_pos_embed_flatten = []
            depth_spatial_shapes = []
            for lvl, (depth_src, depth_mask, depth_pos_embed) in enumerate(zip(depth_srcs, depth_masks, depth_pos_embeds)):
                bs, c, h, w = depth_src.shape
                depth_spatial_shape = (h, w)
                depth_spatial_shapes.append(depth_spatial_shape)
                depth_src = depth_src.flatten(2).transpose(1, 2)
                depth_mask = depth_mask.flatten(1)
                depth_pos_embed = depth_pos_embed.flatten(2).transpose(1, 2)
                depth_lvl_pos_embed = depth_pos_embed #+ self.level_embed[lvl].view(1, 1, -1), currently no multiscale for depth
                depth_lvl_pos_embed_flatten.append(depth_lvl_pos_embed)
                depth_src_flatten.append(depth_src)
                depth_mask_flatten.append(depth_mask)
            depth_src_flatten = torch.cat(depth_src_flatten, 1)
            depth_mask_flatten = torch.cat(depth_mask_flatten, 1)
            depth_lvl_pos_embed_flatten = torch.cat(depth_lvl_pos_embed_flatten, 1)
            depth_spatial_shapes = torch.as_tensor(depth_spatial_shapes, dtype=torch.long, device=depth_src_flatten.device)    
            depth_lvl_start_index = torch.cat((depth_spatial_shapes.new_zeros((1, )), depth_spatial_shapes.prod(1).cumsum(0)[:-1]))
            depth_valid_ratios = torch.stack([self.get_valid_ratio(m) for m in depth_masks], 1)
            
            # Get reference points for depth
            rgb_reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src_flatten.device)
            depth_reference_points = self.get_reference_points(depth_spatial_shapes, depth_valid_ratios, device=depth_src_flatten.device)
            memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, rgbd_flatten if self.rgbd_query else None, 
                                  depth_src_flatten, depth_spatial_shapes, depth_lvl_start_index, depth_valid_ratios, depth_lvl_pos_embed_flatten, depth_mask_flatten)
        else:
            memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
            
        # prepare input for decoder:
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        inter_references_out = inter_references
        if self.two_stage:
            return hs, init_reference_out, inter_references_out, enc_outputs_class, enc_outputs_coord_unact
        
        if self.fixed_pretrained_model:
            print("fixed")
            memory = memory.detach()
            hs = hs.detach()
            inter_references = inter_references.detach()
            
        self.TDAM = True
        if self.TDAM:
            # memory
            memory_list = torch.chunk(memory, self.num_ref_frames+1,  dim=0)
            cur_memory = memory_list[0]
            ref_memory_list = memory_list[1:]
            # ref_memory = torch.cat(memory_list[1:], 1)

            # pos ToDO
            ref_spatial_shapes = spatial_shapes.expand(self.num_ref_frames, 2).contiguous()
            cur_pos_embed = lvl_pos_embed_flatten[0:1]
            ref_pos_embed_list = torch.chunk(lvl_pos_embed_flatten[1:], self.num_ref_frames, dim=0)
            
            #-------------------------------------------------------------------------------------------
            # Get ref memory with ref position embedding of each reference frame
            # ref_pos_embed = torch.cat(ref_pos_embed_list, 1)
            ref_memory_with_pos_embed_list = []
            for i in range(len(ref_memory_list)):
                ref_memory_each = ref_memory_list[i]
                ref_pos_embed_each = ref_pos_embed_list[i]
                ref_memory_each = ref_memory_each + ref_pos_embed_each
                ref_memory_with_pos_embed_list.append(ref_memory_each)
            # ref_memory = ref_memory + ref_pos_embed
            frame_start_index = torch.cat((ref_spatial_shapes.new_zeros((1, )), ref_spatial_shapes.prod(1).cumsum(0)[:-1])).contiguous()
            valid_ratios = valid_ratios[0:1].expand(1, self.num_ref_frames, 2)
            reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=cur_memory.device)
            # output_memory = self.temporal_encoder_layer(cur_memory, cur_pos_embed, reference_points, ref_memory, ref_spatial_shapes,
                                                                   # frame_start_index)
            mask_flatten = None 
            
            #--------------------------------------------------------------------------------------------
            # get current/reference hs and currenct/reference reference points 
            last_hs = hs[-1]
            last_hs_list = torch.chunk(last_hs, self.num_ref_frames + 1, dim = 0)
            cur_hs = last_hs_list[0]
            ref_hs_list = last_hs_list[1:]
            # ref_hs = torch.cat(last_hs_list[1:], 1)

     
            last_reference_out = inter_references_out[-1]
            last_reference_out_list = torch.chunk(last_reference_out, self.num_ref_frames + 1, dim = 0)
            cur_reference_out = last_reference_out_list[0]
            ref_reference_out_list = last_reference_out_list[1:]
            # ref_reference_out = torch.cat(last_reference_out_list[1:], 1)
            
            #------------------------------------------------------------------------------------------
            # Get score of current and reference frame

            # score of current frame
            cur_hs_logits = class_embed(cur_hs)
            cur_prob = cur_hs_logits.sigmoid()      # torch.Size([1, 300, 31])

            # score of refernce frames
            # ref_hs_logits = class_embed(ref_hs)
            # ref_prob = ref_hs_logits.sigmoid()      # torch.Size([1, 300*N, 31])
            ref_prob_list = []
            ref_hs_logits_each = class_embed(ref_hs_list[0])
            ref_prob_each = ref_hs_logits_each.sigmoid()      # torch.Size([1, 300, 31])
            ref_prob_list.append(ref_prob_each)
            ref_hs_logits_concat = ref_hs_logits_each
            for ref_hs_each in ref_hs_list[1:]:
                ref_hs_logits_each = class_embed(ref_hs_each)
                ref_prob_each = ref_hs_logits_each.sigmoid()      # torch.Size([1, 300, 31])
                ref_prob_list.append(ref_prob_each)
                ref_hs_logits_concat = torch.cat((ref_hs_logits_concat, ref_hs_logits_each), dim=1)
            # print(ref_hs_logits_concat.shape) # torch.Size([1, 300*N, 31])
            ref_prob_concat = ref_hs_logits_concat.sigmoid()      # torch.Size([1, 300*N, 31])
            
            #----------------------------------------------------------------------------------------
            # Get bbox of current and reference frame

            # BBox of current frame
            cur_hs_bbox = cur_bbox_embed(cur_hs)
            cur_hs_reference_points = inverse_sigmoid(cur_reference_out)
            cur_hs_bbox += cur_hs_reference_points
            cur_hs_bbox_sigmoid = cur_hs_bbox.sigmoid()  # (c_x, c_y, w, h)

            # BBox of reference frame
            ref_hs_bbox_sigmoid_list = []
            for i in range(len(ref_hs_list)):
                ref_hs_each = ref_hs_list[i]
                ref_reference_out_each = ref_reference_out_list[i]
                ref_hs_bbox_each = cur_bbox_embed(ref_hs_each)
                ref_hs_reference_points_each = inverse_sigmoid(ref_reference_out_each)
                ref_hs_bbox_each += ref_hs_reference_points_each
                ref_hs_bbox_sigmoid_each = ref_hs_bbox_each.sigmoid()  # (c_x, c_y, w, h)
                ref_hs_bbox_sigmoid_list.append(ref_hs_bbox_sigmoid_each)
            
            #----------------------------------------------------------------------------------------
            # Get RoI feature of current and reference frame
            
            # RoI Feature of current frame
            cur_hs_bbox_xyxy_norm = box_ops.box_cxcywh_to_xyxy(cur_hs_bbox_sigmoid)
            cur_hs_bbox_xyxy = cur_hs_bbox_xyxy_norm  * imgs_whwh_shape
            cur_hs_bbox_xyxy_list = [cur_hs_bbox_xyxy[i] for i in range(len(cur_hs_bbox_xyxy))] 
            cur_rois = bbox2roi(cur_hs_bbox_xyxy_list)

            cur_memory_for_rcnn = cur_memory.permute(0, 2, 1).unsqueeze(-1).view(1, self.d_model, h, w).contiguous()
            cur_roi_features = self.temporal_roi_layers1[0](cur_memory_for_rcnn, cur_rois)
            # Query and RoI Fusion (QRF) of current frame
            cur_hs = self.dynamic_layer_for_current_query1(cur_roi_features, cur_hs) 

            # RoI Feature of reference frame
            ref_hs_enhanced_list = []
            for i in range(len(ref_hs_bbox_sigmoid_list)):
                ref_hs_bbox_sigmoid = ref_hs_bbox_sigmoid_list[i]
                ref_hs_bbox_xyxy_norm = box_ops.box_cxcywh_to_xyxy(ref_hs_bbox_sigmoid)
                ref_hs_bbox_xyxy = ref_hs_bbox_xyxy_norm  * imgs_whwh_shape #feats_whwh  # ref_imgs_whwh ??????
                ref_hs_bbox_xyxy_list = [ref_hs_bbox_xyxy[i] for i in range(len(ref_hs_bbox_xyxy))] 
                ref_rois = bbox2roi(ref_hs_bbox_xyxy_list)

                ref_memory = ref_memory_with_pos_embed_list[i]
                ref_memory_for_rcnn = ref_memory.permute(0, 2, 1).unsqueeze(-1).view(1, self.d_model, h, w).contiguous()
                ref_roi_features = self.temporal_roi_layers1[0](ref_memory_for_rcnn, ref_rois)
                ref_hs_each = ref_hs_list[i]
                # Query and RoI Fusion (QRF) of reference frame
                ref_hs_enhanced = self.dynamic_layer_for_current_query1(ref_roi_features, ref_hs_each) 
                ref_hs_enhanced_list.append(ref_hs_enhanced)

            # 
            ref_hs_concat = ref_hs_enhanced_list[0]
            for ref_hs_each in ref_hs_enhanced_list[1:]:
                ref_hs_concat = torch.cat((ref_hs_concat, ref_hs_each), dim=1)
            
            num_elements_greater_than_001 = torch.sum(torch.gt(ref_prob_concat[:,:,1], 0.001))
            num_elements_greater_than_01 = torch.sum(torch.gt(ref_prob_concat[:,:,1], 0.01))
            num_elements_greater_than_1 = torch.sum(torch.gt(ref_prob_concat[:,:,1], 0.1))
            
            with_decoder = True
            topk_values, topk_indexes =  torch.topk(ref_prob_concat[:,:,1], 80 * self.num_ref_frames, dim=1)
            # topk_values, topk_indexes =  torch.topk(ref_prob_concat[:,:,:-1].reshape(ref_hs_logits_concat.shape[0], -1), 80 * self.num_ref_frames, dim=1)
            # topk_indexes = topk_indexes // (ref_hs_logits_concat.shape[2] - 1)
            # topk_values, topk_indexes = torch.topk(ref_prob_concat.view(ref_hs_logits_concat.shape[0], -1), 80 * self.num_ref_frames, dim=1)
            # topk_indexes = topk_indexes // ref_hs_logits_concat.shape[2]
            ref_hs_input1 = torch.gather(ref_hs_concat, 1, topk_indexes.unsqueeze(-1).repeat(1,1,ref_hs_concat.shape[-1]))
            cur_hs = self.temporal_query_layer1(cur_hs, ref_hs_input1)

            if with_decoder:
                cur_hs, cur_references_out = self.temporal_decoder1(cur_hs, cur_reference_out, cur_memory,spatial_shapes[0:1], level_start_index[0:1], valid_ratios[0:1], None, None) 
            else:
                cur_references_out = cur_reference_out

            out = {}
            reference1 = inverse_sigmoid(cur_references_out)
            output_class1 = temp_class_embed_list[0](cur_hs)
            tmp1 = temp_bbox_embed_list[0](cur_hs)
            if reference1.shape[-1] == 4:
                tmp1 += reference1
            else:
                assert reference1.shape[-1] == 2
                tmp1[..., :2] += reference1
            output_coord1 = tmp1.sigmoid()
            out['aux_outputs'] = [{"pred_logits":output_class1, "pred_boxes":output_coord1}]

            ###
            topk_values, topk_indexes =  torch.topk(ref_prob_concat[:,:,1], 50 * self.num_ref_frames, dim=1)
            # topk_values, topk_indexes =  torch.topk(ref_prob_concat[:,:,:-1].reshape(ref_hs_logits_concat.shape[0], -1), 50 * self.num_ref_frames, dim=1)
            # topk_indexes = topk_indexes // (ref_hs_logits_concat.shape[2] - 1)
            # topk_values, topk_indexes = torch.topk(ref_prob_concat.view(ref_hs_logits_concat.shape[0], -1), 50 * self.num_ref_frames, dim=1)
            # topk_indexes = topk_indexes // ref_hs_logits_concat.shape[2]
            ref_hs_input2 = torch.gather(ref_hs_concat, 1, topk_indexes.unsqueeze(-1).repeat(1,1,ref_hs_concat.shape[-1]))
            cur_hs = self.temporal_query_layer2(cur_hs, ref_hs_input2)

            if  with_decoder:
                cur_hs, cur_references_out = self.temporal_decoder2(cur_hs, cur_reference_out, cur_memory,
                                             spatial_shapes[0:1], level_start_index[0:1], valid_ratios[0:1], None, None)  
            cur_references_out = cur_reference_out
            
            reference2 = inverse_sigmoid(cur_references_out)
            output_class2 = temp_class_embed_list[1](cur_hs)
            tmp2 = temp_bbox_embed_list[1](cur_hs)
            if reference2.shape[-1] == 4:
                tmp2 += reference2
            else:
                assert reference2.shape[-1] == 2
                tmp2[..., :2] += reference2
            output_coord2 = tmp2.sigmoid()
            out['aux_outputs'].append({"pred_logits":output_class2, "pred_boxes":output_coord2})

            ###
            topk_values, topk_indexes =  torch.topk(ref_prob_concat[:,:,1], 30 * self.num_ref_frames, dim=1)
            # topk_values, topk_indexes =  torch.topk(ref_prob_concat[:,:,:-1].reshape(ref_hs_logits_concat.shape[0], -1), 30 * self.num_ref_frames, dim=1)
            # topk_indexes = topk_indexes // (ref_hs_logits_concat.shape[2] - 1)
            # topk_values, topk_indexes = torch.topk(ref_prob_concat.view(ref_hs_logits_concat.shape[0], -1), 30 * self.num_ref_frames, dim=1)
            # topk_indexes = topk_indexes // ref_hs_logits_concat.shape[2]
            ref_hs_input3 = torch.gather(ref_hs_concat, 1, topk_indexes.unsqueeze(-1).repeat(1,1,ref_hs_concat.shape[-1]))
            cur_hs = self.temporal_query_layer3(cur_hs, ref_hs_input3)
            # print("ref_hs", ref_hs_input3.shape)
            # print("cur_hs", cur_hs.shape)

            if with_decoder:
                final_hs, final_references_out = self.temporal_decoder3(cur_hs, cur_reference_out, cur_memory,
                                            spatial_shapes[0:1], level_start_index[0:1], valid_ratios[0:1], None, None)
            else:
                cur_references_out = cur_reference_out
                final_hs = cur_hs
                final_references_out = cur_references_out
            # print("final_hs", final_hs.shape)
            # print("final_references", final_references_out.shape)
            
            return hs[:,0:1,:,:], init_reference_out[0:1], inter_references_out[:,0:1,:,:], None, None, final_hs, final_references_out, out

            
        return hs[:,0:1,:,:], init_reference_out[0:1], inter_references_out[:,0:1,:,:], None, None, final_hs, final_references_out, out

# LateFusion Class
class DepthDeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, 
                 activation='relu', n_depth_levels=1, n_heads=8, dpth_n_points=4, 
                 depth_self_attn=False, gate=True, adaptation_layers=True):
        super().__init__()
        
        # Flags for depth self-attention and adaptation layers
        self.depth_self_attn = depth_self_attn
        self.adaptation_layers = adaptation_layers
        
        # Cross attention mechanism (MSDeformAttn for cross-modality attention)
        self.cross_attn = MSDeformAttn(d_model, n_depth_levels, n_heads, dpth_n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Feed-Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = _get_activation_fn('gelu')
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Scale adaptation layers (adjusts the feature representations)
        self.depth_scale_adapt = nn.Linear(d_model, d_model)
        self.norm_depth_scale = nn.LayerNorm(d_model)
        self.cross_scale_adapt = nn.Linear(d_model, d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Utility function to apply positional encoding if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Applies a Feed-Forward Network (FFN) on the target tensor."""
        tgt2 = self.activation(self.linear1(tgt))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, src_pos, tgt_spatial_shapes, reference_points, 
                depth_reference_points, src, src_spatial_shapes, frame_start_index, 
                tgt_padding_mask=None, src_padding_mask=None):
        """Forward pass for the encoder layer that performs cross-attention and FFN."""
        
        # Apply depth scale adaptation to source input
        src = self.depth_scale_adapt(src)
        src = self.norm_depth_scale(src)
        
        # Cross-attention: RGB attending to depth features
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, 
                               src_spatial_shapes, frame_start_index, src_padding_mask)
        
        # Apply cross scale adaptation
        tgt2 = self.cross_scale_adapt(tgt2)
        
        # Add residual connection for cross-attention output
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Apply Feed-Forward Network (FFN)
        tgt = self.forward_ffn(tgt)
        
        return tgt


# Encoder CrossFusion Class
class DeformableTransformerFusionLayerV2(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="gelu", 
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # Cross-attention mechanism (MSDeformAttn for cross-modality attention)
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Feed-Forward Network (FFN)
        self.linear1 = nn.Linear(d_model, d_model)
        self.activation = _get_activation_fn('gelu')
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Scale adaptation layers (adjusts the feature representations)
        self.depth_scale_adapt = nn.Linear(d_model, d_model)
        self.norm_depth_scale = nn.LayerNorm(d_model)
        self.cross_scale_adapt = nn.Linear(d_model, d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Utility function to apply positional encoding if provided."""
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        """Applies a Feed-Forward Network (FFN) on the target tensor."""
        tgt2 = self.activation(self.linear1(tgt))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, 
                level_start_index, src_padding_mask=None):
        """Forward pass for the fusion encoder layer that performs cross-attention and FFN."""
        
        # Apply depth scale adaptation to source input
        src = self.depth_scale_adapt(src)
        src = self.norm_depth_scale(src)
        
        # Cross-attention: RGB attending to depth features
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, src, 
                               src_spatial_shapes, level_start_index, src_padding_mask)
        
        # Apply cross scale adaptation
        tgt2 = self.cross_scale_adapt(tgt2)
        
        # Add residual connection for cross-attention output
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Apply Feed-Forward Network (FFN)
        tgt = self.forward_ffn(tgt)

        return tgt


# Encoder Layers for Encoder CrossFusion
class RGBDDeformableTransformerEncoderV2(nn.Module):
    def __init__(self, encoder_layer, fusion_encoder_layer, num_layers, depth_num_layers, 
                 fusion_num_layers, fusion_layers_order=[]):
        super().__init__()
        
        # Initialize layers and fusion layers
        self.layers = _get_clones(encoder_layer, num_layers)
        self.fusion_layers = _get_clones(fusion_encoder_layer, fusion_num_layers)
        self.num_layers = num_layers
        self.depth_num_layers = depth_num_layers
        self.fusion_num_layers = fusion_num_layers
        self.fusion_layers_order = fusion_layers_order if len(fusion_layers_order) > 0 else list(range(fusion_num_layers))
        
        # Check if the number of fusion layers matches
        assert len(self.fusion_layers_order) == self.fusion_num_layers, "The number of fusion layers should match the fusion layer count"

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Generates reference points for multi-scale attention."""
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

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, 
                padding_mask=None, rgbd_src=None, depth_src=None, depth_spatial_shapes=None, 
                depth_level_start_index=None, depth_valid_ratios=None, depth_pos=None, 
                depth_padding_mask=None):
        """Forward pass for the RGBD Deformable Transformer Encoder."""
        
        output = src
        output_fusion = depth_src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        
        # Loop through each layer of the RGB encoder
        for i, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            
            # Perform fusion with depth encoder at appropriate layers
            if i < self.depth_num_layers:
                if i in self.fusion_layers_order:
                    fusion_layer = self.fusion_layers[self.fusion_layers_order.index(i)]
                    output_fusion = fusion_layer(output, pos, reference_points, output_fusion, depth_spatial_shapes, depth_level_start_index, padding_mask)
                    output = output + output_fusion

        return output


class TemporalQueryEncoderLayer(nn.Module):
    def __init__(self, d_model = 256, d_ffn = 1024, dropout=0.1, activation="relu", n_heads = 8):
        super().__init__()

        # self attention 
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # cross attention 
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
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
    
    def forward(self, query , ref_query, query_pos = None, ref_query_pos = None):
        # self.attention
        q = k = self.with_pos_embed(query, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), query.transpose(0, 1))[0].transpose(0, 1)
        tgt = query + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention 
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos).transpose(0, 1), 
            self.with_pos_embed(ref_query, ref_query_pos).transpose(0, 1),
            ref_query.transpose(0,1)
        )[0].transpose(0,1)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

class TemporalQueryEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, query , ref_query, query_pos = None, ref_query_pos = None):
        output = query
        for _, layer in enumerate(self.layers):
            output = layer(output, ref_query, query_pos, ref_query_pos)
        return output

class TemporalDeformableTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model = 256, d_ffn=1024, dropout=0.1, 
                 activation='relu', num_ref_frames = 3, n_heads = 8, n_points=4):
        super().__init__()

        # cross attention 
        self.cross_attn = MSDeformAttn(d_model, num_ref_frames, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
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

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, frame_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
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

class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # ffn
        src = self.forward_ffn(src)

        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

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

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            # print(str(_) + "deformable_transformer_", [reference_points.shape, level_start_index, spatial_shapes] )
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
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

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class TemporalDeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            self.bbox_embed = None
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points  

class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            # print("Decoder refer", reference_points.shape)
            # print(reference_points)
            # print("src_valid_ratios", src_valid_ratios)
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            # print("reference_points_input", reference_points_input.shape)
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        num_query=args.num_queries,
        n_temporal_decoder_layers = args.n_temporal_decoder_layers, 
        num_ref_frames = args.num_ref_frames,
        fixed_pretrained_model = args.fixed_pretrained_model,
        args = args,
        use_depth=args.use_depth,
        depth_type=args.depth_type,
        dpth_n_points=args.dpth_n_points)