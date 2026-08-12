import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

import numpy as np
import time
from util.misc import _get_clones
from .transformer import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerCrossLayer
from .cross_modal_fusion_qacf import SpatialMaskGenerator, SpatialGuidedQFormer, BranchAdaptiveFusion

class ATFmodule(nn.Module):
    def __init__(self):
        super(ATFmodule, self).__init__()
        self.attetion_block = nn.Sequential(nn.Linear(256*2,256),
                                            nn.ReLU(),
                                            nn.Linear(256,256),
                                            nn.Sigmoid())
        self.mlp_layer = nn.Sequential(nn.Linear(256*2,256),
                                       nn.ReLU(),
                                       nn.Linear(256,256),
                                       nn.LayerNorm(256))
    def forward(self,x,y):
        att = self.attetion_block(torch.cat([x,y],dim=-1))
        ret = x + att * self.mlp_layer(torch.cat([x,y],dim=-1))
        return ret

class ATF(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.args = args
        self.sharing_fusion_module = getattr(self.args, 'sharing_fusion_module', False)

        if self.sharing_fusion_module:
            self.atfm = ATFmodule()
        else:
            self.atfm = _get_clones(ATFmodule(),self.args.dec_layers)

    def forward(self,task_feat,context,n):
        if self.sharing_fusion_module:
            return self.atfm(task_feat,context)
        return self.atfm[n](task_feat,context)
def make_mlp(dim_in,dim_out):
    module = nn.Sequential(nn.Linear(dim_in,dim_out),
                           nn.ReLU(),
                           nn.Linear(dim_out,dim_out),
                           nn.LayerNorm(dim_out)
                           )
    return module

class MURE(nn.Module):
    def __init__(self, dim=256, nhead=8, feeddim=2048, dropout=0.1, args=None):
        super(MURE, self).__init__()
        self.args = args

        self.ternary = make_mlp(dim * 3, dim)
        self.human_obj = make_mlp(dim * 2, dim)
        self.human_rel = make_mlp(dim * 2, dim)
        self.obj_rel = make_mlp(dim * 2, dim)

        self.unary_self = TransformerEncoderLayer(dim, nhead, dim_feedforward=feeddim)
        self.pairwise_self = TransformerEncoderLayer(dim, nhead, dim_feedforward=feeddim)
        self.unary_cross = TransformerCrossLayer(dim, nhead, dim_feedforward=feeddim)
        self.pairwise_cross = TransformerCrossLayer(dim, nhead, dim_feedforward=feeddim)
        self.mc_gen = TransformerCrossLayer(dim, nhead, dim_feedforward=feeddim)

        self.use_spatial_qacf = getattr(args, 'use_spatial_qacf', False)
        self.spatial_mask_generator = SpatialMaskGenerator(
            hidden_dim=dim,
            use_asmbr=getattr(args, 'use_asmbr', False),
            asmbr_offset_scale=getattr(args, 'asmbr_offset_scale', 0.1),
            soft_boundary_temp=getattr(args, 'soft_boundary_temp', 0.05),
        )
        self.spatial_qformer = SpatialGuidedQFormer(
            num_layers=getattr(args, 'qacf_layers', 2),
            d_model=dim,
            nhead=nhead,
            dim_ffn=feeddim,
            dropout=dropout,
        )
        self.branch_fusion = BranchAdaptiveFusion(hidden_dim=dim)
        self.ctx_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, output_sub, output_obj, output_rel, decoding_stuff=None,
                sub_boxes=None, obj_boxes=None, vlm_tokens=None, grid_size=None):
        outputs_ternary = self.ternary(torch.cat([output_sub, output_obj, output_rel], dim=-1))
        memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos = decoding_stuff

        outputs_unary = self.unary_self(torch.stack([output_sub, output_obj, output_rel], dim=0).flatten(1, 2)) \
            .view(3, output_sub.size(0), output_sub.size(1), output_sub.size(2))
        outputs_tu = self.unary_cross(tgt=outputs_ternary.flatten(0, 1).unsqueeze(0), memory=outputs_unary.flatten(1, 2)) \
            .view(output_sub.size(0), output_sub.size(1), output_sub.size(2))

        human_obj_feat = self.human_obj(torch.cat([outputs_unary[0], outputs_unary[1]], dim=-1))
        human_rel_feat = self.human_rel(torch.cat([outputs_unary[0], outputs_unary[2]], dim=-1))
        obj_rel_feat = self.obj_rel(torch.cat([outputs_unary[1], outputs_unary[2]], dim=-1))

        outputs_pairwise = self.pairwise_self(torch.stack([human_obj_feat, human_rel_feat, obj_rel_feat], dim=0).flatten(1, 2)) \
            .view(3, output_sub.size(0), output_sub.size(1), output_sub.size(2))

        outputs_tup = self.pairwise_cross(tgt=outputs_tu.flatten(0, 1).unsqueeze(0), memory=outputs_pairwise.flatten(1, 2)) \
            .view(output_sub.size(0), output_sub.size(1), output_sub.size(2))

        ctx_vis = self.mc_gen(outputs_tup, memory, tgt_mask=tgt_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask,
                              pos=pos)

        fusion_weights = None
        spatial_region = None
        if self.use_spatial_qacf and sub_boxes is not None and obj_boxes is not None and vlm_tokens is not None and grid_size is not None:
            ternary_query = outputs_ternary.transpose(0, 1)
            spatial_keep, spatial_soft_weight, spatial_region = self.spatial_mask_generator(
                sub_boxes.transpose(0, 1), obj_boxes.transpose(0, 1), grid_size, ternary_feat=ternary_query
            )
            ctx_vlm = self.spatial_qformer(
                ternary_query,
                vlm_tokens,
                spatial_keep_mask=spatial_keep,
                spatial_soft_weight=spatial_soft_weight,
                q_attn_mask=tgt_mask,
                q_key_padding_mask=tgt_key_padding_mask,
            )
            ctx_vis_bq = ctx_vis.transpose(0, 1)
            gate = self.ctx_gate(torch.cat([ctx_vis_bq, ctx_vlm], dim=-1))
            ctx_mix = gate * ctx_vis_bq + (1.0 - gate) * ctx_vlm
            sub_ref, obj_ref, rel_ref, fusion_weights = self.branch_fusion(
                output_sub.transpose(0, 1),
                output_obj.transpose(0, 1),
                output_rel.transpose(0, 1),
                ctx_mix,
            )
            output_sub = sub_ref.transpose(0, 1)
            output_obj = obj_ref.transpose(0, 1)
            output_rel = rel_ref.transpose(0, 1)
            multiplex_context = ctx_mix.transpose(0, 1)
        else:
            multiplex_context = ctx_vis

        return output_sub, output_obj, output_rel, multiplex_context, fusion_weights, spatial_region

class MUREN(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_dec_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, return_intermediate_dec=False, args=None,
                 encoder_final_fusion=None):
        super().__init__()

        self.args=args
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, final_fusion=encoder_final_fusion)


        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before,return_attn=False)
        decoder_norm = nn.LayerNorm(d_model)


        self.decoder = TransformerDecoderThreeBranch(decoder_layer, num_dec_layers, decoder_norm, return_intermediate=return_intermediate_dec, args=args)
        self.use_spatial_qacf = getattr(args, 'use_spatial_qacf', False)
        self.spatial_qacf = None
        self.spatial_mask_generator = None
        self.spatial_qformer = None
        self.ctx_gate = None
        if self.use_spatial_qacf:
            self.spatial_mask_generator = SpatialMaskGenerator(
                hidden_dim=d_model,
                use_asmbr=getattr(args, 'use_asmbr', False),
                asmbr_offset_scale=getattr(args, 'asmbr_offset_scale', 0.1),
                soft_boundary_temp=getattr(args, 'soft_boundary_temp', 0.05),
            )
            self.spatial_qformer = SpatialGuidedQFormer(
                num_layers=getattr(args, 'qacf_layers', 2),
                d_model=d_model,
                nhead=nhead,
                dim_ffn=dim_feedforward,
                dropout=dropout,
            )
            self.ctx_gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )

        self.decoder.num_bbox_heads = getattr(args, 'bbox_head_layers', 7)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_human, query_obj, query_rel, pos_embed, fusion_tokens=None, fusion_similarity_matrix=None, vlm_tokens=None, sub_boxes=None, obj_boxes=None, grid_size=None):
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        query_human = query_human.unsqueeze(1).repeat(1, bs, 1)
        query_obj = query_obj.unsqueeze(1).repeat(1, bs, 1)
        query_rel = query_rel.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed, fusion_tokens=fusion_tokens, fusion_similarity_matrix=fusion_similarity_matrix)
        tgt_human = torch.zeros_like(query_human)
        tgt_obj = torch.zeros_like(query_obj)
        tgt_rel = torch.zeros_like(query_rel)

        out_human, out_obj, out_rel = self.decoder(
            tgt_human,
            tgt_obj,
            tgt_rel,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos_human=query_human,
            query_pos_obj=query_obj,
            query_pos_rel=query_rel,
            box_head_sub=None,
            box_head_obj=None,
            vlm_tokens=vlm_tokens,
            grid_size=grid_size,
        )

        out_human = out_human.transpose(1, 2)
        out_obj = out_obj.transpose(1, 2)
        out_rel = out_rel.transpose(1, 2)
        spatial_weights = None
        return out_human, out_obj, out_rel, memory.permute(1, 2, 0).view(bs, c, h, w), spatial_weights


class TransformerDecoderThreeBranch(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,args=None):
        super().__init__()
        self.args = args
        self.return_intermediate = return_intermediate
        setattr(self.args, 'sharing_fusion_module', False)

        self.layers_human = _get_clones(decoder_layer, num_layers)
        self.layers_obj = _get_clones(decoder_layer, num_layers)
        self.layers_rel = _get_clones(decoder_layer, num_layers)
        self.norm_human, self.norm_obj, self.norm_rel = [copy.deepcopy(norm) for _ in range(3)]
        self.MURE_layers = nn.ModuleList([
            MURE(
                dim=getattr(args, 'hidden_dim', 256),
                nhead=getattr(args, 'nheads', 8),
                feeddim=getattr(args, 'dim_feedforward', 2048),
                dropout=getattr(args, 'dropout', 0.1),
                args=args,
            ) for _ in range(num_layers)
        ])

        if self.args.dataset_file == 'vcoco':
            setattr(self.args, 'sharing_fusion_module', True)

        self.aft_human = ATF(args)
        self.aft_obj = ATF(args)
        self.aft_rel = ATF(args)

    def forward(self, tgt_human, tgt_obj, tgt_rel, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos_human: Optional[Tensor] = None,
                query_pos_obj: Optional[Tensor] = None,
                query_pos_rel: Optional[Tensor] = None,
                box_head_sub: Optional[nn.Module] = None,
                box_head_obj: Optional[nn.Module] = None,
                vlm_tokens: Optional[Tensor] = None,
                grid_size: Optional[tuple] = None,
                targets=None,
                epoch=None,
                ):
        output_human, output_obj, output_rel = tgt_human, tgt_obj, tgt_rel

        intermediate_human = []
        intermediate_obj = []
        intermediate_rel = []
        # 初始参考框为 None (Layer 0 将仅使用纯视觉 Context)
        reference_sub_boxes = None
        reference_obj_boxes = None

        for layer_n, (layer_human, layer_obj, layer_rel) in enumerate(zip(self.layers_human, self.layers_obj, self.layers_rel)):
            
            # 1. 基础注意力交互 (Pre-MURE 特征)
            output_human = layer_human(output_human, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos_human)

            output_obj = layer_obj(output_obj, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos_obj)

            output_rel = layer_rel(output_rel, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos_rel)

            # 2. QACF 空间引导 + 多分支交互 (直接使用上一层传下来的 reference_boxes)
            # MURE 内部已妥善处理了 reference_sub_boxes 为 None 的情况
            output_human, output_obj, output_rel, multiplex_context, _, _ = self.MURE_layers[layer_n](
                output_human,
                output_obj,
                output_rel,
                (memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos),
                sub_boxes=reference_sub_boxes,
                obj_boxes=reference_obj_boxes,
                vlm_tokens=vlm_tokens,
                grid_size=grid_size,
            )

            # 3. ATF 自适应特征融合 (Post-MURE -> Post-ATF 特征)
            output_human = self.aft_human(output_human, multiplex_context, layer_n)
            output_obj = self.aft_obj(output_obj, multiplex_context, layer_n)
            output_rel = self.aft_rel(output_rel, multiplex_context, layer_n)

            # 4. 【核心】基于本层最成熟的特征，预测框作为下一层 QACF 的引导
            # 这里送入 head 的特征，与 hoi.py 中计算 Loss 时送入的特征完全一致！保证了域匹配。
            if hasattr(self, 'sub_bbox_heads') and self.sub_bbox_heads is not None:
                # 必须 .detach() 截断梯度，否则会导致跨层反向传播梯度爆炸
                reference_sub_boxes = self.sub_bbox_heads[layer_n](self.norm_human(output_human)).sigmoid().detach()
                reference_obj_boxes = self.obj_bbox_heads[layer_n](self.norm_obj(output_obj)).sigmoid().detach()

            # 5. 记录中间层特征用于辅助损失监督
            if self.return_intermediate:
                intermediate_human.append(self.norm_human(output_human))
                intermediate_obj.append(self.norm_obj(output_obj))
                intermediate_rel.append(self.norm_rel(output_rel))

        if self.return_intermediate:
            return torch.stack(intermediate_human), torch.stack(intermediate_obj), torch.stack(intermediate_rel)

        return output_human, output_obj, output_rel

def build_muren(args, encoder_final_fusion=None):
    return MUREN(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_dec_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        args=args,
        encoder_final_fusion=encoder_final_fusion,
    )