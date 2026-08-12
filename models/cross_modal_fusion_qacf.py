import torch
from torch import nn
import torch.nn.functional as F


class SpatialMaskGenerator(nn.Module):
    def __init__(self, hidden_dim=256, use_asmbr=False, asmbr_offset_scale=0.1, soft_boundary_temp=0.05):
        super().__init__()
        self.use_asmbr = use_asmbr
        self.asmbr_offset_scale = asmbr_offset_scale
        self.soft_boundary_temp = soft_boundary_temp
        if use_asmbr:
            self.offset_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 4),
                nn.Tanh(),
            )

    def forward(self, sub_boxes, obj_boxes, grid_size, ternary_feat=None):
        # boxes: [B,Q,4] in cxcywh normalized
        sub_xyxy = self.cxcywh_to_xyxy(sub_boxes)
        obj_xyxy = self.cxcywh_to_xyxy(obj_boxes)

        union = torch.zeros_like(sub_xyxy)
        union[..., 0] = torch.minimum(sub_xyxy[..., 0], obj_xyxy[..., 0])
        union[..., 1] = torch.minimum(sub_xyxy[..., 1], obj_xyxy[..., 1])
        union[..., 2] = torch.maximum(sub_xyxy[..., 2], obj_xyxy[..., 2])
        union[..., 3] = torch.maximum(sub_xyxy[..., 3], obj_xyxy[..., 3])

        region = union
        if self.use_asmbr and ternary_feat is not None:
            delta = self.offset_head(ternary_feat) * self.asmbr_offset_scale
            region_raw = (union + delta).clamp(0.0, 1.0)

            # Avoid in-place edits on views that can break autograd version tracking.
            x1 = region_raw[..., 0]
            y1 = region_raw[..., 1]
            x2 = torch.maximum(region_raw[..., 2], x1 + 1e-4)
            y2 = torch.maximum(region_raw[..., 3], y1 + 1e-4)
            region = torch.stack([x1, y1, x2, y2], dim=-1)

        keep_mask, soft_weight = self.boxes_to_patch_keep_mask(region, grid_size, self.soft_boundary_temp)
        return keep_mask, soft_weight, region

    @staticmethod
    def cxcywh_to_xyxy(boxes):
        x_c, y_c, w, h = boxes.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1).clamp(0.0, 1.0)

    @staticmethod
    def boxes_to_patch_keep_mask(boxes_xyxy, grid_size, soft_temp=0.05):
        hp, wp = grid_size
        bsz, nq = boxes_xyxy.shape[:2]
        device = boxes_xyxy.device

        ys = (torch.arange(hp, device=device, dtype=boxes_xyxy.dtype) + 0.5) / hp
        xs = (torch.arange(wp, device=device, dtype=boxes_xyxy.dtype) + 0.5) / wp
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        xx = xx.reshape(1, 1, hp * wp)
        yy = yy.reshape(1, 1, hp * wp)

        x1 = boxes_xyxy[..., 0].unsqueeze(-1)
        y1 = boxes_xyxy[..., 1].unsqueeze(-1)
        x2 = boxes_xyxy[..., 2].unsqueeze(-1)
        y2 = boxes_xyxy[..., 3].unsqueeze(-1)

        # Hard keep mask (for safe visibility fallback).
        keep = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
        empty = ~keep.any(dim=-1, keepdim=True)
        keep = torch.where(empty, torch.ones_like(keep), keep)

        # Differentiable soft weights around box boundaries.
        t = max(float(soft_temp), 1e-4)
        sx1 = torch.sigmoid(((xx - x1) / t).clamp(-50, 50))
        sx2 = torch.sigmoid(((x2 - xx) / t).clamp(-50, 50))
        sy1 = torch.sigmoid(((yy - y1) / t).clamp(-50, 50))
        sy2 = torch.sigmoid(((y2 - yy) / t).clamp(-50, 50))
        soft = sx1 * sx2 * sy1 * sy2
        # ensure non-zero mass to avoid degenerate softmax rows
        soft = torch.where(empty, torch.ones_like(soft), soft)
        return keep, soft


class QFormerLayer(nn.Module):
    def __init__(self, d_model=256, nhead=8, dim_ffn=1024, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ffn, d_model),
        )

    def forward(self, q, kv, key_padding_mask=None, attn_mask=None, soft_attn_bias=None,
                q_attn_mask=None, q_key_padding_mask=None):
        q2, _ = self.self_attn(q, q, q, attn_mask=q_attn_mask, key_padding_mask=q_key_padding_mask)
        q = self.norm1(q + self.drop1(q2))

        # Merge hard mask (bool, True=blocked) and soft bias (float, additive logits bias) safely.
        merged_attn_mask = None
        if soft_attn_bias is not None:
            merged_attn_mask = soft_attn_bias.clone()
            if attn_mask is not None:
                merged_attn_mask = merged_attn_mask.masked_fill(attn_mask, float('-inf'))
        else:
            if attn_mask is not None:
                merged_attn_mask = attn_mask

        q2, _ = self.cross_attn(q, kv, kv, key_padding_mask=key_padding_mask, attn_mask=merged_attn_mask)
        q = self.norm2(q + self.drop2(q2))

        q2 = self.ffn(q)
        q = self.norm3(q + self.drop3(q2))
        return q


class SpatialGuidedQFormer(nn.Module):
    def __init__(self, num_layers=2, d_model=256, nhead=8, dim_ffn=1024, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.layers = nn.ModuleList([
            QFormerLayer(d_model=d_model, nhead=nhead, dim_ffn=dim_ffn, dropout=dropout)
            for _ in range(num_layers)
        ])

    def _build_per_query_attn_mask(self, spatial_keep_mask):
        # spatial_keep_mask: [B,Q,Nv], True means this token is visible for this query.
        # MultiheadAttention(attn_mask): True means blocked position.
        bsz, nq, nv = spatial_keep_mask.shape
        keep = spatial_keep_mask.bool()

        # Ensure each query has at least one visible token to avoid invalid all-masked attention rows.
        empty = ~keep.any(dim=-1, keepdim=True)
        if empty.any():
            keep = torch.where(empty, torch.ones_like(keep), keep)

        block = ~keep  # [B,Q,Nv], True=block
        # Expand to per-head: [B,H,Q,Nv] -> [B*H,Q,Nv]
        block = block.unsqueeze(1).expand(bsz, self.nhead, nq, nv).reshape(bsz * self.nhead, nq, nv)
        return block

    def _build_soft_attn_bias(self, spatial_soft_weight):
        # spatial_soft_weight: [B,Q,Nv] in [0,1], higher means more attention.
        # MultiheadAttention float attn_mask is added to attention logits before softmax.
        bsz, nq, nv = spatial_soft_weight.shape
        w = spatial_soft_weight.clamp(min=1e-6, max=1.0)
        bias = torch.log(w)
        bias = bias.unsqueeze(1).expand(bsz, self.nhead, nq, nv).reshape(bsz * self.nhead, nq, nv)
        return bias

    def forward(self, ternary_query, vlm_tokens, spatial_keep_mask=None, spatial_soft_weight=None,
                q_attn_mask=None, q_key_padding_mask=None):
        # ternary_query [B,Q,C], vlm_tokens [B,Nv,C], keep [B,Q,Nv], soft_weight [B,Q,Nv]
        q = ternary_query
        kv = vlm_tokens

        if spatial_keep_mask is None:
            hard_attn_mask = None
            key_padding_mask = None
        else:
            hard_attn_mask = self._build_per_query_attn_mask(spatial_keep_mask)
            key_padding_mask = None

        soft_attn_bias = None
        if spatial_soft_weight is not None:
            soft_attn_bias = self._build_soft_attn_bias(spatial_soft_weight)

        for layer in self.layers:
            q = layer(
                q,
                kv,
                key_padding_mask=key_padding_mask,
                attn_mask=hard_attn_mask,
                soft_attn_bias=soft_attn_bias,
                q_attn_mask=q_attn_mask,
                q_key_padding_mask=q_key_padding_mask,
            )
        return q


class BranchAdaptiveFusion(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.weight_head = nn.Linear(hidden_dim * 4, 3)
        self.proj_h = nn.Linear(hidden_dim, hidden_dim)
        self.proj_o = nn.Linear(hidden_dim, hidden_dim)
        self.proj_r = nn.Linear(hidden_dim, hidden_dim)
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_o = nn.LayerNorm(hidden_dim)
        self.norm_r = nn.LayerNorm(hidden_dim)

    def forward(self, sub_feat, obj_feat, rel_feat, context):
        x = torch.cat([sub_feat, obj_feat, rel_feat, context], dim=-1)
        w = torch.softmax(self.weight_head(x), dim=-1)

        ctx_h = w[..., 0:1] * context
        ctx_o = w[..., 1:2] * context
        ctx_r = w[..., 2:3] * context

        sub = self.norm_h(sub_feat + self.proj_h(ctx_h))
        obj = self.norm_o(obj_feat + self.proj_o(ctx_o))
        rel = self.norm_r(rel_feat + self.proj_r(ctx_r))
        return sub, obj, rel, w


class FrozenPriorBank(nn.Module):
    """Lightweight frozen cache for visual/text priors used by gated fusion.

    This module does not alter the backbone architecture. It only provides
    a trainable wrapper for precomputed or externally encoded priors that can
    be injected into an existing encoder/decoder stack.
    """

    def __init__(self, prior_dim=256, num_prior_tokens=0, freeze=True):
        super().__init__()
        self.prior_dim = prior_dim
        self.num_prior_tokens = num_prior_tokens
        self.register_buffer("text_priors", torch.empty(0, prior_dim), persistent=False)
        self.register_buffer("semantic_similarity_matrix", torch.empty(0, 0), persistent=False)
        self.register_buffer("vision_prior_tokens", torch.empty(0, prior_dim), persistent=False)
        self._frozen = freeze

    @torch.no_grad()
    def set_text_priors(self, text_priors, semantic_similarity_matrix=None):
        self.text_priors = text_priors.detach().clone()
        if semantic_similarity_matrix is not None:
            self.semantic_similarity_matrix = semantic_similarity_matrix.detach().clone()
        self.num_prior_tokens = int(self.text_priors.shape[0])

    @torch.no_grad()
    def set_vision_priors(self, vision_prior_tokens):
        self.vision_prior_tokens = vision_prior_tokens.detach().clone()

    def forward(self):
        return {
            "text_priors": self.text_priors,
            "semantic_similarity_matrix": self.semantic_similarity_matrix,
            "vision_prior_tokens": self.vision_prior_tokens,
        }


class GatedCrossAttention(nn.Module):
    """Cross-attention with a learnable scalar gate.

    Query tokens remain the original visual stream; key/value tokens are frozen
    textual priors. The gate controls how much semantic information is injected.
    """

    def __init__(self, d_model=256, nhead=8, dropout=0.1, init_gate=0.0):
        super().__init__()
        self.d_model = d_model
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.gate_logit = nn.Parameter(torch.tensor(float(init_gate)))
        self.text_proj = None

    def forward(self, query_tokens, key_value_tokens, key_padding_mask=None, attn_mask=None):
        if key_value_tokens is None or key_value_tokens.numel() == 0:
            return query_tokens
        if key_value_tokens.shape[-1] != self.d_model:
            raise ValueError(f'Expected key/value dim {self.d_model}, got {key_value_tokens.shape[-1]}')
        attn_out, _ = self.cross_attn(query_tokens, key_value_tokens, key_value_tokens,
                                      key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        gate = torch.sigmoid(self.gate_logit)
        return self.norm(query_tokens + self.drop(gate * attn_out))


class EarlyGatedSemanticExtraction(nn.Module):
    """Drop-in add-on for early gated fusion before decoder entry.

    Keeps the original encoder intact and only appends a semantic fusion step.
    It can be attached to the final multi-scale visual tokens produced by the
    existing ResNet/encoder stack.
    """

    def __init__(self, d_model=256, nhead=8, dropout=0.1, init_gate=0.0):
        super().__init__()
        self.semantic_gate = GatedCrossAttention(d_model=d_model, nhead=nhead, dropout=dropout, init_gate=init_gate)
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, visual_tokens, textual_priors, key_padding_mask=None, attn_mask=None):
        fused = self.semantic_gate(
            visual_tokens,
            textual_priors,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        return self.output_norm(fused)
