from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh, box_iou,masks_to_boxes, box_area
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import numpy as np
from queue import Queue
import math

from .backbone import build_backbone
from .matcher import build_matcher
from .muren import build_muren
from .cross_modal_fusion_qacf import EarlyGatedSemanticExtraction, FrozenPriorBank
from .frozen_blip2_priors import FrozenBLIP2ImageEncoder

class MURENHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, args=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_human = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_obj = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_rel = nn.Embedding(num_queries, hidden_dim)

        self.obj_class_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_obj_classes + 1) for _ in range(args.dec_layers)
        ])
        self.verb_class_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_verb_classes) for _ in range(args.dec_layers)
        ])
        self.sub_bbox_heads = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(args.dec_layers)
        ])
        self.obj_bbox_heads = nn.ModuleList([
            MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(args.dec_layers)
        ])

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.use_matching = args.use_matching
        self.dec_layers = args.dec_layers

        if self.use_matching:
            self.matching_embed = MLP(hidden_dim*2, hidden_dim, 2, 3)

        # Optional add-ons: keep original architecture intact and freeze backbone if requested.
        self.use_early_gated_fusion = getattr(args, 'use_early_gated_fusion', False)
        self.freeze_backbone = getattr(args, 'freeze_backbone', False)
        self.freeze_transformer = getattr(args, 'freeze_transformer', False)
        self.freeze_query_heads = getattr(args, 'freeze_query_heads', False)

        self.text_prior_bank = FrozenPriorBank(prior_dim=hidden_dim)
        self.vlm_prior_encoder = FrozenBLIP2ImageEncoder(args) if getattr(args, 'use_vlm_priors', False) else None
        self.vlm_token_proj = None
        self.vlm_embed_dim = getattr(args, 'vlm_embed_dim', 1408)
        if self.vlm_prior_encoder is not None and self.vlm_embed_dim != hidden_dim:
            self.vlm_token_proj = nn.Linear(self.vlm_embed_dim, hidden_dim)
        self.early_gated_fusion = EarlyGatedSemanticExtraction(
            d_model=hidden_dim,
            nhead=getattr(args, 'fusion_nheads', 8),
            dropout=getattr(args, 'fusion_dropout', 0.1),
            init_gate=getattr(args, 'fusion_init_gate', 0.0),
        ) if self.use_early_gated_fusion else None

        if hasattr(self.transformer, 'encoder') and self.early_gated_fusion is not None:
            self.transformer.encoder.final_fusion = self.early_gated_fusion
            self.transformer.encoder.final_fusion_bank = self.text_prior_bank

        if getattr(args, 'use_spatial_qacf', False) and hasattr(self.transformer, 'decoder'):
            self.transformer.decoder.use_spatial_qacf = True

        if hasattr(self.transformer, 'decoder'):
            self.transformer.decoder.sub_bbox_heads = self.sub_bbox_heads
            self.transformer.decoder.obj_bbox_heads = self.obj_bbox_heads

        if self.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.input_proj.parameters():
                p.requires_grad = False

        if self.freeze_transformer:
            for p in self.transformer.parameters():
                p.requires_grad = False

        if self.freeze_query_heads:
            for p in list(self.obj_class_heads.parameters()) + list(self.verb_class_heads.parameters()) + \
                     list(self.sub_bbox_heads.parameters()) + list(self.obj_bbox_heads.parameters()):
                p.requires_grad = False
            if self.use_matching:
                for p in self.matching_embed.parameters():
                    p.requires_grad = False

    def load_text_priors(self, prior_path):
        prior = torch.load(prior_path, map_location='cpu')
        text_features = prior.get('text_features', prior.get('verb_text_features', None))
        sim_matrix = prior.get('sim_matrix', prior.get('interaction_sim_matrix', prior.get('verb_sim_matrix', None)))
        if text_features is None:
            raise ValueError(f'No text features found in prior file: {prior_path}')
        self.text_prior_bank.set_text_priors(text_features, sim_matrix)

    def _get_text_priors_for_batch(self, batch_size, device, dtype):
        priors = self.text_prior_bank()
        text_priors = priors.get('text_priors', None)
        sim_matrix = priors.get('semantic_similarity_matrix', None)
        if text_priors is None or text_priors.numel() == 0:
            return None, None
        text_priors = text_priors.to(device=device, dtype=dtype)
        if text_priors.dim() == 2:
            text_priors = text_priors.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        elif text_priors.dim() == 3 and text_priors.shape[0] == batch_size:
            text_priors = text_priors.contiguous()
        else:
            raise ValueError(f'Expected text priors shape [B,N,C] or [N,C], got {tuple(text_priors.shape)}')
        if sim_matrix is not None and sim_matrix.numel() > 0:
            sim_matrix = sim_matrix.to(device=device, dtype=dtype)
        else:
            sim_matrix = None
        return text_priors, sim_matrix

    def forward(self, samples: NestedTensor):
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        vlm_tokens = None
        vlm_grid_size = None
        if self.vlm_prior_encoder is not None:
            vlm_prior = self.vlm_prior_encoder(samples.tensors)
            vlm_tokens = vlm_prior.get('patch_tokens', None)
            vlm_grid_size = vlm_prior.get('grid_size', None)
            if vlm_tokens is not None and self.vlm_token_proj is not None:
                vlm_tokens = self.vlm_token_proj(vlm_tokens)

        src_proj = self.input_proj(src)
        text_priors, sim_matrix = self._get_text_priors_for_batch(src_proj.shape[0], src_proj.device, src_proj.dtype)
        trans_outputs = self.transformer(
            src_proj,
            mask,
            self.query_embed_human.weight,
            self.query_embed_obj.weight,
            self.query_embed_rel.weight,
            pos[-1],
            fusion_tokens=text_priors,
            fusion_similarity_matrix=sim_matrix,
            vlm_tokens=vlm_tokens,
            sub_boxes=None,
            obj_boxes=None,
            grid_size=vlm_grid_size if vlm_grid_size is not None else (src_proj.shape[-2], src_proj.shape[-1]),
        )

        if len(trans_outputs) == 5:
            sub_out, obj_out, rel_out, _, _ = trans_outputs
        elif len(trans_outputs) == 4:
            sub_out, obj_out, rel_out, _ = trans_outputs
        else:
            raise ValueError(f'Unexpected transformer output structure: {len(trans_outputs)} values')

        outputs_obj_class = []
        outputs_verb_class = []
        outputs_sub_coord = []
        outputs_obj_coord = []
        outputs_matching = [] if self.use_matching else None

        for layer_idx in range(sub_out.shape[0]):
            layer_sub = sub_out[layer_idx]
            layer_obj = obj_out[layer_idx]
            layer_rel = rel_out[layer_idx]
            outputs_sub_coord.append(self.sub_bbox_heads[layer_idx](layer_sub).sigmoid())
            outputs_obj_coord.append(self.obj_bbox_heads[layer_idx](layer_obj).sigmoid())
            outputs_obj_class.append(self.obj_class_heads[layer_idx](layer_obj))
            outputs_verb_class.append(self.verb_class_heads[layer_idx](layer_rel))
            if self.use_matching:
                outputs_matching.append(self.matching_embed(torch.cat([layer_sub, layer_obj], dim=-1)))

        outputs_sub_coord = torch.stack(outputs_sub_coord)
        outputs_obj_coord = torch.stack(outputs_obj_coord)
        outputs_obj_class = torch.stack(outputs_obj_class)
        outputs_verb_class = torch.stack(outputs_verb_class)
        if self.use_matching:
            outputs_matching = torch.stack(outputs_matching)

        out = {
            'pred_obj_logits': outputs_obj_class[-1],
            'pred_verb_logits': outputs_verb_class[-1],
            'pred_sub_boxes': outputs_sub_coord[-1],
            'pred_obj_boxes': outputs_obj_coord[-1],
            'sub_out': sub_out,
            'obj_out': obj_out,
            'rel_out': rel_out,
        }

        if self.use_matching:
            out['pred_matching_logits'] = outputs_matching[-1]

        if self.aux_loss:
            if self.use_matching:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_matching)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_matching=None):
        if self.use_matching:
            return [{
                'pred_obj_logits': a,
                'pred_verb_logits': b,
                'pred_sub_boxes': c,
                'pred_obj_boxes': d,
                'pred_matching_logits': e,
            } for a, b, c, d, e in zip(outputs_obj_class[:-1], outputs_verb_class[:-1], outputs_sub_coord[:-1], outputs_obj_coord[:-1], outputs_matching[:-1])]
        else:
            return [{
                'pred_obj_logits': a,
                'pred_verb_logits': b,
                'pred_sub_boxes': c,
                'pred_obj_boxes': d,
            } for a, b, c, d in zip(outputs_obj_class[:-1], outputs_verb_class[:-1], outputs_sub_coord[:-1], outputs_obj_coord[:-1])]





class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.args = args
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.alpha = args.alpha

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o


        obj_weights = self.empty_weight

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, obj_weights)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, alpha=self.alpha)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses


    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def loss_matching_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_matching_logits' in outputs
        src_logits = outputs['pred_matching_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['matching_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_matching = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_matching': loss_matching}

        if log:
            losses['matching_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def _neg_loss(self, pred, gt, alpha=0.25):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'matching_labels': self.loss_matching_labels,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, **kwargs)

    def forward(self, outputs, targets):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if 'cross_outputs' in outputs:
            indices = self.matcher(outputs_without_aux, targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'obj_labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, outputs['cross_outputs'][-1], targets, indices, num_interactions, **kwargs)
                l_dict = {k + f'_cross':  self.args.cl_w * v for k, v in l_dict.items()}
                losses.update(l_dict)

        
        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id
        self.use_matching = args.use_matching

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits = outputs['pred_obj_logits']
        out_verb_logits = outputs['pred_verb_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        if self.use_matching:
            out_matching_logits = outputs['pred_matching_logits']
            matching_scores = F.softmax(out_matching_logits, -1)[..., 1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(obj_scores)):
            os, ol, vs, sb, ob =  obj_scores[index], obj_labels[index], verb_scores[index], sub_boxes[index], obj_boxes[index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            if self.use_matching:
                ms = matching_scores[index]
                vs = vs * ms.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:],'obj_scores':os.to('cpu')})

        return results
def build(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)

    use_early_gated_fusion = getattr(args, 'use_early_gated_fusion', False)
    fusion_nheads = getattr(args, 'fusion_nheads', getattr(args, 'nheads', 8))
    fusion_dropout = getattr(args, 'fusion_dropout', getattr(args, 'dropout', 0.1))
    fusion_init_gate = getattr(args, 'fusion_init_gate', 0.0)
    temp_fusion = EarlyGatedSemanticExtraction(
        d_model=args.hidden_dim,
        nhead=fusion_nheads,
        dropout=fusion_dropout,
        init_gate=fusion_init_gate,
    ) if use_early_gated_fusion else None

    MUREN = build_muren(args, encoder_final_fusion=temp_fusion)

    model = MURENHOI(
        backbone,
        MUREN,
        num_obj_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    if getattr(args, 'load_text_priors_path', ''):
        model.load_text_priors(args.load_text_priors_path)

    if getattr(args, 'freeze_text_prior_bank', True):
        for p in model.text_prior_bank.parameters():
            p.requires_grad = False

    if getattr(args, 'freeze_vlm_prior_encoder', True) and model.vlm_prior_encoder is not None:
        for p in model.vlm_prior_encoder.parameters():
            p.requires_grad = False

    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    if args.use_matching:
        weight_dict['loss_matching'] = args.matching_loss_coef

    if args.aux_loss:
        min_dec_layers_num = args.dec_layers
        aux_weight_dict = {}
        for i in range(min_dec_layers_num - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality']
    if args.use_matching:
        losses.append('matching_labels')


    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                            weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                            args=args)

    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args)}

    return model, criterion, postprocessors
