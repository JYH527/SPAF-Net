import torch
from torch import nn
import torch.nn.functional as F


class FrozenBLIP2ImageEncoder(nn.Module):
    """
    Frozen BLIP-2 image encoder wrapper.

    Priority:
    1) If `use_real_blip2` is enabled, run local BLIP-2 vision encoder (frozen).
    2) Else if `vlm_image_feat_path` is provided, use precomputed features.
    3) Else return zero placeholders (safe fallback).

    Returns dict:
      - global_feat: [B, Dv]
      - patch_tokens: [B, Nv, Dv]
      - grid_size: (Hp, Wp)
    """

    def __init__(self, args):
        super().__init__()
        self.vlm_embed_dim = getattr(args, 'vlm_embed_dim', 1408)
        self.precomputed_path = getattr(args, 'vlm_image_feat_path', '')
        self.default_grid = tuple(getattr(args, 'vlm_patch_grid', [16, 16]))

        self.use_real_blip2 = getattr(args, 'use_real_blip2', False)
        self.blip2_model_name = getattr(args, 'blip2_model_name', 'BLIP2-opt-2.7b')

        self.processor = None
        self.model = None
        if self.use_real_blip2:
            from transformers import Blip2Processor, Blip2Model

            self.processor = Blip2Processor.from_pretrained(self.blip2_model_name, local_files_only=True)
            self.model = Blip2Model.from_pretrained(self.blip2_model_name, local_files_only=True)
            self.model.eval()
            self.model.requires_grad_(False)
            for p in self.model.parameters():
                p.requires_grad = False

            # Prefer actual BLIP-2 vision hidden size when available
            if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'config'):
                self.vlm_embed_dim = int(getattr(self.model.vision_model.config, 'hidden_size', self.vlm_embed_dim))

        self.use_precomputed = False
        self.precomputed = None
        if self.precomputed_path:
            ckpt = torch.load(self.precomputed_path, map_location='cpu')
            # formats:
            # 1) {'global_feat': [N,D], 'patch_tokens': [N,Nv,D], 'grid_size': (Hp, Wp)}
            # 2) {'features': [N,D]} legacy
            # 3) Tensor [N,D] legacy
            if isinstance(ckpt, dict):
                if 'global_feat' in ckpt or 'patch_tokens' in ckpt:
                    self.precomputed = ckpt
                    self.use_precomputed = True
                elif 'features' in ckpt and torch.is_tensor(ckpt['features']):
                    self.precomputed = {
                        'global_feat': ckpt['features'],
                        'patch_tokens': None,
                        'grid_size': self.default_grid,
                    }
                    self.use_precomputed = True
            elif torch.is_tensor(ckpt):
                self.precomputed = {
                    'global_feat': ckpt,
                    'patch_tokens': None,
                    'grid_size': self.default_grid,
                }
                self.use_precomputed = True

    @staticmethod
    def _to_pil_batch(images: torch.Tensor):
        """
        images: normalized tensor [B,3,H,W] using ImageNet mean/std.
        convert back to PIL for BLIP-2 processor.
        """
        from PIL import Image

        mean = torch.tensor([0.485, 0.456, 0.406], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=images.device).view(1, 3, 1, 1)
        x = images * std + mean
        x = x.clamp(0.0, 1.0)
        x = (x * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

        pil_images = [Image.fromarray(arr) for arr in x]
        return pil_images

    @torch.no_grad()
    def _forward_real_blip2(self, images: torch.Tensor):
        device = images.device
        self.model.to(device)

        pil_images = self._to_pil_batch(images)
        proc = self.processor(images=pil_images, return_tensors='pt')
        pixel_values = proc['pixel_values'].to(device)

        vis_out = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        # [B, 1+Nv, Dv]
        hs = vis_out.last_hidden_state
        global_feat = hs[:, 0, :]
        patch_tokens = hs[:, 1:, :]

        nv = patch_tokens.shape[1]
        hp = int(nv ** 0.5)
        wp = hp if hp * hp == nv else nv
        grid_size = (hp, wp)

        return {
            'global_feat': global_feat,
            'patch_tokens': patch_tokens,
            'grid_size': grid_size,
        }

    def forward(self, images: torch.Tensor):
        bsz = images.shape[0]
        hp, wp = self.default_grid
        nv = hp * wp

        if self.use_real_blip2 and self.model is not None and self.processor is not None:
            return self._forward_real_blip2(images)

        if self.use_precomputed and self.precomputed is not None:
            global_feat = self.precomputed.get('global_feat', None)
            patch_tokens = self.precomputed.get('patch_tokens', None)
            grid_size = tuple(self.precomputed.get('grid_size', self.default_grid))

            if torch.is_tensor(global_feat):
                global_feat = global_feat.to(device=images.device, dtype=images.dtype)
                if global_feat.dim() == 1:
                    global_feat = global_feat.unsqueeze(0)
                if global_feat.shape[0] != bsz:
                    global_feat = global_feat.expand(bsz, -1).contiguous()
            else:
                global_feat = torch.zeros((bsz, self.vlm_embed_dim), device=images.device, dtype=images.dtype)

            if torch.is_tensor(patch_tokens):
                patch_tokens = patch_tokens.to(device=images.device, dtype=images.dtype)
                if patch_tokens.dim() == 2:
                    patch_tokens = patch_tokens.unsqueeze(0)
                if patch_tokens.shape[0] != bsz:
                    patch_tokens = patch_tokens.expand(bsz, -1, -1).contiguous()
            else:
                hp, wp = grid_size
                patch_tokens = torch.zeros((bsz, hp * wp, self.vlm_embed_dim), device=images.device, dtype=images.dtype)

            return {
                'global_feat': global_feat,
                'patch_tokens': patch_tokens,
                'grid_size': grid_size,
            }

        return {
            'global_feat': torch.zeros((bsz, self.vlm_embed_dim), device=images.device, dtype=images.dtype),
            'patch_tokens': torch.zeros((bsz, nv, self.vlm_embed_dim), device=images.device, dtype=images.dtype),
            'grid_size': (hp, wp),
        }


def build_text_semantic_similarity(text_features: torch.Tensor) -> torch.Tensor:
    text_features = F.normalize(text_features, dim=-1)
    return torch.matmul(text_features, text_features.t())
