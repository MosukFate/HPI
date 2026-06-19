# hpi_cross_attention_standard.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HPIStandardCrossAttention(nn.Module):
    """
    Standard patch-query/text-key-value cross-attention ablation.

    This keeps the same forward API as HPICrossAttention:
        patch_tokens: [B, N, D]
        shp_embed:   [K, T]

    Returns:
        delta_p:          [B, N, D]
        semantic_logits: [B, K]
        prompt_patch_attn:           [B, N, K]
    """

    def __init__(
        self,
        dim_img: int,
        dim_txt: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        assert dim_img % num_heads == 0, "dim_img must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim_img // num_heads
        self.dim_img = dim_img

        self.query_proj = nn.Linear(dim_img, dim_img, bias=qkv_bias)
        self.key_proj = nn.Linear(dim_txt, dim_img, bias=qkv_bias)
        self.value_proj = nn.Linear(dim_txt, dim_img, bias=qkv_bias)
        self.patch_value_proj = nn.Linear(dim_img, dim_img, bias=qkv_bias)

        self.proj = nn.Linear(dim_img, dim_img)

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))
        self.temperature = nn.Parameter(torch.tensor(0.7))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, h, *, d] -> [B, *, h*d]"""
        B, h, L, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, h * d)

    def forward(self, patch_tokens: torch.Tensor, shp_embed: torch.Tensor):
        B, N, D = patch_tokens.shape
        Kp = shp_embed.shape[0]
        h = self.num_heads
        d = self.head_dim
        device = patch_tokens.device
        dtype = patch_tokens.dtype

        shp_embed = shp_embed.to(device=device, dtype=dtype)

        Q_full = self.query_proj(patch_tokens)  # [B, N, D]
        K_full = self.key_proj(shp_embed)      # [K, D]
        V_full = self.value_proj(shp_embed)    # [K, D]

        Q = Q_full.view(B, N, h, d).permute(0, 2, 1, 3)                         # [B,h,N,d]
        K = K_full.view(1, Kp, h, d).permute(0, 2, 1, 3).expand(B, -1, -1, -1)   # [B,h,K,d]
        V = V_full.view(1, Kp, h, d).permute(0, 2, 1, 3).expand(B, -1, -1, -1)   # [B,h,K,d]

        T = self.temperature.clamp(0.05, 5.0)

        attn_scores = torch.einsum("bhnd,bhkd->bhnk", Q, K) / math.sqrt(d)
        attn_scores = attn_scores / T
        attn_weights = self.attn_drop(F.softmax(attn_scores, dim=-1))  # [B,h,N,K]

        delta_p_heads = torch.einsum("bhnk,bhkd->bhnd", attn_weights, V)  # [B,h,N,d]
        delta_p = self._merge_heads(delta_p_heads)
        delta_p = self.proj_drop(self.proj(delta_p))

        # Build prompt-wise visual summaries so the existing image-level
        # semantic-consistency loss can be reused unchanged.
        patch_values = self.patch_value_proj(patch_tokens)
        patch_values = patch_values.view(B, N, h, d).permute(0, 2, 1, 3)  # [B,h,N,d]
        prompt_weights = attn_weights.transpose(2, 3)                    # [B,h,K,N]
        prompt_weights = prompt_weights / prompt_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        out_prompt = torch.einsum("bhkn,bhnd->bhkd", prompt_weights, patch_values)

        out_prompt_merge = out_prompt.permute(0, 2, 1, 3).contiguous().view(B, Kp, h * d)
        txt_norm = F.normalize(K_full, dim=-1).unsqueeze(0).expand(B, -1, -1)
        img_norm = F.normalize(out_prompt_merge, dim=-1)
        sim_cos = (img_norm * txt_norm).sum(dim=-1)

        logit_scale = self.logit_scale.exp().clamp(1.0, 100.0)
        semantic_logits = sim_cos * logit_scale

        prompt_patch_attn = attn_weights.mean(dim=1).contiguous()  # [B,N,K]

        return delta_p, semantic_logits, prompt_patch_attn
