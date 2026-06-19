# hpi_cross_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HPICrossAttention(nn.Module):
    """
    Patch-level HPI cross-attention. Text is Q, image patches are K/V, and external QKV weights are not reused.

    - Inputs:
        patch_tokens: [B, N, D]   Image patch tokens with visual width D
        shp_embed:   [C, T]      Text/description prototypes with text width T

    - Outputs:
        delta_p: [B, N, D]   Patch update written back to image tokens, enriched by text semantics while staying in visual value space
        sim:   [B, 1]      Monitoring similarity between the text-conditioned image summary and text prototypes
        prompt_patch_attn:  [B, N, C]   Per-patch attention to each text prototype, averaged across heads and usable for weighting

    - Computation notes:
        1) Weights = softmax(QK^T / sqrt(d))
        2) Text-to-image read: out_text = weights * Vimage, yielding [B, h, C, d]
        3) Distribute out_text back to each patch: delta_p = weights^T * out_text, yielding [B, h, N, d]
           (This writes text-weighted image values rather than raw text vectors, keeping the update in visual value space.)
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

        self.query_proj = nn.Linear(dim_txt, dim_img, bias=qkv_bias)
        self.key_proj   = nn.Linear(dim_img, dim_img, bias=qkv_bias)
        self.value_proj = nn.Linear(dim_img, dim_img, bias=qkv_bias)

        self.proj = nn.Linear(dim_img, dim_img)

        # Learnable temperature, clamped in forward
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))
        self.temperature = nn.Parameter(torch.tensor(0.7))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, h, *, d] -> [B, *, h*d]"""
        B, h, L, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(B, L, h * d)

    def forward(self, patch_tokens: torch.Tensor, shp_embed: torch.Tensor):
        """
        patch_tokens: [B, N, D]
        shp_embed:   [C, T]
        """
        B, N, D = patch_tokens.shape
        C = shp_embed.shape[0]
        h = self.num_heads
        d = self.head_dim
        device = patch_tokens.device
        dtype  = patch_tokens.dtype

        Q_full = self.query_proj(shp_embed.to(device=device, dtype=dtype))  # [C, D]
        K_full = self.key_proj(patch_tokens)                                  # [B, N, D]
        V_full = self.value_proj(patch_tokens)                                # [B, N, D]

        Q = Q_full.view(1, C, h, d).permute(0, 2, 1, 3).expand(B, -1, -1, -1)  # [B,h,C,d]
        K = K_full.view(B, N, h, d).permute(0, 2, 1, 3)                        # [B,h,N,d]
        V = V_full.view(B, N, h, d).permute(0, 2, 1, 3)                        # [B,h,N,d]

        # Clamp temperature in forward
        T = self.temperature.clamp(0.05, 5.0)

        attn_scores  = torch.einsum('bhcd,bhnd->bhcn', Q, K) / math.sqrt(d)
        attn_scores  = attn_scores / T
        attn_weights = self.attn_drop(F.softmax(attn_scores, dim=-1))          # [B,h,C,N]

        out_text = torch.einsum('bhcn,bhnd->bhcd', attn_weights, V)            # [B,h,C,d]
        delta_p_heads = torch.einsum('bhnc,bhcd->bhnd', attn_weights.transpose(2, 3), out_text)  # [B,h,N,d]

        delta_p = self._merge_heads(delta_p_heads)                                  # [B,N,D]
        delta_p = self.proj_drop(self.proj(delta_p))                                # [B,N,D]

        # Per-class cosine similarity to alignment logits
        # Merge out_text by class to [B,C,D] and compute cosine similarity with text prototypes Q_full
        out_text_merge = out_text.permute(0, 2, 1, 3).contiguous().view(B, C, h * d)  # [B,C,D]
        txt_norm = F.normalize(Q_full, dim=-1).unsqueeze(0).expand(B, -1, -1)         # [B,C,D]
        img_norm = F.normalize(out_text_merge, dim=-1)                                 # [B,C,D]
        sim_cos = (img_norm * txt_norm).sum(dim=-1)                                    # [B,C] ∈ [-1,1]

        # Scale cosine similarity into logits for BCEWithLogitsLoss
        logit_scale = self.logit_scale.exp().clamp(1.0, 100.0)
        semantic_logits = sim_cos * logit_scale                                         # [B,C]

        # Average heads to obtain [B,N,C] attention maps for the spatial loss
        prompt_patch_attn = attn_weights.mean(dim=1).transpose(1, 2).contiguous()  # [B, N, C]

        return delta_p, semantic_logits, prompt_patch_attn
