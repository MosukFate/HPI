# --------------------------------------------------------
# HPI EVA vision backbone. Adapted from EVA-CLIP/BEiT components.
# --------------------------------------------------------
import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from timm.models.layers import drop_path, to_2tuple, trunc_normal_
except Exception:
    from timm.layers import drop_path, to_2tuple, trunc_normal_

from .transformer import PatchDropout
from .rope import VisionRotaryEmbeddingFast
from ..adapter_pruning import *
from ..dino_v2 import DinoVisionTransformer
from ..hpi_cross_attention import HPICrossAttention
from mmseg.models.builder import BACKBONES

if os.getenv('ENV_TYPE') == 'deepspeed':
    try:
        from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint
    except Exception:
        from torch.utils.checkpoint import checkpoint
else:
    from torch.utils.checkpoint import checkpoint

try:
    import xformers.ops as xops
except ImportError:
    xops = None


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Mlp(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        norm_layer=nn.LayerNorm, 
        drop=0.,
        subln=False,

        ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.ffn_ln(x)

        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0., 
                norm_layer=nn.LayerNorm, subln=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.ffn_ln = norm_layer(hidden_features) if subln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None, xattn=False, rope=None, subln=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.subln = subln
        if self.subln:
            self.q_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.k_proj = nn.Linear(dim, all_head_dim, bias=False)
            self.v_proj = nn.Linear(dim, all_head_dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(window_size[0])
            coords_w = torch.arange(window_size[1])
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros(size=(window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
            relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0:] = self.num_relative_distance - 3
            relative_position_index[0:, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer("relative_position_index", relative_position_index)
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.inner_attn_ln = norm_layer(all_head_dim) if subln else nn.Identity()
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.xattn = xattn
        self.xattn_drop = attn_drop

        self.rope = rope

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        B, N, C = x.shape
        if self.subln: 
            q = F.linear(input=x, weight=self.q_proj.weight, bias=self.q_bias)
            k = F.linear(input=x, weight=self.k_proj.weight, bias=None)
            v = F.linear(input=x, weight=self.v_proj.weight, bias=self.v_bias)

            q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)     # B, num_heads, N, C
            k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)  
            v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3) 
        else: 

            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)   # 3, B, num_heads, N, C
            q, k, v = qkv[0], qkv[1], qkv[2]

        if self.rope:
            # slightly fast impl
            q_t = q[:, :, 1:, :]
            ro_q_t = self.rope(q_t)
            q = torch.cat((q[:, :, :1, :], ro_q_t), -2).type_as(v)

            k_t = k[:, :, 1:, :]
            ro_k_t = self.rope(k_t)
            k = torch.cat((k[:, :, :1, :], ro_k_t), -2).type_as(v)

        if self.xattn:
            q = q.permute(0, 2, 1, 3)   # B, num_heads, N, C -> B, N, num_heads, C
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            x = xops.memory_efficient_attention(
                q, k, v,
                p=self.xattn_drop,
                scale=self.scale,
                )
            x = x.reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        else:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))

            if self.relative_position_bias_table is not None:
                relative_position_bias = \
                    self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0).type_as(attn)

            if rel_pos_bias is not None:
                attn = attn + rel_pos_bias.type_as(attn)

            if attn_mask is not None:
                attn_mask = attn_mask.bool()
                attn = attn.masked_fill(~attn_mask[:, None, None, :], float("-inf"))
            
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
            x = self.inner_attn_ln(x)
            x = self.proj(x)
            x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None, xattn=False, rope=None, postnorm=False,
                 subln=False, naiveswiglu=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim,
            xattn=xattn, rope=rope, subln=subln, norm_layer=norm_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim, 
                hidden_features=mlp_hidden_dim, 
                act_layer=act_layer,
                subln=subln,
                drop=drop
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)))
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros(size=(window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww



class MultiScaleSpatialOperator(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.fuse  = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.dirac_(self.conv0.weight)
        nn.init.dirac_(self.conv1.weight)
        nn.init.dirac_(self.conv2.weight)
        
        with torch.no_grad():
            self.fuse.weight.zero_()
            C = self.fuse.weight.shape[1] // 3
            self.fuse.weight[:, 0:C, 0, 0] = torch.eye(C) / 3.0
            self.fuse.weight[:, C:2*C, 0, 0] = torch.eye(C) / 3.0
            self.fuse.weight[:, 2*C:3*C, 0, 0] = torch.eye(C) / 3.0
            self.fuse.bias.zero_()

    def forward(self, x):
        x1 = self.conv0(x)
        x2 = self.conv1(x)
        x3 = self.conv2(x)
        return self.fuse(torch.cat([x1, x2, x3], dim=1))


class SCCGate(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int = 64):
        super().__init__()
        self.reduce = nn.Linear(dim * 2, bottleneck_dim * 2)
        self.msconv = MultiScaleSpatialOperator(bottleneck_dim * 2)
        self.out = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, 1),
            nn.Sigmoid()
        )
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.reduce.weight, std=0.02)
        nn.init.zeros_(self.reduce.bias)
        nn.init.trunc_normal_(self.out[0].weight, std=0.02)
        nn.init.constant_(self.out[0].bias, 0.0)

    def forward(self, x_cat: torch.Tensor, hw: tuple) -> torch.Tensor:
        B, T, _ = x_cat.shape
        H, W = hw
        assert H * W == T, f"SCCGate: H*W({H}*{W}) != T({T})"
        feat = self.reduce(x_cat)
        feat_2d = feat.transpose(1, 2).reshape(B, -1, H, W)
        multi = self.msconv(feat_2d)
        multi = multi.flatten(2).transpose(1, 2)
        w = self.out(multi)
        return w
    
@BACKBONES.register_module()
class EVAVisionTransformerHPI(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, patch_dropout=0.,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, rope=False,
                 use_mean_pooling=True, init_scale=0.001, grad_checkpointing=False, xattn=False, postnorm=False,
                 pt_hw_seq_len=16, intp_freq=False, naiveswiglu=False, subln=False, out_indices=[], pretrained=None,
                 adapter_type=None,
                 hpi_layers=None, hpi_layers_dino=None):
        super().__init__()
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.adapter_type = adapter_type
        
        self.image_size = img_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None
        
        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=pt_hw_seq_len,
                ft_seq_len=hw_seq_len if intp_freq else None,
            )
        else: 
            self.rope = None

        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                xattn=xattn, rope=self.rope, postnorm=postnorm, subln=subln, naiveswiglu=naiveswiglu)
            for i in range(depth)])

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.grad_checkpointing = grad_checkpointing

        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim*2, embed_dim*2, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim*2),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim*2, embed_dim*2, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim*2, embed_dim*2, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dinov2 = DinoVisionTransformer(patch_size=14,
                        embed_dim=1024,
                        depth=24,
                        num_heads=16,
                        mlp_ratio=4,
                        img_size=512,
                        ffn_layer="mlp",
                        init_values=1e-05,
                        block_chunks=0,
                        qkv_bias=True,
                        proj_bias=True,
                        ffn_bias=True,)
        dinov2_state_dict = torch.load('pretrained/dinov2_vitl14_pretrain.pth')
        all_keys = list(dinov2_state_dict.keys())
        # interpolate position embedding
        if 'pos_embed' in dinov2_state_dict:
            pos_embed_checkpoint = dinov2_state_dict['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = self.dinov2.patch_embed.num_patches
            num_extra_tokens = self.dinov2.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                dinov2_state_dict['pos_embed'] = new_pos_embed

                patch_embed_proj = dinov2_state_dict['patch_embed.proj.weight']
                patch_size = self.dinov2.patch_embed.patch_size
                dinov2_state_dict['patch_embed.proj.weight'] = torch.nn.functional.interpolate(
                    patch_embed_proj.float(), size=patch_size, mode='bicubic', align_corners=False)
        self.dinov2.load_state_dict(dinov2_state_dict, strict=True)

        if self.adapter_type == 'vlmborrow':        
            vlm_flag = [False, False, False, False, False, False,
                    False, False, False, True,  True,  True,
                    True,  True,  True,  True,  True,  False,
                    False, False, False, False, True, True]

            self.vlm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=vlm_flag[i]) for i in range(24)]) 
            self.vfm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=False) for i in range(24)])
        elif self.adapter_type == 'vfmbase':
            self.vlm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=False) for i in range(24)]) 
            self.vfm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=True) for i in range(24)])
        elif self.adapter_type == 'vlmadp':
            self.vlm_adapter   = nn.Sequential(*[DynGateAdapter() for i in range(24)]) 
            self.vfm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=False) for i in range(24)])        
        elif self.adapter_type == 'bar':
            vlm_flag = [False, False, False, False, False, False,
                        False, False, False, False, False, False,
                        False, False, False, False, False,  True,
                        True,  True,  True,  True,  True,  True]
            self.vlm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=vlm_flag[i]) for i in range(24)]) 
            self.vfm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=False) for i in range(24)])
        else:
            assert False, f"Not implement"
        self.hpi_layers = hpi_layers if hpi_layers is not None else [18, 22]
        self.hpi_layers_dino = hpi_layers_dino if hpi_layers_dino is not None else [23]

        # Text-side prototype library injected by the segmentor
        if not hasattr(self, 'shp_embed_vlm'):
            self.register_buffer("shp_embed_vlm", torch.empty(0), persistent=False)  # [Kp, embed_dim]
        if not hasattr(self, 'shp_embed_dino'):
            self.register_buffer("shp_embed_dino", torch.empty(0), persistent=False)  # [Kp, 1024]

        self.hpi_attn_vlm = HPICrossAttention(
            dim_img=self.embed_dim, dim_txt=self.embed_dim, num_heads=num_heads, qkv_bias=True
        )
        self.hpi_attn_dino = HPICrossAttention(
            dim_img=1024, dim_txt=1024, num_heads=16, qkv_bias=True
        )

        self.use_scc_gate = True
        self.scc_gate_vlm = nn.ModuleList([SCCGate(dim=embed_dim, bottleneck_dim=64) for _ in range(depth)])
        self.scc_gate_dino = nn.ModuleList([SCCGate(dim=1024,     bottleneck_dim=64) for _ in range(depth)])


        self.sac_scc_beta_vlm_logit = nn.Parameter(torch.tensor(0.0))
        self.sac_scc_beta_dino_logit = nn.Parameter(torch.tensor(0.0))



                
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)
                
    def _aggregate_semantic_alpha(self, prompt_patch_attn, K, hw=None, eps=1e-8):
        """
        Semantic-aware w_sac aggregation: within-class fusion, inter-class competition, and strongest-class confidence.
        
        Aligned with the spatial loss computation.
        
        Args:
            prompt_patch_attn: [B, N, Cp] attention weights where Cp = K * Np.
            K: number of classes.
            hw: optional (H, W).
            eps: numerical-stability term.
        
        Returns:
            w_sac: [B, N, 1]
        """
        B_prompt_patch_attn, N_tokens, Cp = prompt_patch_attn.size()
        
        if Cp % K == 0:
            Np = Cp // K
            proto = prompt_patch_attn.view(B_prompt_patch_attn, N_tokens, K, Np)
            
            # Within-class aggregation with LSE
            logp = torch.log(proto.clamp_min(eps))
            logp_cls = torch.logsumexp(logp / self.alpha_tau_cls, dim=-1) * self.alpha_tau_cls
            p_cls = torch.exp(logp_cls)
            
            # Inter-class normalization
            p_sum = p_cls.sum(dim=-1, keepdim=True) + eps
            p_cls = p_cls / p_sum
            
            # Inter-class competition with softmax
            p_cls = torch.softmax(p_cls / self.alpha_tau_spatial, dim=-1)
            
            # Use the strongest-class confidence
            w_sac = (self.alpha_scale * p_cls.max(dim=-1).values).unsqueeze(-1)
            w_sac = w_sac.clamp(0.0, 2.0)
        else:
            # Fallback path
            w_sac = prompt_patch_attn.max(dim=2).values.unsqueeze(-1)
        
        return w_sac

    def get_cast_dtype(self) -> torch.dtype:
        return self.blocks[0].mlp.fc2.weight.dtype

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)
    
    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, return_all_features=False):
        x, _ = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            if self.grad_checkpointing:
                x = checkpoint(blk, x, (rel_pos_bias,))
            else:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        if not return_all_features:
            x = self.norm(x)
            if self.fc_norm is not None:
                return self.fc_norm(x.mean(1))
            else:
                return x[:, 0]
        return x

    def forward(self, x, return_all_features=False):
        if return_all_features:
            return self.forward_features(x, return_all_features)
        x = self.forward_features(x)
        x = self.head(x)
        return x
    def init_weights(self):
        pass

    @staticmethod
    def convert_list_to_tensor(list_convert):
        if len(list_convert):
            result = torch.stack(list_convert, dim=1)
        else :
            result = None
        return result 

    def extract_feats(self, x, use_fpn=True, use_adapter=True, train_loss=False):
        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_x = x * IMG_STD + IMG_MEAN
        
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_x = (original_x - DINOV2_IMG_MEAN) / DINOV2_IMG_STD        
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_x)

        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        # Explicitly determine the DINO grid (Hd, Wd); never fall back to the EVA grid.
        if hasattr(self.dinov2.patch_embed, 'grid_size'):
            _gs = self.dinov2.patch_embed.grid_size
            if isinstance(_gs, (tuple, list)):
                Hd, Wd = int(_gs[0]), int(_gs[1])
            else:
                Hd = Wd = int(_gs)
        else:
            _ps = getattr(self.dinov2.patch_embed, 'patch_size', 14)
            if isinstance(_ps, (tuple, list)):
                _ps_h, _ps_w = int(_ps[0]), int(_ps[1])
            else:
                _ps_h = _ps_w = int(_ps)
            Hd, Wd = H // _ps_h, W // _ps_w
        self._dino_hw = (Hd, Wd)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)



        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        features = []
        vlm_feature_list = []
        vfm_feature_list = []
        
        # Record sim and prompt_patch_attn for spatial and semantic losses
        sims_clip_list, sims_dino_list = [], []
        prompt_patch_attns_clip_list, prompt_patch_attns_dino_list = [], []

        desc_clip = None
        if hasattr(self, 'shp_embed_vlm') and isinstance(self.shp_embed_vlm, torch.Tensor) and self.shp_embed_vlm.numel() > 0:
            desc_clip = self.shp_embed_vlm

        desc_dino = None
        if hasattr(self, 'shp_embed_dino') and isinstance(self.shp_embed_dino, torch.Tensor) and self.shp_embed_dino.numel() > 0:
            desc_dino = self.shp_embed_dino
        
        for i, blk in enumerate(self.blocks):
            # EVA branch: block-external HPI injection on patch tokens only; keep cls unchanged
            if i in self.hpi_layers:
                if not (hasattr(self, 'shp_embed_vlm') and isinstance(self.shp_embed_vlm, torch.Tensor) and self.shp_embed_vlm.numel() > 0):
                    raise RuntimeError("shp_embed_vlm not registe")
                x_cls, x_patch = x[:, :1, :], x[:, 1:, :]  # [B,1,D], [B,N,D]
                delta_p, semantic_logits, prompt_patch_attn = self.hpi_attn_vlm(
                    x_patch, self.shp_embed_vlm.to(x_patch.device, dtype=x_patch.dtype)
                )  # delta_p:[B,N,D], prompt_patch_attn:[B,N,Cp]
                
                w_sac = prompt_patch_attn.max(dim=2).values.unsqueeze(-1)  # [B,N,1]

                if self.use_scc_gate:
                    w_scc = self.scc_gate_vlm[i](torch.cat([x_patch, delta_p], dim=-1), hw=(Hp, Wp))  # [B,N,1]
                    b = torch.sigmoid(self.sac_scc_beta_vlm_logit)
                    hpi_weight = (1.0 - b) * w_sac + b * w_scc
                else:
                    hpi_weight = w_sac

                x = torch.cat([x_cls, x_patch + hpi_weight * delta_p], dim=1)
        
                # Monitoring outputs for spatial and semantic losses
                sims_clip_list.append(semantic_logits)
                prompt_patch_attns_clip_list.append(
                    prompt_patch_attn.transpose(1, 2).reshape(prompt_patch_attn.size(0), prompt_patch_attn.size(-1), Hp, Wp).contiguous()
                )
        
            # DINO branch: block-external HPI injection on patch tokens only; keep cls unchanged
            if i in self.hpi_layers_dino:
                if not (hasattr(self, 'shp_embed_dino') and isinstance(self.shp_embed_dino, torch.Tensor) and self.shp_embed_dino.numel() > 0):
                    raise RuntimeError("shp_embed_dino not registe")
                d_cls, d_patch = dinov2_x[:, :1, :], dinov2_x[:, 1:, :]  # [B,1,1024], [B,Nd,1024]
                delta_p_dino, semantic_logits_dino, prompt_patch_attn_dino = self.hpi_attn_dino(
                    d_patch, self.shp_embed_dino.to(d_patch.device, dtype=d_patch.dtype)
                )  # delta_p_dino:[B,Nd,1024]

                w_sac_dino = prompt_patch_attn_dino.max(dim=2).values.unsqueeze(-1)  # [B,Nd,1]


                if self.use_scc_gate:
                    Hd, Wd = self._dino_hw
                    w_scc_dino = self.scc_gate_dino[i](torch.cat([d_patch, delta_p_dino], dim=-1), hw=(Hd, Wd))  # [B,Nd,1]
                    b_d = torch.sigmoid(self.sac_scc_beta_dino_logit)
                    hpi_weight_dino = (1.0 - b_d) * w_sac_dino + b_d * w_scc_dino
                else:
                    hpi_weight_dino = w_sac_dino

                dinov2_x = torch.cat([d_cls, d_patch + hpi_weight_dino * delta_p_dino], dim=1)

                Hd, Wd = self._dino_hw
                B_prompt_patch_attn, Nd_tokens, Cp = prompt_patch_attn_dino.size()
                assert Nd_tokens == Hd * Wd, f"DINO token-grid mismatch: Nd={Nd_tokens}, Hd*Wd={Hd*Wd}"
                prompt_patch_attns_dino_list.append(
                    prompt_patch_attn_dino.transpose(1, 2).reshape(B_prompt_patch_attn, Cp, Hd, Wd).contiguous()
                )
                sims_dino_list.append(semantic_logits_dino)

            
            x = blk(x, rel_pos_bias)
            
            dinov2_x = self.dinov2.blocks[i](dinov2_x)
            x_delta_p      = self.vlm_adapter[i](x_self=x       , x_borrow =dinov2_x)
            dinov2_delta_p = self.vfm_adapter[i](x_self=dinov2_x, x_borrow =x)

            x        = x      + x_delta_p
            dinov2_x = dinov2_x + dinov2_delta_p

            vlm_feature_list.append(x)
            vfm_feature_list.append(dinov2_x)
            
            
            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous(), x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()], dim=1)
                features.append(xp.contiguous())
        vlm_feature = self.convert_list_to_tensor(vlm_feature_list)[:, :, 1:, :]
        vfm_feature = self.convert_list_to_tensor(vfm_feature_list)[:, :, 1:, :]
        
        if use_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])
            
        x = self.norm(x) + self.dinov2.norm(dinov2_x)
        
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = self.head(x)
        
        global_embedding = x[:, :1]
        visual_embedding = x[:, 1:].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        features.append([global_embedding, visual_embedding])

        if train_loss:
            return tuple(features), dict(vlm_feature=vlm_feature, 
                                         vfm_feature=vfm_feature,)
        else:
            # Used by the segmentor to compute spatial and semantic losses
            return tuple(features), sims_clip_list, sims_dino_list, prompt_patch_attns_clip_list, prompt_patch_attns_dino_list
            


    def get_all_features(self, x, use_fpn=True, use_adapter=True):
        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_x = x * IMG_STD + IMG_MEAN
        
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_x = (original_x - DINOV2_IMG_MEAN) / DINOV2_IMG_STD        
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_x)

        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        if os.getenv('RoPE') == '1':
            if self.training and not isinstance(self.patch_dropout, nn.Identity):
                x, patch_indices_keep = self.patch_dropout(x)
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=patch_indices_keep)
            else:
                self.rope.forward = partial(self.rope.forward, patch_indices_keep=None)
                x = self.patch_dropout(x)
        else:
            x = self.patch_dropout(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        features = []

        analysis_feataure_vlm = dict()
        analysis_feataure_vfm = dict()
        
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias)
            
            dinov2_x = self.dinov2.blocks[i](dinov2_x)

            analysis_feataure_vlm[i] = x.contiguous()
            analysis_feataure_vfm[i] = dinov2_x.contiguous()
            
            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous(), x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()], dim=1)
                features.append(xp.contiguous())
                
        if use_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])
            
        x = self.norm(x) + self.dinov2.norm(dinov2_x)
        
        if self.fc_norm is not None:
            x = self.fc_norm(x)
        x = self.head(x)
        
        global_embedding = x[:, :1]
        visual_embedding = x[:, 1:].permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()

        features.append([global_embedding, visual_embedding])

        return analysis_feataure_vlm, analysis_feataure_vfm
