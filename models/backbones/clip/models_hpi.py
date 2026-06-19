import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from timm.models.layers import drop_path, trunc_normal_

from ..adapter_pruning import *
from ..dino_v2 import DinoVisionTransformer
from ..hpi_cross_attention import HPICrossAttention
from ..hpi_cross_attention_standard import HPIStandardCrossAttention

try:
    from einops import rearrange, repeat
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as MambaRMSNormGated
except Exception:
    rearrange = None
    repeat = None
    selective_scan_fn = None
    MambaRMSNormGated = None


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

class QuickGELU(nn.Module):

    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class ResidualAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, drop_path=0.):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor, H=None, W=None):
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x

class Transformer(nn.Module):

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, drop_path_rate=0.):
        super().__init__()
        self.width = width
        self.layers = layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dpr[i]) for i in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        B, N, C = q.shape
        assert k.shape == v.shape
        B, M, C = k.shape
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads)

        attn = torch.einsum('bnkc,bmkc->bknm', q, k) * self.scale

        attn = attn.softmax(dim=-1)

        x = torch.einsum('bknm,bmkc->bnkc', attn, v).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class HPISequenceEnhancer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.1):
        super().__init__()
        self.self_attn = Attention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = HPISequenceMixer(d_model, d_state=16)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        x_m_x = self.cross_attn(torch.cat([self.norm2(x), visual, self.norm2(x)], dim=1))
        x = x + x_m_x[:, :x.shape[1], :] + x_m_x[:, -x.shape[1]:, :]
        x = x + self.dropout(self.mlp(self.norm3(x)))  
        
        return x


class HPIRMSNormGated(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        x = x * torch.sigmoid(gate)
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class HPISequenceMixer(nn.Module):
    """Local sequence mixer with checkpoint-compatible parameter names.

    When mamba selective-scan ops are available this follows the legacy
    selective-scan path used by released checkpoints. A pure PyTorch fallback
    keeps the public repo importable in environments without mamba, but exact
    old/new tensor parity requires the selective-scan path.
    """

    def __init__(
        self,
        d_model,
        d_state=8,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        proj_drop=0.1,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.use_selective_scan = (
            selective_scan_fn is not None
            and rearrange is not None
            and repeat is not None
            and MambaRMSNormGated is not None
        )
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        if repeat is not None:
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner // 2,
            ).contiguous()
        else:
            A = torch.arange(
                1,
                self.d_state + 1,
                dtype=torch.float32,
                device=device,
            ).repeat(self.d_inner // 2, 1).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner // 2, self.d_model, bias=bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        if self.use_selective_scan:
            self.norm = MambaRMSNormGated(
                self.d_inner // 2,
                eps=1e-5,
                norm_before_gate=True,
                **factory_kwargs,
            )
        else:
            self.norm = HPIRMSNormGated(self.d_inner // 2, eps=1e-5)
        self.scale = nn.Parameter(torch.tensor(0.001))

    def forward(self, hidden_states):
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        if self.use_selective_scan:
            xz = rearrange(xz, "b l d -> b d l")
            x, z = xz.chunk(2, dim=1)
            A = -torch.exp(self.A_log.float())
            x = F.silu(
                F.conv1d(
                    input=x,
                    weight=self.conv1d_x.weight,
                    bias=self.conv1d_x.bias,
                    padding='same',
                    groups=self.d_inner // 2,
                )
            )
            z = F.silu(
                F.conv1d(
                    input=z,
                    weight=self.conv1d_z.weight,
                    bias=self.conv1d_z.bias,
                    padding='same',
                    groups=self.d_inner // 2,
                )
            )
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(
                x_dbl,
                [self.dt_rank, self.d_state, self.d_state],
                dim=-1,
            )
            dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y = rearrange(y, "b d l -> b l d")
            z = rearrange(z, "b d l -> b l d")
            y = self.norm(y, z)
            return self.proj_drop(self.out_proj(y)) * self.scale

        xz = xz.transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        x = F.silu(F.conv1d(x, self.conv1d_x.weight, self.conv1d_x.bias, padding='same', groups=self.d_inner // 2))
        z = F.silu(F.conv1d(z, self.conv1d_z.weight, self.conv1d_z.bias, padding='same', groups=self.d_inner // 2))
        y = x.transpose(1, 2)
        z = z.transpose(1, 2)
        y = self.norm(y, z)
        return self.proj_drop(self.out_proj(y)) * self.scale


@BACKBONES.register_module()
class CLIPTextEncoder(nn.Module):

    def __init__(self, context_length=77,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                    state_dict[k] = checkpoint[k]
             
            self.load_state_dict(state_dict, False)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

@BACKBONES.register_module()
class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=22,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k]
                
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                    state_dict[k] = checkpoint[k]
             
            self.load_state_dict(state_dict, False)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, context=None):
        if context is not None:
            x_text = self.token_embedding(text)  # n_clas, n_text, C
            K, N1, C = x_text.shape
            if len(context.shape) == 3:
                B, N2, C = context.shape

                eos_indx = text.argmax(dim=-1) + N2
                eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

                x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
                context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)
            
            elif len(context.shape) == 4:
                B, K, N2, C = context.shape

                eos_indx = text.argmax(dim=-1) + N2
                eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

                x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
            x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
            x = x.reshape(B, K, self.embed_dim) # 1 19 512
            return x
        
        else:
            x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
            x = x + self.positional_embedding
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = self.ln_final(x)
            x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
            return x

@BACKBONES.register_module()
class ContextDecoder(nn.Module):
    def __init__(self,
                 transformer_width=256,
                 transformer_heads=4,
                 transformer_layers=6,
                 visual_dim=1024,
                 dropout=0.1,
                 **kwargs):
        super().__init__()

        self.memory_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
            nn.LayerNorm(transformer_width),
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, transformer_width),
        )

        self.decoder = nn.ModuleList([
                    HPISequenceEnhancer(transformer_width, transformer_heads, dropout) for _ in range(transformer_layers)
                ])
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(transformer_width),
            nn.Linear(transformer_width, visual_dim)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    
    def forward(self, text, visual):
        B, N, C = visual.shape
        visual = self.memory_proj(visual)
        x = self.text_proj(text)

        for layer in self.decoder:
            x = layer(x, visual)
        
        return self.out_proj(x)


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
class HPIClipVisionTransformer(nn.Module):

    def __init__(self, 
                 input_resolution=224,
                 patch_size=32, 
                 width=768, 
                 layers=12, 
                 heads=12, 
                 output_dim=512, 
                 drop_path_rate=0.0, 
                 out_indices=[3, 5, 7, 11], 
                 pretrained=None, 
                 get_embeddings=False, 
                 ignore_last_attn=False, 
                 
                 adapter_type = None,
                 **kwargs):

        super().__init__()

        self.embed_dim = width
        self.output_dim = output_dim
        self.pretrained = pretrained
        self.patch_size = patch_size
        self.layers = layers
        self.heads = heads
        
        self.adapter_type    = adapter_type
        self.hpi_layers = kwargs.get('hpi_layers', [])
        self.hpi_layers_dino = kwargs.get('hpi_layers_dino', [])
        self.hpi_cross_attn_type = str(kwargs.get('hpi_cross_attn_type', 'hpi')).lower()
        self.use_scc_gate = bool(kwargs.get('use_scc_gate', True))
        sac_scc_beta_init_logit = float(kwargs.get('sac_scc_beta_init_logit', 0.0))
        sac_scc_beta_vlm_init_logit = float(
            kwargs.get('sac_scc_beta_vlm_init_logit', sac_scc_beta_init_logit)
        )
        sac_scc_beta_dino_init_logit = float(
            kwargs.get('sac_scc_beta_dino_init_logit', sac_scc_beta_init_logit)
        )
        sac_scc_beta_fixed = bool(kwargs.get('sac_scc_beta_fixed', False))
        if isinstance(input_resolution, int):
            self.input_resolution = (input_resolution, input_resolution)
        elif isinstance(input_resolution, tuple):
            self.input_resolution = input_resolution       
            
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn((self.input_resolution[0] // patch_size) * (self.input_resolution[1] // patch_size) + 1, width))
        self.spatial_size = (self.input_resolution[0] // patch_size, self.input_resolution[1] // patch_size)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices
        self.ignore_last_attn = ignore_last_attn

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))      

        self.fpn_dim = width + 1024
        self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(self.fpn_dim),
                nn.GELU(),
                nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.fpn_dim, self.fpn_dim, kernel_size=2, stride=2))
        self.fpn3 = nn.Identity()
        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)      
        
        # DINOv2-L
        self.dinov2 = DinoVisionTransformer(patch_size=16,
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
            self.vlm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=True) for i in range(24)]) 
            self.vfm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=False) for i in range(24)])
        elif self.adapter_type == 'vfmbase':
            self.vlm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=False) for i in range(24)]) 
            self.vfm_adapter   = nn.Sequential(*[GateAdapter(borrow_flag=True) for i in range(24)])
        else:
            assert False, f"Not implement"
            
        self.adapter_proj3 = nn.Linear(1024, self.embed_dim)
        
        
        # Text-side prototype library injected by the segmentor
        if not hasattr(self, 'shp_embed_vlm'):
            self.register_buffer("shp_embed_vlm", torch.empty(0), persistent=False)  # [Kp, embed_dim]
        if not hasattr(self, 'shp_embed_dino'):
            self.register_buffer("shp_embed_dino", torch.empty(0), persistent=False)  # [Kp, 1024]
        
        if self.hpi_cross_attn_type in ('hpi', 'router', 'visual_router'):
            hpi_attn_cls = HPICrossAttention
        elif self.hpi_cross_attn_type in ('standard', 'standard_ca', 'cross_attention'):
            hpi_attn_cls = HPIStandardCrossAttention
        else:
            raise ValueError(f"Unknown hpi_cross_attn_type: {self.hpi_cross_attn_type}")
        
        self.hpi_attn_vlm = hpi_attn_cls(
            dim_img=self.embed_dim, dim_txt=self.embed_dim, num_heads=heads, qkv_bias=True
        )
        self.hpi_attn_dino = hpi_attn_cls(
            dim_img=1024, dim_txt=1024, num_heads=16, qkv_bias=True
        )
        
        self.scc_gate_vlm = nn.ModuleList([SCCGate(dim=width, bottleneck_dim=64) for _ in range(layers)])
        self.scc_gate_dino = nn.ModuleList([SCCGate(dim=1024, bottleneck_dim=64) for _ in range(24)])

        if sac_scc_beta_fixed:
            self.register_buffer(
                "sac_scc_beta_vlm_logit",
                torch.tensor(sac_scc_beta_vlm_init_logit, dtype=torch.float32),
            )
            self.register_buffer(
                "sac_scc_beta_dino_logit",
                torch.tensor(sac_scc_beta_dino_init_logit, dtype=torch.float32),
            )
        else:
            self.sac_scc_beta_vlm_logit = nn.Parameter(
                torch.tensor(sac_scc_beta_vlm_init_logit, dtype=torch.float32)
            )
            self.sac_scc_beta_dino_logit = nn.Parameter(
                torch.tensor(sac_scc_beta_dino_init_logit, dtype=torch.float32)
            )
        
    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    orig_size = int(state_dict["positional_embedding"][1:,].shape[0] ** 0.5)
                    spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, orig_size, orig_size, self.embed_dim).permute(0, 3, 1, 2), size=self.spatial_size, mode='bilinear')
                    spatial_pos = spatial_pos.reshape(self.embed_dim, self.spatial_size[0]*self.spatial_size[1]).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            if self.conv1.weight.shape != state_dict['conv1.weight'].shape:
                state_dict["conv1.weight"] = F.interpolate(state_dict["conv1.weight"], size=self.conv1.weight.shape[-2:], mode='bilinear')
                assert self.conv1.weight.shape == state_dict['conv1.weight'].shape
                
            self.load_state_dict(state_dict, False)
    

    def prepare_tokens_with_masks(self, x: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size[0], self.spatial_size[1], C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        return x

    def get_all_features(self, x: torch.Tensor, use_adapter=True, verbose=False):
        
        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_x = x * IMG_STD + IMG_MEAN
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_x = (original_x - DINOV2_IMG_MEAN) / DINOV2_IMG_STD        
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_x)
        
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        
        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size[0], self.spatial_size[1], C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        features = []
        analysis_feataure_vlm = dict()
        analysis_feataure_vfm = dict()
        
        for i, blk in enumerate(self.transformer.resblocks):
            if self.ignore_last_attn:
                mask = torch.empty(x.shape[0], x.shape[0])
                mask.fill_(float('-inf'))
                mask.fill_diagonal_(0)
                self.transformer.resblocks[-1].attn_mask = mask

            x = blk(x)
            dinov2_x = self.dinov2.blocks[i](dinov2_x)

            if use_adapter:
                x_delta_p       = self.vlm_adapter[i](x_self=x.permute(1,0,2), x_borrow=dinov2_x)
                dinov2_delta_p  = self.vfm_adapter[i](x_self=dinov2_x,         x_borrow=x.permute(1,0,2))
                
                
                x        = x        + x_delta_p.permute(1,0,2)
                dinov2_x = dinov2_x + dinov2_delta_p

            analysis_feataure_vlm[i] = x.contiguous()
            analysis_feataure_vfm[i] = dinov2_x.contiguous()
                
            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous(), x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()], dim=1)
                features.append(xp.contiguous())
        
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.get_embeddings:
            x = x.permute(1, 0, 2) + self.adapter_proj3(dinov2_x)
            x = self.ln_post(x)
            x = x @ self.proj
            
            global_embedding = x[:, :1]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)

            features.append([global_embedding, visual_embedding])
        
        return analysis_feataure_vlm, analysis_feataure_vfm
    
    @staticmethod
    def convert_list_to_tensor(list_convert):
        if len(list_convert):
            result = torch.stack(list_convert, dim=1)
        else :
            result = None
        return result 
    
    def forward(self, x: torch.Tensor, use_adapter=True, train_loss=False):
        # Various Foundation Models use different normalization, convert inputs correspondingly
        IMG_MEAN = torch.tensor([ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]).view(1, 3, 1, 1).cuda()
        IMG_STD = torch.tensor([ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]).view(1, 3, 1, 1).cuda()
        original_x = x * IMG_STD + IMG_MEAN
        DINOV2_IMG_MEAN = torch.tensor([v * 255 for v in [0.485, 0.456, 0.406]]).view(1, 3, 1, 1).cuda()
        DINOV2_IMG_STD = torch.tensor([v * 255 for v in [0.229, 0.224, 0.225]]).view(1, 3, 1, 1).cuda()
        normalized_x = (original_x - DINOV2_IMG_MEAN) / DINOV2_IMG_STD        
        dinov2_x = self.dinov2.prepare_tokens_with_masks(normalized_x)
        
        x = self.conv1(x)
        B, C, H, W = x.shape
        clip_H, clip_W = H, W
        
        # Determine the DINO grid size, matching the EVA path
        if hasattr(self.dinov2.patch_embed, 'grid_size'):
            _gs = self.dinov2.patch_embed.grid_size
            if isinstance(_gs, (tuple, list)):
                Hd, Wd = int(_gs[0]), int(_gs[1])
            else:
                Hd = Wd = int(_gs)
        else:
            _ps = getattr(self.dinov2.patch_embed, 'patch_size', 16)
            if isinstance(_ps, (tuple, list)):
                _ps_h, _ps_w = int(_ps[0]), int(_ps[1])
            else:
                _ps_h = _ps_w = int(_ps)
            inH, inW = normalized_x.shape[-2], normalized_x.shape[-1]
            Hd, Wd = inH // _ps_h, inW // _ps_w
        self._dino_hw = (Hd, Wd)
        
        x = x.reshape(x.shape[0], x.shape[1], -1) 
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size[0], self.spatial_size[1], C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)

        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

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
        
        for i, blk in enumerate(self.transformer.resblocks):
            # CLIP branch: block-external HPI injection on patch tokens only; keep cls unchanged
            if i in self.hpi_layers:
                if not (hasattr(self, 'shp_embed_vlm') and isinstance(self.shp_embed_vlm, torch.Tensor) and self.shp_embed_vlm.numel() > 0):
                    raise RuntimeError("shp_embed_vlm not registered")
                x_cls, x_patch = x[:1, :, :], x[1:, :, :]  # CLIP uses [T,B,C] format
                x_patch = x_patch.permute(1, 0, 2)  # [B,N,C]
                
                delta_p, semantic_logits, prompt_patch_attn = self.hpi_attn_vlm(
                    x_patch, self.shp_embed_vlm.to(x_patch.device, dtype=x_patch.dtype)
                )  # delta_p:[B,N,D], prompt_patch_attn:[B,N,Cp]
                
                w_sac = prompt_patch_attn.max(dim=2).values.unsqueeze(-1)  # [B,N,1]
                
                if self.use_scc_gate:
                    w_scc = self.scc_gate_vlm[i](torch.cat([x_patch, delta_p], dim=-1), hw=(clip_H, clip_W))
                    b = torch.sigmoid(self.sac_scc_beta_vlm_logit)
                    hpi_weight = (1.0 - b) * w_sac + b * w_scc
                else:
                    hpi_weight = w_sac
                
                x_patch = x_patch + hpi_weight * delta_p
                x_patch = x_patch.permute(1, 0, 2)
                x = torch.cat([x_cls, x_patch], dim=0)
                
                # Monitoring outputs for spatial and semantic losses
                sims_clip_list.append(semantic_logits)
                prompt_patch_attns_clip_list.append(
                    prompt_patch_attn.transpose(1, 2).reshape(prompt_patch_attn.size(0), prompt_patch_attn.size(-1), clip_H, clip_W).contiguous()
                )
            
            # DINO branch: block-external HPI injection on patch tokens only; keep cls unchanged
            if i in self.hpi_layers_dino:
                if not (hasattr(self, 'shp_embed_dino') and isinstance(self.shp_embed_dino, torch.Tensor) and self.shp_embed_dino.numel() > 0):
                    raise RuntimeError("shp_embed_dino not registered")
                d_cls, d_patch = dinov2_x[:, :1, :], dinov2_x[:, 1:, :]  # [B,1,1024], [B,Nd,1024]
                delta_p_dino, semantic_logits_dino, prompt_patch_attn_dino = self.hpi_attn_dino(
                    d_patch, self.shp_embed_dino.to(d_patch.device, dtype=d_patch.dtype)
                )  # delta_p_dino:[B,Nd,1024]
                
                w_sac_dino = prompt_patch_attn_dino.max(dim=2).values.unsqueeze(-1)  # [B,Nd,1]
                
                if self.use_scc_gate:
                    Hd, Wd = self._dino_hw
                    w_scc_dino = self.scc_gate_dino[i](torch.cat([d_patch, delta_p_dino], dim=-1), hw=(Hd, Wd))
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

            if self.ignore_last_attn:
                mask = torch.empty(x.shape[0], x.shape[0])
                mask.fill_(float('-inf'))
                mask.fill_diagonal_(0)
                self.transformer.resblocks[-1].attn_mask = mask

            x = blk(x)
            dinov2_x = self.dinov2.blocks[i](dinov2_x)

            x_delta_p       = self.vlm_adapter[i](x_self=x.permute(1,0,2), x_borrow=dinov2_x)
            dinov2_delta_p  = self.vfm_adapter[i](x_self=dinov2_x,         x_borrow=x.permute(1,0,2))
            
            
            x        = x        + x_delta_p.permute(1,0,2)
            dinov2_x = dinov2_x + dinov2_delta_p

            vlm_feature_list.append(x.permute(1,0,2))
            vfm_feature_list.append(dinov2_x)
            
            if i in self.out_indices:
                xp = torch.cat([dinov2_x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous(), x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W).contiguous()], dim=1)
                features.append(xp.contiguous())
        
        vlm_feature = self.convert_list_to_tensor(vlm_feature_list)[:, :, 1:, :]
        vfm_feature = self.convert_list_to_tensor(vfm_feature_list)[:, :, 1:, :]
        
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(features)):
            features[i] = ops[i](features[i])

        if self.get_embeddings:
            x = x.permute(1, 0, 2) + self.adapter_proj3(dinov2_x)
            x = self.ln_post(x)
            x = x @ self.proj
            
            global_embedding = x[:, :1]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)

            features.append([global_embedding, visual_embedding])
        
        if train_loss:
            return tuple(features), dict(vlm_feature=vlm_feature, 
                                         vfm_feature=vfm_feature,)
        else:

            return tuple(features), sims_clip_list, sims_dino_list, prompt_patch_attns_clip_list, prompt_patch_attns_dino_list
