import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
# from einops import rearrange
import einops

from .pruning.adapter import DeltaBlock, Compensator
from .pruning.mask import *


class WrapATT(nn.Module): 
    '''
    in:  B, T, C
    out: B, T, C
    '''
    
    def __init__(self, model_name):
        super().__init__()      
        self.model_name = model_name
        
    def forward(self, x, blk):
        if self.model_name == 'dinov2':
            return blk.ls1(blk.attn(blk.norm1(x)))
        
        elif self.model_name == 'clip':
            return blk.drop_path(blk.attention(blk.ln_1(x)))
        
        else:
            assert False, "Unkonw name"
        
class WrapFFN(nn.Module):
    '''
    in:  B, T, C
    out: B, T, C
    '''
    
    def __init__(self, model_name):
        super().__init__()      
        self.model_name = model_name
        
    def forward(self, x, blk):
        if self.model_name == 'dinov2':
            return blk.ls2(blk.mlp(blk.norm2(x)))
        
        elif self.model_name == 'clip':
            return blk.drop_path(blk.mlp(blk.ln_2(x)))   
        
        else:
            assert False, "Unkonw name"


class TRSpatial(nn.Module):
    def __init__(self, model_name, dim=1024, bottleneck_dim=64, scale=0.1):
        super().__init__()
        self.spatial_adapter = Compensator(dim=dim, bottleneck_dim=bottleneck_dim)
        self.linear_adapter = DeltaBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        self.mask_gen = TokenSelect(dim=dim*2, mask_dim=1)
        self.scale = nn.Parameter(scale * torch.ones(1))
        self.dim = dim 
        
        self.model_name = model_name

    def forward(self, x, x_other):
        #att layer
        policy_token = x

        #generate token mask
        token_mask, token_logits = self.mask_gen(torch.cat([policy_token, x_other], dim=-1))

        #linear adapter and spatial adapter
        adpt_x = self.linear_adapter(x)
        spt_x  = self.spatial_adapter(x)

        #adding
        adpt_x[:, 1:, :] = adpt_x[:, 1:, :] + spt_x
        adpt_x = adpt_x * self.scale
        
        
        if self.model_name == 'clip':       
            return adpt_x.permute(1,0,2), token_mask.permute(1,0,2), dict(sub_token_select=token_mask, token_logits=token_logits)
        
        else:       
            return adpt_x, token_mask, dict(sub_token_select=token_mask, token_logits=token_logits)

    
class TRLinar(nn.Module):
    def __init__(self, model_name, layer, dim=1024, bottleneck_dim=64, scale=0.1):
        super().__init__()
        self.light_layer = DeltaBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        self.mask_gen = TokenSelectLinear(dim=dim*2, bottleneck_dim=bottleneck_dim*2, mask_dim=1, layer=layer)
        self.scale = nn.Parameter(scale * torch.ones(1))
        self.dim = dim 
        
        self.model_name = model_name

    def forward(self, x, x_other):
        #att layer
        policy_token = x

        #generate token mask
        token_mask, token_logits = self.mask_gen(torch.cat([policy_token, x_other], dim=-1))

        #linear adapter and spatial adapter
        adpt_x = self.light_layer(x)

        #adding
        adpt_x = adpt_x * self.scale
        
        if self.model_name == 'clip':       
            return adpt_x.permute(1,0,2), token_mask.permute(1,0,2), dict(sub_token_select=token_mask, token_logits=token_logits)
        
        else:       
            return adpt_x, token_mask, dict(sub_token_select=token_mask, token_logits=token_logits)

class TRLinarSoft(nn.Module):
    def __init__(self, model_name, layer, dim=1024, bottleneck_dim=64, scale=0.1):
        super().__init__()
        self.light_layer = DeltaBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        self.mask_gen = TokenSelectLinearSoft(dim=dim*2, bottleneck_dim=bottleneck_dim*2, mask_dim=1, layer=layer)
        self.scale = nn.Parameter(scale * torch.ones(1))
        self.dim = dim 
        
        self.model_name = model_name

    def forward(self, x, x_other):
        #att layer
        policy_token = x

        #generate token mask
        token_mask, token_logits = self.mask_gen(torch.cat([policy_token, x_other], dim=-1))

        #linear adapter and spatial adapter
        adpt_x = self.light_layer(x)

        #adding
        adpt_x = adpt_x * self.scale
        
        if self.model_name == 'clip':       
            return adpt_x.permute(1,0,2), token_mask.permute(1,0,2), dict(sub_token_select=token_mask, token_logits=token_logits)
        
        else:       
            return adpt_x, token_mask, dict(sub_token_select=token_mask, token_logits=token_logits)


    
class GateAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, borrow_flag=False):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve
        '''
        super().__init__()
        # hr module
        self.adapter = DeltaBlock(dim, bottleneck_dim)
        self.gate_layer = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
            nn.Sigmoid(),
        )
        self.scale = scale
        self.borrow_flag = borrow_flag
            

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.gate_layer[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.gate_layer[0].bias)
            
            #zero init
            nn.init.zeros_(self.gate_layer[2].weight)
            nn.init.zeros_(self.gate_layer[2].bias)

    def forward(self, x_self, x_borrow):
        # Transform VFM via LoRA-style adapter
        if self.borrow_flag is True:
            #borrowing
            delta = self.adapter(x_borrow)
        else:
            #self evolving
            delta = self.adapter(x_self)
            
        #learning weight
        x_cat = torch.cat([x_self, x_borrow], dim=-1)  # [B, T, 2C]
        w = self.gate_layer(x_cat)      

        return self.scale * w * delta

class RedAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, VLM='clip', scale=0.1):
        super().__init__()
        self.att_layer_vlm = WrapATT(VLM) 
        self.att_layer_vfm = WrapATT('dinov2')
        
        self.ffn_layer_vlm = WrapFFN(VLM) 
        self.ffn_layer_vfm = WrapFFN('dinov2')
        
        self.proj_vfm2vlm = DeltaBlock(dim, bottleneck_dim)
        self.scale = nn.Parameter(scale * torch.ones(1))
        self.VLM = VLM
        
    def forward(self, x_vlm, x_vfm, blk_vlm, blk_vfm):
        
        #forward att layer
        x_vlm = x_vlm + self.att_layer_vlm(x_vlm, blk_vlm)
        x_vfm = x_vfm + self.att_layer_vfm(x_vfm, blk_vfm)
        
        #applying on the ffn and forward
        # mlp_vlm = self.ffn_layer_vlm(x_vlm, blk_vlm)
        mlp_vfm = self.ffn_layer_vfm(x_vfm, blk_vfm)
        mlp_vlm = self.proj_vfm2vlm(mlp_vfm) * self.scale
        if self.VLM == 'clip':
            mlp_vlm = mlp_vlm.permute(1,0,2)
        
        #apply mask and adapted output
        x_vlm = x_vlm + mlp_vlm 
        x_vfm = x_vfm + mlp_vfm 
        
        return x_vlm, x_vfm#, mlp_vlm, mlp_vfm

class NoiAdapter(nn.Module):
    def __init__(self, layer=2, dim=1024, bottleneck_dim=64, VLM='clip', scale_TR=0.1):
        super().__init__()
        self.att_layer_vlm = WrapATT(VLM) 
        self.att_layer_vfm = WrapATT('dinov2')
        
        self.ffn_layer_vlm = WrapFFN(VLM) 
        self.ffn_layer_vfm = WrapFFN('dinov2')

        #TR  token reduction
        self.TR_VLM = TRLinar(model_name=VLM,      layer=layer, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR)
        self.TR_VFM = TRLinar(model_name='dinov2', layer=layer, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR) 
        
    def forward(self, x_vlm, x_vfm, blk_vlm, blk_vfm):
        
        #forward att layer
        x_vlm = x_vlm + self.att_layer_vlm(x_vlm, blk_vlm)
        x_vfm = x_vfm + self.att_layer_vfm(x_vfm, blk_vfm)
        
        #generate mask, and adapted output
        apt_vlm, mask_vlm, dict_vlm = self.TR_VLM(x_vlm.permute(1,0,2), x_other=x_vfm)
        apt_vfm, mask_vfm, dict_vfm = self.TR_VFM(x_vfm, x_other=x_vlm.permute(1,0,2))
        
        #applying on the ffn and forward
        mlp_vlm = self.ffn_layer_vlm(x_vlm, blk_vlm)
        mlp_vfm = self.ffn_layer_vfm(x_vfm, blk_vfm)
        
        #apply mask and adapted output
        x_vlm = x_vlm + mlp_vlm * mask_vlm + apt_vlm * (1-mask_vlm)
        x_vfm = x_vfm + mlp_vfm * mask_vfm + apt_vfm * (1-mask_vfm)
        
        return x_vlm, x_vfm, dict_vlm, dict_vfm

class PrunAdapter(nn.Module):
    def __init__(self, layer=2, dim=1024, bottleneck_dim=64, VLM='clip', scale_TR=0.1, TR_type='Linear'):
        super().__init__()
        self.att_layer_vlm = WrapATT(VLM) 
        self.att_layer_vfm = WrapATT('dinov2')
        
        self.ffn_layer_vlm = WrapFFN(VLM) 
        self.ffn_layer_vfm = WrapFFN('dinov2')

        #TR  token reduction
        if TR_type == 'Linear':
            self.TR_VLM = TRLinar(model_name=VLM,      layer=layer, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR)
            self.TR_VFM = TRLinar(model_name='dinov2', layer=layer, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR) 
        elif TR_type == 'Spatial':
            self.TR_VLM = TRSpatial(model_name=VLM,      layer=layer, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR)
            self.TR_VFM = TRSpatial(model_name='dinov2', layer=layer, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR)   
        elif TR_type == 'LinearSoft':
            self.TR_VLM = TRLinarSoft(model_name=VLM,      layer=layer, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR)
            self.TR_VFM = TRLinarSoft(model_name='dinov2', layer=layer, dim=dim, bottleneck_dim=bottleneck_dim, scale=scale_TR)   
            
        
    def forward(self, x_vlm, x_vfm, blk_vlm, blk_vfm):
        
        #forward att layer
        x_vlm = x_vlm + self.att_layer_vlm(x_vlm, blk_vlm)
        x_vfm = x_vfm + self.att_layer_vfm(x_vfm, blk_vfm)
        
        #generate mask, and adapted output
        apt_vlm, mask_vlm, dict_vlm = self.TR_VLM(x_vlm.permute(1,0,2), x_other=x_vfm)
        apt_vfm, mask_vfm, dict_vfm = self.TR_VFM(x_vfm, x_other=x_vlm.permute(1,0,2))
        
        #applying on the ffn and forward
        mlp_vlm = self.ffn_layer_vlm(x_vlm, blk_vlm)
        mlp_vfm = self.ffn_layer_vfm(x_vfm, blk_vfm)
        
        #apply mask and adapted output
        x_vlm = x_vlm + mlp_vlm * mask_vlm + apt_vlm * (1-mask_vlm)
        x_vfm = x_vfm + mlp_vfm * mask_vfm + apt_vfm * (1-mask_vfm)
        
        return x_vlm, x_vfm, dict_vlm, dict_vfm


class DynGateAdapter(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, scale=1.0, 
                 warmup_iters=2500, policy="mean_std", log_stats=True):
        '''
        borrow_flag: 
            True: borrow from another model
            False: self evolve

        policy: "entropy", "mean_std"
        '''
        super().__init__()
        # hr module
        self.adapter_self   = DeltaBlock(dim, bottleneck_dim)
        self.adapter_borrow = DeltaBlock(dim, bottleneck_dim)
        
        self.gate_layer_self = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
            nn.Sigmoid(),
        )

        self.gate_layer_borrow = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1),
            nn.Sigmoid(),
        )

        self.scale = scale

        # Initialization
        with torch.no_grad():
            for gate in [self.gate_layer_self, self.gate_layer_borrow]:
                nn.init.kaiming_uniform_(gate[0].weight, a=math.sqrt(5))
                nn.init.zeros_(gate[0].bias)
                nn.init.zeros_(gate[2].weight)
                nn.init.zeros_(gate[2].bias)

        # --- stats buffers (persist in state_dict, move with .to(device)) ---self.log_stats = True                 
        # turn off after freeze
        
        # --- accumulators (float32) ---
        self.register_buffer("w_sum_self",   torch.zeros((), dtype=torch.float32))
        self.register_buffer("w_sq_self",    torch.zeros((), dtype=torch.float32))
        self.register_buffer("H_sum_self",   torch.zeros((), dtype=torch.float32))

        self.register_buffer("w_sum_borrow", torch.zeros((), dtype=torch.float32))
        self.register_buffer("w_sq_borrow",  torch.zeros((), dtype=torch.float32))
        self.register_buffer("H_sum_borrow", torch.zeros((), dtype=torch.float32))

        # --- counters (int64) ---
        self.register_buffer("count",     torch.zeros((), dtype=torch.long))
        self.register_buffer("iter_ctr",  torch.zeros((), dtype=torch.long))

        # --- config/state as buffers (so they are saved with state_dict) ---
        self.log_stats = log_stats
        self.warmup_iters = warmup_iters
        self.policy = policy

        self.register_buffer("act_self", torch.tensor(True, dtype=torch.bool))
        self.register_buffer("act_borrow", torch.tensor(True, dtype=torch.bool))

    @torch.no_grad()
    def binary_entropy(self, w, eps=1e-8):
        w = w.clamp(eps, 1 - eps)
        return -(w * w.log() + (1 - w) * (1 - w).log())
        
    @torch.no_grad()
    def _update_stats(self, w_self, w_borrow):
        # w_*: [B, T, 1] or [B, T]
        ws = w_self.reshape(-1).detach()
        wb = w_borrow.reshape(-1).detach()
        
        Hs = self.binary_entropy(ws)
        Hb = self.binary_entropy(wb)

        self.w_sum_self   += ws.sum()
        self.w_sq_self    += (ws * ws).sum()
        self.H_sum_self   += Hs.sum()

        self.w_sum_borrow += wb.sum()
        self.w_sq_borrow  += (wb * wb).sum()
        self.H_sum_borrow += Hb.sum()

        self.count += ws.numel()


    @torch.no_grad()
    def _means_stds(self):
        # return dict with mean w, std w, mean entropy for both branches
        def pack(sum_w, sum_sq, sum_H, cnt):
            if cnt.item() == 0:
                return dict(mean_w=0.0, std_w=0.0, mean_H=0.0, n=0)
            mean_w = (sum_w / cnt).item()
            var_w  = (sum_sq / cnt - (sum_w / cnt) ** 2).clamp(min=0).item()
            mean_H = (sum_H / cnt).item()
            return dict(mean_w=mean_w, std_w=var_w ** 0.5, mean_H=mean_H, n=int(cnt.item()))
        stats_s = pack(self.w_sum_self,   self.w_sq_self,   self.H_sum_self,   self.count)
        stats_b = pack(self.w_sum_borrow, self.w_sq_borrow, self.H_sum_borrow, self.count)
        return stats_s, stats_b

    @torch.no_grad()
    def reset_stats(self):
        self.w_sum_self.zero_();   self.w_sq_self.zero_();   self.H_sum_self.zero_();   
        self.w_sum_borrow.zero_(); self.w_sq_borrow.zero_(); self.H_sum_borrow.zero_(); 
        self.count.zero_(); self.iter_ctr.zero_()

    
    @torch.no_grad()
    def finalize(self, margin=0.05, beta=0.0):
        """
        Decide one winner per layer and hard-wire branch_mode.
        policy="entropy": choose lower mean entropy; tie-break by mean_w, then std_w.
        policy="mean_std": choose higher mean_w if > margin; else lower std_w; then lower mean_H.
        beta>0 enables Score = mean_w - beta * mean_H (optional).
        """
        stats_s, stats_b = self._means_stds()

        def choose_by_entropy(sb, bb):
            print("#####################")
            print("Drop use entropy")
            
            # primary: mean entropy
            if bb["mean_H"] < sb["mean_H"]:
                self.act_self.fill_(False)
                self.act_borrow.fill_(True)
            else:
                self.act_self.fill_(True)
                self.act_borrow.fill_(False)

        # def choose_by_mean_then_std(sb, bb):
        #     if (bb["mean_w"] - sb["mean_w"]) > margin: return "borrow"
        #     if (sb["mean_w"] - bb["mean_w"]) > margin: return "self"
        #     # close ¡ú stability
        #     if bb["std_w"] < sb["std_w"]:
                # self.act_self.fill_(False)
                # self.act_borrow.fill_(True)
        #     else:
                # self.act_self.fill_(True)
                # self.act_borrow.fill_(False)

        def choose_by_mean_then_std(sb, bb, lam=1.0):
            print("#####################")
            print("Drop use mean")
            
            s_self = sb["mean_w"] - lam * sb["std_w"]
            s_borr = bb["mean_w"] - lam * bb["std_w"]
            if s_borr > s_self:
                self.act_self.fill_(False)
                self.act_borrow.fill_(True)
            else:
                self.act_self.fill_(True)
                self.act_borrow.fill_(False)

        
        if self.policy == "entropy":
            self.branch_mode = choose_by_entropy(stats_s, stats_b)
        else:
            self.branch_mode = choose_by_mean_then_std(stats_s, stats_b)

        # stop logging once we¡¯ve chosen
        self.log_stats = False
    
    def forward(self, x_self, x_borrow):
        #learning weight
        x_cat = torch.cat([x_self, x_borrow], dim=-1)  # [B, T, 2C]

        # mixing per current mode
        if self.act_self.item():
            delta_self   = self.adapter_self(x_self)
            w_self   = self.gate_layer_self(x_cat)
        else:
            delta_self = 0
            w_self = 0

        if self.act_borrow.item():
            delta_borrow = self.adapter_borrow(x_borrow)
            w_borrow = self.gate_layer_borrow(x_cat)
        else:
            delta_borrow = 0
            w_borrow = 0
            
 
        # warm-up logging
        if self.training and self.log_stats:
            self._update_stats(w_self, w_borrow)
            self.iter_ctr += 1
            
            # optional: auto-finalize right here
            if self.iter_ctr.item() >= self.warmup_iters and self.act_self.item() and self.act_borrow.item():
                self.finalize()  # uses accumulated stats

        return self.scale * (w_self * delta_self + w_borrow * delta_borrow)