import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def add(x, scale, shift):
    assert scale.shape == shift.shape
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('the input tensor shape does not match the shape of the scale factor.')

class VLMAdapterLightweight(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=1):
        super().__init__()
        
        # Linear projection for gating input
        self.proj = nn.Linear(dim, dim)
        
        # LoRA-style MLP adapter (non-zero init to allow learning)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x
        self.up = nn.Linear(bottleneck_dim, dim)

        # Gating MLP from [VFM_proj, VLM]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)
        )

        self.scale = scale

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)
            
            #zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x_vfm, x_vlm, use_sigmoid=True):
        # Transform VFM via LoRA-style adapter
        delta_p = self.down(x_vfm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]
        if use_sigmoid:
            w = torch.sigmoid(w)

        # Residual injection into VLM
        return x_vlm + self.scale * w * delta_p

    def getweights(self, x_vfm, x_vlm):
        # Transform VFM via LoRA-style adapter
        delta_p = self.down(x_vfm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # Residual injection into VLM
        return x_vlm + self.scale * w * delta_p, w, self.scale * w * delta_p

class VFMAdapterLightweight(nn.Module):
    # def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=0.1):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=1):
        super().__init__()
        
        # Projection for cross-model gating (VLM feature)
        self.proj = nn.Linear(dim, dim)

        # LoRA-style MLP adapter for VFM
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.up = nn.Linear(bottleneck_dim, dim)

        # Token-wise gating from [x_vfm, projected x_vlm]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)  # token-wise scalar gate
        )

        self.scale = scale  # global residual scale (optional)

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)

            # zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x_vfm, x_vlm, use_sigmoid=True):
        # Adapt VFM features (delta_p)
        delta_p = self.down(x_vfm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gating weight using [x_vfm, projected x_vlm]
        x_cat = torch.cat([x_vfm, self.proj(x_vlm)], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                    # [B, T, 1]
        if use_sigmoid:
            w = torch.sigmoid(w)

        # Inject back into VFM (residual)
        return x_vfm + self.scale * w * delta_p

    def getweights(self, x_vfm, x_vlm):
        # Adapt VFM features (delta_p)
        delta_p = self.down(x_vfm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gating weight using [x_vfm, projected x_vlm]
        x_cat = torch.cat([x_vfm, self.proj(x_vlm)], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                    # [B, T, 1]

        # Inject back into VFM (residual)
        return x_vfm + self.scale * w * delta_p, w, self.scale * w * delta_p
        

class VFMAdapterLightweightBorrow(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=0.1):
        super().__init__()
        
        # Projection for cross-model gating (VLM feature)
        self.proj = nn.Linear(dim, dim)

        # LoRA-style MLP adapter for VFM
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else nn.Identity()
        self.up = nn.Linear(bottleneck_dim, dim)

        # Token-wise gating from [x_vfm, projected x_vlm]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)  # token-wise scalar gate
        )

        self.scale = scale  # global residual scale (optional)

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)

            # zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

    def forward(self, x_vfm, x_vlm):
        # Adapt VFM features (delta_p)
        delta_p = self.down(x_vlm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gating weight using [x_vfm, projected x_vlm]
        x_cat = torch.cat([x_vfm, self.proj(x_vlm)], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                    # [B, T, 1]

        # Inject back into VFM (residual)
        return x_vfm + self.scale * w * delta_p

    def getweights(self, x_vfm, x_vlm):
        # Adapt VFM features (delta_p)
        delta_p = self.down(x_vlm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gating weight using [x_vfm, projected x_vlm]
        x_cat = torch.cat([x_vfm, self.proj(x_vlm)], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                    # [B, T, 1]

        # Inject back into VFM (residual)
        return x_vfm + self.scale * w * delta_p, w

class DistributionUncertainty(nn.Module):
    """
    Distribution Uncertainty Module
        Args:
        p   (float): probabilty of foward distribution uncertainty module, p in [0,1].

    """

    def __init__(self, p=0.5, eps=1e-6):
        super(DistributionUncertainty, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        # print('1', t.shape)
        t = t.repeat(x.shape[0], 1)
        # print('2', t.shape)
        return t

    def forward(self, x):
        if (not self.training) or (np.random.random()) > self.p:
            return x
        B, T, C = x.shape

        #B, 1, C
        mean = x.mean(dim=[1], keepdim=False)
        std = (x.var(dim=[1], keepdim=False) + self.eps).sqrt()

        sqrtvar_mu = self.sqrtvar(mean)
        sqrtvar_std = self.sqrtvar(std)

        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        x = (x - mean.reshape(B, 1, C).detach()) / std.reshape(B, 1, C).detach()
        x = x * gamma.reshape(B, 1, C) + beta.reshape(B, 1, C)

        return x


class SelfGramLayer(nn.Module):
    def __init__(self, dim=1024, bottleneck=32):
        super().__init__()
        self.out_proj_vlm = nn.Conv2d(2, 1, kernel_size=32, stride=32, groups=1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        
    def forward(self, x1, x2):
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        
        B, T, C = x1.shape
        out1 = x1.permute(0, 2, 1) @ x1 / T
        out2 = x2.permute(0, 2, 1) @ x2 / T

        out = self.out_proj_vlm(torch.cat([out1.unsqueeze(1), out2.unsqueeze(1)], dim=1))
        out = out.flatten(1)   # [B, 1024]
        out = out.unsqueeze(1).expand(-1, T, -1)
        return out

class CrossGramLayer(nn.Module):
    def __init__(self, dim=1024, bottleneck=32):
        super().__init__()
        self.out_proj_vlm = nn.Conv2d(1, 1, kernel_size=32, stride=32, groups=1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        
    def forward(self, x1, x2):
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        
        B, T, C = x1.shape
        
        out = x1.permute(0, 2, 1) @ x2 / T
        out = self.out_proj_vlm(out.unsqueeze(1))
        out = out.flatten(1)   # [B, 1024]
        out = out.unsqueeze(1).expand(-1, T, -1)
        return out


class AttentionGramLayer(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64):
        super().__init__()
        #self.out_proj_vlm = nn.Conv2d(1, 1, kernel_size=32, stride=32, groups=1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        
    def forward(self, x1, x2):
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        
        B, T, C = x1.shape
        
        gram_vfm = x1.permute(0, 2, 1) @ x1 / T 
        gram_vlm = x2.permute(0, 2, 1) @ x2 / T
        x_cat = torch.cat([gram_vfm, gram_vlm], dim=-1)
        
        return x_cat        


class VLMAdapterDomain(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=1):
        super().__init__()
        
        # Linear projection for gating input
        self.proj = nn.Linear(dim, dim)
        
        # LoRA-style MLP adapter (non-zero init to allow learning)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x
        self.up = nn.Linear(bottleneck_dim, dim)
        self.gram = AttentionGramLayer()

        # Gating MLP from [VFM_proj, VLM]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)
        )

        self.weight_domain = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)
        )     
    

        self.scale = scale
        
        # self.scale_init= 0.001
        # self.scale_domain = nn.Parameter(torch.tensor(self.scale_init))

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)
            
            #zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

            nn.init.zeros_(self.weight_domain[-1].weight)
            nn.init.zeros_(self.weight_domain[-1].bias)

    def forward(self, x_vfm, x_vlm):
        # Transform VFM via LoRA-style adapter
        delta_p = self.down(x_vfm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gate
        x_vfm_proj = self.proj(x_vfm)
        x_cat = torch.cat([x_vfm_proj, x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # Compute domain-aware gate
        gram = self.gram(x_vfm, x_vlm)
        w_gram = self.weight(gram).permute(0, 2, 1)
        
        # Residual injection into VLM
        return x_vlm + self.scale * w * delta_p + self.scale * w_gram * delta_p 

    def getweights(self, x_vfm, x_vlm):
        # Transform VFM via LoRA-style adapter
        delta_p = self.down(x_vfm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # Residual injection into VLM
        return x_vlm + self.scale * w * delta_p, w


class VFMAdapterDomain(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=0.1):
        super().__init__()
        
        # Linear projection for gating input
        self.proj = nn.Linear(dim, dim)
        
        # LoRA-style MLP adapter (non-zero init to allow learning)
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x
        self.up = nn.Linear(bottleneck_dim, dim)
        self.gram = AttentionGramLayer()

        # Gating MLP from [VFM_proj, VLM]
        self.weight = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)
        )

        self.weight_domain = nn.Sequential(
            nn.Linear(dim * 2, bottleneck_dim * 2),
            nn.GELU(),
            nn.Linear(bottleneck_dim * 2, 1)
        )     
    
        self.scale = scale
    
        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.proj.bias)

            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)

            nn.init.kaiming_uniform_(self.up.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up.bias)
            
            #zero init
            nn.init.zeros_(self.weight[-1].weight)
            nn.init.zeros_(self.weight[-1].bias)

            nn.init.zeros_(self.weight_domain[-1].weight)
            nn.init.zeros_(self.weight_domain[-1].bias)

    def forward(self, x_vfm, x_vlm):
        # Transform VFM via LoRA-style adapter
        delta_p = self.down(x_vfm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gate
        x_vfm_proj = self.proj(x_vfm)
        x_cat = torch.cat([x_vfm_proj, x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # print(x_vfm_proj.shape)
        # Compute domain-aware gate
        gram = self.gram(x_vfm, x_vlm)
        w_gram = self.weight(gram).permute(0, 2, 1)
        
        # Residual injection into VLM
        return x_vfm + self.scale * w * delta_p + self.scale * w_gram * delta_p

    def getweights(self, x_vfm, x_vlm):
        # Transform VFM via LoRA-style adapter
        delta_p = self.down(x_vfm)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        # Compute token-wise gate
        x_cat = torch.cat([self.proj(x_vfm), x_vlm], dim=-1)  # [B, T, 2C]
        w = self.weight(x_cat)                                # [B, T, 1]

        # Residual injection into VLM
        return x_vfm + self.scale * w * delta_p, w


class GatingBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim):
        super().__init__()
        self.weight_vfm = nn.Linear(dim*2, bottleneck_dim)
        self.weight_vlm = nn.Linear(dim*2, bottleneck_dim)
        self.act = nn.GELU()

        self.fc_vfm = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.fc_vlm = nn.Linear(bottleneck_dim, bottleneck_dim)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight_vfm.weight, a=math.sqrt(5))
            nn.init.zeros_(self.weight_vfm.bias)
            nn.init.kaiming_uniform_(self.weight_vlm.weight, a=math.sqrt(5))
            nn.init.zeros_(self.weight_vlm.bias)
                
            nn.init.kaiming_uniform_(self.fc_vfm.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fc_vfm.bias)
            nn.init.kaiming_uniform_(self.fc_vlm.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fc_vlm.bias)
                           
        
    def forward(self, x_vfm, x_vlm, w_vfm, w_vlm):
        x_vfm = self.fc_vfm(x_vfm)
        x_vlm = self.fc_vlm(x_vlm)

        x_cat = torch.cat([w_vfm, w_vlm], dim=-1)
        w_vfm = self.act(self.weight_vfm(x_cat))
        w_vlm = self.act(self.weight_vlm(x_cat))
        
        return x_vfm * w_vfm, x_vlm * w_vlm

class SharedGatingBlock(nn.Module):
    def __init__(self, dim, bottleneck_dim):
        super().__init__()
        self.weight = nn.Linear(dim*2, bottleneck_dim)                   
        self.act = nn.GELU()

        self.fc = nn.Linear(bottleneck_dim*2, bottleneck_dim)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.weight.weight, a=math.sqrt(5))
            nn.init.zeros_(self.weight.bias)
                
            nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
            nn.init.zeros_(self.fc.bias)
                           
        
    def forward(self, x_vfm, x_vlm, w_vfm, w_vlm):
        x = self.fc(torch.cat([x_vfm, x_vlm], dim=-1))

        w = torch.cat([w_vfm, w_vlm], dim=-1)
        w = self.act(self.weight(w))
        
        return x*w
        

class VLMAdapterShared(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=1, shared=False):
        super().__init__()
        
        # Linear projection for gating input
        self.shared=shared

        self.down_vfm = nn.Linear(dim, bottleneck_dim)
        self.down_vlm = nn.Linear(dim, bottleneck_dim)
            
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x

        self.up_vlm = nn.Linear(bottleneck_dim, dim)
        self.up_vfm = nn.Linear(bottleneck_dim, dim)

        if shared:
            self.block = SharedGatingBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        else:
            self.block = GatingBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        
        self.scale = scale
        
        # self.scale_init= 0.001
        # self.scale_domain = nn.Parameter(torch.tensor(self.scale_init))

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_vfm.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_vfm.bias)  
            nn.init.kaiming_uniform_(self.down_vlm.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_vlm.bias) 
                
            nn.init.zeros_(self.up_vlm.weight)
            nn.init.zeros_(self.up_vlm.bias)

            nn.init.zeros_(self.up_vfm.weight)
            nn.init.zeros_(self.up_vfm.bias)
            

    def forward(self, x_vfm, x_vlm):
    # def forward(self, x):
    #     x_vfm, x_vlm = x[:,:1025], x[:,1025:]

        delta_p_vfm = self.down_vfm(x_vfm)
        delta_p_vlm = self.down_vlm(x_vlm)
            
        delta_p_vfm = self.act(delta_p_vfm)
        delta_p_vlm = self.act(delta_p_vlm)

        if self.shared:
            delta_p = self.block(delta_p_vfm, delta_p_vlm, x_vfm, x_vlm)
            delta_p_vfm = self.dropout(delta_p)
            delta_p_vfm = self.up_vfm(delta_p)
    
            delta_p_vlm = self.dropout(delta_p)
            delta_p_vlm = self.up_vlm(delta_p)   
            
        else:
            delta_p_vfm, delta_p_vlm = self.block(delta_p_vfm, delta_p_vlm, x_vfm, x_vlm)
            delta_p_vfm = self.dropout(delta_p_vfm)
            delta_p_vfm = self.up_vfm(delta_p_vfm)
    
            delta_p_vlm = self.dropout(delta_p_vlm)
            delta_p_vlm = self.up_vlm(delta_p_vlm)           
     
        
        # Residual injection into VLM
        return x_vlm + self.scale * delta_p_vlm + self.scale * delta_p_vfm

class VFMAdapterShared(nn.Module):
    def __init__(self, dim=1024, bottleneck_dim=64, dropout_rate=0.1, scale=0.1, shared=False):
        super().__init__()
        
        # Linear projection for gating input
        self.shared=shared

        self.down_vfm = nn.Linear(dim, bottleneck_dim)
        self.down_vlm = nn.Linear(dim, bottleneck_dim)
            
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x

        self.up_vlm = nn.Linear(bottleneck_dim, dim)
        self.up_vfm = nn.Linear(bottleneck_dim, dim)

        if shared:
            self.block = SharedGatingBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        else:
            self.block = GatingBlock(dim=dim, bottleneck_dim=bottleneck_dim)
        
        self.scale = scale
        
        # self.scale_init= 0.001
        # self.scale_domain = nn.Parameter(torch.tensor(self.scale_init))

        # Initialization
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_vfm.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_vfm.bias)  
            nn.init.kaiming_uniform_(self.down_vlm.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_vlm.bias) 
                
            nn.init.zeros_(self.up_vlm.weight)
            nn.init.zeros_(self.up_vlm.bias)

            nn.init.zeros_(self.up_vfm.weight)
            nn.init.zeros_(self.up_vfm.bias)
            

    def forward(self, x_vfm, x_vlm):
    # def forward(self, x):
    #     x_vfm, x_vlm = x[:,:1025], x[:,1025:]

        delta_p_vfm = self.down_vfm(x_vfm)
        delta_p_vlm = self.down_vlm(x_vlm)
            
        delta_p_vfm = self.act(delta_p_vfm)
        delta_p_vlm = self.act(delta_p_vlm)

        if self.shared:
            delta_p = self.block(delta_p_vfm, delta_p_vlm, x_vfm, x_vlm)
            delta_p_vfm = self.dropout(delta_p)
            delta_p_vfm = self.up_vfm(delta_p)
    
            delta_p_vlm = self.dropout(delta_p)
            delta_p_vlm = self.up_vlm(delta_p)   
            
        else:
            delta_p_vfm, delta_p_vlm = self.block(delta_p_vfm, delta_p_vlm, x_vfm, x_vlm)
            delta_p_vfm = self.dropout(delta_p_vfm)
            delta_p_vfm = self.up_vfm(delta_p_vfm)
    
            delta_p_vlm = self.dropout(delta_p_vlm)
            delta_p_vlm = self.up_vlm(delta_p_vlm)           
     
        
        # Residual injection into VLM
        return x_vfm + self.scale * delta_p_vlm + self.scale * delta_p_vfm