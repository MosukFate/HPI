import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret

    
class TokenSelectSpatial(nn.Module):
    def __init__(self, dim, bottleneck_dim, mask_dim, layer, p_token_start_idx=1, tau=5, is_hard=True, threshold=0.5):
        super().__init__()
        if layer == 2:
            self.selector = nn.Sequential(nn.Conv2d(dim, bottleneck_dim, kernel_size=3, padding=1),
                                          nn.GELU(),
                                          nn.Conv2d(bottleneck_dim, mask_dim, kernel_size=3, padding=1),
                                         )
        else:
            self.selector = nn.Conv2d(dim, mask_dim, kernel_size=3, padding=1, bias=bias)

        # 1 = p_token_start_idx

        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def forward(self, x):
        b, l = x.shape[:2]
        
        h = w = int((l - 1) ** 0.5)
        logits = rearrange(x[:, 1:, :], 'b (h w) c -> b c h w', h=h, w=w)
        
        logits = self.selector(logits)
        logits = rearrange(logits, 'b c h w -> b (h w) c')
        
        token_mask = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        token_mask = torch.cat([token_mask.new_ones(b, 1, 1), token_mask], dim=1)
        
        return token_mask, logits

class TokenSelectLinear(nn.Module):
    def __init__(self, dim, bottleneck_dim, mask_dim, layer, p_token_start_idx=1, tau=5, is_hard=True, threshold=0.5):
        super().__init__()
        if layer == 2:
            self.selector = nn.Sequential(nn.Linear(dim, bottleneck_dim),
                                          nn.GELU(),
                                          nn.Linear(bottleneck_dim, mask_dim),
                                         )
        else:
            self.selector = nn.Linear(dim, mask_dim)

        # 1 = 
        self.p_token_start_idx = p_token_start_idx

        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def forward(self, x):
        b, l = x.shape[:2]

        logits = self.selector(x[:, self.p_token_start_idx:, :])
        
        token_mask = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        token_mask = torch.cat([token_mask.new_ones(b, 1, 1), token_mask], dim=1)
        
        return token_mask, logits


class TokenSelectLinearSoft(nn.Module):
    def __init__(self, dim, bottleneck_dim, mask_dim, layer, p_token_start_idx=1, tau=5, is_hard=True, threshold=0.5):
        super().__init__()
        if layer == 2:
            self.selector = nn.Sequential(nn.Linear(dim, bottleneck_dim),
                                          nn.GELU(),
                                          nn.Linear(bottleneck_dim, mask_dim),
                                          nn.Sigmoid(),
                                         )
        else:
            self.selector = nn.Sequential(nn.Linear(dim, mask_dim),
                                         nn.Sigmoid(),)

        # 1 = 
        self.p_token_start_idx = p_token_start_idx

        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def forward(self, x):
        b, l = x.shape[:2]

        logits = self.selector(x[:, self.p_token_start_idx:, :])      
        token_mask = torch.cat([logits.new_ones(b, 1, 1), logits], dim=1)
        
        return token_mask, logits
    