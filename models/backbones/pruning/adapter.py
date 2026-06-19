import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class DeltaBlock(nn.Module):
    '''
    Shape:
        In: (N, L, C)
        Out: (N, L, C)
    
    '''
    
    def __init__(self, dim, bottleneck_dim, dropout_rate=0.1):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0. else lambda x: x
        self.up = nn.Linear(bottleneck_dim, dim)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)
            
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        
    def forward(self, x):
        delta_p = self.down(x)
        delta_p = self.act(delta_p)
        delta_p = self.dropout(delta_p)
        delta_p = self.up(delta_p)

        return delta_p


class Compensator(nn.Module):
    '''
    Shape:
        In: (N, L, C)
        Out: (N, L, C)
    
    '''
    
    def __init__(self,  dim, bottleneck_dim,):
        super().__init__()
        self.dim = dim  
        self.bottleneck_dim = bottleneck_dim

        
        self.down_conv1 = nn.Conv2d(in_channels=self.dim, out_channels=self.bottleneck_dim, kernel_size=1)
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=self.bottleneck_dim, kernel_size=3, padding=1, groups=self.bottleneck_dim), 
            nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=self.bottleneck_dim, kernel_size=1)
        )
        
        # self.conv_actfunc = build_activation_layer(**pt_act_cfg)
        self.conv_actfunc = nn.GELU()
        self.up_conv = nn.Conv2d(in_channels=self.bottleneck_dim, out_channels=self.dim, kernel_size=1)
        
        
        # initalize weights
        self.init_weights()
        
        
    def init_weights(self):      
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_conv1.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down_conv1.bias)
            
            for e in self.down_conv2:
                nn.init.kaiming_uniform_(e.weight, a=math.sqrt(5))
                nn.init.zeros_(e.bias)
            
            
            nn.init.zeros_(self.up_conv.weight)
            nn.init.zeros_(self.up_conv.bias)


    def forward(self, x):
        B, L, C = x.shape
        
        p_token = x[:, 1:, :] 
        h = int(math.sqrt(L - 1))
        w = h
        # p_token = self.norm(p_token)
        p_token = rearrange(p_token, 'b (h w) c -> b c h w', h=h, w=w)
        
        pt_down = self.down_conv1(p_token)
        pt_down = self.down_conv2(pt_down)
        pt_down = self.conv_actfunc(pt_down)
        pt_up = self.up_conv(pt_down)
        
        pt_up = rearrange(pt_up, 'b c h w -> b (h w) c')
        
        up = pt_up
        
        return up