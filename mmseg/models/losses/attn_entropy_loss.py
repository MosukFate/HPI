import math
import torch
import torch.nn as nn
from ..builder import LOSSES

@LOSSES.register_module()
class AttnEntropyLoss(nn.Module):

    def __init__(self, mode='max', eps=1e-6, loss_weight=0.01, target_coeff=0.7):
        super().__init__()
        assert mode in ('max', 'target')
        self.mode = mode
        self.eps = eps
        self.loss_weight = float(loss_weight)
        self.target_coeff = float(target_coeff)

    def forward(self, attn_list):
        if not isinstance(attn_list, (list, tuple)) or len(attn_list) == 0:
            # Support paths without HPI outputs
            return torch.tensor(0.0, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        losses = []
        for attn in attn_list:
            # attn: [B, N, C] -> [B, C, N], normalized over N
            p = attn.transpose(1, 2)                                  # [B, C, N]
            p = p / (p.sum(dim=-1, keepdim=True) + self.eps)          # renorm over N
            H = -(p.clamp_min(self.eps) * p.clamp_min(self.eps).log()).sum(dim=-1).mean()

            if self.mode == 'max':
                losses.append(-H)                                      # Maximize entropy
            else:
                N = p.shape[-1]
                H_star = self.target_coeff * math.log(N)
                losses.append((H - H_star).abs())                      # Match the target entropy

        return self.loss_weight * torch.stack(losses).mean()
