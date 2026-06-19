import os
import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.ops import sigmoid_focal_loss

from mmseg.ops import resize
from mmseg.core import add_prefix
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.utils import get_root_logger

from ..backbones.clip import *
from ..backbones.utils import tokenize
from ..backbones.shp_prompt_utils import build_shp_prompt_dict, encode_shp_prompt_library
from ..backbones.shp_prompt_utils_class import (
    build_shp_prompt_dict as build_class_prompt_dict,
    encode_shp_prompt_library as encode_class_prompt_library,
)

# Add Talk2DINO to sys.path for ProjectionLayer imports.
_T2D_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'Talk2DINO')
)
_T2D_SRC = os.path.join(_T2D_ROOT, 'src')
if os.path.isdir(_T2D_ROOT) and (_T2D_ROOT not in sys.path):
    sys.path.insert(0, _T2D_ROOT)
if os.path.isdir(_T2D_SRC) and (_T2D_SRC not in sys.path):
    sys.path.insert(0, _T2D_SRC)

# ProjectionLayer maps CLIP text embeddings to the DINO space.
try:
    from src.model import ProjectionLayer
except Exception:
    import importlib.util, types
    _model_py = os.path.join(_T2D_SRC, 'model.py')
    if not os.path.isfile(_model_py):
        raise FileNotFoundError(f'Cannot find Talk2DINO model.py at {_model_py}')
    if 'src' not in sys.modules:
        src_pkg = types.ModuleType('src')
        src_pkg.__path__ = [_T2D_SRC]
        sys.modules['src'] = src_pkg
    spec = importlib.util.spec_from_file_location('src.model', _model_py)
    _t2d_model = importlib.util.module_from_spec(spec)
    sys.modules['src.model'] = _t2d_model
    spec.loader.exec_module(_t2d_model)
    ProjectionLayer = _t2d_model.ProjectionLayer

@SEGMENTORS.register_module()
class HPI_CLIP(BaseSegmentor):

    def __init__(self,
                 backbone,
                 text_encoder,
                 decode_head,
                 class_names,
                 context_length,
                 context_decoder=None,
                 token_embed_dim=512, 
                 text_dim=512,
                 neck=None,
                 identity_head=None,
                 visual_reg=True,
                 textual_reg=True,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,

                 loss_backbone=None,
                 shp_prompt_mode='shp',
                 **args):

        super(HPI_CLIP, self).__init__(init_cfg)

        self.tau = 0.07
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.num_classes = len(class_names)
        self.class_names = class_names
        self.context_length = context_length
        self.visual_reg = visual_reg
        self.textual_reg = textual_reg
        self.shp_prompt_mode = str(shp_prompt_mode).lower()
        self.save_dir = args.get('save_dir', None)
        self.beta_tracking = args.get('beta_tracking', None)
        self._beta_trace_iter = 0
        self._beta_trace_path = None
        self._beta_trace_ready = False
        if self.beta_tracking is not None:
            self.beta_tracking = copy.deepcopy(self.beta_tracking)
            if self._trace_rank0():
                get_root_logger().info(
                    '[ModelBetaTrace] enabled: %s', self.beta_tracking)

        if pretrained is not None:
            assert backbone.get('pretrained') is None, \
                'both backbone and segmentor set pretrained weight'
            backbone.pretrained = pretrained
            assert text_encoder.get('pretrained') is None, \
                'both text encoder and segmentor set pretrained weight'
            text_encoder.pretrained = 'pretrained/ViT-L-14-336px.pt'

        self.backbone = builder.build_backbone(backbone); self.backbone.init_weights()
        self.text_encoder = builder.build_backbone(text_encoder); self.text_encoder.init_weights()
        self.neck = builder.build_neck(neck) if neck is not None else None
        self.context_decoder = builder.build_backbone(context_decoder) if context_decoder is not None else None

        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.backbone.named_parameters():
            if any(key in name for key in ['adapter', 'fpn', 'adapter_proj']): 
                param.requires_grad = True
            # Enable gradients for HPI-related parameters
            elif ('hpi_attn_vlm' in name or 'hpi_attn_dino' in name):
                if any(k in name for k in ['temperature','proj','query_proj','key_proj','value_proj','q_proj','k_proj','v_proj']):
                    param.requires_grad = True
            elif ('scc_gate' in name) or ('sac_scc_beta' in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.decode_head = builder.build_head(decode_head) if decode_head is not None else None
        self.identity_head = builder.build_head(identity_head) if identity_head is not None else None

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_encoder.to(device)
        prompt_num = self.text_encoder.context_length - self.context_length
        self.texts = torch.cat([tokenize(c, context_length=context_length) for c in class_names]).to(device)
        self.contexts = nn.Parameter(torch.randn(1, prompt_num, token_embed_dim))
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

        nn.init.trunc_normal_(self.contexts)
        nn.init.trunc_normal_(self.gamma)
        
        self.loss_backbone = builder.build_loss(loss_backbone) if loss_backbone is not None else None
        
        # Build the SHP description library with the same logic as the EVA path.
        # Encode the 77-token description library once using the local TorchScript CLIP text model.
        if self.shp_prompt_mode in ('shp', 'hpi', 'region'):
            prompt_builder = build_shp_prompt_dict
            prompt_encoder = encode_shp_prompt_library
            prompt_prefix = "a photo of a {cls} which"
        elif self.shp_prompt_mode in ('class', 'class_only', 'classname', 'class_name'):
            prompt_builder = build_class_prompt_dict
            prompt_encoder = encode_class_prompt_library
            prompt_prefix = "a photo of a {cls}"
        else:
            raise ValueError(f"Unknown shp_prompt_mode: {self.shp_prompt_mode}")
        _shp_dict = prompt_builder()

        # Local TorchScript model: pretrained/ViT-L-14-336px.pt
        ts_model = torch.jit.load('pretrained/ViT-L-14-336px.pt', map_location=device).eval().to(device)
        full_text_encoder = ts_model.encode_text
        ctx77 = int(getattr(ts_model, 'context_length', 77))

        with torch.no_grad():
            # Build the 768-dimensional text library
            shp_text_embed = prompt_encoder(
                text_encoder   = full_text_encoder,
                class_names    = class_names,
                prompt_dict    = _shp_dict,
                device         = device,
                context_length = ctx77,
                prefix_fmt     = prompt_prefix,
                batch_size     = 256,
                normalize      = True,
            ).view(-1, 768)  # [K, 768]

        # CLIP branch: 768 -> self.backbone.embed_dim (768/1024)
        pad_dim = self.backbone.embed_dim - shp_text_embed.size(-1)
        assert pad_dim >= 0
        if pad_dim > 0:
            shp_embed_vlm = F.pad(shp_text_embed, (0, pad_dim), value=0.0)  # [K, embed_dim]
        else:
            shp_embed_vlm = shp_text_embed
        shp_embed_vlm = F.normalize(shp_embed_vlm, dim=-1)

        # DINO branch: Talk2DINO projects text features from 768 to 1024
        t2d_cfg = os.path.join(_T2D_ROOT, 'configs', 'vitl_mlp_infonce.yaml')
        t2d_wts = os.path.join(_T2D_ROOT, 'weights', 'vitl_mlp_infonce.pth')
        if not os.path.isfile(t2d_cfg):
            raise FileNotFoundError(f'Talk2DINO config not found: {t2d_cfg}')
        if not os.path.isfile(t2d_wts):
            raise FileNotFoundError(f'Talk2DINO weights not found: {t2d_wts}')

        self.talk2dino = ProjectionLayer.from_config(t2d_cfg)
        self.talk2dino.load_state_dict(torch.load(t2d_wts, map_location='cpu'))
        self.talk2dino.eval().to(device)

        with torch.no_grad():
            shp_embed_dino = self.talk2dino.project_clip_txt(shp_text_embed.to(device))  # [K, 1024]
            shp_embed_dino = F.normalize(shp_embed_dino, dim=-1)

        shp_embed_vlm = shp_embed_vlm.float().contiguous()
        shp_embed_dino = shp_embed_dino.float().contiguous()

        # Register buffers on the backbone for forward-time HPI injection and w_sac.
        self.backbone.register_buffer("shp_embed_vlm", shp_embed_vlm, persistent=False)
        self.backbone.register_buffer("shp_embed_dino", shp_embed_dino, persistent=False)
        if not hasattr(self.backbone, 'shp_embed'):
            self.backbone.register_buffer("shp_embed", shp_embed_vlm, persistent=False)

        # Release the large text-side model
        ts_model.to('cpu'); del ts_model
        torch.cuda.empty_cache()

    @staticmethod
    def _trace_rank0():
        rank = os.environ.get('RANK', None)
        if rank is not None:
            return int(rank) == 0
        local_rank = os.environ.get('LOCAL_RANK', '0')
        return int(local_rank) == 0

    @staticmethod
    def _scalar_from_module(module, name):
        if not hasattr(module, name):
            return None
        value = getattr(module, name)
        if isinstance(value, torch.nn.Parameter):
            value = value.detach()
        elif isinstance(value, torch.Tensor):
            value = value.detach()
        else:
            return None
        return float(value.float().view(-1)[0].cpu().item())

    def _maybe_write_beta_trace(self):
        if self.beta_tracking is None or not self._trace_rank0():
            self._beta_trace_iter += 1
            return

        logger = get_root_logger()
        interval = int(self.beta_tracking.get('interval', 50))
        filename = self.beta_tracking.get('filename', 'beta_trace.csv')
        save_dir = self.beta_tracking.get('save_dir', self.save_dir or '.')
        if self._beta_trace_path is None:
            self._beta_trace_path = os.path.join(save_dir, filename)

        if not self._beta_trace_ready:
            os.makedirs(os.path.dirname(self._beta_trace_path), exist_ok=True)
            with open(self._beta_trace_path, 'w') as f:
                f.write('iter,clip_logit,clip_beta,dino_logit,dino_beta\n')
            self._beta_trace_ready = True
            logger.info('[ModelBetaTrace] writing to %s',
                        self._beta_trace_path)

        cur_iter = self._beta_trace_iter
        should_write = cur_iter == 0 or (interval > 0 and cur_iter % interval == 0)
        if should_write:
            clip_logit = self._scalar_from_module(
                self.backbone, 'sac_scc_beta_vlm_logit')
            dino_logit = self._scalar_from_module(
                self.backbone, 'sac_scc_beta_dino_logit')
            clip_beta = None if clip_logit is None else float(
                torch.sigmoid(torch.tensor(clip_logit)).item())
            dino_beta = None if dino_logit is None else float(
                torch.sigmoid(torch.tensor(dino_logit)).item())

            def fmt(x):
                return '' if x is None else f'{x:.8f}'

            with open(self._beta_trace_path, 'a') as f:
                f.write(
                    f'{cur_iter},{fmt(clip_logit)},{fmt(clip_beta)},'
                    f'{fmt(dino_logit)},{fmt(dino_beta)}\n')
            logger.info(
                '[ModelBetaTrace] iter=%d clip_beta=%s dino_beta=%s',
                cur_iter, fmt(clip_beta), fmt(dino_beta))

        self._beta_trace_iter += 1
    
    def _image_level_labels(self, gt_semantic_seg: torch.Tensor) -> torch.Tensor:
        """Build image-level class-presence labels y in {0,1}^{B*C} from pixel annotations, ignoring 255."""
        ignore_index = int(getattr(self.decode_head, 'ignore_index', 255))
        seg = gt_semantic_seg.squeeze(1)                                  # [B,H,W]
        B, H, W = seg.shape
        C = self.num_classes
    
        # Flatten to [B, H*W] to avoid any(dim=tuple) compatibility issues
        seg_flat = seg.view(B, -1)                                         # [B,HW]
        valid_flat = (seg_flat != ignore_index)                            # [B,HW]
    
        y = torch.zeros(B, C, device=seg.device, dtype=torch.float32)      # [B,C]
        for c in range(C):
            present = ((seg_flat == c) & valid_flat).any(dim=1)            # [B]
            y[:, c] = present.float()
        return y

    def _aggregate_class_logits(self, sim_list, num_classes: int) -> torch.Tensor:
        """
        Aggregate per-layer sim logits, each [B,K], into [B,C]:
        - If K == C, return directly.
        - If K is a multiple of C, average over templates per class.
        - Otherwise, fall back to the first C dimensions.
        """
        logits = torch.stack(sim_list, dim=0).mean(0)  # [B,K]
        B, K = logits.shape
        if K == num_classes:
            return logits
        if K % num_classes == 0:
            tpc = K // num_classes  # templates per class
            return logits.view(B, num_classes, tpc).mean(dim=-1)
        # Fallback: keep the first C channels when template counts are uneven
        return logits[:, :num_classes]

    def extract_feat(self, img, train_loss=False):
        x = self.backbone(img, train_loss=train_loss)
        return x

    def after_extract_feat(self, x):
        # Support both return formats
        if isinstance(x, tuple) and len(x) > 1:
            # If the backbone returned (features, sims_clip, sims_dino, prompt_patch_attns_clip, prompt_patch_attns_dino)
            if isinstance(x[0], (list, tuple)):
                features = x[0]
                self._last_sims_clip = x[1] if len(x) > 1 else None
                self._last_sims_dino = x[2] if len(x) > 2 else None
                self._last_prompt_patch_attns_clip = x[3] if len(x) > 3 else None
                self._last_prompt_patch_attns_dino = x[4] if len(x) > 4 else None
                x = features
            else:
                self._last_sims_clip = None
                self._last_sims_dino = None
                self._last_prompt_patch_attns_clip = None
                self._last_prompt_patch_attns_dino = None
        else:
            self._last_sims_clip = None
            self._last_sims_dino = None
            self._last_prompt_patch_attns_clip = None
            self._last_prompt_patch_attns_dino = None
        
        x_orig = list(x[:-1])
        global_feat, visual_embeddings = x[-1]
        b_size = global_feat.shape[0]

        visual_context = torch.cat([global_feat, visual_embeddings.flatten(-2).permute(0, 2, 1)], dim=1)
        text_embeddings = self.text_encoder(self.texts, context=self.contexts).expand(b_size, -1, -1)

        if self.context_decoder is not None:
            text_diff = self.context_decoder(text_embeddings, visual_context)
            text_embeddings = text_embeddings + self.gamma * text_diff
        ret_text_emb = text_embeddings

        visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
        text_embeddings = F.normalize(text_embeddings, dim=-1, p=2)
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings, text_embeddings)

        return x_orig, score_map, ret_text_emb, global_feat

    def forward_train(self, img, img_metas, gt_semantic_seg, **kwargs):      
        self._maybe_write_beta_trace()

        if self.loss_backbone is not None:
            x_return = self.extract_feat(img, train_loss=True)
            x, token_dict = x_return
        else:
            x_return = self.extract_feat(img, train_loss=False)
            x = x_return

        x_orig, score_map, text_emb, global_feat = self.after_extract_feat(x)
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig

        losses = dict()

        if self.identity_head is not None:
            loss_score_map = self.identity_head.forward_train(
                score_map/self.tau, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_score_map, 'scr_map'))

        loss_decode = self.decode_head.forward_train(
            x, text_emb, img_metas, gt_semantic_seg, self.train_cfg, kwargs['gt_labels'], kwargs['gt_masks'])
        losses.update(add_prefix(loss_decode, 'decode'))
        
        if self.loss_backbone is not None:
            red_loss = self.loss_backbone(token_dict['vlm_feature'], token_dict['vfm_feature'])
            for name in red_loss:
                losses.update({f'{name}': red_loss[name]})
        
        # HPI spatial localization: probability supervision with NLLLoss, default aggregation is lse
        w_where_clip = float(self.train_cfg.get('lambda_spatial_vlm', 0.05)) if self.train_cfg else 0.05
        w_where_dino = float(self.train_cfg.get('lambda_spatial_dino', 0.05)) if self.train_cfg else 0.05
        tau_c = float(self.train_cfg.get('spatial_loss_tau', 0.7)) if self.train_cfg else 0.7
        agg = str(self.train_cfg.get('spatial_loss_agg', 'lse')).lower() if self.train_cfg else 'lse'
        
        ignore_index = int(getattr(self.decode_head, 'ignore_index', 255))
        target_full = gt_semantic_seg.squeeze(1)  # [B,H,W]
        K = self.num_classes
        
        def _prompt_patch_attn_ce_branch(prompt_patch_attn_list):
            """
            prompt_patch_attn_list: list of tensors, each [B, C_p, Hs, Ws], where C_p = K * Np and Np is prototypes/templates per class.
            Steps:
              1) Aggregate over prototypes to class probabilities p_cls[B,K,Hs,Ws].
              2) Normalize over classes to obtain per-pixel class distributions.
              3) Upsample to label resolution and apply NLLLoss with ignore_index support.
            """
            if not prompt_patch_attn_list:
                return None
        
            L = len(prompt_patch_attn_list)
            w_layer = torch.linspace(0.5, 1.0, steps=L, device=prompt_patch_attn_list[0].device, dtype=prompt_patch_attn_list[0].dtype)
            w_layer = w_layer / w_layer.sum()
        
            loss_accum = 0.0
            eps = 1e-8
            for li, att2d in enumerate(prompt_patch_attn_list):
                # att2d already contains post-softmax attention probabilities with shape [B, C_p, Hs, Ws]
                B, C_p, Hs, Ws = att2d.shape
                Np = max(C_p // K, 1)             # Number of templates per class; truncate to K*Np if uneven
                C_use = K * Np
                proto = att2d[:, :C_use].contiguous().view(B, K, Np, Hs, Ws)  # [B,K,Np,Hs,Ws]
        
                if agg == 'sum':
                    p_cls = proto.sum(dim=2)                       # [B,K,Hs,Ws]
                elif agg == 'mean':
                    p_cls = proto.mean(dim=2)
                elif agg == 'max':
                    p_cls = proto.max(dim=2).values
                else:  # 'lse': temperature-smoothed log-sum-exp
                    logp = torch.log(proto.clamp_min(eps))         # [B,K,Np,Hs,Ws]
                    logp_cls = torch.logsumexp(logp / tau_c, dim=2) * tau_c
                    p_cls = torch.exp(logp_cls)                    # [B,K,Hs,Ws]
        
                # Normalize over classes to obtain per-pixel class distributions and avoid numeric drift
                p_sum = p_cls.sum(dim=1, keepdim=True) + eps
                p_cls = p_cls / p_sum                               # [B,K,Hs,Ws]
                log_p = torch.log(p_cls.clamp_min(eps))             # Log probabilities for NLL
        
                # Upsample to GT resolution and apply NLLLoss with ignore_index support
                log_p_up = F.interpolate(log_p, size=gt_semantic_seg.shape[-2:], mode='bilinear', align_corners=False)
                loss_nll = F.nll_loss(log_p_up, target_full.to(log_p_up.device), ignore_index=ignore_index)
        
                loss_accum = loss_accum + w_layer[li] * loss_nll
        
            return loss_accum
        
        if w_where_clip > 0 and hasattr(self, '_last_prompt_patch_attns_clip') and self._last_prompt_patch_attns_clip:
            loss_where_clip = _prompt_patch_attn_ce_branch(self._last_prompt_patch_attns_clip)
            if loss_where_clip is not None:
                losses['loss_spatial_vlm'] = w_where_clip * loss_where_clip
        
        if w_where_dino > 0 and hasattr(self, '_last_prompt_patch_attns_dino') and self._last_prompt_patch_attns_dino:
            loss_where_dino = _prompt_patch_attn_ce_branch(self._last_prompt_patch_attns_dino)
            if loss_where_dino is not None:
                losses['loss_spatial_dino'] = w_where_dino * loss_where_dino
        
        w_align_clip = float(self.train_cfg.get('lambda_semantic_vlm', 1.0)) if self.train_cfg else 1.0
        w_align_dino = float(self.train_cfg.get('lambda_semantic_dino', 1.0)) if self.train_cfg else 1.0
        need_image_labels = False
        if hasattr(self, '_last_sims_clip') and self._last_sims_clip and len(self._last_sims_clip) > 0:
            need_image_labels = True
        if hasattr(self, '_last_sims_dino') and self._last_sims_dino and len(self._last_sims_dino) > 0:
            need_image_labels = True
            
        if need_image_labels:
            with torch.no_grad():
                y = self._image_level_labels(gt_semantic_seg)  # [B,C]
                B = y.size(0)
                p = y.sum(0).clamp_min(1.0)                 
                n = B - p                                     
                pos_w = (n / (p + 1e-6)).clamp(1.0, 100.0)      
                
        if w_align_clip > 0 and hasattr(self, '_last_sims_clip') and self._last_sims_clip:
            logits_clip = self._aggregate_class_logits(self._last_sims_clip, self.num_classes)
            losses['loss_semantic_vlm'] = w_align_clip * F.binary_cross_entropy_with_logits(
                logits_clip, y, pos_weight=pos_w)
        
        if w_align_dino > 0 and hasattr(self, '_last_sims_dino') and self._last_sims_dino:
            logits_d = self._aggregate_class_logits(self._last_sims_dino, self.num_classes) 
            losses['loss_semantic_dino'] = w_align_dino* F.binary_cross_entropy_with_logits(
                logits_d, y, pos_weight=pos_w)
            
        return losses
        
    def encode_decode(self, img, img_metas):
        x = self.extract_feat(img)
        x_orig, score_map, text_emb, global_feat = self.after_extract_feat(x)
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig
        
        out = self.decode_head.forward_test(
            x, text_emb, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=False)
            
        return out

    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap."""
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=False)
        
        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style."""
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))

        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred

    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations."""
        assert rescale
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)
        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        seg_pred = list(seg_pred)
        return seg_pred    
        
    def forward_with_mask(self, img, img_metas, mask_vlm, mask_vfm, ratio, mask_gene):
        def extract_feat_with_mask(img, mask_vlm, mask_vfm, mask_gene):
            x = self.backbone.foward_with_mask(img, mask_vlm, mask_vfm, ratio, mask_gene)
            return x
        
        x = extract_feat_with_mask(img, mask_vlm, mask_vfm, mask_gene)
        x_orig, score_map, text_emb, global_feat = self.after_extract_feat(x)
        x = list(self.neck(x_orig)) if self.neck is not None else x_orig
        
        out = self.decode_head.forward_with_mask(
            x, text_emb, img_metas, self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[-2:],
            mode='bilinear',
            align_corners=False)
            
        return out
