#!/usr/bin/env python
"""
HPI Model Profiling Script (v3) - with HPI freeze comparison
"""

import os
import sys
import mmcv
import torch
import argparse
import warnings
from contextlib import contextmanager

warnings.filterwarnings('ignore')

from mmcv.runner import load_checkpoint
from mmseg.models import build_segmentor

import models


def get_hpi_module_names():
    """Return HPI-related module names."""
    return [
        'hpi_attn_vlm', 'hpi_attn_dino',
        'scc_gate_vlm', 'scc_gate_dino',
        'sac_scc_beta_vlm_logit', 'sac_scc_beta_dino_logit',
    ]


def count_params(model):
    """Count parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_hpi_modules(model):
    """Freeze HPI-related modules."""
    bb = model.backbone if hasattr(model, 'backbone') else None
    if bb is None:
        print("  [WARN] No backbone found")
        return 0
    
    frozen_count = 0
    for name in get_hpi_module_names():
        if hasattr(bb, name):
            module = getattr(bb, name)
            if module is not None:
                if isinstance(module, torch.nn.Parameter):
                    if module.requires_grad:
                        module.requires_grad = False
                        frozen_count += module.numel()
                else:
                    for p in module.parameters():
                        if p.requires_grad:
                            p.requires_grad = False
                            frozen_count += p.numel()
    return frozen_count


def unfreeze_hpi_modules(model):
    """Unfreeze HPI-related modules."""
    bb = model.backbone if hasattr(model, 'backbone') else None
    if bb is None:
        return
    
    for name in get_hpi_module_names():
        if hasattr(bb, name):
            module = getattr(bb, name)
            if module is not None:
                if isinstance(module, torch.nn.Parameter):
                    module.requires_grad = True
                else:
                    for p in module.parameters():
                        p.requires_grad = True


def detailed_param_breakdown(model):
    """Print detailed parameter breakdown."""
    print("\n" + "="*70)
    print("DETAILED PARAMETER BREAKDOWN")
    print("="*70)
    
    unwrapped = model.module if hasattr(model, 'module') else model
    if hasattr(unwrapped, 'backbone'):
        bb = unwrapped.backbone
        print("\n[BACKBONE]")
        core_frozen = ['conv1', 'class_embedding', 'positional_embedding', 'ln_pre', 
                       'transformer', 'ln_post', 'proj', 'dinov2']
        hpi_modules = ['hpi_attn_vlm', 'hpi_attn_dino', 'scc_gate_vlm', 'scc_gate_dino',
                       'sac_scc_beta_vlm_logit', 'sac_scc_beta_dino_logit']
        adapter_modules = ['vlm_adapter', 'vfm_adapter', 'adapter_proj3']
        fpn_modules = ['fpn1', 'fpn2', 'fpn3', 'fpn4']
        
        def print_module_params(name, module):
            if module is None:
                return
            if isinstance(module, torch.nn.Parameter):
                total = module.numel()
                train = total if module.requires_grad else 0
            else:
                total = sum(p.numel() for p in module.parameters())
                train = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"    {name:30s}: {total/1e6:8.2f}M (trainable: {train/1e6:.2f}M)")
            return total, train
        hpi_total, hpi_train = 0, 0
        adapter_total, adapter_train = 0, 0
        fpn_total, fpn_train = 0, 0
        
        print("  -- HPI Modules --")
        for name in hpi_modules:
            if hasattr(bb, name):
                t, tr = print_module_params(name, getattr(bb, name)) or (0, 0)
                hpi_total += t
                hpi_train += tr
        print(f"    {'HPI Subtotal':30s}: {hpi_total/1e6:8.2f}M (trainable: {hpi_train/1e6:.2f}M)")
        
        print("\n  -- Adapter Modules --")
        for name in adapter_modules:
            if hasattr(bb, name):
                t, tr = print_module_params(name, getattr(bb, name)) or (0, 0)
                adapter_total += t
                adapter_train += tr
        print(f"    {'Adapter Subtotal':30s}: {adapter_total/1e6:8.2f}M (trainable: {adapter_train/1e6:.2f}M)")
        
        print("\n  -- FPN Modules --")
        for name in fpn_modules:
            if hasattr(bb, name):
                t, tr = print_module_params(name, getattr(bb, name)) or (0, 0)
                fpn_total += t
                fpn_train += tr
        print(f"    {'FPN Subtotal':30s}: {fpn_total/1e6:8.2f}M (trainable: {fpn_train/1e6:.2f}M)")
        if hasattr(bb, 'transformer'):
            t = sum(p.numel() for p in bb.transformer.parameters())
            tr = sum(p.numel() for p in bb.transformer.parameters() if p.requires_grad)
            print(f"\n    {'CLIP Transformer (frozen)':30s}: {t/1e6:8.2f}M (trainable: {tr/1e6:.2f}M)")
        if hasattr(bb, 'dinov2'):
            t = sum(p.numel() for p in bb.dinov2.parameters())
            tr = sum(p.numel() for p in bb.dinov2.parameters() if p.requires_grad)
            print(f"    {'DINOv2 (frozen)':30s}: {t/1e6:8.2f}M (trainable: {tr/1e6:.2f}M)")
    if hasattr(unwrapped, 'decode_head'):
        dec = unwrapped.decode_head
        t = sum(p.numel() for p in dec.parameters())
        tr = sum(p.numel() for p in dec.parameters() if p.requires_grad)
        print(f"\n[DECODE HEAD]")
        print(f"    {'Total':30s}: {t/1e6:8.2f}M (trainable: {tr/1e6:.2f}M)")
    if hasattr(unwrapped, 'text_encoder'):
        te = unwrapped.text_encoder
        t = sum(p.numel() for p in te.parameters())
        tr = sum(p.numel() for p in te.parameters() if p.requires_grad)
        print(f"\n[TEXT ENCODER]")
        print(f"    {'Total':30s}: {t/1e6:8.2f}M (trainable: {tr/1e6:.2f}M)")
    print(f"\n[OTHER TOP-LEVEL PARAMS]")
    for name, param in unwrapped.named_parameters():
        if '.' not in name:  # top-level params
            print(f"    {name:30s}: {param.numel()/1e6:8.4f}M (trainable: {param.requires_grad})")


def compare_with_without_hpi(model):
    """Compare parameter counts with and without HPI."""
    print("\n" + "="*70)
    print("COMPARISON: WITH vs WITHOUT HPI")
    print("="*70)
    total_with, train_with = count_params(model)
    frozen_params = freeze_hpi_modules(model)
    total_without, train_without = count_params(model)
    unfreeze_hpi_modules(model)
    
    print(f"\n  WITH HPI enabled:")
    print(f"    Total params:     {total_with/1e6:.2f}M")
    print(f"    Trainable params: {train_with/1e6:.2f}M")
    
    print(f"\n  WITHOUT HPI (frozen):")
    print(f"    Total params:     {total_without/1e6:.2f}M")
    print(f"    Trainable params: {train_without/1e6:.2f}M")
    
    print(f"\n  HPI Contribution:")
    print(f"    HPI trainable params: {(train_with - train_without)/1e6:.2f}M")
    print(f"    HPI ratio: {(train_with - train_without)/train_with*100:.1f}% of trainable")


def profile_model_params(model):
    """Print main parameter statistics."""
    print("\n" + "="*70)
    print("OVERALL PARAMETER SUMMARY")
    print("="*70)
    
    total, trainable = count_params(model)
    frozen = total - trainable
    
    print(f"\n  Total Parameters:     {total/1e6:.2f}M")
    print(f"  Trainable Parameters: {trainable/1e6:.2f}M ({trainable/total*100:.1f}%)")
    print(f"  Frozen Parameters:    {frozen/1e6:.2f}M ({frozen/total*100:.1f}%)")


def parse_args():
    parser = argparse.ArgumentParser(description='Profile HPI Model Parameters')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    
    if not hasattr(cfg.model, 'class_names') or cfg.model.class_names is None:
        cfg.model.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence',
            'pole', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
    print("\nBuilding model...")
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    print("Loading checkpoint...")
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.to(args.device).eval()
    profile_model_params(model)
    detailed_param_breakdown(model)
    compare_with_without_hpi(model)
    
    print("\n" + "="*70)
    print("PROFILING COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()