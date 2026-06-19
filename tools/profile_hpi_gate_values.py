#!/usr/bin/env python3
"""Profile actual SAC/SCC gate values on real images.

This runs normal model inference and records the quantities used by HPI:

    hpi_weight = (1 - beta) * w_sac + beta * w_scc

where w_sac is SAC and w_scc is SCC/SCCGate.
"""

import argparse
import contextlib
import glob
import io
import os
import sys
from collections import defaultdict, deque

import mmcv
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.profile_hpi import (  # noqa: E402
    beta_from_model,
    beta_from_state_dict,
    build_model,
)


class TensorStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.min = None
        self.max = None

    def update(self, tensor):
        x = tensor.detach().float().cpu()
        if x.numel() == 0:
            return
        self.count += x.numel()
        self.sum += float(x.sum())
        cur_min = float(x.min())
        cur_max = float(x.max())
        self.min = cur_min if self.min is None else min(self.min, cur_min)
        self.max = cur_max if self.max is None else max(self.max, cur_max)

    @property
    def mean(self):
        return self.sum / self.count if self.count else 0.0


def normalize_image(path, input_size, mean, std, to_rgb, device):
    img = mmcv.imread(path, flag="color")
    ori_shape = img.shape
    img = mmcv.imresize(img, tuple(input_size[::-1]))
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    img = mmcv.imnormalize(img, mean, std, to_rgb=to_rgb)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    meta = dict(
        ori_shape=ori_shape,
        img_shape=img.shape,
        pad_shape=img.shape,
        scale_factor=1.0,
        flip=False,
        flip_direction=None,
        filename=path,
        ori_filename=os.path.basename(path),
    )
    return tensor, [meta]


def install_gate_probes(backbone):
    stats = defaultdict(TensorStats)
    ratios = defaultdict(list)
    pending_alpha = {"vlm": deque(), "dino": deque()}

    def beta_for(branch):
        logit = getattr(backbone, f"sac_scc_beta_{branch}_logit")
        return torch.sigmoid(logit.detach().float())

    def wrap_hpi_attn(module, branch):
        orig_forward = module.forward

        def wrapped_forward(patch_tokens, shp_embed):
            delta_p, semantic_logits, prompt_patch_attn = orig_forward(patch_tokens, shp_embed)
            alpha = prompt_patch_attn.max(dim=2).values.unsqueeze(-1).detach()
            pending_alpha[branch].append(alpha)
            return delta_p, semantic_logits, prompt_patch_attn

        module.forward = wrapped_forward

    def wrap_gate(gate, branch, layer_idx):
        orig_forward = gate.forward

        def wrapped_forward(x_cat, hw):
            w_scc = orig_forward(x_cat, hw)
            if not pending_alpha[branch]:
                return w_scc

            alpha = pending_alpha[branch].popleft().to(w_scc.device, dtype=w_scc.dtype)
            beta = beta_for(branch).to(w_scc.device, dtype=w_scc.dtype)
            sac = (1.0 - beta) * alpha
            scc = beta * w_scc
            hpi_weight = sac + scc

            prefix = f"{branch}.layer{layer_idx}"
            stats[f"{prefix}.w_sac_raw"].update(alpha)
            stats[f"{prefix}.w_scc_raw"].update(w_scc)
            stats[f"{prefix}.sac_contrib"].update(sac)
            stats[f"{prefix}.scc_contrib"].update(scc)
            stats[f"{prefix}.hpi_weight"].update(hpi_weight)

            sac_mean = float(sac.detach().float().mean().cpu())
            scc_mean = float(scc.detach().float().mean().cpu())
            ratios[f"{prefix}.scc_over_sac"].append(scc_mean / max(sac_mean, 1e-12))
            return w_scc

        gate.forward = wrapped_forward

    wrap_hpi_attn(backbone.hpi_attn_vlm, "vlm")
    wrap_hpi_attn(backbone.hpi_attn_dino, "dino")
    for idx in getattr(backbone, "hpi_layers", []):
        wrap_gate(backbone.scc_gate_vlm[idx], "vlm", idx)
    for idx in getattr(backbone, "hpi_layers_dino", []):
        wrap_gate(backbone.scc_gate_dino[idx], "dino", idx)

    return stats, ratios


def collect_image_paths(cfg, limit):
    data_cfg = cfg.data.test
    data_root = data_cfg.get("data_root", "")
    img_dir = data_cfg.get("img_dir", "")
    root = os.path.join(data_root, img_dir)
    patterns = [
        os.path.join(root, "*", "*.png"),
        os.path.join(root, "*.png"),
        os.path.join(root, "*", "*.jpg"),
        os.path.join(root, "*.jpg"),
    ]
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    paths = sorted(set(paths))
    return paths[:limit]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--input-size", type=int, nargs=2, default=(512, 512), metavar=("H", "W"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-mode", default="whole", choices=["whole", "slide", "config"])
    parser.add_argument("--verbose-build", action="store_true")
    args = parser.parse_args()

    device = args.device
    if args.verbose_build:
        model, cfg = build_model(args.config, args.checkpoint, device, args.inference_mode)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            model, cfg = build_model(args.config, args.checkpoint, device, args.inference_mode)
    backbone = model.backbone
    stats, ratios = install_gate_probes(backbone)
    checkpoint_betas = beta_from_state_dict(args.checkpoint)
    model_betas = beta_from_model(model)

    img_norm_cfg = cfg.get("img_norm_cfg", None)
    if img_norm_cfg is None:
        img_norm_cfg = dict(
            mean=[122.7709383, 116.7460125, 104.09373615000001],
            std=[68.5005327, 66.6321579, 70.32316304999999],
            to_rgb=True,
        )

    paths = collect_image_paths(cfg, args.limit)
    if not paths:
        raise RuntimeError("No images found from cfg.data.test")

    print(f"[gate-profile] images={len(paths)} input_size={tuple(args.input_size)} device={device}")
    if checkpoint_betas:
        print("[gate-profile] checkpoint beta:")
        for name, value in checkpoint_betas.items():
            print(
                f"  {name}: logit={value['raw_logit']:.8f} "
                f"beta={value['beta_sigmoid']:.8f}"
            )
    if model_betas:
        print("[gate-profile] model beta after load:")
        for name, value in model_betas.items():
            print(
                f"  {name}: logit={value['raw_logit']:.8f} "
                f"beta={value['beta_sigmoid']:.8f}"
            )

    ckpt_by_suffix = {
        name.split(".")[-1]: value["beta_sigmoid"]
        for name, value in checkpoint_betas.items()
    }
    for name, value in model_betas.items():
        suffix = name.split(".")[-1]
        if suffix in ckpt_by_suffix:
            delta_p = abs(value["beta_sigmoid"] - ckpt_by_suffix[suffix])
            if delta_p > 1e-5:
                print(
                    "[gate-profile][WARNING] beta mismatch for "
                    f"{suffix}: checkpoint={ckpt_by_suffix[suffix]:.8f} "
                    f"model={value['beta_sigmoid']:.8f}. "
                    "Use the dumped config that belongs to this checkpoint."
                )

    with torch.no_grad():
        for i, path in enumerate(paths, 1):
            img, meta = normalize_image(
                path,
                tuple(args.input_size),
                img_norm_cfg["mean"],
                img_norm_cfg["std"],
                img_norm_cfg.get("to_rgb", True),
                device,
            )
            model.inference(img, meta, rescale=False)
            print(f"[gate-profile] {i:02d}/{len(paths)} {path}")

    print("\n[gate-profile] element-wise means over all sampled images/tokens")
    for key in sorted(stats.keys()):
        item = stats[key]
        print(
            f"{key}: mean={item.mean:.8f} min={item.min:.8f} "
            f"max={item.max:.8f} count={item.count}"
        )

    print("\n[gate-profile] contribution ratios")
    for key in sorted(ratios.keys()):
        values = ratios[key]
        mean_value = sum(values) / len(values) if values else 0.0
        print(f"{key}: mean={mean_value:.8f} n={len(values)}")


if __name__ == "__main__":
    main()
