#!/usr/bin/env python3
"""Profile HPI params, SAC/SCC beta, FPS, and CUDA memory.

This script reports:

- total/trainable/frozen parameters;
- parameter groups for HPI, adapters, FPN, decode head, text encoder, and
  frozen foundation branches;
- learned beta values, i.e. sigmoid(sac_scc_beta_*_logit);
- optional dummy single-scale inference FPS and peak CUDA memory.

Example:
    python tools/profile_hpi.py \
      --config configs/mfuser_clip_vit-l_1e-4_20k_g2c_512_layer21.py \
      --checkpoint work_dirs_d/mfuser_clip_vit-l_1e-4_20k-g2c-512_hpiclip/\
mfuser_clip_vit-l_1e-4_20k_g2c_512_layer21/best_mIoU_iter_12000.pth \
      --benchmark --input-size 512 512 --iters 50 --warmup 10 \
      --out docs/hpi_profile_g2c.json
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


DEFAULT_CLASSES = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


GROUPS = OrderedDict(
    [
        (
            "hpi",
            (
                "backbone.hpi_attn_vlm",
                "backbone.hpi_attn_dino",
                "backbone.scc_gate_vlm",
                "backbone.scc_gate_dino",
                "backbone.sac_scc_beta_vlm_logit",
                "backbone.sac_scc_beta_dino_logit",
            ),
        ),
        (
            "adapter",
            (
                "backbone.vlm_adapter",
                "backbone.vfm_adapter",
                "backbone.adapter_proj",
                "backbone.adapter_proj3",
            ),
        ),
        ("fpn", ("backbone.fpn",)),
        ("decode_head", ("decode_head",)),
        ("text_encoder", ("text_encoder",)),
        ("dino_v2_frozen", ("backbone.dinov2",)),
        (
            "vlm_visual_frozen",
            (
                "backbone.conv1",
                "backbone.class_embedding",
                "backbone.positional_embedding",
                "backbone.ln_pre",
                "backbone.transformer",
                "backbone.ln_post",
                "backbone.proj",
            ),
        ),
        ("context_prompt_top_level", ("contexts", "gamma", "context_decoder")),
    ]
)


def count_params(params: Iterable[torch.nn.Parameter]) -> Tuple[int, int]:
    total = 0
    trainable = 0
    for p in params:
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return total, trainable


def classify_param(name: str) -> str:
    for group, prefixes in GROUPS.items():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            return group
    return "other"


def parameter_summary(model: torch.nn.Module) -> Dict[str, object]:
    groups: Dict[str, Dict[str, int]] = OrderedDict()
    for group in list(GROUPS.keys()) + ["other"]:
        groups[group] = {"total": 0, "trainable": 0, "frozen": 0}

    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        n = p.numel()
        group = classify_param(name)
        groups[group]["total"] += n
        if p.requires_grad:
            groups[group]["trainable"] += n
            trainable += n
        else:
            groups[group]["frozen"] += n
        total += n

    for item in groups.values():
        item["frozen"] = item["total"] - item["trainable"]

    hpi_trainable = groups["hpi"]["trainable"]
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "trainable_percent": (100.0 * trainable / total) if total else 0.0,
        "hpi_trainable": hpi_trainable,
        "hpi_trainable_percent_of_trainable": (
            100.0 * hpi_trainable / trainable
        )
        if trainable
        else 0.0,
        "groups": groups,
    }


def load_raw_state_dict(checkpoint: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(checkpoint, map_location="cpu")
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")


def load_checkpoint_for_profile(model: torch.nn.Module, checkpoint: str) -> None:
    """Load checkpoint for profiling while ignoring regenerated prompt buffers."""
    state_dict = load_raw_state_dict(checkpoint)
    filtered = OrderedDict()
    skipped = []
    for key, value in state_dict.items():
        clean_key = strip_module_prefix(key)
        if clean_key in {
            "backbone.shp_embed_vlm",
            "backbone.shp_embed_dino",
            "backbone.shp_embed",
        }:
            skipped.append(clean_key)
            continue
        filtered[clean_key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if skipped:
        print("[profile] skipped regenerated prompt buffers:", ", ".join(skipped))
    if unexpected:
        print("[profile] unexpected checkpoint keys after filtering:")
        for key in unexpected[:20]:
            print(f"  - {key}")
        if len(unexpected) > 20:
            print(f"  ... {len(unexpected) - 20} more")
    # Missing shp_embed buffers are expected because they are registered with
    # persistent=False and regenerated by the segmentor constructor.
    real_missing = [
        key
        for key in missing
        if key
        not in {
            "backbone.shp_embed_vlm",
            "backbone.shp_embed_dino",
            "backbone.shp_embed",
        }
    ]
    if real_missing:
        print("[profile] missing model keys:")
        for key in real_missing[:20]:
            print(f"  - {key}")
        if len(real_missing) > 20:
            print(f"  ... {len(real_missing) - 20} more")


def strip_module_prefix(name: str) -> str:
    return name[7:] if name.startswith("module.") else name


def beta_from_state_dict(checkpoint: str) -> Dict[str, Dict[str, float]]:
    if not checkpoint:
        return {}
    state_dict = load_raw_state_dict(checkpoint)
    out: Dict[str, Dict[str, float]] = OrderedDict()
    for key, value in state_dict.items():
        clean_key = strip_module_prefix(key)
        if clean_key.endswith("sac_scc_beta_vlm_logit") or clean_key.endswith(
            "sac_scc_beta_dino_logit"
        ):
            tensor = value.detach().float().view(-1)
            if tensor.numel() != 1:
                continue
            raw = float(tensor.item())
            out[clean_key] = {"raw_logit": raw, "beta_sigmoid": float(torch.sigmoid(tensor)[0])}
    return out


def beta_from_model(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = OrderedDict()
    for name, p in model.named_parameters():
        if name.endswith("sac_scc_beta_vlm_logit") or name.endswith(
            "sac_scc_beta_dino_logit"
        ):
            tensor = p.detach().float().view(-1).cpu()
            if tensor.numel() != 1:
                continue
            raw = float(tensor.item())
            out[name] = {
                "raw_logit": raw,
                "beta_sigmoid": float(torch.sigmoid(tensor)[0]),
                "requires_grad": bool(p.requires_grad),
            }
    for name, b in model.named_buffers():
        if name.endswith("sac_scc_beta_vlm_logit") or name.endswith(
            "sac_scc_beta_dino_logit"
        ):
            tensor = b.detach().float().view(-1).cpu()
            if tensor.numel() != 1:
                continue
            raw = float(tensor.item())
            out[name] = {
                "raw_logit": raw,
                "beta_sigmoid": float(torch.sigmoid(tensor)[0]),
                "requires_grad": False,
                "fixed_buffer": True,
            }
    return out


def set_inference_mode(cfg, mode: str) -> None:
    if mode == "config":
        return
    for obj in (cfg, cfg.model):
        if hasattr(obj, "test_cfg") and obj.test_cfg is not None:
            obj.test_cfg.mode = mode


def build_model(config: str, checkpoint: str, device: str, inference_mode: str):
    import mmcv
    from mmseg.models import build_segmentor

    import models  # noqa: F401  # Registers local model components.

    cfg = mmcv.Config.fromfile(config)
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    set_inference_mode(cfg, inference_mode)

    if not hasattr(cfg.model, "class_names") or cfg.model.class_names is None:
        cfg.model.class_names = DEFAULT_CLASSES

    model = build_segmentor(cfg.model, test_cfg=cfg.get("test_cfg"))
    if checkpoint:
        load_checkpoint_for_profile(model, checkpoint)
    model = model.to(device).eval()
    return model, cfg


def make_img_meta(batch_size: int, height: int, width: int) -> List[Dict[str, object]]:
    return [
        {
            "ori_shape": (height, width, 3),
            "img_shape": (height, width, 3),
            "pad_shape": (height, width, 3),
            "scale_factor": 1.0,
            "flip": False,
            "flip_direction": None,
            "filename": f"dummy_{i}.png",
            "ori_filename": f"dummy_{i}.png",
        }
        for i in range(batch_size)
    ]


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    device: str,
    input_size: Tuple[int, int],
    batch_size: int,
    warmup: int,
    iters: int,
    amp: bool,
) -> Dict[str, float]:
    height, width = input_size
    img = torch.randn(batch_size, 3, height, width, device=device)
    img_metas = make_img_meta(batch_size, height, width)

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model_allocated_mb = torch.cuda.memory_allocated() / (1024**2)
    else:
        model_allocated_mb = 0.0

    def run_once():
        with torch.cuda.amp.autocast(enabled=amp):
            return model.inference(img, img_metas, rescale=False)

    for _ in range(warmup):
        run_once()
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    if device.startswith("cuda"):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            run_once()
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1000.0
    else:
        t0 = time.perf_counter()
        for _ in range(iters):
            run_once()
        elapsed_s = time.perf_counter() - t0

    fps = (batch_size * iters) / elapsed_s if elapsed_s > 0 else 0.0
    latency_ms = (elapsed_s / iters) * 1000.0 if iters > 0 else 0.0

    if device.startswith("cuda"):
        peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
    else:
        peak_allocated_mb = 0.0
        peak_reserved_mb = 0.0

    return {
        "input_height": height,
        "input_width": width,
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "amp": bool(amp),
        "elapsed_s": elapsed_s,
        "fps_images_per_s": fps,
        "latency_ms_per_iter": latency_ms,
        "model_memory_allocated_mb_before_inference": model_allocated_mb,
        "peak_memory_allocated_mb": peak_allocated_mb,
        "peak_memory_reserved_mb": peak_reserved_mb,
    }


def fmt_m(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def print_report(result: Dict[str, object]) -> None:
    params = result["parameters"]
    print("\n" + "=" * 80)
    print("HPI PROFILE")
    print("=" * 80)
    print(f"Config:     {result['config']}")
    print(f"Checkpoint: {result.get('checkpoint') or 'None'}")
    print(f"Device:     {result['device']}")

    print("\n[Parameter Summary]")
    print(f"Total params:     {fmt_m(params['total'])}")
    print(
        f"Trainable params: {fmt_m(params['trainable'])} "
        f"({params['trainable_percent']:.2f}%)"
    )
    print(f"Frozen params:    {fmt_m(params['frozen'])}")
    print(
        f"HPI trainable:    {fmt_m(params['hpi_trainable'])} "
        f"({params['hpi_trainable_percent_of_trainable']:.2f}% of trainable)"
    )

    print("\n[Parameter Groups]")
    print(f"{'group':28s} {'total':>12s} {'trainable':>12s} {'frozen':>12s}")
    for group, values in params["groups"].items():
        print(
            f"{group:28s} {fmt_m(values['total']):>12s} "
            f"{fmt_m(values['trainable']):>12s} {fmt_m(values['frozen']):>12s}"
        )

    print("\n[Learned Beta = sigmoid(sac_scc_beta_*_logit)]")
    betas = result.get("beta_from_model") or result.get("beta_from_checkpoint") or {}
    if not betas:
        print("No beta parameters found.")
    for name, values in betas.items():
        print(
            f"{name}: raw_logit={values['raw_logit']:.6f}, "
            f"beta={values['beta_sigmoid']:.6f}"
        )

    if "benchmark" in result:
        b = result["benchmark"]
        print("\n[Dummy Inference Benchmark]")
        print(
            f"Input: {b['batch_size']} x 3 x {b['input_height']} x {b['input_width']}, "
            f"AMP={b['amp']}"
        )
        print(f"FPS:        {b['fps_images_per_s']:.3f} images/s")
        print(f"Latency:    {b['latency_ms_per_iter']:.3f} ms/iter")
        print(f"Peak alloc: {b['peak_memory_allocated_mb']:.1f} MB")
        print(f"Peak reserv:{b['peak_memory_reserved_mb']:.1f} MB")


def parse_args():
    parser = argparse.ArgumentParser(description="Profile HPI runtime evidence.")
    parser.add_argument("--config", default=None, help="MMCV config path.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path.")
    parser.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="Only inspect beta values from checkpoint; does not import mmcv/mmseg.",
    )
    parser.add_argument("--device", default="cuda", help="cuda or cpu.")
    parser.add_argument(
        "--inference-mode",
        default="config",
        choices=["config", "whole", "slide"],
        help="Override test_cfg.mode for FPS benchmarking.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run dummy inference FPS and CUDA memory benchmark.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("H", "W"),
        help="Dummy input size for benchmarking.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--amp", action="store_true", help="Benchmark with AMP.")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.checkpoint_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint-only requires --checkpoint")
        result = OrderedDict()
        result["checkpoint"] = args.checkpoint
        result["beta_from_checkpoint"] = beta_from_state_dict(args.checkpoint)
        print("\n[Learned Beta = sigmoid(sac_scc_beta_*_logit)]")
        if not result["beta_from_checkpoint"]:
            print("No beta parameters found.")
        for name, values in result["beta_from_checkpoint"].items():
            print(
                f"{name}: raw_logit={values['raw_logit']:.6f}, "
                f"beta={values['beta_sigmoid']:.6f}"
            )
        if args.out:
            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            print(f"\nSaved JSON report to {args.out}")
        return

    if not args.config:
        raise ValueError("--config is required unless --checkpoint-only is set")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    result: Dict[str, object] = OrderedDict()
    result["config"] = args.config
    result["checkpoint"] = args.checkpoint
    result["device"] = args.device
    result["inference_mode"] = args.inference_mode

    if args.checkpoint:
        result["beta_from_checkpoint"] = beta_from_state_dict(args.checkpoint)

    model, _ = build_model(args.config, args.checkpoint, args.device, args.inference_mode)
    result["parameters"] = parameter_summary(model)
    result["beta_from_model"] = beta_from_model(model)

    if args.benchmark:
        result["benchmark"] = benchmark_inference(
            model=model,
            device=args.device,
            input_size=tuple(args.input_size),
            batch_size=args.batch_size,
            warmup=args.warmup,
            iters=args.iters,
            amp=args.amp,
        )

    print_report(result)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved JSON report to {args.out}")


if __name__ == "__main__":
    main()
