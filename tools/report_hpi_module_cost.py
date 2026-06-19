#!/usr/bin/env python3
"""Report core HPI module-level cost tables.

The scope is intentionally explicit:
- module-level HPI injection cost: HPI cross-attention + active SCCGate layers;
- learnable beta scalars are included in HPI params.

FLOPs are analytical estimates for the main Linear/Conv/matmul operations and
count multiply-add as 2 ops. Norms, GELU, sigmoid, softmax, reshapes, and small
elementwise ops are not included unless noted.
"""

import argparse
import json
import math
import os
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import mmcv
import torch


DEFAULT_CLASSES = 19
DEFAULT_PROMPTS_PER_CLASS = 3


def fmt_params(n: int) -> str:
    return f"{n / 1e6:.2f}M"


def fmt_flops(n: int) -> str:
    if n >= 1e12:
        return f"{n / 1e12:.3f} TFLOPs"
    return f"{n / 1e9:.3f} GFLOPs"


def strip_module_prefix(name: str) -> str:
    return name[7:] if name.startswith("module.") else name


def load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        if isinstance(obj.get("state_dict"), dict):
            obj = obj["state_dict"]
        elif isinstance(obj.get("model"), dict):
            obj = obj["model"]
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")
    return OrderedDict((strip_module_prefix(k), v) for k, v in obj.items())


def count_prefix(state_dict: Dict[str, torch.Tensor], prefixes: Iterable[str]) -> int:
    total = 0
    prefixes = tuple(prefixes)
    for name, value in state_dict.items():
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            total += int(value.numel())
    return total


def count_active_scc_gates(
    state_dict: Dict[str, torch.Tensor], vlm_layers: List[int], dino_layers: List[int]
) -> int:
    prefixes = [f"backbone.scc_gate_vlm.{i}" for i in vlm_layers]
    prefixes += [f"backbone.scc_gate_dino.{i}" for i in dino_layers]
    return count_prefix(state_dict, prefixes)


def linear_flops(items: int, in_dim: int, out_dim: int) -> int:
    return int(2 * items * in_dim * out_dim)


def hpi_cross_attention_flops(batch: int, patch_tokens: int, prompts: int, dim: int) -> OrderedDict:
    # hpi_cross_attention.py: text Q, patch K/V, then A V and A^T write-back.
    items_patch = batch * patch_tokens
    query_proj = linear_flops(prompts, dim, dim)
    key_proj = linear_flops(items_patch, dim, dim)
    value_proj = linear_flops(items_patch, dim, dim)
    attn_scores = int(2 * batch * prompts * patch_tokens * dim)
    out_text = int(2 * batch * prompts * patch_tokens * dim)
    write_back = int(2 * batch * prompts * patch_tokens * dim)
    out_proj = linear_flops(items_patch, dim, dim)
    sim_dot = int(2 * batch * prompts * dim)
    total = (
        query_proj
        + key_proj
        + value_proj
        + attn_scores
        + out_text
        + write_back
        + out_proj
        + sim_dot
    )
    return OrderedDict(
        query_proj=query_proj,
        key_proj=key_proj,
        value_proj=value_proj,
        attn_scores=attn_scores,
        out_text=out_text,
        write_back=write_back,
        out_proj=out_proj,
        sim_dot=sim_dot,
        total=total,
    )


def hpi_standard_flops(batch: int, patch_tokens: int, prompts: int, dim: int) -> OrderedDict:
    # hpi_cross_attention_standard.py: patch Q, prompt K/V, plus prompt summaries
    # for reusing the same semantic consistency loss.
    items_patch = batch * patch_tokens
    query_proj = linear_flops(items_patch, dim, dim)
    key_proj = linear_flops(prompts, dim, dim)
    value_proj = linear_flops(prompts, dim, dim)
    attn_scores = int(2 * batch * patch_tokens * prompts * dim)
    delta_p_av = int(2 * batch * patch_tokens * prompts * dim)
    patch_value_proj = linear_flops(items_patch, dim, dim)
    prompt_summary = int(2 * batch * prompts * patch_tokens * dim)
    out_proj = linear_flops(items_patch, dim, dim)
    sim_dot = int(2 * batch * prompts * dim)
    total = (
        query_proj
        + key_proj
        + value_proj
        + attn_scores
        + delta_p_av
        + patch_value_proj
        + prompt_summary
        + out_proj
        + sim_dot
    )
    return OrderedDict(
        query_proj=query_proj,
        key_proj=key_proj,
        value_proj=value_proj,
        attn_scores=attn_scores,
        delta_p_av=delta_p_av,
        patch_value_proj=patch_value_proj,
        prompt_summary=prompt_summary,
        out_proj=out_proj,
        sim_dot=sim_dot,
        total=total,
    )


def scc_gate_flops(
    batch: int, patch_tokens: int, dim: int, grid: Tuple[int, int], bottleneck: int
) -> OrderedDict:
    hidden = 2 * bottleneck
    h, w = grid
    spatial = batch * h * w
    reduce = linear_flops(batch * patch_tokens, 2 * dim, hidden)
    conv0 = int(2 * spatial * hidden * hidden)
    conv1_dw = int(2 * spatial * hidden * 3 * 3)
    conv2_dw = int(2 * spatial * hidden * 5 * 5)
    fuse = int(2 * spatial * (3 * hidden) * hidden)
    out = linear_flops(batch * patch_tokens, hidden, 1)
    total = reduce + conv0 + conv1_dw + conv2_dw + fuse + out
    return OrderedDict(
        reduce=reduce,
        conv0=conv0,
        conv1_depthwise=conv1_dw,
        conv2_depthwise=conv2_dw,
        fuse=fuse,
        out=out,
        total=total,
    )


def gate_adapter_flops(batch: int, full_tokens: int, dim: int, bottleneck: int) -> OrderedDict:
    hidden = bottleneck
    gate_hidden = 2 * bottleneck
    items = batch * full_tokens
    delta_p_dinoown = linear_flops(items, dim, hidden)
    delta_p_up = linear_flops(items, hidden, dim)
    gate_reduce = linear_flops(items, 2 * dim, gate_hidden)
    gate_out = linear_flops(items, gate_hidden, 1)
    # Small multiply for w * delta_p; not usually included in model FLOPs tables,
    # but keep it so the row is self-contained.
    elementwise = int(items * dim)
    total = delta_p_dinoown + delta_p_up + gate_reduce + gate_out + elementwise
    return OrderedDict(
        delta_p_dinoown=delta_p_dinoown,
        delta_p_up=delta_p_up,
        gate_reduce=gate_reduce,
        gate_out=gate_out,
        elementwise=elementwise,
        total=total,
    )


def sum_parts(parts: Iterable[OrderedDict]) -> OrderedDict:
    out = OrderedDict()
    for part in parts:
        for key, value in part.items():
            out[key] = int(out.get(key, 0) + value)
    return out


def load_benchmark(path: str):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        data = data[0]
    return data


def make_report(args) -> OrderedDict:
    cfg = mmcv.Config.fromfile(args.config)
    backbone = cfg.model.backbone
    state_dict = load_state_dict(args.checkpoint)

    batch = int(args.batch_size)
    height, width = map(int, args.input_size)
    patch_size = int(backbone.get("patch_size", 16))
    width_dim = int(backbone.width)
    dino_dim = 1024
    bottleneck = int(args.bottleneck)
    prompts = int(args.num_prompts)

    grid_h = math.ceil(height / patch_size)
    grid_w = math.ceil(width / patch_size)
    patch_tokens = grid_h * grid_w
    full_tokens = patch_tokens + 1
    vlm_layers = [int(x) for x in backbone.get("hpi_layers", [])]
    dino_layers = [int(x) for x in backbone.get("hpi_layers_dino", [])]
    attn_type = str(backbone.get("hpi_cross_attn_type", "hpi")).lower()

    if attn_type in ("standard", "standard_ca", "cross_attention"):
        hpi_fn = hpi_standard_flops
    else:
        hpi_fn = hpi_cross_attention_flops

    hpi_clip_parts = [hpi_fn(batch, patch_tokens, prompts, width_dim) for _ in vlm_layers]
    hpi_dino_parts = [hpi_fn(batch, patch_tokens, prompts, dino_dim) for _ in dino_layers]
    scc_gate_vlm_parts = [
        scc_gate_flops(batch, patch_tokens, width_dim, (grid_h, grid_w), bottleneck)
        for _ in vlm_layers
    ]
    scc_gate_dino_parts = [
        scc_gate_flops(batch, patch_tokens, dino_dim, (grid_h, grid_w), bottleneck)
        for _ in dino_layers
    ]

    adapter_one = gate_adapter_flops(batch, full_tokens, width_dim, bottleneck)
    adapter_all = OrderedDict((k, v * 48) for k, v in adapter_one.items())
    adapter_proj3 = linear_flops(batch * full_tokens, dino_dim, width_dim)

    params = OrderedDict()
    params["hpi_attn_vlm"] = count_prefix(state_dict, ["backbone.hpi_attn_vlm"])
    params["hpi_attn_dino"] = count_prefix(state_dict, ["backbone.hpi_attn_dino"])
    params["hpi_attn_total"] = params["hpi_attn_vlm"] + params["hpi_attn_dino"]
    params["scc_gate_vlm_all_24"] = count_prefix(state_dict, ["backbone.scc_gate_vlm"])
    params["scc_gate_dino_all_24"] = count_prefix(state_dict, ["backbone.scc_gate_dino"])
    params["scc_gate_all_48"] = params["scc_gate_vlm_all_24"] + params["scc_gate_dino_all_24"]
    params["scc_gate_active"] = count_active_scc_gates(state_dict, vlm_layers, dino_layers)
    params["beta_scalars"] = count_prefix(
        state_dict,
        [
            "backbone.sac_scc_beta_vlm_logit",
            "backbone.sac_scc_beta_dino_logit",
        ],
    )
    params["hpi_active"] = (
        params["hpi_attn_total"] + params["scc_gate_active"] + params["beta_scalars"]
    )
    params["hpi_instantiated"] = (
        params["hpi_attn_total"] + params["scc_gate_all_48"] + params["beta_scalars"]
    )
    params["vlm_adapter"] = count_prefix(state_dict, ["backbone.vlm_adapter"])
    params["vfm_adapter"] = count_prefix(state_dict, ["backbone.vfm_adapter"])
    params["branch_adapters"] = params["vlm_adapter"] + params["vfm_adapter"]
    params["adapter_proj3"] = count_prefix(state_dict, ["backbone.adapter_proj3"])
    params["branch_adapters_plus_proj3"] = params["branch_adapters"] + params["adapter_proj3"]
    params["hpi_active_plus_branch_adapters"] = (
        params["hpi_active"] + params["branch_adapters_plus_proj3"]
    )
    params["hpi_instantiated_plus_branch_adapters"] = (
        params["hpi_instantiated"] + params["branch_adapters_plus_proj3"]
    )
    params["fpn_neck"] = count_prefix(
        state_dict,
        ["backbone.fpn1", "backbone.fpn2", "backbone.fpn3", "backbone.fpn4"],
    )
    params["decode_head"] = count_prefix(state_dict, ["decode_head"])

    flops = OrderedDict()
    flops["hpi_clip"] = sum_parts(hpi_clip_parts)
    flops["hpi_dino"] = sum_parts(hpi_dino_parts)
    flops["hpi_total"] = sum_parts([flops["hpi_clip"], flops["hpi_dino"]])
    flops["scc_gate_vlm_active"] = sum_parts(scc_gate_vlm_parts)
    flops["scc_gate_dino_active"] = sum_parts(scc_gate_dino_parts)
    flops["scc_gate_active_total"] = sum_parts(
        [flops["scc_gate_vlm_active"], flops["scc_gate_dino_active"]]
    )
    flops["hpi_active_total"] = sum_parts([flops["hpi_total"], flops["scc_gate_active_total"]])
    flops["branch_gate_adapters_48"] = adapter_all
    flops["adapter_proj3"] = OrderedDict(total=adapter_proj3)
    flops["branch_adapters_plus_proj3"] = sum_parts(
        [flops["branch_gate_adapters_48"], flops["adapter_proj3"]]
    )
    flops["hpi_active_plus_branch_adapters"] = sum_parts(
        [flops["hpi_active_total"], flops["branch_adapters_plus_proj3"]]
    )

    full_bench = load_benchmark(args.full_benchmark_json)
    backbone_bench = load_benchmark(args.backbone_benchmark_json)

    return OrderedDict(
        config=args.config,
        checkpoint=args.checkpoint,
        input=dict(batch_size=batch, channels=3, height=height, width=width),
        tokenization=dict(
            patch_size=patch_size,
            patch_grid=f"{grid_h}x{grid_w}",
            patch_tokens=patch_tokens,
            full_tokens_with_cls=full_tokens,
            prompts=prompts,
            classes=args.num_classes,
            prompts_per_class=args.prompts_per_class,
        ),
        hpi_setting=dict(
            attention_type=attn_type,
            hpi_layers_vlm=vlm_layers,
            hpi_layers_dino=dino_layers,
            scc_gate_bottleneck=bottleneck,
        ),
        params=params,
        flops=flops,
        measured=dict(backbone_only=backbone_bench, end_to_end=full_bench),
        note=(
            "FLOPs are analytical module-level estimates for Linear/Conv/matmul "
            "main operations and count multiply-add as 2 ops. They exclude norm, "
            "GELU, sigmoid, softmax, interpolation, reshape, indexing, and most "
            "small elementwise ops. Measured FPS/memory are loaded from benchmark "
            "JSON files when provided."
        ),
    )


def write_markdown(report: OrderedDict, path: str) -> None:
    p = report["params"]
    f = report["flops"]

    lines = [
        "# HPI Module-Level Cost",
        "",
        "## Setting",
        "",
        f"- Input: {report['input']['batch_size']}x3x{report['input']['height']}x{report['input']['width']}",
        f"- Tokens: {report['tokenization']['patch_tokens']} patch tokens, {report['tokenization']['full_tokens_with_cls']} with CLS",
        f"- Prompts: {report['tokenization']['prompts']} ({report['tokenization']['classes']} classes x {report['tokenization']['prompts_per_class']})",
        f"- HPI layers: CLIP {report['hpi_setting']['hpi_layers_vlm']}, DINO {report['hpi_setting']['hpi_layers_dino']}",
        "",
    ]

    lines += [
        "## Core HPI Parameters / FLOPs",
        "",
        "| Scope | Params | FLOPs | Notes |",
        "|---|---:|---:|---|",
        (
            f"| HPI cross-attention only | {fmt_params(p['hpi_attn_total'])} | "
            f"{fmt_flops(f['hpi_total']['total'])} | active CLIP+DINO HPI calls |"
        ),
        (
            f"| Active SCCGate only | {fmt_params(p['scc_gate_active'])} | "
            f"{fmt_flops(f['scc_gate_active_total']['total'])} | only layers actually injected |"
        ),
        (
            f"| Active HPI injection | {fmt_params(p['hpi_active'])} | "
            f"{fmt_flops(f['hpi_active_total']['total'])} | HPI cross-attention + active SCC gate + beta |"
        ),
        (
            f"| Instantiated HPI injection | {fmt_params(p['hpi_instantiated'])} | "
            f"{fmt_flops(f['hpi_active_total']['total'])} | all 48 SCCGates are parameters, only active layers run |"
        ),
    ]

    lines += [
        "",
        "## Notes",
        "",
        "- Core HPI means HPI cross-attention, active SCCGate, and learnable beta scalars.",
        "- This excludes branch adapters, adapter_proj3, FPN neck, decode head, and full-model runtime.",
        "- FLOPs count multiply-add as 2 operations.",
        "- Module FLOPs include main Linear/Conv/matmul operations only.",
        "- Instantiated HPI includes inactive SCCGate parameters; FLOPs stay active-layer only.",
        "",
    ]

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fobj:
        fobj.write("\n".join(lines))


def print_report(report: OrderedDict) -> None:
    p = report["params"]
    f = report["flops"]
    print("\nHPI module-level cost")
    print(f"Input: {report['input']['batch_size']}x3x{report['input']['height']}x{report['input']['width']}")
    print(f"Patch tokens: {report['tokenization']['patch_tokens']}, prompts: {report['tokenization']['prompts']}")
    print(f"HPI cross-attention: {fmt_params(p['hpi_attn_total'])}, {fmt_flops(f['hpi_total']['total'])}")
    print(f"Active SCCGate:     {fmt_params(p['scc_gate_active'])}, {fmt_flops(f['scc_gate_active_total']['total'])}")
    print(f"Active HPI:        {fmt_params(p['hpi_active'])}, {fmt_flops(f['hpi_active_total']['total'])}")
    print(f"Instantiated HPI:  {fmt_params(p['hpi_instantiated'])}")


def parse_args():
    parser = argparse.ArgumentParser(description="Report HPI module-level cost.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_CLASSES)
    parser.add_argument("--prompts-per-class", type=int, default=DEFAULT_PROMPTS_PER_CLASS)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--bottleneck", type=int, default=64)
    parser.add_argument("--full-benchmark-json", default=None)
    parser.add_argument("--backbone-benchmark-json", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--markdown", default=None)
    args = parser.parse_args()
    if args.num_prompts is None:
        args.num_prompts = int(args.num_classes) * int(args.prompts_per_class)
    return args


def main() -> None:
    args = parse_args()
    report = make_report(args)
    print_report(report)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fobj:
            json.dump(report, fobj, indent=2)
        print(f"Saved JSON report to {args.out}")
    if args.markdown:
        write_markdown(report, args.markdown)
        print(f"Saved Markdown report to {args.markdown}")


if __name__ == "__main__":
    main()
