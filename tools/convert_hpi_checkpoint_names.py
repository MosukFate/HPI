#!/usr/bin/env python3
"""Convert internal HPI checkpoint keys to the public HPI naming scheme.

The public repository renamed Python registries/classes such as ``HPI_EVA_CLIP``
and ``EVAVisionTransformerHPI``. Those names are not stored in a PyTorch
``state_dict``, so checkpoint migration only needs to rewrite parameter keys.
The ContextDecoder sequence mixer intentionally keeps its ``cross_attn.*``
parameter names for weight compatibility; no rewrite is needed there.
"""

import argparse
import os
from collections import Counter, OrderedDict

import torch

KEY_REPLACEMENTS = (
    ("backbone.sap_attn_clip.", "backbone.hpi_attn_vlm."),
    ("backbone.sap_attn_dino.", "backbone.hpi_attn_dino."),
    ("backbone.msgate_clip.", "backbone.scc_gate_vlm."),
    ("backbone.msgate_dino.", "backbone.scc_gate_dino."),
    ("backbone.msgate_blend_clip_logit", "backbone.sac_scc_beta_vlm_logit"),
    ("backbone.msgate_blend_dino_logit", "backbone.sac_scc_beta_dino_logit"),
    ("backbone.desc_embed_clip", "backbone.shp_embed_vlm"),
    ("backbone.desc_embed_dino", "backbone.shp_embed_dino"),
    ("backbone.desc_embed", "backbone.shp_embed"),
)

OLD_KEY_MARKERS = tuple(old for old, _ in KEY_REPLACEMENTS)
NEW_KEY_MARKERS = tuple(new for _, new in KEY_REPLACEMENTS)


def strip_module_prefix(key: str):
    if key.startswith("module."):
        return "module.", key[len("module."):]
    return "", key


def convert_key(key: str) -> str:
    prefix, body = strip_module_prefix(key)
    for old, new in KEY_REPLACEMENTS:
        if body == old or body.startswith(old):
            return prefix + new + body[len(old):]
    return key


def marker_family(key: str) -> str:
    _, body = strip_module_prefix(key)
    for old, new in KEY_REPLACEMENTS:
        if body == old or body.startswith(old):
            return old
        if body == new or body.startswith(new):
            return new
    return "unchanged"


def convert_state_dict(state_dict):
    converted = OrderedDict()
    target_sources = {}
    changed = []
    collisions = []
    counts = Counter()

    for key, value in state_dict.items():
        new_key = convert_key(key)
        counts[marker_family(key)] += 1

        if new_key in converted:
            collisions.append((target_sources[new_key], key, new_key))
            continue

        converted[new_key] = value
        target_sources[new_key] = key
        if new_key != key:
            changed.append((key, new_key))

    if collisions:
        msg = "\n".join(
            f"{first} and {second} both map to {target}"
            for first, second, target in collisions[:20]
        )
        raise RuntimeError(f"Key collisions during conversion:\n{msg}")

    return converted, changed, counts


def get_state_dict_container(checkpoint):
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        return checkpoint, "state_dict", checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return None, None, checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def has_old_hpi_keys(state_dict) -> bool:
    for key in state_dict:
        _, body = strip_module_prefix(key)
        if any(body == old or body.startswith(old) for old in OLD_KEY_MARKERS):
            return True
    return False


def has_new_hpi_keys(state_dict) -> bool:
    for key in state_dict:
        _, body = strip_module_prefix(key)
        if any(body == new or body.startswith(new) for new in NEW_KEY_MARKERS):
            return True
    return False


def summarize(changed, counts):
    print(f"Converted {len(changed)} keys")
    converted_groups = {old: counts[old] for old, _ in KEY_REPLACEMENTS if counts[old]}
    existing_new_groups = {new: counts[new] for _, new in KEY_REPLACEMENTS if counts[new]}
    if converted_groups:
        print("Converted groups:")
        for key, count in converted_groups.items():
            print(f"  {key}: {count}")
    if existing_new_groups:
        print("Already-public groups found:")
        for key, count in existing_new_groups.items():
            print(f"  {key}: {count}")
    for old, new in changed[:20]:
        print(f"  {old} -> {new}")
    if len(changed) > 20:
        print(f"  ... {len(changed) - 20} more")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="Source checkpoint with internal HPI parameter keys")
    parser.add_argument("dst", nargs="?", help="Destination checkpoint with public HPI parameter keys")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Inspect conversion without writing a checkpoint")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if no old HPI keys are found. Useful for catching accidental double conversion.",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.dst:
        parser.error("dst is required unless --dry-run is used")
    if args.dst and os.path.exists(args.dst) and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"Destination exists: {args.dst}. Pass --overwrite to replace it.")

    checkpoint = torch.load(args.src, map_location="cpu")
    container, state_key, state_dict = get_state_dict_container(checkpoint)

    old_found = has_old_hpi_keys(state_dict)
    new_found = has_new_hpi_keys(state_dict)
    if args.strict and not old_found:
        raise RuntimeError("No internal HPI keys found; checkpoint may already be converted.")

    converted, changed, counts = convert_state_dict(state_dict)
    summarize(changed, counts)
    if not old_found and new_found:
        print("Checkpoint already appears to use public HPI keys.")

    if args.dry_run:
        print("Dry run only; no checkpoint written.")
        return

    if container is not None:
        checkpoint = dict(checkpoint)
        checkpoint[state_key] = converted
    else:
        checkpoint = converted

    os.makedirs(os.path.dirname(args.dst) or ".", exist_ok=True)
    torch.save(checkpoint, args.dst)
    print(f"Saved: {args.dst}")


if __name__ == "__main__":
    main()
