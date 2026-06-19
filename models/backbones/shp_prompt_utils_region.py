# mmseg/models/backbones/shp_prompt_utils.py
#
# Prompt utilities for HPI (Semantic Alignment Prompting).
# This version uses **natural, conversational descriptions** (3–7 words each)
# generated in the previous assistant response.
#
# Usage:
#   from mmseg.models.backbones.shp_prompt_utils import (
#       build_shp_prompt_dict, encode_shp_prompt_library
#   )
'''
For each class, write 3 short English sentences (≤12 words) in the same style as the examples. Each sentence starts with a region cue YOU choose for that class. Template: [Region cue] + [part] + [geometry] + [relation to neighbor/boundary]. Use simple present verbs, strong geometry words , and an explicit boundary neighbor . No extra modifiers. Output only the sentences.

Object class list:
road, sidewalk, building, wall, fence, pole, traffic light, traffic sign, vegetation, terrain, sky, person, rider, car, truck, bus, train, motorcycle, bicycle

'''
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

# Adjust path below to match your project layout.
from .utils import tokenize

# -------------------------------------------------------------------------
# Natural visual descriptions (3‑7 words each). Keys are lower‑case class names.
# -------------------------------------------------------------------------
_SHP_BULLETS: Dict[str, List[str]] = {
    "road": [
        "Near flat strip meets sidewalk edge.",
        "Mid straight lane meets car boundary.",
        "Far curved edge meets terrain line.",
    ],
    "sidewalk": [
        "Near straight curb meets road edge.",
        "Mid flat slab meets building base.",
        "Far thin edge meets vegetation line.",
    ],
    "building": [
        "Bottom vertical facade meets sidewalk edge.",
        "Middle rectangular window meets wall boundary.",
        "Top straight roofline meets sky line.",
    ],
    "wall": [
        "Bottom straight base meets terrain line.",
        "Middle flat plane meets building edge.",
        "Top thin edge meets sky line.",
    ],
    "fence": [
        "Near vertical post meets terrain line.",
        "Mid straight rail meets sidewalk edge.",
        "Far thin top meets sky line.",
    ],
    "pole": [
        "Bottom circular base meets sidewalk plane.",
        "Middle vertical shaft meets road edge.",
        "Top thin tip meets sky line.",
    ],
    "traffic light": [
        "Top rectangular head meets pole tip.",
        "Middle circular lens meets housing edge.",
        "Bottom short bracket meets pole shaft.",
    ],
    "traffic sign": [
        "Front circular plate meets pole shaft.",
        "Middle flat face meets road edge.",
        "Top straight edge meets sky line.",
    ],
    "vegetation": [
        "Bottom vertical trunk meets terrain plane.",
        "Middle irregular canopy meets building edge.",
        "Top jagged leaf line meets sky line.",
    ],
    "terrain": [
        "Near uneven plane meets road edge.",
        "Mid curved ridge meets vegetation line.",
        "Far thin boundary meets sky line.",
    ],
    "sky": [
        "Top open field meets roofline edge.",
        "Middle curved cloud meets building outline.",
        "Horizon straight line meets terrain edge.",
    ],
    "person": [
        "Top circular head meets shoulder line.",
        "Middle vertical torso meets arm edge.",
        "Bottom straight leg meets ground line.",
    ],
    "rider": [
        "Top round helmet meets shoulder line.",
        "Middle angled torso meets motorcycle frame.",
        "Bottom curved leg meets wheel rim.",
    ],
    "car": [
        "Front curved hood meets road plane.",
        "Side rectangular door meets window edge.",
        "Rear straight bumper meets road line.",
    ],
    "truck": [
        "Front box cab meets road plane.",
        "Side rectangular cargo meets wheel arch.",
        "Rear flat gate meets road line.",
    ],
    "bus": [
        "Front large windshield meets road plane.",
        "Side straight window row meets roof edge.",
        "Rear flat body meets road line.",
    ],
    "train": [
        "Front rounded nose meets rail line.",
        "Side long carriage meets platform edge.",
        "Roof straight edge meets sky line.",
    ],
    "motorcycle": [
        "Front circular wheel meets road plane.",
        "Middle angular frame meets engine block.",
        "Rear flat seat meets wheel rim.",
    ],
    "bicycle": [
        "Front thin wheel meets road plane.",
        "Middle triangular frame meets wheel rim.",
        "Rear circular rim meets ground line.",
    ],
}


# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------

def _normalize_key(k: str) -> str:
    """Normalize dataset class name to our bullet‑text key space."""
    k = k.lower().strip()
    # handle underscores vs spaces
    k = k.replace("_", " ")
    # compress multi‑spaces
    k = " ".join(k.split())
    return k


def build_shp_prompt_dict() -> Dict[str, List[str]]:
    """Return a *copy* of the bullet‑text dict so caller can safely edit."""
    return {k: list(v) for k, v in _SHP_BULLETS.items()}


def build_shp_prompts(
    cls_name: str,
    desc_list: Sequence[str],
    prefix_fmt: str = "a photo of a {cls} which",
) -> List[str]:
    """Wrap each description in a short template that repeats the class name."""
    cls_token = _normalize_key(cls_name)
    prefix = prefix_fmt.format(cls=cls_token)

    # sanitize curly apostrophes to ASCII
    def _clean(s: str) -> str:
        return s.replace("’", "'").replace("`", "'")

    return [f"{prefix} { _clean(d).strip() }" for d in desc_list]


# -------------------------------------------------------------------------
# Encoding utility
# -------------------------------------------------------------------------

@torch.no_grad()
def encode_shp_prompt_library(
    text_encoder: torch.nn.Module,
    class_names: Sequence[str],
    prompt_dict: Dict[str, Sequence[str]],
    device: torch.device,
    context_length: int,
    prefix_fmt: str = "a photo of a {cls} which",
    batch_size: int = 256,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode a multi‑prompt‑per‑class library → per‑class *prompt* embeddings.

    Returns: Tensor shape **[C, N, D]** where
      *C* = number of classes,
      *N* = prompts per class (variable, e.g. 3 here),
      *D* = text embedding dim.
    """
    all_prompts: List[str] = []
    class_offsets: List[int] = []

    # 1. Build prompt list
    for cname in class_names:
        key = _normalize_key(cname)
        descs = prompt_dict.get(key, [cname])
        prompts = build_shp_prompts(cname, descs, prefix_fmt=prefix_fmt)
        class_offsets.append(len(all_prompts))
        all_prompts.extend(prompts)
    class_offsets.append(len(all_prompts))

    # 2. Tokenize in manageable chunks
    token_chunks = [
        tokenize(all_prompts[i : i + batch_size], context_length=context_length)
        for i in range(0, len(all_prompts), batch_size)
    ]
    tokens = torch.cat(token_chunks, dim=0).to(device)  # [total_prompts, ctx]

    # 3. Encode
    feats = text_encoder(tokens)  # [total_prompts, D]

    # 4. Optional global normalization
    if normalize:
        feats = F.normalize(feats, dim=-1)

    # 5. Slice back into class‑wise tensors
    class_embeds: List[torch.Tensor] = []
    for i in range(len(class_names)):
        s, e = class_offsets[i], class_offsets[i + 1]
        part_feats = feats[s:e]  # (N_i, D)
        if normalize:
            part_feats = F.normalize(part_feats, dim=-1)
        class_embeds.append(part_feats)

    return torch.stack(class_embeds, dim=0)  # [C, N, D]
