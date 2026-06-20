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
Instruction: For each class, write 3 short English sentences (≤12 words) in the same style as the examples. Initiate each sentence with a distinct regional identifier. Ensure these regions constitute a mutually exclusive and collectively exhaustive partition of the entire object morphology.

Template: [Region cue]+[component of object]+[geometry]+[relation to neighbor/boundary]. Use simple present verbs, strong geometry words, and an explicit boundary neighbor.
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
        "Near asphalt surface shows parallel lane stripe edges.",
        "Mid lane centerline runs between wheel tracks, slightly curved.",
        "Far road shoulder meets curb along straight boundary.",
    ],
    "sidewalk": [
        "Top curb edge sits above asphalt road boundary.",
        "Middle walkway slabs show square joints between tiles.",
        "Bottom gutter line runs below slabs near street.",
    ],
    "building": [
        "Top roofline forms straight edge above facade.",
        "Middle wall holds rectangular windows between corner edges.",
        "Lower doorway sits below windows along front plane.",
    ],
    "wall": [
        "Top cap edge runs above stacked brick pattern.",
        "Middle section shows rectangular blocks between mortar lines.",
        "Bottom base meets ground along continuous straight boundary.",
    ],
    "fence": [
        "Top rail forms straight line above vertical posts.",
        "Middle mesh shows square gaps between metal wires.",
        "Lower rail sits below mesh near ground.",
    ],
    "pole": [
        "Top fixture mounts above cylindrical pole shaft.",
        "Middle shaft stands between brackets, vertical outline.",
        "Bottom base plate sits below near ground edge.",
    ],
    "traffic light": [
        "Top signal head holds circular lens under hood.",
        "Middle stacked lenses align vertically between side edges.",
        "Lower mount attaches below head, near supporting pole.",
    ],
    "traffic sign": [
        "Top plate shows rectangular border above bracket.",
        "Center face displays symbol between straight edges.",
        "Bottom bracket bolts below plate onto pole.",
    ],
    "vegetation": [
        "Top crown spreads above trunk with irregular outline.",
        "Middle branches extend between leaves, curved edges.",
        "Lower trunk stands below crown near ground.",
    ],
    "terrain": [
        "Top ridge sits above slope, defining curved contour.",
        "Middle ground shows rocks between soil patches.",
        "Lower ditch lies below ridge along uneven boundary.",
    ],
    "sky": [
        "Upper sky arches above horizon, smooth curved outline.",
        "Middle sky stretches between clouds, soft edges.",
        "Lower sky rests above terrain along straight horizon.",
    ],
    "person": [
        "Head sits above shoulders, rounded hairline outline.",
        "Torso lies between arms, straight side edges.",
        "Legs extend below waist, feet near ground.",
    ],
    "rider": [
        "Helmeted head above handlebars, curved bar outline.",
        "Hands grip bars between brake levers near stem.",
        "Feet rest below seat on pedals or pegs.",
    ],
    "car": [
        "Front hood slopes below windshield, straight leading edge.",
        "Side doors sit between pillars, rectangular windows above.",
        "Rear bumper under trunk, near circular wheel arches.",
    ],
    "truck": [
        "Front cab below roof, wide rectangular grille.",
        "Side mirrors extend beside cab, near door edge.",
        "Rear cargo area behind cab above big wheels.",
    ],
    "bus": [
        "Front display above windshield, long straight roofline.",
        "Side doors between long rectangular windows.",
        "Rear engine panel below windows near large wheels.",
    ],
    "train": [
        "Front car nose below roof, curved windshield edge.",
        "Side doors align between windows along straight carriage.",
        "Wheel bogies below body near rails.",
    ],
    "motorcycle": [
        "Front fork holds circular headlight below handlebars.",
        "Fuel tank between seat and bars, curved surface.",
        "Rear wheel and chain below seat near swingarm.",
    ],
    "bicycle": [
        "Handlebars above front wheel, curved rim edge.",
        "Frame triangle between seat tube and bars.",
        "Pedals and chainring below frame near rear wheel.",
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
