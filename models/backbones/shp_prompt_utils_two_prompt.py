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
You are a vision–language prompt engineer and a computer vision expert in urban street scenes. Your goal is to generate CLIP-friendly textual descriptions that stay fully visual, encourage part-level attention, and remain valid under occlusion, truncation, and viewpoint changes.

For each object class below, write exactly 3 short English sentences (each ≤ 10 words). The 3 sentences must form this visual chain:
1) whole / global cue → 2) key visible part → 3) border or context fragment.

GLOBAL RULES (APPLY TO EVERY SENTENCE)
1. Describe ONLY what can be visually observed.
   - DO NOT write meta or evaluative phrases such as “valid even when truncated”, “this still counts”, “this is correct”.
   - Instead, describe the actually VISIBLE remainder, e.g. “lower edge visible”, “short segment near curb”, “narrow gap between cars”.

2. Always ALLOW occlusion or truncation.
   - Use visual guards like: “may be partly hidden by cars/people/vegetation/fences/poles”, “if visible”, “when not occluded”, “partial section visible”.
   - These guards must look visual, not logical.

3. Mention ONE external feature typical for that class.
   - Example: flat road surface, paved sidewalk strip, tall building facade, vertical fence bars, cylindrical pole, signal head, sign face, leaf cluster, long vehicle body.

4. Name EXACTLY ONE specific part in the sentence.
   - Example parts: curb edge, side window, wheel, pole top, sign face, signal head, door panel, cargo box, passenger windows, train door, bike handlebar.

5. Include at least ONE geometric or boundary cue.
   - Examples: straight line, curved edge, rectangular frame, round plate, vertical bar, long strip, irregular outline, narrow band.

6. Use at least ONE spatial word.
   - Examples: above, below, beside, between, near, in front of, behind, on top of, at the bottom.

7. Make the sentence UNIQUE to that class in street scenes.
   - DO NOT reuse the same generic ending like “near sidewalk” or “partial area visible” for many classes.
   - Tailor the part and the nearby object to that class.

8. Prefer common, CLIP-friendly vocabulary.
   - No brands, no rare street furniture names, no location-specific names.

9. If a contextual object (e.g. intersection, bus stop, tracks) is not always visible in typical datasets, DO NOT hard-code it.
   - Say “above road”, “beside sidewalk”, “near pole”, “above vehicles”, or “on tracks” instead of “above road intersection”.
   - Always pick a context that is often visible for that class.

SENTENCE ROLES

Sentence 1 — OBJECT-LEVEL, WITH OCCLUSION
- Purpose: declare the class, give its usual shape/placement, and allow occlusion.
- Must mention a typical outline or extent.
- Must mention a realistic street occluder when natural (cars, people, vegetation, fences, poles).
- Patterns:
  - “A [class] with [outline], may be partly hidden by cars.”
  - “A [class] beside road, irregular outline, may be behind vegetation.”
  - “Tall [class] facing street, partly covered by poles or signs.”

Sentence 2 — KEY VISIBLE PART, SPATIAL, GEOMETRIC
- Purpose: move attention to a diagnostic part that distinguishes this class from similar ones.
- Must place the part with a spatial word.
- Must add a shape/boundary cue.
- Must guard with “if visible” or “when not occluded”.
- Patterns:
  - “Visible [part] above [other part], rectangular edge, if visible.”
  - “Signal head on pole, round front, when not occluded.”
  - “Side windows along body, straight frames, if visible.”

Sentence 3 — BORDER / CONTEXT FRAGMENT (VISUAL, NOT META)
- Purpose: describe what remains visible when the object is truncated or partly blocked.
- MUST describe an actually visible fragment: edge, lower strip, top line, narrow gap, wheel area, base at curb.
- MUST reference a nearby, likely-visible street element: road, sidewalk, building, fence, sky, tracks, ground, curb.
- MUST NOT use meta phrases like “valid even when truncated.”
- Patterns:
  - “Lower fence segment near sidewalk, short vertical bars visible.”
  - “Roof line below sky between buildings, narrow strip visible.”
  - “Wheels at road level below body, partial circle visible.”
  - “Base of pole at curb, small section visible.”
  - “Ground edge near pavement, uneven strip visible.”

CLASS-DISAMBIGUATION RULES (IMPORTANT)

1. Building vs Wall vs Fence
   - Building: multi-level, has windows above street, vertical facade.
   - Wall: continuous flat barrier, NO windows, often beside sidewalk or terrain.
   - Fence: not solid, has gaps / slats / vertical bars, often near sidewalk or vegetation.
   - Enforce this in the parts you mention.

2. Traffic Light vs Traffic Sign
   - Traffic light: signal heads / lights / lenses on a structure, usually above road.
   - Traffic sign: flat sign face / plate on pole, faces road, often at sidewalk level.
   - Do NOT call a traffic sign “black housing above intersection”.
   - Do NOT call a traffic light “metal surface above sidewalk”.

3. Road vs Terrain vs Sidewalk
   - Road: flat, for vehicles, lane markings, beside sidewalk.
   - Sidewalk: paved or tiled, for pedestrians, has curb above road.
   - Terrain: natural, uneven, soil/grass, near road or vegetation.
   - Reflect this in geometric and context words.

4. Vehicle subclasses (car, truck, bus, train, motorcycle, bicycle, rider)
   - Always mention their diagnostic parts:
     - Car: side windows above doors; wheels below body; curved/compact body.
     - Truck: tall cab + large cargo box; rear/dual wheels; higher than car.
     - Bus: long/high body; many side windows in a row; lower panels near curb.
     - Train: on tracks; doors between windows; lower carriage above rails.
     - Motorcycle/Bicycle: handlebar above front wheel; two wheels; thin frame.
     - Rider: human torso + hands near handlebar; feet on pedals/pegs.
   - Avoid vague sentences like “Large body above road, valid even when truncated.”
   - Always put the diagnostic part in sentence 2 or 3.

OUTPUT RULES
- Output as a bullet list with class.
- For each class, output 3 sentences as 3 bullets, in the order (whole → part → border).
- Each sentence ≤ 10 words.
- No extra text, no explanations, no class names outside the bullets.

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
        "Top asphalt strip runs between curb edges.",
        "Near lane boundary forms straight edge beside sidewalk.",
    ],
    "sidewalk": [
        "Upper curb line forms straight edge beside road.",
        "Near paving strip runs parallel to road boundary.",
    ],
    "building": [
        "Upper facade rectangle rises above sidewalk edge.",
        "Middle window grid aligns between wall columns.",
    ],
    "wall": [
        "Top concrete slab forms straight edge above road.",
        "Middle panel strip runs along sidewalk boundary.",
    ],
    "fence": [
        "Upper rail line forms thin grid beside road.",
        "Lower post row stands along sidewalk edge.",
    ],
    "pole": [
        "Upper shaft cylinder stands beside sidewalk edge.",
        "Lower base circle sits on road surface.",
    ],
    "traffic light": [
        "Top lamp box forms square above pole shaft.",
        "Middle lens circle faces toward road boundary.",
    ],
    "traffic sign": [
        "Upper plate rectangle mounts above pole shaft.",
        "Middle icon circle centers within plate boundary.",
    ],
    "vegetation": [
        "Upper canopy mass forms irregular edge against sky.",
        "Lower trunk column stands beside terrain boundary.",
    ],
    "terrain": [
        "Near soil patch forms uneven strip beside road.",
        "Middle grass area spreads along sidewalk edge.",
    ],
    "sky": [
        "Upper blue field forms broad curve above building roof.",
        "Far cloud band drifts across sky boundary.",
    ],
    "person": [
        "Upper head circle sits above torso column.",
        "Lower leg pair stands on road surface.",
    ],
    "rider": [
        "Upper helmet dome rests above rider torso.",
        "Lower wheel pair rolls along road boundary.",
    ],
    "car": [
        "Upper roof arc spans above window row.",
        "Lower wheel pair rolls along road edge.",
    ],
    "truck": [
        "Upper cargo box forms long rectangle above wheels.",
        "Lower axle pair aligns beneath cargo boundary.",
    ],
    "bus": [
        "Upper window band forms long strip above wheels.",
        "Lower door panel aligns beside sidewalk edge.",
    ],
    "train": [
        "Upper carriage rectangle links along track boundary.",
        "Lower wheel row runs on rail edge.",
    ],
    "motorcycle": [
        "Upper handlebar line extends above fuel tank.",
        "Lower wheel pair aligns along road boundary.",
    ],
    "bicycle": [
        "Upper handlebar arc curves above frame triangle.",
        "Lower wheel pair rolls beside road edge.",
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
