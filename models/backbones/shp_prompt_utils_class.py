# mmseg/models/backbones/shp_prompt_utils_class.py
#
# Class-name prompt utility for HPI ablations:
#   SHP region prompts vs. conventional class-level prompts under the same HPI.
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F

from .utils import tokenize


def _normalize_key(k: str) -> str:
    k = k.lower().strip()
    k = k.replace("_", " ")
    return " ".join(k.split())


def build_shp_prompt_dict() -> Dict[str, List[str]]:
    """Keep the same public API as shp_prompt_utils.py."""
    return {}


def build_shp_prompts(
    cls_name: str,
    desc_list: Sequence[str],
    prefix_fmt: str = "a photo of a {cls}",
) -> List[str]:
    cls_token = _normalize_key(cls_name)
    return [prefix_fmt.format(cls=cls_token)]


@torch.no_grad()
def encode_shp_prompt_library(
    text_encoder: torch.nn.Module,
    class_names: Sequence[str],
    prompt_dict: Dict[str, Sequence[str]],
    device: torch.device,
    context_length: int,
    prefix_fmt: str = "a photo of a {cls}",
    batch_size: int = 256,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode one conventional class-level prompt per class.

    Returns: Tensor shape [C, 1, D].
    """
    all_prompts: List[str] = []
    class_offsets: List[int] = []

    for cname in class_names:
        class_offsets.append(len(all_prompts))
        all_prompts.extend(build_shp_prompts(cname, prompt_dict.get(_normalize_key(cname), []), prefix_fmt))
    class_offsets.append(len(all_prompts))

    token_chunks = [
        tokenize(all_prompts[i : i + batch_size], context_length=context_length)
        for i in range(0, len(all_prompts), batch_size)
    ]
    tokens = torch.cat(token_chunks, dim=0).to(device)

    feats = text_encoder(tokens)
    if normalize:
        feats = F.normalize(feats, dim=-1)

    class_embeds: List[torch.Tensor] = []
    for i in range(len(class_names)):
        s, e = class_offsets[i], class_offsets[i + 1]
        part_feats = feats[s:e]
        if normalize:
            part_feats = F.normalize(part_feats, dim=-1)
        class_embeds.append(part_feats)

    return torch.stack(class_embeds, dim=0)
