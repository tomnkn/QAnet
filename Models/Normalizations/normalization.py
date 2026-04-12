import torch.nn as nn

from .layernorm import LayerNorm
from .groupnorm import GroupNorm


# Registry: norm_name -> class
normalizations = {
    "layer_norm": LayerNorm,
    "group_norm":  GroupNorm,
}


def get_norm(name: str, d_model: int, length: int, num_groups: int = 8) -> nn.Module:
    """
    Instantiate a normalization module by registry name.

    Args:
        name:       one of "layer_norm", "group_norm"
        d_model:    number of channels (C)
        length:     sequence length (L); unused for layer_norm, kept for API compatibility
        num_groups: number of groups; used only by group_norm

    Returns:
        nn.Module instance of the requested normalization.

    Shapes:
        "layer_norm" → LayerNorm(d_model)
            normalizes across channel dim per position of [B, d_model, length]
        "group_norm"  → GroupNorm(num_groups, d_model)
            normalizes over [C/G, *spatial] per group of [B, d_model, *]
    """
    if name not in normalizations:
        raise ValueError(
            f"Unknown normalization '{name}'. Available: {list(normalizations.keys())}"
        )
    if name == "layer_norm":
        return LayerNorm(d_model)
    else:  # group_norm
        return GroupNorm(num_groups, d_model)
