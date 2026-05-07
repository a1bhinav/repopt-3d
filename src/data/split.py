import math

import torch


def per_scene_split(
    labels: torch.Tensor,
    fraction: float = 0.05,
    seed: int = 0,
    class_balanced: bool = True,
    ignore_label: int = -1,
) -> torch.BoolTensor:
    """Pick a labeled subset for one scene.

    Returns a bool mask the same shape as ``labels``, True where a point is
    selected to be 'labeled' for transductive training.

    - Reproducible: uses ``torch.Generator`` seeded with ``seed``.
    - Points where ``labels == ignore_label`` are never selected.
    - ``class_balanced=True``: per class ``c``, pick ``ceil(fraction * n_c)``
      of the ``n_c`` points with that class. Empty classes are skipped.
    - ``class_balanced=False``: pick ``floor(fraction * n_valid)`` random
      points from the union of valid points (clamped to >= 1 if any valid
      points exist).
    - Always returns at least 1 selected point if any valid points exist.
    """
    mask = torch.zeros_like(labels, dtype=torch.bool)
    valid = labels != ignore_label
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return mask

    g = torch.Generator(device="cpu").manual_seed(seed)

    if class_balanced:
        unique_classes = torch.unique(labels[valid])
        for c in unique_classes.tolist():
            idx = (labels == c).nonzero(as_tuple=False).squeeze(-1)
            n_c = idx.numel()
            if n_c == 0:
                continue
            k = max(1, math.ceil(fraction * n_c))
            perm = torch.randperm(n_c, generator=g)
            mask[idx[perm[:k]]] = True
    else:
        valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
        k = max(1, math.floor(fraction * n_valid))
        perm = torch.randperm(n_valid, generator=g)
        mask[valid_idx[perm[:k]]] = True

    return mask
