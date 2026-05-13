import torch
import torch.nn as nn


class TrustRegionPenalty:
    """Quadratic anchor penalty: ``L_tr = 0.5 * sum_i ||theta_i - theta_0_i||^2``.

    Snapshots the module's trainable parameters at construction time using
    ``.detach().clone()``. The detach is critical: without it the snapshot
    would alias the live parameter tensors and the penalty would stay at
    zero for the entire run.

    Apply this penalty ONLY to the adapter, never to the classifier. The
    classifier has no meaningful initialization to anchor to (it starts
    from torch's default Linear init), so anchoring it would just freeze
    a random head.
    """

    def __init__(self, module: nn.Module):
        self._module = module
        self._anchors = {
            name: p.detach().clone()
            for name, p in module.named_parameters()
            if p.requires_grad
        }
        if not self._anchors:
            raise ValueError(
                "TrustRegionPenalty requires the module to have at least one "
                "parameter with requires_grad=True."
            )

    def __call__(self) -> torch.Tensor:
        total = None
        for name, p in self._module.named_parameters():
            anchor = self._anchors.get(name)
            if anchor is None:
                continue
            term = (p - anchor).pow(2).sum()
            total = term if total is None else total + term
        return 0.5 * total

    @property
    def num_anchored_params(self) -> int:
        return len(self._anchors)


def split_param_groups(
    adapter: nn.Module,
    classifier: nn.Module,
    lr: float,
    classifier_wd: float = 1e-4,
):
    """Build AdamW param groups: adapter has weight_decay=0 (the trust-region
    penalty replaces it), classifier carries its own weight decay."""
    return [
        {"params": list(adapter.parameters()), "lr": lr, "weight_decay": 0.0},
        {"params": list(classifier.parameters()), "lr": lr, "weight_decay": classifier_wd},
    ]
