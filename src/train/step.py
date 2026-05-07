import torch
import torch.nn as nn
import torch.nn.functional as F


def train_step(
    adapter: nn.Module,
    classifier: nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    labeled_mask: torch.Tensor,
    trust_region=None,
    lambda_tr: float = 0.0,
) -> dict:
    """One forward + backward + optimizer step over a batch.

    Forward runs adapter -> classifier on every point. The supervised loss
    is cross-entropy restricted to ``labeled_mask``; the optional
    trust-region term is added when ``trust_region`` is provided and
    ``lambda_tr > 0``. If no points are labeled and no trust-region term
    is active, the step is a no-op (returns zeros) rather than crashing.
    """
    adapter.train()
    classifier.train()
    optimizer.zero_grad()

    logits = classifier(adapter(features))

    n_labeled = int(labeled_mask.sum().item())
    if n_labeled > 0:
        sup_loss = F.cross_entropy(logits[labeled_mask], labels[labeled_mask])
    else:
        sup_loss = torch.zeros((), device=features.device)

    if trust_region is not None and lambda_tr > 0:
        tr_loss = lambda_tr * trust_region()
    else:
        tr_loss = torch.zeros((), device=features.device)

    total = sup_loss + tr_loss

    if total.requires_grad:
        total.backward()
        optimizer.step()

    return {
        "loss": float(total.item()),
        "sup_loss": float(sup_loss.item()),
        "tr_loss": float(tr_loss.item()),
        "n_labeled": n_labeled,
    }
