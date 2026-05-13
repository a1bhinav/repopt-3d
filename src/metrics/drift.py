import torch
import torch.nn as nn


def _run_predictions(
    adapter: nn.Module,
    classifier: nn.Module,
    feature_loader,
    device: torch.device,
) -> torch.Tensor:
    adapter.eval()
    classifier.eval()
    preds = []
    with torch.no_grad():
        for batch in feature_loader:
            x = batch["features"] if isinstance(batch, dict) else batch
            x = x.to(device)
            logits = classifier(adapter(x))
            preds.append(logits.argmax(dim=-1).to(torch.int32).cpu())
    return torch.cat(preds)


def snapshot_baseline_preds(
    adapter: nn.Module,
    classifier: nn.Module,
    feature_loader,
    device: torch.device,
) -> torch.Tensor:
    """Run adapter+classifier over the loader and return concatenated argmax
    predictions as a CPU int32 tensor of shape (N,)."""
    return _run_predictions(adapter, classifier, feature_loader, device)


def compute_drift(
    adapter: nn.Module,
    classifier: nn.Module,
    feature_loader,
    baseline_preds: torch.Tensor,
    device: torch.device,
    num_classes: int = None,
) -> dict:
    """Re-run inference and compare to ``baseline_preds`` element-wise.

    Returns ``{'fraction_changed', 'n_changed', 'n_total'}``. Raises
    ``ValueError`` if the new predictions don't match the baseline shape —
    that almost always means the loader iterated in a different order
    (shuffling, dropped batch, etc.) between snapshot and re-evaluation,
    which would silently invalidate the drift number.
    """
    final_preds = _run_predictions(adapter, classifier, feature_loader, device)
    if final_preds.shape != baseline_preds.shape:
        raise ValueError(
            f"Drift comparison requires equal-shape predictions, got "
            f"baseline {tuple(baseline_preds.shape)} vs final {tuple(final_preds.shape)}. "
            f"This usually means the feature_loader iterated in a different "
            f"order between snapshot and re-evaluation."
        )
    n_total = int(baseline_preds.numel())
    n_changed = int((final_preds != baseline_preds).sum().item())
    return {
        "fraction_changed": n_changed / n_total,
        "n_changed": n_changed,
        "n_total": n_total,
    }
