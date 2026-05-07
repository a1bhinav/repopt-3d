import torch
import torch.nn as nn


def compute_miou(
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int = -1,
) -> dict:
    """Per-class IoU and mean IoU under the Matterport convention.

    The mean is taken only over classes that actually appear in
    ``labels`` — classes absent from the ground truth do not pull mIoU
    down toward zero. Their per-class IoU is reported as NaN so callers
    can tell "absent" from "predicted but wrong".

    Confusion matrix is built with ``bincount`` on a flattened index
    ``true * num_classes + pred`` (after dropping ``ignore_label`` rows).
    For class ``c``, ``IoU_c = conf[c, c] / (row_sum_c + col_sum_c - conf[c, c])``.
    """
    valid = labels != ignore_label
    valid_labels = labels[valid].long()
    valid_preds = preds[valid].long()
    n_evaluated = int(valid.sum().item())

    flat = valid_labels * num_classes + valid_preds
    conf = torch.bincount(flat, minlength=num_classes ** 2).view(
        num_classes, num_classes
    )

    diag = conf.diag().float()
    row_sums = conf.sum(dim=1).float()
    col_sums = conf.sum(dim=0).float()
    union = row_sums + col_sums - diag

    per_class_iou = torch.full((num_classes,), float("nan"))
    has_union = union > 0
    per_class_iou[has_union] = diag[has_union] / union[has_union]

    present_in_labels = row_sums > 0
    if present_in_labels.any():
        miou = float(per_class_iou[present_in_labels].mean().item())
    else:
        miou = float("nan")

    return {
        "mIoU": miou,
        "per_class_iou": per_class_iou,
        "n_evaluated": n_evaluated,
    }


def evaluate_miou(
    adapter: nn.Module,
    classifier: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    batch_size: int = 8192,
    ignore_label: int = -1,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Run adapter+classifier over ``features`` in batches, argmax to
    predictions, then call :func:`compute_miou`."""
    adapter.eval()
    classifier.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, features.shape[0], batch_size):
            x = features[i : i + batch_size].to(device)
            logits = classifier(adapter(x))
            preds.append(logits.argmax(dim=-1).cpu())
    preds = torch.cat(preds)
    return compute_miou(preds, labels, num_classes, ignore_label=ignore_label)
