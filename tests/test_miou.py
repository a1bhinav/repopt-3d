import torch

from src.eval import compute_miou


def test_miou_perfect():
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0])
    preds = labels.clone()
    result = compute_miou(preds, labels, num_classes=3)
    assert abs(result["mIoU"] - 1.0) < 1e-6
    assert torch.allclose(result["per_class_iou"][:3], torch.ones(3), atol=1e-6)


def test_miou_zero_when_all_wrong():
    labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    preds = torch.tensor([1, 2, 3, 0, 1, 2, 3, 0])
    result = compute_miou(preds, labels, num_classes=4)
    assert result["mIoU"] == 0.0


def test_miou_ignores_ignore_label():
    labels = torch.tensor([0, 1, 2, -1, -1, -1])
    preds = torch.tensor([0, 1, 2, 99, 99, 99])
    result = compute_miou(preds, labels, num_classes=3, ignore_label=-1)
    assert abs(result["mIoU"] - 1.0) < 1e-6
    assert result["n_evaluated"] == 3


def test_miou_absent_class_is_nan_not_zero():
    labels = torch.tensor([0, 0, 1, 1])
    preds = torch.tensor([0, 0, 1, 1])
    result = compute_miou(preds, labels, num_classes=3)
    assert abs(result["mIoU"] - 1.0) < 1e-6
    assert torch.isnan(result["per_class_iou"][2])
