import pytest
import torch

from src.adapter import ResidualMLPAdapter
from src.classifier import LinearClassifier
from src.metrics import compute_drift, snapshot_baseline_preds


def _make_setup(seed=0):
    torch.manual_seed(seed)
    adapter = ResidualMLPAdapter(d_in=64)
    clf = LinearClassifier(64, 5)
    feats = torch.randn(200, 64)
    loader = [feats[i : i + 50] for i in range(0, 200, 50)]
    return adapter, clf, loader


def test_zero_drift_unchanged_model():
    adapter, clf, loader = _make_setup()
    baseline = snapshot_baseline_preds(adapter, clf, loader, torch.device("cpu"))
    d = compute_drift(adapter, clf, loader, baseline, torch.device("cpu"))
    assert d["fraction_changed"] == 0.0
    assert d["n_total"] == 200


def test_positive_drift_after_perturb():
    adapter, clf, loader = _make_setup()
    baseline = snapshot_baseline_preds(adapter, clf, loader, torch.device("cpu"))
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(torch.randn_like(p) * 0.5)
        for p in clf.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    d = compute_drift(adapter, clf, loader, baseline, torch.device("cpu"))
    assert d["fraction_changed"] > 0
    assert d["n_total"] == 200


def test_shape_mismatch_raises():
    adapter, clf, loader = _make_setup()
    bad_baseline = torch.zeros(199, dtype=torch.int32)
    with pytest.raises(ValueError):
        compute_drift(adapter, clf, loader, bad_baseline, torch.device("cpu"))
