import torch

from src.adapter import ResidualMLPAdapter
from src.classifier import LinearClassifier
from src.losses import TrustRegionPenalty, split_param_groups


def test_zero_at_init():
    adapter = ResidualMLPAdapter(d_in=128)
    tr = TrustRegionPenalty(adapter)
    assert abs(tr().item()) < 1e-9


def test_positive_after_perturb():
    adapter = ResidualMLPAdapter(d_in=128)
    tr = TrustRegionPenalty(adapter)
    with torch.no_grad():
        for p in adapter.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    assert tr().item() > 0


def test_split_param_groups():
    adapter = ResidualMLPAdapter(d_in=128)
    clf = LinearClassifier(128, 21)
    groups = split_param_groups(adapter, clf, lr=1e-3)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 0.0
    assert groups[1]["weight_decay"] > 0
