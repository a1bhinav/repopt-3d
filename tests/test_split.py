import torch

from src.data import per_scene_split


def test_split_fraction_unbalanced():
    labels = torch.zeros(10000, dtype=torch.long)
    mask = per_scene_split(labels, fraction=0.05, class_balanced=False, seed=0)
    n = mask.sum().item()
    assert 490 <= n <= 510


def test_split_class_balanced():
    labels = torch.cat(
        [torch.zeros(800), torch.ones(150), torch.full((50,), 2)]
    ).long()
    mask = per_scene_split(labels, fraction=0.05, class_balanced=True, seed=0)
    counts = [(labels[mask] == c).sum().item() for c in range(3)]
    assert 38 <= counts[0] <= 42
    assert 6 <= counts[1] <= 10
    assert 1 <= counts[2] <= 5


def test_split_ignores_unlabeled():
    labels = torch.full((1000,), -1, dtype=torch.long)
    labels[:100] = 0
    mask = per_scene_split(labels, fraction=0.5, ignore_label=-1, seed=0)
    assert (labels[mask] != -1).all()
    assert 45 <= mask.sum().item() <= 55


def test_split_seed_deterministic():
    labels = torch.randint(0, 5, (1000,))
    m1 = per_scene_split(labels, seed=42)
    m2 = per_scene_split(labels, seed=42)
    assert torch.equal(m1, m2)
