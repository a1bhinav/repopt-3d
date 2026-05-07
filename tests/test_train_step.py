import torch

from src.adapter import ResidualMLPAdapter
from src.classifier import LinearClassifier
from src.data import make_synthetic_scenes, per_scene_split
from src.train import train_step


def test_loss_decreases_on_synthetic():
    """End-to-end check: on learnable synthetic data the loss must drop."""
    torch.manual_seed(0)
    scenes = make_synthetic_scenes(
        num_scenes=3, points_per_scene=500, d_feat=64, num_classes=5
    )
    for s in scenes:
        s["labeled_mask"] = per_scene_split(s["labels"], fraction=0.10, seed=0)

    adapter = ResidualMLPAdapter(d_in=64)
    clf = LinearClassifier(d_in=64, num_classes=5)
    opt = torch.optim.AdamW(
        list(adapter.parameters()) + list(clf.parameters()),
        lr=1e-2,
    )

    feats = torch.cat([s["features"] for s in scenes], dim=0)
    lbls = torch.cat([s["labels"] for s in scenes], dim=0)
    lmask = torch.cat([s["labeled_mask"] for s in scenes], dim=0)

    initial_loss = train_step(adapter, clf, opt, feats, lbls, lmask)["sup_loss"]
    for _ in range(50):
        result = train_step(adapter, clf, opt, feats, lbls, lmask)
    final_loss = result["sup_loss"]

    assert final_loss < 0.7 * initial_loss, (
        f"loss did not decrease: {initial_loss} -> {final_loss}"
    )
