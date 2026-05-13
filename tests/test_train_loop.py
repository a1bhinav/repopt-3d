import torch

from src.adapter import ResidualMLPAdapter
from src.classifier import LinearClassifier
from src.data import make_synthetic_scenes, per_scene_split
from src.eval import evaluate_miou
from src.train import train_loop


def test_loop_decreases_loss():
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

    feats = torch.cat([s["features"] for s in scenes])
    lbls = torch.cat([s["labels"] for s in scenes])
    lmask = torch.cat([s["labeled_mask"] for s in scenes])

    history = train_loop(
        adapter, clf, opt, feats, lbls, lmask,
        num_epochs=5, batch_size=256, seed=0,
    )
    assert len(history) == 5
    assert history[-1]["avg_sup_loss"] < 0.7 * history[0]["avg_sup_loss"]


def test_loop_then_eval_recovers_high_miou():
    torch.manual_seed(0)
    scenes = make_synthetic_scenes(
        num_scenes=2, points_per_scene=400, d_feat=32, num_classes=4
    )
    feats = torch.cat([s["features"] for s in scenes])
    lbls = torch.cat([s["labels"] for s in scenes])
    lmask = per_scene_split(lbls, fraction=0.5, seed=0)

    adapter = ResidualMLPAdapter(d_in=32)
    clf = LinearClassifier(d_in=32, num_classes=4)
    opt = torch.optim.AdamW(
        list(adapter.parameters()) + list(clf.parameters()),
        lr=1e-2,
    )
    train_loop(
        adapter, clf, opt, feats, lbls, lmask,
        num_epochs=20, batch_size=128, seed=0,
    )

    result = evaluate_miou(adapter, clf, feats, lbls, num_classes=4, batch_size=128)
    assert result["mIoU"] > 0.9, f"got {result['mIoU']}"
