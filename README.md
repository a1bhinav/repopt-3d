# RepOpt-3D

Lightweight optimization-guided adaptation of pretrained 2D features for 3D scene understanding.

## Status
Reproducing OpenScene baseline — fusion mode, Matterport3D 21-class test set.

## Reproduction Plan

**Target:** OpenScene fusion-mode evaluation on Matterport3D test set (21-class mIoU).

**Why Matterport instead of ScanNet:** The ScanNet OpenSeg fused features are a single 234.8 GB train+val zip with no val-only option. Matterport ships a separate test-set zip at 66.7 GB, which fits our 150 GB storage budget.

**Why fusion mode:** Reads the pre-fused 2D OpenSeg features directly — no checkpoint loading, no MinkowskiEngine model instantiation at eval time. The simplest path to a real mIoU number.

**Expected output:** mIoU on Matterport 21-class test set, logged to `out/matterport_openseg/`.

**Data required:**
- `data/matterport_3d/test/*.pth` — 3D point clouds with GT labels
- `data/matterport_multiview_openseg_test/` — fused OpenSeg features (66.7 GB)

**Eval command:**
```bash
sh run/eval.sh out/matterport_openseg \
    config/matterport/ours_openseg_pretrained.yaml \
    fusion
```

See [PLAN.md](PLAN.md) for the full phased plan.

## Structure
- `third_party/openscene/` — upstream OpenScene repo (submodule)
- `src/` — our adapter, losses, and training code (TBD)
- `configs/` — our experiment configs (TBD)
