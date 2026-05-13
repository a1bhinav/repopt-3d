# RepOpt-3D

Lightweight optimization-guided adaptation of pretrained 2D features for 3D scene understanding.

## Team Setup

New to the project? Follow [SETUP.md](SETUP.md) to replicate the repo and verify your environment.

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

**Eval command (fusion baseline, no adapter):**
```bash
cd third_party/openscene
sh run/eval.sh out/matterport_openseg config/matterport/ours_openseg_pretrained.yaml fusion
```

**Full baseline → CE → H1 → H2 → H3 (train + eval each adapter):** set `MP_ROOT` and `FUSED_ROOT`, then `cd third_party/openscene && bash scripts/run_repopt_five_runs.sh` (see script header).

See [PLAN.md](PLAN.md) for the full phased plan.

## Structure
- `third_party/openscene/` — upstream OpenScene repo (submodule) with Matterport adapter training (`run/train_adapter.py`) and per-hypothesis YAMLs (`config/matterport/adapter_baseline.yaml`, `adapter_h1.yaml`, `adapter_h2.yaml`, `adapter_h3.yaml`, …).
- `src/` — standalone supervised-adapter + trust-region utilities from the h3 branch (unit tests under `tests/`).
- `configs/` — reserved for future top-level experiment configs (optional).
