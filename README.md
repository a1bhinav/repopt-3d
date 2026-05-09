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

**Eval command:**
```bash
sh run/eval.sh out/matterport_openseg \
    config/matterport/ours_openseg_pretrained.yaml \
    fusion
```

See [PLAN.md](PLAN.md) for the full phased plan.

## Experiments (Matterport adapt split)
1. Build adapt splits: `sbatch split_dataset.sh` (or `python scripts/split_dataset.py`) — symlinks `adapt-train` / `adapt-val` / `adapt-test` from `test/*.pth`.
2. Baseline mIoU on `adapt-test`: `sbatch run_eval_baseline.sh` or `python scripts/eval_baseline.py`.
3. Baseline on **full** official test split: `sbatch run_eval_full_test.sh` or `python scripts/eval_full_test.py` (`$MATTERPORT_3D_ROOT` / `$DATA_ROOT/matterport_3d`).
4. HPO + final `adapt-test` for CE-only: `sbatch run_hpo_ce.sh`
5. HPO + final `adapt-test` for CE + H1: `sbatch run_hpo_ce_h1.sh`

## Structure
- `scripts/` — `split_dataset.py`, `eval_baseline.py`, `eval_full_test.py`, `hpo_fusion.py`
- `third_party/openscene/` — OpenScene submodule (includes `run/train_fusion_h1.py`)
