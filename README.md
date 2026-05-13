# RepOpt-3D

Lightweight optimization-guided adaptation of pretrained 2D features for 3D scene understanding.

## Team setup

New to the project? Follow [SETUP.md](SETUP.md) to clone the repo (including submodules), create the `**repopt**` conda environment (Python 3.11 + PyTorch), install OpenScene dependencies, and run a quick import check.

## Status

Reproducing and extending the **OpenScene** baseline — **fusion mode** on **Matterport3D** (21-class test mIoU), plus **adapter** experiments (low-label regime) implemented in the OpenScene fork under `third_party/openscene/`.

## Reproduction plan (summary)

**Target:** OpenScene fusion-mode evaluation on the Matterport3D **test** split (21-class mIoU).

**Why Matterport instead of ScanNet:** The ScanNet OpenSeg fused features ship as a very large train+val archive; Matterport provides a **separate test-set** fused-feature release (~66.7 GB), which fits typical storage budgets.

**Why fusion mode:** Reads **pre-fused** 2D OpenSeg features from disk — no MinkowskiEngine 3D UNet at eval time in the default `**repopt`** environment. That is the fastest path to a real mIoU number.

**Expected output:** Metrics and logs under `third_party/openscene/out/` (or paths you pass as `save_folder`).

**Data layout (conceptual):**

- `matterport_3d/train/*.pth`, `matterport_3d/test/*.pth` — 3D points + labels  
- `matterport_multiview_openseg_test/` — fused OpenSeg `.pt` per scene

**Eval (from the OpenScene tree):**

```bash
cd third_party/openscene
mkdir -p out/matterport_openseg
sh run/eval.sh out/matterport_openseg \
  config/matterport/ours_openseg_pretrained.yaml \
  fusion
```

## Repository layout


| Path                     | Role                                                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `third_party/openscene/` | **OpenScene fork** (git submodule): fusion eval, adapter train/eval, YAML configs, Unity/H2 shell scripts. |
| `src/`, `configs/`       | Reserved for project-native code and configs.                                                              |
|                          |                                                                                                            |
|                          |                                                                                                            |


## Key files (OpenScene subtree)

All paths below are under `**third_party/openscene/`** unless noted.


| What                                                    | Where                                                                                                   |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Fusion / baseline evaluation                            | `run/evaluate.py`                                                                                       |
| Adapter training / evaluation                           | `run/train_adapter.py`, `run/evaluate_adapter.py`                                                       |
| Matterport fusion + adapter YAMLs                       | `config/matterport/` (`ours_openseg_pretrained.yaml`, `eval_fusion_baseline.yaml`, `adapter_*.yaml`, …) |
| ScanNet fusion config (small val)                       | `config/scannet/ours_openseg_pretrained.yaml`                                                           |
| H2 pipeline (prepare → baseline conf → trains)          | `scripts/run_h2_pipeline.sh` + `scripts/h2_pipeline_env_unity.sh`                                       |
| **Unity cluster:** Slurm, `/work` data paths, RAM notes | `.claude/RUNNING_ON_UNITY.md`                                                                           |
| Adapter losses, protocol, results table                 | `README.md` in this subtree (not the repo root README)                                                  |
| Upstream OpenScene (datasets, paper pointers)           | `README_openscene.md`, `installation.md`                                                                |


**Convention:** run Python from `third_party/openscene` with `PYTHONPATH=.` (the `run/eval.sh` helper does this). The default `**repopt`** env does **not** include MinkowskiEngine — use `**feature_type fusion`** (or the adapter stack on frozen fused features), not `distill` / `ensemble`, unless you install MinkowskiEngine yourself.

## Submodule

`third_party/openscene` is a **submodule**. Changes to OpenScene belong in commits **inside** that directory (on the submodule’s remote/branch); the parent repo records the updated submodule commit when you bump the pointer.