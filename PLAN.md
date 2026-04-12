# Reproduction Plan

## Goal

Reproduce the OpenScene fusion-mode baseline on the Matterport3D 21-class test set and obtain a published mIoU number. This is the first milestone before building our own adapter on top.

## Dataset and budget rationale

| Option | Features zip | Val/test only? | Size |
|---|---|---|---|
| ScanNet + OpenSeg | `scannet_multiview_openseg.zip` | No — train+val monolithic | 234.8 GB |
| **Matterport + OpenSeg** | `matterport_multiview_openseg_test.zip` | **Yes — test set only** | **66.7 GB** |

ScanNet has no separate val-only download; the entire 234.8 GB must be fetched. Matterport exposes a dedicated test-set features zip. At a 150 GB storage budget, Matterport is the only viable choice.

## Why fusion mode

- Reads per-point multi-view fused OpenSeg features directly from disk.
- No checkpoint download, no `MinkowskiEngine` model loading at eval time.
- `evaluate.py` explicitly skips all weight loading when `feature_type == 'fusion'`.
- Matterport GT labels ship with the processed 3D data → `eval_iou` defaults to `True` → real mIoU is computed.

## Phases

### Phase 0 — Environment
1. Create conda env `openscene` (Python 3.8).
2. Install PyTorch 1.7.1+cu110.
3. Install MinkowskiEngine (required by the dataloader even in fusion mode — `SparseTensor` is imported unconditionally).
4. `pip install -r third_party/openscene/requirements.txt`.

### Phase 1 — Data download
All commands run from `third_party/openscene/`.

```bash
# Matterport 3D point clouds + GT labels (21-class)
bash scripts/download_dataset.sh          # enter: 2

# Matterport test-set fused OpenSeg features (66.7 GB)
bash scripts/download_fused_features.sh   # enter: 3
```

Expected layout after download:
```
third_party/openscene/data/
├── matterport_3d/
│   ├── train/
│   ├── val/
│   └── test/        ← eval reads only this
└── matterport_multiview_openseg_test/   ← 66.7 GB
```

### Phase 2 — Verify paths

Config file: `config/matterport/ours_openseg_pretrained.yaml`

Key settings to confirm match the downloaded layout:

| Config key | Expected value |
|---|---|
| `DATA.data_root` | `data/matterport_3d` |
| `DATA.data_root_2d_fused_feature` | `data/matterport_multiview_openseg_test` |
| `TEST.split` | `test` |
| `TEST.eval_iou` | *(absent → defaults to True)* |
| `TEST.feature_type` | `ensemble` *(overridden to `fusion` on CLI)* |

### Phase 3 — Evaluation

```bash
cd third_party/openscene

sh run/eval.sh out/matterport_openseg \
    config/matterport/ours_openseg_pretrained.yaml \
    fusion
```

Output lands in `out/matterport_openseg/`:
- `eval-YYYYMMDD_HHMM.log` — mIoU printed at the end
- `result_eval/` — per-scene PLY visualizations

### Phase 4 — Record baseline

Copy the final mIoU line from the log and record it here as the fusion-mode Matterport 21-class baseline before any of our modifications.

| Mode | Labelset | mIoU |
|---|---|---|
| fusion | Matterport 21 | TBD |
| ensemble | Matterport 21 | TBD (requires checkpoint) |

## Storage budget summary

| Item | Size |
|---|---|
| Matterport 3D point clouds | Unknown (small) |
| Matterport test fused features | 66.7 GB |
| ScanNet fused features (not downloading) | ~~234.8 GB~~ |
| Checkpoint (not needed for fusion) | — |
| **Total** | **~66.7 GB + 3D zip** |

## Known issues to watch

1. **MinkowskiEngine is still required** even in fusion mode — the dataloader imports `SparseTensor` unconditionally. The environment must have it built.
2. **`test_repeats: 5`** in the config means evaluation runs 5 times with different random voxelization seeds and averages. Expected wall-clock time is proportionally higher than a single pass.
3. **`mark_no_feature_to_unknown: True`** is set — points with no 2D feature coverage are assigned the `unknown` label rather than the nearest-neighbor class. This matches the published eval protocol.
All OpenScene deps OK
