#!/usr/bin/env python3
"""
Fusion baseline evaluation on **adapt-test** (no training).

Runs OpenScene ``evaluate.py`` with ``feature_type fusion`` on the adapt split tree
produced by ``split_dataset.py`` (default: ``$SPLIT_ROOT`` → ``adapt-test/``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    repo = _repo_root()
    opense = repo / "third_party/openscene"
    eval_py = opense / "run" / "evaluate.py"
    default_cfg = opense / "config" / "matterport" / "ours_openseg_pretrained.yaml"

    parser = argparse.ArgumentParser(description="Fusion baseline mIoU on adapt-test.")
    parser.add_argument(
        "--split-root",
        type=Path,
        default=None,
        help="Directory with adapt-test/ (default: $SPLIT_ROOT)",
    )
    parser.add_argument(
        "--feat-root",
        type=Path,
        default=None,
        help="Fused OpenSeg features (default: $SRC_FEAT)",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--test-repeats", type=int, default=int(os.environ.get("TEST_REPEATS", "1")))
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra evaluate.py key-value overrides after --",
    )
    args = parser.parse_args()

    data_root = Path(os.environ.get("DATA_ROOT", "/project/pi_hongyu_umass_edu/mdhassan/3d"))
    seed = int(os.environ.get("SEED", "3407"))
    split_root = args.split_root or Path(
        os.environ.get("SPLIT_ROOT", data_root / f"matterport_3d_adapt_split_seed{seed}")
    )
    feat_root = args.feat_root or Path(
        os.environ.get("SRC_FEAT", data_root / "matterport_multiview_openseg_test")
    )
    cfg_path = args.config or default_cfg

    eval_split = "adapt-test"
    if not (split_root / eval_split).is_dir():
        print(f"Missing {split_root}/{eval_split} — run scripts/split_dataset.py first.", file=sys.stderr)
        return 1

    if args.out_dir is not None:
        out_root = args.out_dir
    elif os.environ.get("OUT_DIR"):
        out_root = Path(os.environ["OUT_DIR"])
    else:
        out_root = opense / "out" / "baseline_adapt_test"
    out_root.mkdir(parents=True, exist_ok=True)
    result_dir = out_root / f"result_eval_baseline_{eval_split}"

    # Correct CLIP label space for adapt split: ``ours_openseg_pretrained.yaml`` sets ``labelset``;
    # OpenScene’s merge only allows keys that exist in the YAML (see config/matterport/...).
    cmd: list[str] = [
        sys.executable,
        "-u",
        str(eval_py),
        "--config",
        str(cfg_path.resolve()),
        "data_root",
        str(split_root.resolve()),
        "data_root_2d_fused_feature",
        str(feat_root.resolve()),
        "feature_type",
        "fusion",
        "vis_input",
        "True",
        "vis_pred",
        "True",
        "vis_gt",
        "True",
        "split",
        eval_split,
        "test_repeats",
        str(args.test_repeats),
        "save_folder",
        str(result_dir.resolve()),
    ]
    if args.extra:
        cmd.extend(args.extra[1:] if args.extra[0] == "--" else args.extra)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(opense.resolve())
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=str(opense), env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
