#!/usr/bin/env python3
"""
Fusion baseline on the **full** Matterport official **test** split (no training).

``data_root`` is the Matterport 3D root (must contain ``test/*.pth``); evaluate uses ``split=test``.
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

    parser = argparse.ArgumentParser(description="Fusion baseline mIoU on full Matterport test split.")
    parser.add_argument(
        "--matterport-root",
        type=Path,
        default=None,
        help="Directory containing test/ (default: $MATTERPORT_3D_ROOT or $DATA_ROOT/matterport_3d)",
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

    data_top = Path(os.environ.get("DATA_ROOT", "/project/pi_hongyu_umass_edu/mdhassan/3d"))
    matterport_root = args.matterport_root or Path(
        os.environ.get("MATTERPORT_3D_ROOT", data_top / "matterport_3d")
    )
    feat_root = args.feat_root or Path(
        os.environ.get("SRC_FEAT", data_top / "matterport_multiview_openseg_test")
    )
    cfg_path = args.config or default_cfg

    eval_split = "test"
    if not (matterport_root / eval_split).is_dir():
        print(f"Missing {matterport_root}/{eval_split}", file=sys.stderr)
        return 1

    if args.out_dir is not None:
        out_root = args.out_dir
    elif os.environ.get("OUT_DIR"):
        out_root = Path(os.environ["OUT_DIR"])
    else:
        out_root = opense / "out" / "baseline_full_test"
    out_root.mkdir(parents=True, exist_ok=True)
    result_dir = out_root / f"result_eval_baseline_{eval_split}"

    cmd: list[str] = [
        sys.executable,
        "-u",
        str(eval_py),
        "--config",
        str(cfg_path.resolve()),
        "data_root",
        str(matterport_root.resolve()),
        "data_root_2d_fused_feature",
        str(feat_root.resolve()),
        "feature_type",
        "fusion",
        "vis_input",
        "False",
        "vis_pred",
        "False",
        "vis_gt",
        "False",
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
