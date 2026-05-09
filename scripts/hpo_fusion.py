#!/usr/bin/env python3
"""
Hyperparameter grid search for the fusion adapter.

Per trial: train with ``third_party/openscene/run/train_fusion_h1.py`` on adapt-train,
score on adapt-val (full ``evaluate.py``), pick best val mIoU, run **one** adapt-test eval.

Edit ``SEARCH_SPACE_*`` below. Run once with ``--loss-mode ce`` and once with ``ce_h1``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SEARCH_SPACE_SHARED: dict[str, list] = {
    "base_lr": [8e-4],
    "batch_size": [1, 4],
    "weight_decay": [0.0, 0.001],
    "epochs": [15, 30, 50],
}

SEARCH_SPACE_H1: dict[str, list] = {
    "h1_lambda": [0.2],
    "h1_k": [20],
    "h1_sigma": [0.2],
}

FIXED_TRAIN_OVERRIDES: dict[str, object] = {
    # "epochs": 15,
}

MEAN_IOU_RE = re.compile(r"Mean IoU\s+([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _flatten_opts(overrides: dict[str, object]) -> list[str]:
    out: list[str] = []
    for k in sorted(overrides.keys()):
        out.extend([k, str(overrides[k])])
    return out


def _sorted_product(space: dict[str, list]) -> tuple[list[str], list[tuple]]:
    keys = sorted(space.keys())
    values = [space[k] for k in keys]
    combos = list(itertools.product(*values))
    return keys, combos


def parse_mean_iou(log_text: str) -> float | None:
    matches = MEAN_IOU_RE.findall(log_text)
    if not matches:
        return None
    return float(matches[-1])


def _import_build_adapt_split():
    sd = _scripts_dir()
    if str(sd) not in sys.path:
        sys.path.insert(0, str(sd))
    from split_dataset import build_adapt_split  # noqa: E402

    return build_adapt_split


def _openscene(repo: Path) -> Path:
    return repo / "third_party" / "openscene"


def _adapter_eval_cmd(
    repo: Path,
    *,
    split_root: Path,
    feat_root: Path,
    eval_split: str,
    checkpoint: Path,
    save_folder: Path,
    test_repeats: int,
    vis_input: bool = False,
    vis_pred: bool = False,
    vis_gt: bool = False,
) -> tuple[list[str], dict]:
    """Subprocess ``evaluate.py`` with fusion-with-h1; CLIP class set from YAML ``labelset``."""
    opense = _openscene(repo)
    eval_py = opense / "run" / "evaluate.py"
    cfg = opense / "config" / "matterport" / "ours_openseg_pretrained.yaml"
    cmd = [
        sys.executable,
        "-u",
        str(eval_py),
        "--config",
        str(cfg.resolve()),
        "data_root",
        str(split_root.resolve()),
        "data_root_2d_fused_feature",
        str(feat_root.resolve()),
        "feature_type",
        "fusion-with-h1",
        "vis_input",
        str(bool(vis_input)),
        "vis_pred",
        str(bool(vis_pred)),
        "vis_gt",
        str(bool(vis_gt)),
        "split",
        eval_split,
        "test_repeats",
        str(test_repeats),
        "save_folder",
        str(save_folder.resolve()),
        "model_path",
        str(checkpoint.resolve()),
        "adapter_input_dim",
        str(int(os.environ.get("ADAPTER_INPUT_DIM", "768"))),
        "adapter_hidden_dim",
        str(int(os.environ.get("ADAPTER_HIDDEN_DIM", "512"))),
        "adapter_dropout",
        str(float(os.environ.get("ADAPTER_DROPOUT", "0.1"))),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(opense.resolve())
    return cmd, env


def run_hpo_for_mode(
    *,
    loss_mode: str,
    hpo_root: Path,
    split_root: Path,
    feat_root: Path,
    seed: int,
    config_path: Path,
    skip_split: bool,
    max_trials: int | None,
    dry_run: bool,
    test_repeats_val: int,
    test_repeats_test: int,
    skip_final_test: bool,
) -> dict:
    build_adapt_split = _import_build_adapt_split()
    repo = _repo_root()
    opense = _openscene(repo)
    train_py = opense / "run" / "train_fusion_h1.py"
    data_root = Path(os.environ.get("DATA_ROOT", "/project/pi_hongyu_umass_edu/mdhassan/3d"))
    src_3d = Path(os.environ.get("SRC_3D", data_root / "matterport_3d"))

    space = dict(SEARCH_SPACE_SHARED)
    if loss_mode == "ce_h1":
        space.update(SEARCH_SPACE_H1)
    keys, combos = _sorted_product(space)
    trials: list[dict[str, object]] = []
    for combo in combos:
        d = dict(zip(keys, combo))
        d.update(FIXED_TRAIN_OVERRIDES)
        trials.append(d)
    if max_trials is not None:
        trials = trials[: max_trials]

    mode_root = hpo_root / loss_mode
    mode_root.mkdir(parents=True, exist_ok=True)
    csv_path = mode_root / "hpo_trials.csv"
    fieldnames = (
        ["trial_idx", "save_path", "train_exit", "eval_val_exit", "val_miou"]
        + sorted(set(keys) | set(FIXED_TRAIN_OVERRIDES.keys()))
    )

    print(f"\n=== HPO loss_mode={loss_mode} | {len(trials)} trials | out={mode_root} ===")

    if not dry_run and not skip_split:
        build_adapt_split(src_3d / "test", split_root, seed, clean=True)

    if dry_run:
        for i, t in enumerate(trials):
            print(f"[dry-run] trial {i}: {t}")
        return {"loss_mode": loss_mode, "dry_run": True, "n_trials": len(trials)}

    best: dict | None = None
    file_exists = csv_path.is_file()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for trial_idx, overrides in enumerate(trials):
            trial_cfg = dict(overrides)
            trial_dir = mode_root / f"trial_{trial_idx:04d}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            train_opts = _flatten_opts(trial_cfg)
            train_cmd = [
                sys.executable,
                "-u",
                str(train_py),
                "--config",
                str(config_path.resolve()),
                "data_root",
                str(split_root.resolve()),
                "adapt_split",
                "adapt-train",
                "data_root_2d_fused_feature",
                str(feat_root.resolve()),
                "save_path",
                str(trial_dir.resolve()),
                "manual_seed",
                str(seed),
                "loss_mode",
                loss_mode,
                *train_opts,
            ]
            env_train = os.environ.copy()
            env_train["PYTHONPATH"] = str(opense.resolve())

            print("\n--- train ---\n" + " ".join(train_cmd))
            tr = subprocess.run(
                train_cmd,
                cwd=str(opense),
                env=env_train,
                capture_output=True,
                text=True,
            )
            (trial_dir / "train_subprocess.log").write_text((tr.stdout or "") + (tr.stderr or ""))

            val_miou: float | None = None
            eval_val_exit = -1
            ckpt = trial_dir / "model" / "model_last.pth.tar"
            if tr.returncode == 0 and ckpt.is_file():
                eval_val_dir = trial_dir / "eval_adapt_val"
                eval_val_dir.mkdir(parents=True, exist_ok=True)
                result_dir = eval_val_dir / "result_eval_adapter_adapt-val"
                eval_cmd, env_eval = _adapter_eval_cmd(
                    repo,
                    split_root=split_root,
                    feat_root=feat_root,
                    eval_split="adapt-val",
                    checkpoint=ckpt,
                    save_folder=result_dir,
                    test_repeats=test_repeats_val,
                    vis_input=False,
                    vis_pred=False,
                    vis_gt=False,
                )
                print("--- eval adapt-val ---\n" + " ".join(eval_cmd))
                ev = subprocess.run(eval_cmd, cwd=str(opense), env=env_eval, capture_output=True, text=True)
                eval_val_exit = ev.returncode
                (trial_dir / "eval_val_subprocess.log").write_text((ev.stdout or "") + (ev.stderr or ""))
                if ev.returncode == 0:
                    val_miou = parse_mean_iou((ev.stdout or "") + (ev.stderr or ""))

            row = {
                "trial_idx": trial_idx,
                "save_path": str(trial_dir),
                "train_exit": tr.returncode,
                "eval_val_exit": eval_val_exit,
                "val_miou": "" if val_miou is None else val_miou,
            }
            for k in keys:
                row[k] = trial_cfg[k]
            for k in FIXED_TRAIN_OVERRIDES:
                row[k] = trial_cfg[k]
            writer.writerow(row)
            f.flush()

            if val_miou is not None and (best is None or val_miou > best["val_miou"]):
                best = {
                    "trial_idx": trial_idx,
                    "val_miou": val_miou,
                    "save_path": str(trial_dir),
                    "checkpoint": str(ckpt),
                    "overrides": {k: trial_cfg[k] for k in keys},
                }

            print(
                f"trial {trial_idx} train_exit={tr.returncode} eval_val_exit={eval_val_exit} val_mIoU={val_miou}",
                flush=True,
            )

    summary: dict = {"loss_mode": loss_mode, "best": best, "trials_csv": str(csv_path)}
    if best is None or skip_final_test:
        summary["test_miou"] = None
        summary["note"] = "no successful trial or --skip-final-test"
        (mode_root / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    final_dir = mode_root / "final_eval_adapt_test"
    final_dir.mkdir(parents=True, exist_ok=True)
    result_test = final_dir / "result_eval_adapter_adapt-test"
    eval_test_cmd, env_eval = _adapter_eval_cmd(
        repo,
        split_root=split_root,
        feat_root=feat_root,
        eval_split="adapt-test",
        checkpoint=Path(best["checkpoint"]),
        save_folder=result_test,
        test_repeats=test_repeats_test,
        vis_input=True,
        vis_pred=True,
        vis_gt=True,
    )
    print("\n=== final eval adapt-test (best trial) ===\n" + " ".join(eval_test_cmd))
    ev_t = subprocess.run(eval_test_cmd, cwd=str(opense), env=env_eval, capture_output=True, text=True)
    (mode_root / "final_test_subprocess.log").write_text((ev_t.stdout or "") + (ev_t.stderr or ""))
    test_miou = parse_mean_iou((ev_t.stdout or "") + (ev_t.stderr or "")) if ev_t.returncode == 0 else None
    summary["test_miou"] = test_miou
    summary["final_eval_exit"] = ev_t.returncode
    (mode_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Best trial={best['trial_idx']} val_mIoU={best['val_miou']} test_mIoU={test_miou}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="HPO: train → adapt-val → adapt-test.")
    parser.add_argument(
        "--loss-mode",
        choices=("ce", "ce_h1"),
        required=True,
        help="ce = CE-only adapter; ce_h1 = CE + graph smoothness.",
    )
    parser.add_argument(
        "--hpo-root",
        type=Path,
        default=None,
        help="HPO output root (default: openscene/out/hpo_fusion_seed<SEED>)",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "3407")))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split-root", type=Path, default=None)
    parser.add_argument("--feat-root", type=Path, default=None)
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--test-repeats-val", type=int, default=1)
    parser.add_argument("--test-repeats-test", type=int, default=1)
    parser.add_argument("--skip-final-test", action="store_true")
    args = parser.parse_args()

    repo = _repo_root()
    opense = repo / "third_party" / "openscene"
    data_root = Path(os.environ.get("DATA_ROOT", "/project/pi_hongyu_umass_edu/mdhassan/3d"))
    split_root = args.split_root or Path(
        os.environ.get("SPLIT_ROOT", data_root / f"matterport_3d_adapt_split_seed{args.seed}")
    )
    feat_root = args.feat_root or Path(
        os.environ.get("SRC_FEAT", data_root / "matterport_multiview_openseg_test")
    )
    cfg_path = args.config or (opense / "config" / "matterport" / "h1_fusion.yaml")
    hpo_root = args.hpo_root or (opense / "out" / f"hpo_fusion_seed{args.seed}")

    if not args.dry_run:
        if not split_root.is_dir() or not (split_root / "adapt-train").is_dir():
            print(f"Missing split at {split_root} (run scripts/split_dataset.py first).", file=sys.stderr)
            return 1
        if not feat_root.is_dir():
            print(f"Missing feat root {feat_root}", file=sys.stderr)
            return 1
        if not cfg_path.is_file():
            print(f"Missing config {cfg_path}", file=sys.stderr)
            return 1

    hpo_root.mkdir(parents=True, exist_ok=True)
    summary = run_hpo_for_mode(
        loss_mode=args.loss_mode,
        hpo_root=hpo_root,
        split_root=split_root,
        feat_root=feat_root,
        seed=args.seed,
        config_path=cfg_path,
        skip_split=args.skip_split,
        max_trials=args.max_trials,
        dry_run=args.dry_run,
        test_repeats_val=args.test_repeats_val,
        test_repeats_test=args.test_repeats_test,
        skip_final_test=args.skip_final_test,
    )
    (hpo_root / f"summary_{args.loss_mode}.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
