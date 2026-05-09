#!/usr/bin/env python3
"""
Build adapt splits from the official Matterport **test** `.pth` scenes.

Creates three directories under ``out_root`` with symlinks into ``src_test_dir``:

- ``adapt-train`` — adaptation training scenes (default 40%%)
- ``adapt-val`` — held-out for checkpoint selection during training (default 20%%)
- ``adapt-test`` — held-out for final evaluation metrics (default 40%%)

Writes ``split_manifest.json`` listing scene basenames per split (reproducible given ``seed``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path


def _three_way_sizes(n: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    """Hamilton / largest-remainder allocation so counts sum to *n*."""
    r0, r1, r2 = ratios
    s = r0 + r1 + r2
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1, got {ratios}")
    if n < 3:
        raise ValueError(f"Need at least 3 test scenes for a 3-way split, got {n}")
    p = (r0 / s, r1 / s, r2 / s)
    exact = [n * pi for pi in p]
    floors = [math.floor(x) for x in exact]
    rem = n - sum(floors)
    fracs = sorted([(exact[i] - floors[i], i) for i in range(3)], reverse=True)
    out = floors[:]
    for k in range(rem):
        out[fracs[k][1]] += 1
    return out[0], out[1], out[2]


def build_adapt_split(
    src_test_dir: Path,
    out_root: Path,
    seed: int,
    ratios: tuple[float, float, float] = (0.4, 0.2, 0.4),
    clean: bool = True,
) -> dict[str, list[str]]:
    """
    Populate ``out_root/{adapt-train,adapt-val,adapt-test}`` with symlinks.

    Returns manifest dict with keys ``adapt-train``, ``adapt-val``, ``adapt-test``
    mapping to sorted lists of scene basenames (``*.pth`` stems).
    """
    scenes = sorted(src_test_dir.glob("*.pth"))
    if len(scenes) < 3:
        raise RuntimeError(f"Need >= 3 scenes under {src_test_dir}, found {len(scenes)}")

    rng = random.Random(seed)
    shuffled = scenes[:]
    rng.shuffle(shuffled)

    n_t, n_v, n_te = _three_way_sizes(len(shuffled), ratios)
    split_train = shuffled[:n_t]
    split_val = shuffled[n_t : n_t + n_v]
    split_test = shuffled[n_t + n_v :]
    assert len(split_train) == n_t and len(split_val) == n_v and len(split_test) == n_te

    subdirs = ("adapt-train", "adapt-val", "adapt-test")
    buckets = (split_train, split_val, split_test)

    out_root.mkdir(parents=True, exist_ok=True)
    for name in subdirs:
        (out_root / name).mkdir(parents=True, exist_ok=True)

    if clean:
        for name in subdirs:
            d = out_root / name
            for p in d.glob("*.pth"):
                if p.is_symlink() or p.is_file():
                    p.unlink()

    for name, bucket in zip(subdirs, buckets):
        dest_dir = out_root / name
        for p in bucket:
            target = dest_dir / p.name
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(p.resolve())

    manifest = {
        "seed": seed,
        "ratios": list(ratios),
        "src_test_dir": str(src_test_dir.resolve()),
        "out_root": str(out_root.resolve()),
        "adapt-train": sorted(p.name for p in split_train),
        "adapt-val": sorted(p.name for p in split_val),
        "adapt-test": sorted(p.name for p in split_test),
    }
    (out_root / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build adapt-train / adapt-val / adapt-test from test scenes.")
    parser.add_argument(
        "--src-3d",
        type=Path,
        default=None,
        help="Matterport 3D root containing ``test/*.pth`` (default: $SRC_3D or …/matterport_3d)",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory for adapt-* folders (default: $SPLIT_ROOT or …/matterport_3d_adapt_split_seed$SEED)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: $SEED or 3407)")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.4,
        help="Fraction of scenes for adapt-train",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for adapt-val")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.4,
        help="Fraction for adapt-test",
    )
    args = parser.parse_args()

    data_root = Path(os.environ.get("DATA_ROOT", "/project/pi_hongyu_umass_edu/mdhassan/3d"))
    src_3d = args.src_3d if args.src_3d is not None else Path(os.environ.get("SRC_3D", data_root / "matterport_3d"))
    src_test = src_3d / "test"
    seed = args.seed if args.seed is not None else int(os.environ.get("SEED", "3407"))
    default_out = Path(os.environ.get("SPLIT_ROOT", data_root / f"matterport_3d_adapt_split_seed{seed}"))
    out_root = args.out_root if args.out_root is not None else default_out

    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if abs(sum(ratios) - 1.0) > 1e-6:
        print("train/val/test ratios must sum to 1", file=sys.stderr)
        return 1

    if not src_test.is_dir():
        print(f"Missing test directory: {src_test}", file=sys.stderr)
        return 1

    m = build_adapt_split(src_test, out_root, seed, ratios=ratios, clean=True)
    print(json.dumps({k: m[k] for k in ("adapt-train", "adapt-val", "adapt-test") if k in m}, indent=2))
    print(f"Wrote manifest: {out_root / 'split_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
