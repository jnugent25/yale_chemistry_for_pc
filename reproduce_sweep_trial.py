"""Reproduce a sweep trial's held-out gap_ev / logP R² using tune_representation's
EXACT split and ElasticNet probe.

The sweep evaluated each config on a single 85/15 train/val split (seed 3245) and
scored with ElasticNet. This replays that identical computation for one trial, so
you can confirm the stored number (e.g. trial 93's gap R²≈0.57) reproduces — and
then see how much it moves under different random splits, which quantifies the
split variance + selection bias that make a fresh split (as in tune_gap_model)
read lower.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import optuna

from tune_representation import elasticnet_r2
from train_gap_model import (
    build_representation,
    config_from_params,
    load_working_set,
    make_nmf,
    select_trial,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce a sweep trial's ElasticNet gap/logP R² on the exact sweep split.")
    p.add_argument("--raw-data", type=Path, default=Path("/Users/jacknugent/Downloads/alberts_merged_10k.pkl"))
    p.add_argument("--storage", type=Path,
                   default=Path("/Users/jacknugent/Downloads/alberts_gap_repr_sweep.db"))
    p.add_argument("--study-name", type=str, default="nmf_repr")
    p.add_argument("--trial", type=int, default=None, help="Trial number; default = best gap_ev R².")
    # These MUST match the values tune_representation.py was run with, or the split differs.
    p.add_argument("--sample", type=int, default=12000, help="Match the sweep's --sample.")
    p.add_argument("--seed", type=int, default=3245, help="Match the sweep's --seed (data + split).")
    p.add_argument("--val-frac", type=float, default=0.15, help="Match the sweep's --val-frac.")
    p.add_argument("--max-iter", type=int, default=600, help="Match the sweep's NMF --max-iter.")
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--n-reseeds", type=int, default=0,
                   help="If >0, also evaluate this many different random 85/15 splits "
                        "to show the spread (each costs one NMF fit).")
    return p.parse_args()


def sweep_split(n: int, seed: int, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact split from tune_representation.py: permute with default_rng(seed),
    first val_frac -> val, rest -> train, both sorted."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(n * val_frac)
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def eval_split(x, gap, logp, cfg, train_idx, val_idx, max_iter) -> tuple[float, float]:
    """Fit NMF on train, transform val, ElasticNet R² on val — mirrors the objective."""
    nmf = make_nmf(cfg, max_iter)
    w_tr = nmf.fit_transform(x[train_idx])
    w_va = nmf.transform(x[val_idx])

    def r2(y: np.ndarray) -> float:
        tr_m = ~np.isnan(y[train_idx])
        va_m = ~np.isnan(y[val_idx])
        if not tr_m.any() or not va_m.any():
            return float("nan")
        return elasticnet_r2(w_tr[tr_m], y[train_idx][tr_m], w_va[va_m], y[val_idx][va_m], random_state=0)

    return r2(gap), r2(logp)


def main() -> None:
    args = parse_args()
    if not args.storage.exists():
        raise SystemExit(f"Storage not found: {args.storage}")

    trial = select_trial(args.storage, args.study_name, args.trial)
    cfg = config_from_params(trial.params)
    stored_gap, stored_logp = trial.values[0], trial.values[1]
    print(f"Trial {trial.number}: stored sweep gap R²={stored_gap:.4f}, logP R²={stored_logp:.4f}")

    print(f"Loading {args.raw_data} (sample={args.sample}, seed={args.seed}) ...")
    df, gap = load_working_set(args.raw_data, args.sample, args.seed)
    import pandas as pd
    logp = pd.to_numeric(df["logP"], errors="coerce").to_numpy(dtype=float) if "logP" in df.columns \
        else np.full(len(df), np.nan)
    x = build_representation(df, cfg, args.h_step_ppm, args.c_step_ppm)
    print(f"Working set: {len(df)} molecules; representation {x.shape}\n")

    # 1) Exact sweep split — should match the stored value closely.
    tr, va = sweep_split(len(df), args.seed, args.val_frac)
    gap_r2, logp_r2 = eval_split(x, gap, logp, cfg, tr, va, args.max_iter)
    print(f"Reproduced on the EXACT sweep split (seed {args.seed}):")
    print(f"  gap R²  = {gap_r2:.4f}   (stored {stored_gap:.4f}, diff {gap_r2 - stored_gap:+.4f})")
    print(f"  logP R² = {logp_r2:.4f}   (stored {stored_logp:.4f}, diff {logp_r2 - stored_logp:+.4f})")

    # 2) Optional: different random splits, to show variance + selection bias.
    if args.n_reseeds > 0:
        print(f"\nRe-evaluating on {args.n_reseeds} different random 85/15 splits ...")
        gaps = []
        for i in range(args.n_reseeds):
            tr_i, va_i = sweep_split(len(df), 10_000 + i, args.val_frac)
            g, _ = eval_split(x, gap, logp, cfg, tr_i, va_i, args.max_iter)
            gaps.append(g)
            print(f"  split {i + 1:>2}: gap R² = {g:.4f}", flush=True)
        gaps = np.array(gaps)
        print(f"\n  gap R² over reseeds: mean {gaps.mean():.4f} ± {gaps.std():.4f} "
              f"(range {gaps.min():.4f}–{gaps.max():.4f})")
        print(f"  the stored {stored_gap:.4f} is the best-of-many from the sweep, so it sits "
              f"at/above this spread — that's the selection bias.")


if __name__ == "__main__":
    main()
