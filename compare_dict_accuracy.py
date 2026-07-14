"""Compare downstream predictive accuracy of two (or more) NMF dictionaries.

Each dictionary's codes file (from build_dict_codes_torch.py) carries the
``nmr_dict_code_*`` features plus the original target columns. For every target we
run k-fold CV of the same model on each file's codes and print a side-by-side
R2/RMSE/MAE table. Both dictionaries are judged on identical rows/folds, so the
only thing that varies is the representation — a clean A/B (e.g. balanced w1_grid
vs unbalanced Sinkhorn).

Example
-------
    python compare_dict_accuracy.py \
        --codes Datasets/alberts_merged_10k_torch_nmf_codes.pkl=sinkhorn \
                Datasets/alberts_merged_10k_w1_grid_codes.pkl=w1_grid \
        --targets gap_ev logP homo_ev lumo_ev
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CODE_PREFIX = "nmr_dict_code_"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B two NMF dictionaries on downstream target accuracy.")
    p.add_argument("--codes", nargs="+", required=True,
                   help="One or more 'path=label' (label optional; defaults to file stem).")
    p.add_argument("--targets", nargs="+", default=["gap_ev", "logP", "homo_ev", "lumo_ev"])
    p.add_argument("--model", choices=["hgb", "elasticnet"], default="hgb",
                   help="hgb = HistGradientBoosting (nonlinear, matches train_gap_model); "
                        "elasticnet = linear probe (matches the tuning sweep).")
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_model(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=25, random_state=seed,
        )
    return make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=3, random_state=seed))


def load_codes(spec: str) -> tuple[str, pd.DataFrame]:
    path, _, label = spec.partition("=")
    df = pd.read_pickle(path)
    return (label or Path(path).stem), df


def cv_metrics(X: np.ndarray, y: np.ndarray, kind: str, cv: int, seed: int) -> dict:
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    yt, yp = [], []
    for tr, te in kf.split(X):
        m = make_model(kind, seed)
        m.fit(X[tr], y[tr])
        yt.append(y[te]); yp.append(m.predict(X[te]))
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    return {"r2": r2_score(yt, yp), "rmse": root_mean_squared_error(yt, yp),
            "mae": mean_absolute_error(yt, yp), "n": len(yt)}


def main() -> None:
    args = parse_args()
    dicts = [load_codes(s) for s in args.codes]
    for label, df in dicts:
        n_codes = sum(c.startswith(CODE_PREFIX) for c in df.columns)
        print(f"  {label}: {len(df)} rows, {n_codes} codes")

    print(f"\nModel={args.model}  CV={args.cv}-fold\n")
    header = f"{'target':<10} " + " ".join(f"{lab:>22}" for lab, _ in dicts)
    print(header)
    print("-" * len(header))
    for tgt in args.targets:
        cells = []
        for label, df in dicts:
            if tgt not in df.columns:
                cells.append(f"{'(absent)':>22}"); continue
            codes = [c for c in df.columns if c.startswith(CODE_PREFIX)]
            sub = df[codes + [tgt]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(sub) < args.cv * 2:
                cells.append(f"{'(too few rows)':>22}"); continue
            m = cv_metrics(sub[codes].to_numpy(float), sub[tgt].to_numpy(float),
                           args.model, args.cv, args.seed)
            cells.append(f"R2={m['r2']:.3f} RMSE={m['rmse']:.3f}".rjust(22))
        print(f"{tgt:<10} " + " ".join(cells))

    print("\n(higher R2 / lower RMSE = better; identical rows+folds per target, so "
          "differences are purely the representation.)")


if __name__ == "__main__":
    main()
