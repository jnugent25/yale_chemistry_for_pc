"""Tune the HistGradientBoosting hyperparameters for gap_ev and diagnose overfitting.

Builds the tuned NMF representation (from the sweep study) and tunes the booster
with Optuna over cross-validated R². Crucially, the whole NMF→HGB pipeline is
cross-validated, so NMF is refit inside every fold — a validation-fold molecule
is never encoded by an NMF that has seen it. This removes the representation
leakage that otherwise inflates CV R² far above the honest held-out number.

For both the default and tuned booster it reports train R² vs (honest) CV R² vs
held-out test R² — a large train-minus-test gap is the overfitting signal. Also
writes a learning curve and an actual-vs-predicted plot.

Note: refitting NMF per fold makes this markedly slower than tuning on fixed
codes (n_trials × cv NMF fits); lower --n-trials or --max-iter to trade off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display (works over SSH)
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, learning_curve, train_test_split
from sklearn.pipeline import Pipeline

from tune_representation import SweepConfig
from train_gap_model import (
    build_representation,
    config_from_params,
    load_working_set,
    make_nmf,
    select_trial,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune HistGradientBoosting for gap_ev and diagnose overfitting.")
    p.add_argument("--raw-data", type=Path, default=Path("/Users/jacknugent/Downloads/alberts_merged_10k.pkl"))
    p.add_argument("--storage", type=Path,
                   default=Path("/Users/jacknugent/Downloads/alberts_gap_repr_sweep.db"))
    p.add_argument("--study-name", type=str, default="nmf_repr")
    p.add_argument("--trial", type=int, default=None,
                   help="Representation trial to use; default = best sweep gap_ev R².")
    p.add_argument("--sample", type=int, default=12000)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--cv", type=int, default=5, help="CV folds for booster tuning.")
    p.add_argument("--n-trials", type=int, default=25,
                   help="Optuna trials for the booster (each costs cv NMF refits).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=600, help="NMF max_iter.")
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--out-dir", type=Path, default=Path("/Users/jacknugent/Downloads/gap_model_tuned"))
    return p.parse_args()


def make_hgb(params: dict, seed: int) -> HistGradientBoostingRegressor:
    # Early stopping caps the number of trees per fit via an internal validation
    # split, which is itself an overfitting guard independent of the tuned knobs.
    return HistGradientBoostingRegressor(
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=seed,
        **params,
    )


def make_pipeline(cfg: SweepConfig, hgb_params: dict, nmf_max_iter: int, seed: int) -> Pipeline:
    """NMF → HGB pipeline. Cross-validating this refits NMF inside each fold, so
    a validation-fold molecule is never encoded by an NMF that has seen it."""
    return Pipeline([
        ("nmf", make_nmf(cfg, nmf_max_iter)),
        ("hgb", make_hgb(hgb_params, seed)),
    ])


def evaluate(cfg, hgb_params, x_tr, gap_tr, x_te, gap_te, cv, seed, nmf_max_iter) -> tuple[dict, Pipeline]:
    """Train R², honest (per-fold-NMF) CV R², and held-out test R² for one booster.

    CV cross-validates the full NMF→HGB pipeline on the raw representation matrix,
    so NMF is refit per fold; the returned pipeline is fit on all training rows.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(
        make_pipeline(cfg, hgb_params, nmf_max_iter, seed),
        x_tr, gap_tr, cv=kf, scoring="r2", n_jobs=-1,
    )
    pipe = make_pipeline(cfg, hgb_params, nmf_max_iter, seed)
    pipe.fit(x_tr, gap_tr)
    train_r2 = r2_score(gap_tr, pipe.predict(x_tr))
    test_pred = pipe.predict(x_te)
    metrics = {
        "train_r2": float(train_r2),
        "cv_r2_mean": float(cv_scores.mean()),
        "cv_r2_std": float(cv_scores.std()),
        "test_r2": float(r2_score(gap_te, test_pred)),
        "test_rmse": float(root_mean_squared_error(gap_te, test_pred)),
        "overfit_gap": float(train_r2 - r2_score(gap_te, test_pred)),
    }
    return metrics, pipe


def report(label: str, m: dict) -> None:
    print(f"\n{label}:")
    print(f"  train R²    = {m['train_r2']:.4f}")
    print(f"  CV R²       = {m['cv_r2_mean']:.4f} ± {m['cv_r2_std']:.4f}")
    print(f"  test R²     = {m['test_r2']:.4f}  (RMSE {m['test_rmse']:.4f} eV)")
    print(f"  overfit gap = {m['overfit_gap']:.4f}   (train R² − test R²)")


def plot_learning_curve(estimator, x_tr, gap_tr, cv, seed, out_dir: Path) -> None:
    # estimator is the NMF→HGB pipeline, so NMF is refit at every training size/fold.
    sizes, train_scores, val_scores = learning_curve(
        estimator, x_tr, gap_tr, cv=KFold(cv, shuffle=True, random_state=seed),
        scoring="r2", train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1,
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    for scores, color, name in [(train_scores, "tab:blue", "train"),
                                (val_scores, "tab:orange", "validation")]:
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, "o-", color=color, label=name)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_xlabel("training samples")
    ax.set_ylabel("R²")
    ax.set_title("Learning curve (gap of train above validation = overfitting)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curve.png", dpi=150)
    plt.close(fig)


def plot_actual_vs_pred(y_true, y_pred, m: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=18, alpha=0.4, edgecolor="none")
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="ideal")
    ax.set_xlabel("actual gap_ev (eV)")
    ax.set_ylabel("predicted gap_ev (eV)")
    ax.set_title(f"Tuned NMF→HGB (held-out)\nR²={m['test_r2']:.3f}  RMSE={m['test_rmse']:.3f} eV")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_dir / "gap_actual_vs_predicted_tuned.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # our own concise output below
    if not args.storage.exists():
        raise SystemExit(f"Storage not found: {args.storage}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trial = select_trial(args.storage, args.study_name, args.trial)
    cfg = config_from_params(trial.params)
    print(f"Representation: trial {trial.number} (sweep gap_ev R²={trial.values[0]:.4f})")

    print(f"Loading {args.raw_data} ...")
    df, gap = load_working_set(args.raw_data, args.sample, args.seed)
    x = build_representation(df, cfg, args.h_step_ppm, args.c_step_ppm)
    print(f"Working set: {len(df)} molecules; representation {x.shape}")

    # Split on the raw representation matrix; NMF is refit downstream (per fold
    # during CV, and on the whole training block for the final test estimate).
    idx = np.arange(len(x))
    tr, te = train_test_split(idx, test_size=args.test_frac, random_state=args.seed)
    x_tr, x_te = x[tr], x[te]
    gap_tr, gap_te = gap[tr], gap[te]

    kf = KFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    def objective(t: optuna.Trial) -> float:
        params = dict(
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=t.suggest_int("max_leaf_nodes", 8, 128, log=True),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 100, log=True),
            l2_regularization=t.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            max_features=t.suggest_float("max_features", 0.3, 1.0),
        )
        pipe = make_pipeline(cfg, params, args.max_iter, args.seed)
        scores = cross_val_score(pipe, x_tr, gap_tr, cv=kf, scoring="r2", n_jobs=-1)
        return float(scores.mean())

    print(f"\nTuning booster: {args.n_trials} trials, {args.cv}-fold CV "
          f"(NMF refit per fold — this is the slow part) ...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    print(f"Best honest CV R² = {study.best_value:.4f}")
    print("Best booster params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    default_metrics, _ = evaluate(cfg, {}, x_tr, gap_tr, x_te, gap_te, args.cv, args.seed, args.max_iter)
    tuned_metrics, tuned_pipe = evaluate(
        cfg, study.best_params, x_tr, gap_tr, x_te, gap_te, args.cv, args.seed, args.max_iter
    )

    print("\n=== Overfitting diagnosis (gap_ev, honest per-fold NMF) ===")
    report("Default booster", default_metrics)
    report("Tuned booster", tuned_metrics)
    print("\nInterpretation: CV R² now refits NMF per fold, so it should track the "
          "held-out test R² closely; a large train−test gap still means overfitting.")

    # tuned_pipe is already fit on all training rows by evaluate().
    plot_actual_vs_pred(gap_te, tuned_pipe.predict(x_te), tuned_metrics, args.out_dir)
    plot_learning_curve(make_pipeline(cfg, study.best_params, args.max_iter, args.seed),
                        x_tr, gap_tr, args.cv, args.seed, args.out_dir)

    summary = {
        "representation_trial": trial.number,
        "best_cv_r2": study.best_value,
        "best_booster_params": study.best_params,
        "default": default_metrics,
        "tuned": tuned_metrics,
    }
    (args.out_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote plots + tuning_summary.json to {args.out_dir}")


if __name__ == "__main__":
    main()
