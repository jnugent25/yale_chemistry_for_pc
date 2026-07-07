"""Tune HistGradientBoosting for gap_ev and diagnose overfitting — fast and honest.

Uses a three-way train/val/test split. The tuned NMF representation (from the
sweep study) is fit ONCE on train and applied to val/test via transform, so val
and test codes are out-of-sample for NMF — this removes the representation
leakage that inflates CV without needing to refit NMF per fold. The booster is
tuned on the val set and reported on the untouched test set.

Because NMF is fit once (not hundreds of times inside a CV loop), the whole run
is dominated by a single NMF fit; the booster HPO then runs on small code
matrices in seconds. For both the default and tuned booster it reports
train / val / test R² — a large train-minus-test gap is the overfitting signal.
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
from sklearn.model_selection import train_test_split

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
    p.add_argument("--val-frac", type=float, default=0.15, help="Validation fraction (booster HPO signal).")
    p.add_argument("--test-frac", type=float, default=0.15, help="Test fraction (held out from HPO).")
    p.add_argument("--n-trials", type=int, default=40, help="Optuna trials for the booster.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iter", type=int, default=600, help="NMF max_iter.")
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--out-dir", type=Path, default=Path("/Users/jacknugent/Downloads/gap_model_tuned"))
    return p.parse_args()


def make_hgb(params: dict, seed: int) -> HistGradientBoostingRegressor:
    # Early stopping caps the number of trees per fit via an internal validation
    # split, an overfitting guard independent of the tuned knobs.
    return HistGradientBoostingRegressor(
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=seed,
        **params,
    )


def evaluate(model, w_tr, g_tr, w_val, g_val, w_te, g_te) -> dict:
    """Fit on train codes; report train / val / test R² (val & test are
    out-of-sample for NMF, so these are honest)."""
    model.fit(w_tr, g_tr)
    train_r2 = r2_score(g_tr, model.predict(w_tr))
    val_r2 = r2_score(g_val, model.predict(w_val))
    test_pred = model.predict(w_te)
    test_r2 = r2_score(g_te, test_pred)
    return {
        "train_r2": float(train_r2),
        "val_r2": float(val_r2),
        "test_r2": float(test_r2),
        "test_rmse": float(root_mean_squared_error(g_te, test_pred)),
        "overfit_gap": float(train_r2 - test_r2),
    }


def report(label: str, m: dict) -> None:
    print(f"\n{label}:")
    print(f"  train R²    = {m['train_r2']:.4f}")
    print(f"  val   R²    = {m['val_r2']:.4f}")
    print(f"  test  R²    = {m['test_r2']:.4f}  (RMSE {m['test_rmse']:.4f} eV)")
    print(f"  overfit gap = {m['overfit_gap']:.4f}   (train R² − test R²)")


def plot_learning_curve(best_params, w_tr, g_tr, w_val, g_val, seed, out_dir: Path) -> None:
    """Manual learning curve on the fixed val set (val is out-of-sample for NMF)."""
    order = np.arange(len(w_tr))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    fracs = np.linspace(0.2, 1.0, 6)
    train_r2, val_r2, sizes = [], [], []
    for frac in fracs:
        n = max(50, int(len(order) * frac))
        sub = order[:n]
        model = make_hgb(best_params, seed)
        model.fit(w_tr[sub], g_tr[sub])
        train_r2.append(r2_score(g_tr[sub], model.predict(w_tr[sub])))
        val_r2.append(r2_score(g_val, model.predict(w_val)))
        sizes.append(n)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, train_r2, "o-", color="tab:blue", label="train")
    ax.plot(sizes, val_r2, "o-", color="tab:orange", label="validation")
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
    ax.set_title(f"Tuned NMF→HGB (held-out test)\nR²={m['test_r2']:.3f}  RMSE={m['test_rmse']:.3f} eV")
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

    # Three-way split: test held out entirely; val is the booster HPO signal.
    idx = np.arange(len(x))
    trainval, te = train_test_split(idx, test_size=args.test_frac, random_state=args.seed)
    rel_val = args.val_frac / (1.0 - args.test_frac)
    tr, va = train_test_split(trainval, test_size=rel_val, random_state=args.seed)
    print(f"Split: {len(tr)} train / {len(va)} val / {len(te)} test")

    # NMF fit ONCE on train; val/test codes via transform are out-of-sample.
    print(f"Fitting NMF once on train (max_iter={args.max_iter}) ...")
    nmf = make_nmf(cfg, args.max_iter)
    w_tr = nmf.fit_transform(x[tr])
    w_va = nmf.transform(x[va])
    w_te = nmf.transform(x[te])
    g_tr, g_va, g_te = gap[tr], gap[va], gap[te]

    def objective(t: optuna.Trial) -> float:
        params = dict(
            learning_rate=t.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_leaf_nodes=t.suggest_int("max_leaf_nodes", 8, 128, log=True),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 5, 100, log=True),
            l2_regularization=t.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
            max_features=t.suggest_float("max_features", 0.3, 1.0),
        )
        model = make_hgb(params, args.seed)
        model.fit(w_tr, g_tr)
        return float(r2_score(g_va, model.predict(w_va)))  # honest: val is out-of-sample for NMF

    print(f"\nTuning booster on the val set: {args.n_trials} trials ...")
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)
    print(f"Best val R² = {study.best_value:.4f}")
    print("Best booster params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    default_metrics = evaluate(make_hgb({}, args.seed), w_tr, g_tr, w_va, g_va, w_te, g_te)
    tuned_model = make_hgb(study.best_params, args.seed)
    tuned_metrics = evaluate(tuned_model, w_tr, g_tr, w_va, g_va, w_te, g_te)

    print("\n=== Overfitting diagnosis (gap_ev) ===")
    report("Default booster", default_metrics)
    report("Tuned booster", tuned_metrics)
    print("\nInterpretation: val and test are out-of-sample for NMF, so they should "
          "agree; a large train−test gap means the booster is overfitting.")

    plot_actual_vs_pred(g_te, tuned_model.predict(w_te), tuned_metrics, args.out_dir)
    plot_learning_curve(study.best_params, w_tr, g_tr, w_va, g_va, args.seed, args.out_dir)

    summary = {
        "representation_trial": trial.number,
        "best_val_r2": study.best_value,
        "best_booster_params": study.best_params,
        "default": default_metrics,
        "tuned": tuned_metrics,
    }
    (args.out_dir / "tuning_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote plots + tuning_summary.json to {args.out_dir}")


if __name__ == "__main__":
    main()
