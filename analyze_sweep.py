"""Analyze the NMF representation tuning sweep.

Reads the resumable Optuna study from its SQLite storage (which holds the full
hyperparameter set, both objective values, and the guardrail metrics), and
produces:

  - a text summary of the Pareto front with full per-config hyperparameters,
  - a CSV of every completed trial,
  - a Pareto-front scatter (gap_ev R² vs logP R²),
  - hyperparameter-importance bar charts for each objective,
  - objective-vs-hyperparameter plots for the tuned knobs.

Defaults mirror tune_representation.py, so on the machine that ran the sweep you
can usually invoke it with no arguments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs without a display (works over SSH)
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import seaborn as sns

GAP = "gap_ev_r2"
LOGP = "logp_r2"
RECON = "val_rel_recon_err"
FG_MICRO = "fg_micro_f1"
FG_MACRO = "fg_macro_f1"

# Hyperparameters as suggested in tune_representation.py, split by plot style.
NUMERIC_PARAMS = [
    "n_components", "h_sigma_ppm", "c_sigma_ppm", "h_width_scale",
    "c_width_scale", "h_modality_weight", "alpha_W", "alpha_H", "l1_ratio",
]
CATEGORICAL_PARAMS = [
    "h_multiplicity", "use_peak_width", "intensity_transform", "solver", "beta_loss",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze the NMF representation tuning sweep.")
    p.add_argument("--storage", type=Path,
                   default=Path("/Users/jacknugent/Downloads/alberts_gap_repr_sweep.db"),
                   help="SQLite file backing the Optuna study (from tune_representation.py).")
    p.add_argument("--study-name", type=str, default="nmf_repr")
    p.add_argument("--out-dir", type=Path,
                   default=Path("/Users/jacknugent/Downloads/sweep_analysis"),
                   help="Directory for the CSV and PNG outputs.")
    return p.parse_args()


def load_completed(storage: Path, study_name: str) -> tuple[optuna.Study, pd.DataFrame]:
    """Load the study and return (study, dataframe of COMPLETE trials).

    The dataframe uses friendly column names: gap_ev_r2 / logp_r2 for the two
    objectives, the guardrail user-attrs unprefixed, and hyperparameters without
    the Optuna 'params_' prefix.
    """
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{storage}")
    df = study.trials_dataframe()
    df = df[df["state"] == "COMPLETE"].copy()

    # Multi-objective stores objectives as values_0 (gap) and values_1 (logP),
    # matching the return order in tune_representation.py's objective.
    rename = {"values_0": GAP, "values_1": LOGP}
    for col in list(df.columns):
        if col.startswith("params_"):
            rename[col] = col[len("params_"):]
        elif col.startswith("user_attrs_"):
            rename[col] = col[len("user_attrs_"):]
    df = df.rename(columns=rename)
    return study, df


def summarize_pareto(study: optuna.Study, df: pd.DataFrame, out_dir: Path) -> set[int]:
    """Print the Pareto front with full params; return its trial numbers."""
    pareto_numbers = {t.number for t in study.best_trials}
    pareto = sorted(study.best_trials, key=lambda t: t.values[0], reverse=True)

    print(f"\n=== Pareto front: {len(pareto)} non-dominated trials "
          f"(of {len(df)} completed) ===")
    print(f"{'trial':<9}{'gap R²':>9}{'logP R²':>9}   params")
    print("-" * 70)
    for t in pareto:
        params = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(t.params.items()))
        print(f"{'t'+str(t.number):<9}{t.values[0]:>9.4f}{t.values[1]:>9.4f}   {params}")
    return pareto_numbers


def _fmt(v: object) -> str:
    return f"{v:.4g}" if isinstance(v, float) else str(v)


def plot_pareto(df: pd.DataFrame, pareto_numbers: set[int], out_dir: Path) -> None:
    is_pareto = df["number"].isin(pareto_numbers)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df.loc[~is_pareto, GAP], df.loc[~is_pareto, LOGP],
               c="lightgray", s=40, label="dominated", zorder=2)
    ax.scatter(df.loc[is_pareto, GAP], df.loc[is_pareto, LOGP],
               c="crimson", s=70, label="Pareto front", zorder=3, edgecolor="black")
    for _, row in df[is_pareto].iterrows():
        ax.annotate(f"t{int(row['number'])}", (row[GAP], row[LOGP]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("held-out gap_ev R²  →")
    ax.set_ylabel("held-out logP R²  →")
    ax.set_title("Pareto front: gap_ev vs logP predictive quality")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_front.png", dpi=150)
    plt.close(fig)


def plot_param_importances(study: optuna.Study, out_dir: Path) -> None:
    targets = [(0, "gap_ev R²"), (1, "logP R²")]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (idx, label) in zip(axes, targets):
        try:
            imp = optuna.importance.get_param_importances(
                study, target=lambda t, i=idx: t.values[i]
            )
        except Exception as exc:  # too few trials / degenerate — skip gracefully
            ax.text(0.5, 0.5, f"importance unavailable:\n{exc}",
                    ha="center", va="center", wrap=True, fontsize=9)
            ax.set_title(label)
            continue
        names = list(imp.keys())[::-1]
        vals = list(imp.values())[::-1]
        ax.barh(names, vals, color="steelblue")
        ax.set_title(f"Hyperparameter importance for {label}")
        ax.set_xlabel("relative importance")
    fig.tight_layout()
    fig.savefig(out_dir / "param_importances.png", dpi=150)
    plt.close(fig)


def plot_objective_vs_params(df: pd.DataFrame, out_dir: Path) -> None:
    numeric = [p for p in NUMERIC_PARAMS if p in df.columns]
    categorical = [p for p in CATEGORICAL_PARAMS if p in df.columns and df[p].notna().any()]

    if numeric:
        ncol = 3
        nrow = (len(numeric) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
        for ax, p in zip(axes.flat, numeric):
            sc = ax.scatter(df[p], df[GAP], c=df[LOGP], cmap="viridis", s=45, edgecolor="k")
            ax.set_xlabel(p)
            ax.set_ylabel("gap_ev R²")
            if p in ("alpha_W", "alpha_H"):
                ax.set_xscale("log")
        for ax in axes.flat[len(numeric):]:
            ax.set_visible(False)
        fig.colorbar(sc, ax=axes.ravel().tolist(), label="logP R²", shrink=0.6)
        fig.suptitle("gap_ev R² vs numeric hyperparameters (color = logP R²)", y=1.0)
        fig.savefig(out_dir / "gap_vs_numeric_params.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    if categorical:
        ncol = 3
        nrow = (len(categorical) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
        for ax, p in zip(axes.flat, categorical):
            sns.stripplot(data=df, x=p, y=GAP, ax=ax, hue=p, legend=False, size=7)
            ax.set_xlabel(p)
            ax.set_ylabel("gap_ev R²")
            ax.tick_params(axis="x", rotation=30)
        for ax in axes.flat[len(categorical):]:
            ax.set_visible(False)
        fig.suptitle("gap_ev R² by categorical hyperparameter", y=1.0)
        fig.tight_layout()
        fig.savefig(out_dir / "gap_vs_categorical_params.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.storage.exists():
        raise SystemExit(f"Storage not found: {args.storage}\n"
                         f"Point --storage at the .db written next to your sweep JSON.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    study, df = load_completed(args.storage, args.study_name)
    if df.empty:
        raise SystemExit("No completed trials in the study yet.")

    print(f"Loaded {len(df)} completed trials from {args.storage}")
    print(f"gap_ev R²:  best={df[GAP].max():.4f}  median={df[GAP].median():.4f}")
    print(f"logP  R²:  best={df[LOGP].max():.4f}  median={df[LOGP].median():.4f}")

    csv_path = args.out_dir / "all_trials.csv"
    df.sort_values("number").to_csv(csv_path, index=False)

    pareto_numbers = summarize_pareto(study, df, args.out_dir)
    plot_pareto(df, pareto_numbers, args.out_dir)
    plot_param_importances(study, args.out_dir)
    plot_objective_vs_params(df, args.out_dir)

    print(f"\nWrote CSV + PNGs to {args.out_dir}")


if __name__ == "__main__":
    main()
