"""Sweep the w1_grid NMF representation by PRETRAINING on a corpus and TRANSFERRING
the frozen dictionary to a labeled target, scoring codes + rich nmr_stats with HGB
5-fold. Optimizes the pretrained dictionary's hyperparameters for held-out gap_ev and
logP R^2 on the target (multi-objective TPE).

Per trial: rasterize corpus -> fit w1_grid NMF (GPU) -> rasterize target -> transform
target through the frozen dictionary -> HGB 5-fold on [codes | rich stats]. Rich stats
and target rasterization don't depend on the fit, so the stats are computed once.

    python sweep_pretrain_transfer.py \
        --corpus Datasets/nmrexp_100k.pkl --target alberts_10k \
        --device cuda --n-trials 40
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import optuna

from tune_representation import SweepConfig, assemble_x, build_raw_matrices, raster_key
from create_nmr_dictionary_features import _make_grid
from nmrlib.data import REPO_ROOT, load_dataset
from nmrlib.features import NMR_STATS_COLS, add_nmr_stats
from nmrlib.torch_nmf import SpectralGeometry, TorchNMF, TorchNMFConfig
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


def _has(v) -> bool:
    return isinstance(v, (list, np.ndarray)) and len(v) > 0


def logp_from_smiles(smiles: pd.Series) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Crippen
    out = np.full(len(smiles), np.nan)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(s)) if pd.notna(s) else None
        if mol is not None:
            out[i] = Crippen.MolLogP(mol)
    return out


def hgb_cv(Z: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 5) -> float:
    mask = np.isfinite(y)
    Zf, yf = Z[mask], y[mask]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for tr, te in kf.split(Zf):
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                          early_stopping=True, random_state=seed)
        m.fit(Zf[tr], yf[tr])
        scores.append(r2_score(yf[te], m.predict(Zf[te])))
    return float(np.mean(scores))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", type=Path, required=True,
                   help="pretraining corpus pickle (peak-lists), e.g. Datasets/nmrexp_100k.pkl")
    p.add_argument("--target", type=str, default="alberts_10k",
                   help="labeled target: registry name or path")
    p.add_argument("--device", default="cuda", help="torch device for the NMF fit")
    p.add_argument("--n-trials", type=int, default=40)
    p.add_argument("--n-iter", type=int, default=300, help="Adam iterations per NMF fit")
    p.add_argument("--h-step-ppm", type=float, default=0.01)
    p.add_argument("--c-step-ppm", type=float, default=0.25)
    p.add_argument("--cv", type=int, default=5, help="HGB CV folds on the target")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None,
                   help="results JSON (default: sweeps/pretrain_transfer_<target>.json)")
    p.add_argument("--storage", type=str, default=None, help="Optuna storage (default <out>.db)")
    args = p.parse_args()
    if args.out is None:
        tag = Path(args.target).stem
        args.out = REPO_ROOT / "sweeps" / f"pretrain_transfer_{tag}.json"
    return args


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    h_grid = _make_grid(0.0, 12.0, args.h_step_ppm)
    c_grid = _make_grid(0.0, 220.0, args.c_step_ppm)

    corpus = pd.read_pickle(args.corpus)
    corpus = corpus[corpus["h_nmr_peaks"].map(_has) & corpus["c_nmr_peaks"].map(_has)].reset_index(drop=True)

    target = load_dataset(args.target)
    target = target[target["h_nmr_peaks"].map(_has) & target["c_nmr_peaks"].map(_has)].reset_index(drop=True)
    gap = pd.to_numeric(target.get("gap_ev"), errors="coerce").to_numpy(float)
    logp = logp_from_smiles(target["smiles"])
    STATS = add_nmr_stats(target)[NMR_STATS_COLS].to_numpy(float)  # fixed across trials
    print(f"corpus {len(corpus)} molecules | target {args.target} {len(target)} rows "
          f"| gap finite {np.isfinite(gap).sum()} | logP finite {np.isfinite(logp).sum()}", flush=True)

    # raster caches (keyed by the params that affect Gaussian rasterization). Corpus
    # matrices are ~0.8 GB each at 100k, so keep only a couple.
    corpus_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
    target_cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()

    def rasterize(df, cache, cfg) -> np.ndarray:
        key = raster_key(cfg)
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        h, c = build_raw_matrices(df, h_grid, c_grid, cfg)
        x = assemble_x(h, c, cfg).astype(np.float32)
        cache[key] = x
        while len(cache) > 2:
            cache.popitem(last=False)
        return x

    def objective(trial: optuna.Trial):
        cfg = SweepConfig(
            name=f"trial_{trial.number}",
            n_components=trial.suggest_int("n_components", 10, 120, step=5),
            h_sigma_ppm=trial.suggest_float("h_sigma_ppm", 0.01, 0.10, step=0.01),
            c_sigma_ppm=trial.suggest_float("c_sigma_ppm", 0.25, 2.0, step=0.25),
            h_width_scale=trial.suggest_float("h_width_scale", 0.1, 1.0, step=0.1),
            c_width_scale=trial.suggest_float("c_width_scale", 0.5, 2.0, step=0.5),
            h_multiplicity=False,
            use_peak_width=trial.suggest_categorical("use_peak_width", [True, False]),
            intensity_transform=trial.suggest_categorical(
                "intensity_transform", ["none", "sqrt", "cbrt", "log1p", "arcsinh"]),
            h_modality_weight=trial.suggest_float("h_modality_weight", 0.1, 10.0, log=True),
            c_modality_weight=1.0,
            solver="cd", beta_loss="frobenius",
            alpha_W=trial.suggest_float("alpha_W", 1e-5, 1e-2, log=True),
            alpha_H=trial.suggest_float("alpha_H", 1e-5, 1e-2, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.0, 1.0),
        )
        t0 = time.time()
        Xc = rasterize(corpus, corpus_cache, cfg)
        Xt = rasterize(target, target_cache, cfg)
        geom = SpectralGeometry(h_coords=h_grid, c_coords=c_grid,
                                h_modality_weight=cfg.h_modality_weight, c_modality_weight=1.0)
        tcfg = TorchNMFConfig(
            loss="w1_grid", n_components=cfg.n_components, alpha_W=cfg.alpha_W,
            alpha_H=cfg.alpha_H, l1_ratio=cfg.l1_ratio, optimizer="adam", lr=0.05,
            n_iter=args.n_iter, transform_n_iter=args.n_iter, sinkhorn_p=1,
            mass_normalize=True, normalize_coords=True, device=args.device, dtype="float32")
        try:
            engine = TorchNMF(tcfg, geometry=geom)
            engine.fit_transform(Xc)          # pretrain on corpus
            W = engine.transform(Xt)          # transfer to target
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            raise optuna.TrialPruned(f"NMF fit failed: {exc}") from exc

        Z = np.hstack([W, STATS])
        gap_r2 = hgb_cv(Z, gap, args.seed, args.cv)
        logp_r2 = hgb_cv(Z, logp, args.seed, args.cv)
        codes_gap = hgb_cv(W, gap, args.seed, args.cv)  # codes-alone, for insight
        print(f"[{cfg.name}] nc={cfg.n_components} hσ={cfg.h_sigma_ppm} cσ={cfg.c_sigma_ppm}"
              f" -> gap(+stats)={gap_r2:.4f} logP(+stats)={logp_r2:.4f}"
              f" codes-only-gap={codes_gap:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        trial.set_user_attr("codes_only_gap_r2", codes_gap)
        return gap_r2, logp_r2

    storage = args.storage or f"sqlite:///{args.out.with_suffix('.db')}"
    study = optuna.create_study(
        study_name=f"pretrain_transfer_{Path(args.target).stem}",
        storage=storage, load_if_exists=True,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
    )
    print(f"Study @ {storage}  |  target objectives: [gap_ev R², logP R²] (codes+stats)", flush=True)

    def dump(study, _trial):
        rows = []
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            rows.append({"name": f"trial_{t.number}", **t.params,
                         "gap_ev_r2": t.values[0], "logp_r2": t.values[1],
                         "codes_only_gap_r2": t.user_attrs.get("codes_only_gap_r2")})
        args.out.write_text(json.dumps(rows, indent=2))

    study.optimize(objective, n_trials=args.n_trials, callbacks=[dump])
    dump(study, None)

    best = max(study.best_trials, key=lambda t: t.values[0])
    print("\nBest (by gap_ev R²):")
    print(f"  gap_ev={best.values[0]:.4f}  logP={best.values[1]:.4f}")
    print(f"  params={json.dumps(best.params, indent=2)}")
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
