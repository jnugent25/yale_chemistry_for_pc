"""Benchmark functional-group prediction with micro-F1, matching Alberts et al. setup.

Paper reference (Table 4):
  - Labels: 37 SMARTS functional groups (binary presence)
  - Metric: micro-F1 (see their run_xgb_baseline.py)
  - Split: 90/10, random_state=3245

This script compares Random Forest models on:
  1. NMR dictionary codes (our interpretable features)
  2. Raw 1H NMR spectrum vectors (paper-like input for 1H)
  3. Raw 13C NMR spectrum vectors (paper-like input for 13C)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

PAPER_REFERENCE = {
    "1H-NMR XGBoost": 0.797,
    "13C-NMR XGBoost": 0.804,
}

FUNCTIONAL_GROUPS = {
    "Acid anhydride": "[CX3](=[OX1])[OX2][CX3](=[OX1])",
    "Acyl halide": "[CX3](=[OX1])[F,Cl,Br,I]",
    "Alcohol": "[#6][OX2H]",
    "Aldehyde": "[CX3H1](=O)[#6,H]",
    "Alkane": "[CX4;H3,H2]",
    "Alkene": "[CX3]=[CX3]",
    "Alkyne": "[CX2]#[CX2]",
    "Amide": "[NX3][CX3](=[OX1])[#6]",
    "Amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "Arene": "[cX3]1[cX3][cX3][cX3][cX3][cX3]1",
    "Azo compound": "[#6][NX2]=[NX2][#6]",
    "Carbamate": "[NX3][CX3](=[OX1])[OX2H0]",
    "Carboxylic acid": "[CX3](=O)[OX2H]",
    "Enamine": "[NX3][CX3]=[CX3]",
    "Enol": "[OX2H][#6X3]=[#6]",
    "Ester": "[#6][CX3](=O)[OX2H0][#6]",
    "Ether": "[OD2]([#6])[#6]",
    "Haloalkane": "[#6][F,Cl,Br,I]",
    "Hydrazine": "[NX3][NX3]",
    "Hydrazone": "[NX3][NX2]=[#6]",
    "Imide": "[CX3](=[OX1])[NX3][CX3](=[OX1])",
    "Imine": "[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]",
    "Isocyanate": "[NX2]=[C]=[O]",
    "Isothiocyanate": "[NX2]=[C]=[S]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Nitrile": "[NX1]#[CX2]",
    "Phenol": "[OX2H][cX3]:[c]",
    "Phosphine": "[PX3]",
    "Sulfide": "[#16X2H0]",
    "Sulfonamide": "[#16X4]([NX3])(=[OX1])(=[OX1])[#6]",
    "Sulfonate": "[#16X4](=[OX1])(=[OX1])([#6])[OX2H0]",
    "Sulfone": "[#16X4](=[OX1])(=[OX1])([#6])[#6]",
    "Sulfonic acid": "[#16X4](=[OX1])(=[OX1])([#6])[OX2H]",
    "Sulfoxide": "[#16X3]=[OX1]",
    "Thial": "[CX3H1](=S)[#6,H]",
    "Thioamide": "[NX3][CX3]=[SX1]",
    "Thiol": "[#16X2H]",
}


def label_matrix(smiles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    compiled = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUPS.items()}
    names = list(FUNCTIONAL_GROUPS)
    y = np.zeros((len(smiles), len(names)), dtype=np.int8)
    valid = np.ones(len(smiles), dtype=bool)

    for row_idx, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            valid[row_idx] = False
            continue
        for col_idx, name in enumerate(names):
            pattern = compiled[name]
            if pattern is not None and mol.HasSubstructMatch(pattern):
                y[row_idx, col_idx] = 1

    return y, valid


def fit_predict_per_label_rf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> np.ndarray:
    predictions = np.zeros((len(x_test), y_train.shape[1]), dtype=np.int8)
    for label_idx in range(y_train.shape[1]):
        if y_train[:, label_idx].sum() == 0:
            continue
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        clf.fit(x_train, y_train[:, label_idx])
        predictions[:, label_idx] = clf.predict(x_test)
    return predictions


def evaluate(name: str, y_test: np.ndarray, y_pred: np.ndarray) -> dict:
    micro = float(f1_score(y_test, y_pred, average="micro", zero_division=0))
    macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    return {"model": name, "micro_f1": micro, "macro_f1": macro}


def stack_spectra(series: pd.Series) -> np.ndarray:
    return np.stack(series.to_numpy()).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark functional-group micro-F1 vs Alberts et al.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("/Users/jacknugent/Downloads/ids_nmr_1k_dictionary_features.pkl"),
    )
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=Path("/Users/jacknugent/Downloads/ids_nmr_100k.pkl"),
    )
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=3245)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip raw 1H/13C spectrum benchmarks (faster, codes only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    features = pd.read_pickle(args.features)
    code_cols = [c for c in features.columns if c.startswith("nmr_dict_code_")]
    if not code_cols:
        raise ValueError("No nmr_dict_code_* columns found in features file.")

    print(f"Loaded features: {features.shape[0]} rows, {len(code_cols)} codes")
    y_all, valid = label_matrix(features["smiles"])
    features = features.loc[valid].reset_index(drop=True)
    y_all = y_all[valid]

    idx = np.arange(len(features))
    idx_train, idx_test = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed
    )

    y_train, y_test = y_all[idx_train], y_all[idx_test]
    results: list[dict] = []

    # 1) Our interpretable dictionary codes (joint H+C)
    x_codes = features[code_cols].to_numpy(dtype=np.float32)
    pred_codes = fit_predict_per_label_rf(
        x_codes[idx_train],
        y_train,
        x_codes[idx_test],
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )
    results.append(
        evaluate("RF on NMR dictionary codes (H+C, interpretable)", y_test, pred_codes)
    )

    if not args.skip_raw:
        print("Loading raw spectra for paper-like baselines...")
        raw = pd.read_pickle(args.raw_data)
        if len(raw) != len(features):
            raw = raw.loc[valid].reset_index(drop=True)

        x_h = stack_spectra(raw["h_nmr_spectra"])
        pred_h = fit_predict_per_label_rf(
            x_h[idx_train], y_train, x_h[idx_test], args.n_estimators, args.seed
        )
        results.append(evaluate("RF on raw 1H spectrum (10k vector)", y_test, pred_h))

        x_c = stack_spectra(raw["c_nmr_spectra"])
        pred_c = fit_predict_per_label_rf(
            x_c[idx_train], y_train, x_c[idx_test], args.n_estimators, args.seed
        )
        results.append(evaluate("RF on raw 13C spectrum (10k vector)", y_test, pred_c))

    print("\nFunctional-group benchmark (micro-F1, paper protocol)")
    print(f"Rows: {len(features):,} | split: {1-args.test_size:.0%}/{args.test_size:.0%} | seed: {args.seed}")
    print(f"Labels: {y_all.shape[1]} SMARTS groups | RF n_estimators={args.n_estimators}\n")

    print(f"{'Model':<48} {'micro-F1':>10} {'macro-F1':>10}")
    print("-" * 70)
    for row in results:
        print(f"{row['model']:<48} {row['micro_f1']:>10.4f} {row['macro_f1']:>10.4f}")

    print("\nPaper reference (XGBoost, micro-F1, Table 4):")
    for name, score in PAPER_REFERENCE.items():
        print(f"  {name:<24} {score:.3f}")

    codes_micro = results[0]["micro_f1"]
    print("\nDirect comparison (dictionary codes vs paper XGBoost):")
    print(f"  Our RF (H+C codes):     micro-F1 = {codes_micro:.4f}")
    print(f"  Paper XGBoost (1H only): micro-F1 = {PAPER_REFERENCE['1H-NMR XGBoost']:.3f}")
    print(f"  Paper XGBoost (13C only): micro-F1 = {PAPER_REFERENCE['13C-NMR XGBoost']:.3f}")
    print(f"  Gap vs paper 1H:        {codes_micro - PAPER_REFERENCE['1H-NMR XGBoost']:+.4f}")
    print(f"  Gap vs paper 13C:       {codes_micro - PAPER_REFERENCE['13C-NMR XGBoost']:+.4f}")


if __name__ == "__main__":
    main()
