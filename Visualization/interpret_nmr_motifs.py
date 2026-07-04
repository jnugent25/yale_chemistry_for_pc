"""Interpretability tooling for the shared H/C NMR dictionary features.

Two capabilities:

1. ``characterize_motifs`` turns each learned NMF code into a human-readable
   "motif signature" (dominant H/C ppm peaks, region label, example molecules,
   and optional RDKit substructure correlations). This is the lookup that maps
   "the model used code_k" back to recognizable chemistry.

2. ``attribute_property`` trains a model on the codes to predict a property and
   reports which motifs drive it, each annotated with its signature.

The motif signature step needs only the saved model + features. The optional
substructure correlations and the proxy demo target need RDKit, which is
already a project dependency.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


H_REGIONS = [
    ("aldehyde H", 9.0, 10.5),
    ("acid/H-bond H", 10.5, 12.0),
    ("aromatic H", 6.5, 8.5),
    ("olefinic H", 5.0, 6.5),
    ("O/N-adjacent H", 3.0, 5.0),
    ("alpha/allylic H", 1.5, 3.0),
    ("alkyl H", 0.0, 1.5),
]
C_REGIONS = [
    ("ketone/aldehyde C=O", 185, 220),
    ("ester/amide/acid C=O", 160, 185),
    ("aromatic/alkene C", 110, 160),
    ("anomeric/alkynyl C", 90, 110),
    ("C-O/C-N/halogen C", 50, 90),
    ("alkyl C", 0, 50),
]

# SMARTS motifs used for optional structure correlation. Kept deliberately
# small and chemically broad.
SMARTS_PATTERNS = {
    "aromatic ring": "a1aaaaa1",
    "heteroaromatic": "[a;!c]",
    "ester": "[CX3](=O)[OX2][#6]",
    "amide": "[NX3][CX3](=O)[#6,#7,#8]",
    "carboxylic acid": "[CX3](=O)[OX2H1]",
    "aldehyde": "[CX3H1](=O)[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "ether": "[OD2]([#6])[#6]",
    "methoxy": "[OX2]([CH3])[#6]",
    "alcohol": "[OX2H][#6]",
    "amine": "[NX3;H2,H1,H0;!$(NC=O)]",
    "nitrile": "[CX2]#N",
    "sulfonyl": "S(=O)(=O)",
    "halogen": "[F,Cl,Br,I]",
    "tert-butyl": "C(C)(C)C",
    "long alkyl chain": "CCCCCC",
    "alkene": "C=C",
    "alkyne": "C#C",
    "morpholine": "N1CCOCC1",
}


def load_artifacts(features_path: Path, model_path: Path) -> tuple[pd.DataFrame, dict]:
    features = pd.read_pickle(features_path)
    with Path(model_path).open("rb") as file:
        model = pickle.load(file)
    return features, model


def _top_region(component: np.ndarray, grid: np.ndarray, regions: list) -> tuple[str, float]:
    total = float(component.sum()) or 1.0
    scored = [
        (name, float(component[(grid >= lo) & (grid < hi)].sum() / total))
        for name, lo, hi in regions
    ]
    return max(scored, key=lambda item: item[1])


def _top_peaks(component: np.ndarray, grid: np.ndarray, n: int = 3, min_height_frac: float = 0.05) -> list:
    if component.max() <= 0:
        return []
    threshold = min_height_frac * component.max()
    peaks = []
    for i in range(1, len(component) - 1):
        if component[i] > threshold and component[i] >= component[i - 1] and component[i] >= component[i + 1]:
            peaks.append((float(grid[i]), float(component[i])))
    peaks.sort(key=lambda item: item[1], reverse=True)
    return [round(ppm, 2) for ppm, _ in peaks[:n]]


def _structure_correlations(
    codes: pd.DataFrame,
    smiles: pd.Series,
    sample_size: int,
    top_n: int,
    random_state: int = 0,
) -> dict[str, list[tuple[str, float]]]:
    """For each code, the SMARTS motifs whose presence best correlates with it."""

    try:
        from rdkit import Chem
    except ImportError:
        return {}

    n = min(sample_size, len(codes))
    idx = np.random.default_rng(random_state).choice(len(codes), size=n, replace=False)
    sample_codes = codes.iloc[idx].reset_index(drop=True)
    sample_smiles = smiles.iloc[idx].reset_index(drop=True)

    compiled = {name: Chem.MolFromSmarts(s) for name, s in SMARTS_PATTERNS.items()}
    presence = {name: np.zeros(n, dtype=float) for name in SMARTS_PATTERNS}

    for row_idx, smi in enumerate(sample_smiles):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            continue
        for name, patt in compiled.items():
            if patt is not None and mol.HasSubstructMatch(patt):
                presence[name][row_idx] = 1.0

    out: dict[str, list[tuple[str, float]]] = {}
    for code in sample_codes.columns:
        col = sample_codes[code].to_numpy(dtype=float)
        if np.std(col) == 0:
            out[code] = []
            continue
        corrs = []
        for name, ind in presence.items():
            if np.std(ind) == 0:
                continue
            r = float(np.corrcoef(col, ind)[0, 1])
            if np.isfinite(r):
                corrs.append((name, round(r, 3)))
        corrs.sort(key=lambda item: item[1], reverse=True)
        out[code] = corrs[:top_n]
    return out


def characterize_motifs(
    features: pd.DataFrame,
    model: dict,
    n_examples: int = 3,
    with_structure: bool = True,
    structure_sample: int = 8000,
) -> pd.DataFrame:
    """Build a per-code motif signature table."""

    code_cols = model["feature_columns"]
    h_grid = model["h_grid_ppm"]
    c_grid = model["c_grid_ppm"]
    h_components = model["h_components"]
    c_components = model["c_components"]
    codes = features[code_cols]

    correlations: dict[str, list] = {}
    if with_structure and "smiles" in features.columns:
        correlations = _structure_correlations(
            codes, features["smiles"], structure_sample, top_n=4
        )

    records = []
    for k, code in enumerate(code_cols):
        activation = codes[code].to_numpy(dtype=float)
        top_positions = activation.argsort()[::-1][:n_examples]
        examples = []
        for pos in top_positions:
            row = features.iloc[pos]
            examples.append(
                {
                    "formula": str(row.get("molecular_formula", "")),
                    "smiles": str(row.get("smiles", ""))[:80],
                    "activation": round(float(row[code]), 4),
                }
            )

        h_region = _top_region(h_components[k], h_grid, H_REGIONS)
        c_region = _top_region(c_components[k], c_grid, C_REGIONS)

        records.append(
            {
                "code": code,
                "mean_activation": round(float(activation.mean()), 5),
                "used_fraction": round(float((activation > 1e-8).mean()), 3),
                "H_region": h_region[0],
                "H_region_share": round(h_region[1], 3),
                "H_peaks_ppm": _top_peaks(h_components[k], h_grid),
                "C_region": c_region[0],
                "C_region_share": round(c_region[1], 3),
                "C_peaks_ppm": _top_peaks(c_components[k], c_grid),
                "structure_correlations": correlations.get(code, []),
                "example_molecules": examples,
            }
        )

    return pd.DataFrame.from_records(records)


def _compute_proxy_target(smiles: pd.Series, kind: str) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    funcs = {
        "aromatic_rings": lambda m: Lipinski.NumAromaticRings(m),
        "tpsa": lambda m: Descriptors.TPSA(m),
        "logp": lambda m: Descriptors.MolLogP(m),
        "mol_wt": lambda m: Descriptors.MolWt(m),
        "fraction_csp3": lambda m: rdMolDescriptors.CalcFractionCSP3(m),
    }
    if kind not in funcs:
        raise ValueError(f"Unsupported proxy target: {kind}")

    values = np.full(len(smiles), np.nan)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is not None:
            values[i] = float(funcs[kind](mol))
    return values


def attribute_property(
    features: pd.DataFrame,
    model: dict,
    target: np.ndarray,
    top_k: int = 12,
    sample_size: int | None = 20000,
    random_state: int = 0,
) -> pd.DataFrame:
    """Train a RandomForest on the codes and rank motifs by permutation importance."""

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split

    code_cols = model["feature_columns"]
    x = features[code_cols].to_numpy(dtype=float)
    y = np.asarray(target, dtype=float)

    mask = np.isfinite(y)
    x, y = x[mask], y[mask]

    if sample_size is not None and len(y) > sample_size:
        idx = np.random.default_rng(random_state).choice(len(y), size=sample_size, replace=False)
        x, y = x[idx], y[idx]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=random_state
    )
    forest = RandomForestRegressor(
        n_estimators=300, n_jobs=-1, random_state=random_state, max_depth=None
    )
    forest.fit(x_train, y_train)
    test_r2 = float(forest.score(x_test, y_test))

    perm = permutation_importance(
        forest, x_test, y_test, n_repeats=8, random_state=random_state, n_jobs=-1
    )

    signatures = characterize_motifs(features, model, with_structure=False).set_index("code")

    rows = []
    order = np.argsort(perm.importances_mean)[::-1][:top_k]
    for j in order:
        code = code_cols[j]
        sig = signatures.loc[code]
        rows.append(
            {
                "code": code,
                "perm_importance": round(float(perm.importances_mean[j]), 5),
                "impurity_importance": round(float(forest.feature_importances_[j]), 5),
                "H_region": sig["H_region"],
                "H_peaks_ppm": sig["H_peaks_ppm"],
                "C_region": sig["C_region"],
                "C_peaks_ppm": sig["C_peaks_ppm"],
            }
        )

    result = pd.DataFrame(rows)
    result.attrs["test_r2"] = test_r2
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpret learned NMR dictionary motifs.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("/Users/jacknugent/Downloads/ids_nmr_1k_dictionary_features.pkl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/Users/jacknugent/Downloads/ids_nmr_1k_dictionary_model.pkl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/jacknugent/Downloads/ids_nmr_motif_signatures.pkl"),
    )
    parser.add_argument("--no-structure", dest="with_structure", action="store_false")
    parser.add_argument("--structure-sample", type=int, default=8000)
    parser.add_argument(
        "--demo-target",
        choices=["aromatic_rings", "tpsa", "logp", "mol_wt", "fraction_csp3"],
        default=None,
        help="If set, run an attribution demo predicting this RDKit property.",
    )
    parser.set_defaults(with_structure=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features, model = load_artifacts(args.features, args.model)

    signatures = characterize_motifs(
        features,
        model,
        with_structure=args.with_structure,
        structure_sample=args.structure_sample,
    )
    signatures.to_pickle(args.output)
    print(f"Wrote motif signatures: {args.output}  ({len(signatures)} codes)")

    if args.demo_target is not None:
        target = _compute_proxy_target(features["smiles"], args.demo_target)
        attribution = attribute_property(features, model, target)
        print(
            f"\nAttribution demo — predicting '{args.demo_target}' "
            f"(test R^2 = {attribution.attrs['test_r2']:.3f}). Top motifs:"
        )
        with pd.option_context("display.max_colwidth", 40, "display.width", 160):
            print(attribution.to_string(index=False))


if __name__ == "__main__":
    main()
