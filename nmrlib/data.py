"""Dataset registry, path resolution, and column-alias normalization.

All notebooks and scripts should load data through ``load_dataset`` so that
dataset switching is a one-word config change and column naming is consistent
regardless of which source pickle the frame came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "Datasets"
DOWNLOADS_DIR = Path.home() / "Downloads"


@dataclass(frozen=True)
class DatasetInfo:
    """One registry entry: the pickle filename and a human description."""

    filename: str
    description: str


# Short name -> (pickle filename, description). Files are looked up in
# Datasets/ first, then ~/Downloads as a fallback for pickles not yet moved
# into the repo. Use describe_datasets() to see this with local availability.
DATASETS: dict[str, DatasetInfo] = {
    "alberts_10k": DatasetInfo(
        "alberts_nmr_qchem_merged.pkl",
        "Alberts et al. 10k subset merged with qchem targets (gap/homo/lumo).",
    ),
    "alberts_10k_logp": DatasetInfo(
        "alberts_merged_10k_with_logp.pkl",
        "Alberts 10k with logP; raw spectra, pre-featurization input.",
    ),
    "alberts_10k_100kdict": DatasetInfo(
        "alberts_10k_100kdict_nmf_features.pkl",
        "Alberts 10k with NMF codes from the dictionary fit on the 100k corpus.",
    ),
    "ids_nmr_1k": DatasetInfo(
        "ids_nmr_1k.pkl", "IDS NMR corpus, 1k unlabeled spectra for dictionary learning."
    ),
    "ids_nmr_10k": DatasetInfo(
        "ids_nmr_10k.pkl", "IDS NMR corpus, 10k unlabeled spectra for dictionary learning."
    ),
    "ids_nmr_100k": DatasetInfo(
        "ids_nmr_100k.pkl", "IDS NMR corpus, 100k unlabeled spectra for dictionary learning."
    ),
    "ids_1k_featurized": DatasetInfo(
        "ids_1k_nmf_115_and_other.pkl",
        "IDS 1k fully featurized (115-component NMF + An 2014, 13C bins, stats).",
    ),
    "ids_1k_tuned_nmf": DatasetInfo(
        "ids_nmr_1k_tuned_nmf_features.pkl", "IDS 1k with tuned-NMF dictionary codes."
    ),
    "gaussian_1k": DatasetInfo(
        "gaussian_nmr_matched_1k.pkl", "Gaussian-matched 1k set with NMR spectra."
    ),
    "qm9": DatasetInfo(
        "qm9_feat.pkl",
        "QM9 127k with gap_ev + RDKit descriptors; NMR peaks as flat ppm lists "
        "(H_ppm/C_ppm, normalized to canonical peak dicts on load).",
    ),
    "nmrshiftdb": DatasetInfo(
        "nmrshift_13C_1H_feat 2.pkl",
        "nmrshiftdb2 3.4k with gap/homo/lumo + descriptors; experimental 13C/1H "
        "shift lists (spectrum_13C/spectrum_1H, normalized to peak dicts on load).",
    ),
}

# Canonical column name -> aliases seen across source datasets.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "smiles": ("canonical_smiles", "SMILES", "smiles_x", "smiles_y"),
    "gap_ev": ("qchem_gap_ev",),
    "homo_ev": ("qchem_homo_ev",),
    "lumo_ev": ("qchem_lumo_ev",),
}

# Some sources store peaks as bare lists of chemical-shift floats (one entry per
# proton/carbon, no intensity/width/multiplicity) instead of the canonical list
# of peak dicts the rasterizer reads. Map those flat columns to the canonical
# peak column, wrapping each shift into the minimal dict build_soft_peak_matrix
# understands (unit weight, no physical width -> pure shift-tolerance Gaussian).
FLAT_PEAK_ALIASES: dict[str, tuple[str, ...]] = {
    "h_nmr_peaks": ("H_ppm", "spectrum_1H"),
    "c_nmr_peaks": ("C_ppm", "spectrum_13C"),
}
# (center_key, weight_key) the rasterizer reads for each canonical peak column.
_PEAK_DICT_KEYS: dict[str, tuple[str, str]] = {
    "h_nmr_peaks": ("delta", "nH"),
    "c_nmr_peaks": ("delta (ppm)", "intensity"),
}


def _wrap_flat_peaks(values: object, center_key: str, weight_key: str) -> object:
    """Turn a flat list of shift floats into canonical peak dicts. Passes through
    entries that are already dicts, and missing/NaN values untouched."""
    if values is None or (isinstance(values, float) and pd.isna(values)):
        return values
    out: list[dict] = []
    for v in values:
        if isinstance(v, dict):
            out.append(v)
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        out.append({center_key: f, weight_key: 1.0})
    return out


def normalize_peak_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical ``h_nmr_peaks``/``c_nmr_peaks`` columns exist, building
    them from flat shift-list aliases (H_ppm/spectrum_1H, C_ppm/spectrum_13C)
    when the canonical column is absent."""
    built: dict[str, pd.Series] = {}
    for canonical, aliases in FLAT_PEAK_ALIASES.items():
        if canonical in df.columns:
            continue
        center_key, weight_key = _PEAK_DICT_KEYS[canonical]
        for alias in aliases:
            if alias in df.columns:
                built[canonical] = df[alias].map(
                    lambda v: _wrap_flat_peaks(v, center_key, weight_key)
                )
                break
    return df.assign(**built) if built else df


def resolve_dataset(name_or_path: str | Path) -> Path:
    """Resolve a registry short name or a path to an existing pickle file."""
    candidates: list[Path] = []
    key = str(name_or_path)
    if key in DATASETS:
        filename = DATASETS[key].filename
        candidates = [DATA_DIR / filename, DOWNLOADS_DIR / filename]
    else:
        p = Path(name_or_path).expanduser()
        candidates = [p, DATA_DIR / p.name, DOWNLOADS_DIR / p.name]
    for c in candidates:
        if c.exists():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    known = ", ".join(sorted(DATASETS))
    raise FileNotFoundError(
        f"Could not find dataset {name_or_path!r}. Tried:\n  {tried}\n"
        f"Known dataset names: {known}"
    )


def describe_datasets() -> pd.DataFrame:
    """Registry table (name, description, filename, whether found locally).

    Handy at the top of a notebook to see which dataset short names are
    available before choosing one in the config cell.
    """
    rows = []
    for name, info in DATASETS.items():
        try:
            found = str(resolve_dataset(name).parent.name) + "/"
        except FileNotFoundError:
            found = "—"
        rows.append({
            "name": name,
            "description": info.description,
            "file": info.filename,
            "found_in": found,
        })
    return pd.DataFrame(rows).set_index("name")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to canonical column names (only when the canonical
    name is not already present), so downstream code can rely on ``smiles``,
    ``gap_ev``, ``homo_ev``, ``lumo_ev``."""
    renames: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                renames[alias] = canonical
                break
    return df.rename(columns=renames) if renames else df


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated columns with the same name, keeping the first occurrence.

    Some source pickles carry duplicated feature columns from repeated merges;
    duplicates with differing values are reported before dropping.
    """
    dup_names = df.columns[df.columns.duplicated()].unique()
    if len(dup_names) == 0:
        return df
    for name in dup_names:
        block = df.loc[:, df.columns == name]
        if not block.T.duplicated(keep=False).all():
            print(f"Warning: duplicated column {name!r} has differing copies; keeping the first")
    print(f"Dropped {int(df.columns.duplicated().sum())} duplicate columns ({len(dup_names)} names)")
    return df.loc[:, ~df.columns.duplicated()]


def load_dataset(name_or_path: str | Path, normalize: bool = True) -> pd.DataFrame:
    """Load a dataset by registry name or path, normalizing column aliases
    and dropping duplicated columns."""
    path = resolve_dataset(name_or_path)
    df = pd.read_pickle(path)
    if normalize:
        df = normalize_peak_columns(dedupe_columns(normalize_columns(df)))
    print(f"Loaded {path} — {df.shape[0]} rows x {df.shape[1]} columns")
    return df
