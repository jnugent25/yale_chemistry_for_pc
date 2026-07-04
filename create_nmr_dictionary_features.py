from __future__ import annotations

import argparse
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


DEFAULT_INPUT = Path('/Users/jacknugent/Downloads/ids_nmr_10k_filtered.pkl')


@dataclass(frozen=True)
class NMRDictionaryConfig:
    """Settings for the shared H/C NMR dictionary feature learner."""

    n_components: int = 64
    random_state: int = 42
    max_iter: int = 5000
    h_min_ppm: float = 0.0
    h_max_ppm: float = 12.0
    h_step_ppm: float = 0.01
    h_sigma_ppm: float = 0.06
    c_min_ppm: float = 0.0
    c_max_ppm: float = 220.0
    c_step_ppm: float = 0.25
    c_sigma_ppm: float = 1.0
    h_modality_weight: float = 1.0
    c_modality_weight: float = 1.0

    # When True, each peak's width is taken from the data (H: rangeMax-rangeMin,
    # C: width (ppm)) and combined with the *_sigma_ppm shift-tolerance baseline
    # by adding variances: sigma_eff = sqrt(physical**2 + tolerance**2). The
    # *_sigma_ppm values then act as a tolerance floor rather than a fixed width.
    use_peak_width: bool = True
    h_width_scale: float = 0.25
    c_width_scale: float = 1.0
    # Elementwise transform applied to the soft fingerprints before NMF, to keep
    # tall peaks from dominating the Frobenius loss on spiky spectra. One of
    # "none", "sqrt" (Hellinger-style), or "log1p". Reconstruction error is
    # always reported in the original intensity space for comparability.
    # Default "none": on the 1k set, sqrt/log1p did not improve original-space
    # reconstruction error (sqrt regressed carbon), so this stays opt-in for
    # downstream-task A/B testing rather than on by default.
    intensity_transform: str = "none"
    # When True, the proton side is split into one ppm channel per multiplicity
    # class (s/d/t/q/m). Each peak's nH mass is routed to the channel matching
    # its ``category`` before the channels are stacked for NMF. A learned motif
    # can then mean e.g. "singlet near 3.9 ppm" (methoxy) vs "triplet near 1.2"
    # (ethyl CH3). C stays single-channel (decoupled = singlets). H rows are
    # normalized jointly across channels, preserving the relative split of
    # protons across multiplicities. Opt-in for A/B vs the single-channel H.
    h_multiplicity_channels: bool = False
    h_multiplicity_classes: tuple = ("s", "d", "t", "q", "m")


VALID_TRANSFORMS = ("none", "sqrt", "cbrt", "log1p", "arcsinh")


def _apply_transform(x: np.ndarray, kind: str) -> np.ndarray:
    if kind == "none":
        return x
    if kind == "sqrt":
        return np.sqrt(np.clip(x, 0.0, None))
    if kind == "cbrt":
        return np.cbrt(np.clip(x, 0.0, None))
    if kind == "log1p":
        return np.log1p(np.clip(x, 0.0, None))
    if kind == "arcsinh":
        return np.arcsinh(np.clip(x, 0.0, None))
    raise ValueError(f"Unsupported intensity_transform: {kind}")


def _invert_transform(x: np.ndarray, kind: str) -> np.ndarray:
    x = np.clip(x, 0.0, None)
    if kind == "none":
        return x
    if kind == "sqrt":
        return x ** 2
    if kind == "cbrt":
        return x ** 3
    if kind == "log1p":
        return np.expm1(x)
    if kind == "arcsinh":
        return np.sinh(x)
    raise ValueError(f"Unsupported intensity_transform: {kind}")


def _make_grid(min_ppm: float, max_ppm: float, step_ppm: float) -> np.ndarray:
    # Include the right endpoint when it lands exactly on the grid.
    return np.arange(min_ppm, max_ppm + 0.5 * step_ppm, step_ppm, dtype=np.float64)


def _safe_float(value: object, default: float | None = None) -> float | None:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_peak_dicts(peaks: object) -> Iterable[dict]:
    if peaks is None:
        return
    if isinstance(peaks, float) and pd.isna(peaks):
        return

    for peak in peaks:
        if isinstance(peak, dict):
            yield peak


def _add_gaussian_peak(
    row: np.ndarray,
    grid: np.ndarray,
    center_ppm: float,
    weight: float,
    sigma_ppm: float,
    radius_sigma: float = 4.0,
) -> None:
    if weight <= 0 or sigma_ppm <= 0:
        return

    lo = center_ppm - radius_sigma * sigma_ppm
    hi = center_ppm + radius_sigma * sigma_ppm
    left = int(np.searchsorted(grid, lo, side="left"))
    right = int(np.searchsorted(grid, hi, side="right"))
    if left >= right:
        return

    local_grid = grid[left:right]
    row[left:right] += weight * np.exp(-0.5 * ((local_grid - center_ppm) / sigma_ppm) ** 2)


def _peak_physical_sigma(
    peak: dict,
    modality: str,
    width_scale: float,
) -> float | None:
    """Per-peak physical width (in ppm) taken from the peak metadata, or None.

    H peaks report a multiplet span via ``rangeMin``/``rangeMax``; we treat that
    span as roughly four standard deviations, so the default ``width_scale`` of
    0.25 maps span -> sigma. C peaks report ``width (ppm)`` directly.
    """

    if modality == "h":
        range_min = _safe_float(peak.get("rangeMin"))
        range_max = _safe_float(peak.get("rangeMax"))
        if range_min is None or range_max is None:
            return None
        span = range_max - range_min
        if span <= 0:
            return None
        return span * width_scale

    if modality == "c":
        width = _safe_float(peak.get("width (ppm)"))
        if width is None or width <= 0:
            return None
        return width * width_scale

    raise ValueError(f"Unsupported modality: {modality}")


def build_soft_peak_matrix(
    peak_series: pd.Series,
    grid: np.ndarray,
    sigma_ppm: float,
    modality: str,
    width_scale: float = 0.25,
    use_peak_width: bool = True,
) -> np.ndarray:
    """Convert sparse peak lists into smooth, shift-tolerant spectra on a ppm grid.

    ``sigma_ppm`` is the shift-tolerance baseline. When ``use_peak_width`` is
    True, each peak also contributes its data-reported physical width, combined
    with the baseline via ``sigma_eff = sqrt(physical**2 + sigma_ppm**2)``.
    """

    matrix = np.zeros((len(peak_series), len(grid)), dtype=np.float32)

    for row_idx, peaks in enumerate(peak_series):
        row = matrix[row_idx]

        for peak in _iter_peak_dicts(peaks):
            if modality == "h":
                center = _safe_float(peak.get("delta"), _safe_float(peak.get("centroid")))
                weight = _safe_float(peak.get("nH"), 1.0)
            elif modality == "c":
                center = _safe_float(peak.get("delta (ppm)"))
                weight = _safe_float(
                    peak.get("integral"),
                    _safe_float(peak.get("intensity"), 1.0),
                )
            else:
                raise ValueError(f"Unsupported modality: {modality}")

            if center is None or weight is None:
                continue
            if center < grid[0] or center > grid[-1]:
                continue

            sigma = sigma_ppm
            if use_peak_width:
                physical = _peak_physical_sigma(peak, modality, width_scale)
                if physical is not None:
                    sigma = float(np.hypot(physical, sigma_ppm))

            _add_gaussian_peak(row, grid, center, weight, sigma)

        total = row.sum()
        if total > 0:
            row /= total

    return matrix


def _peak_multiplicity_class(peak: dict, classes: tuple[str, ...]) -> str:
    """Bucket a peak's ``category`` into one of ``classes``.

    Exact simple multiplicities (s/d/t/q) map to themselves; compound or unknown
    categories (dd, dt, br, complex multiplets, missing) fall into the final
    class, which is the catch-all ``m`` by default.
    """

    catch_all = classes[-1]
    cat = peak.get("category")
    if not isinstance(cat, str):
        return catch_all
    token = cat.strip().lower()
    return token if token in classes else catch_all


def build_h_multichannel_matrix(
    peak_series: pd.Series,
    grid: np.ndarray,
    sigma_ppm: float,
    classes: tuple[str, ...],
    width_scale: float = 0.25,
    use_peak_width: bool = True,
) -> np.ndarray:
    """Proton soft fingerprint split into one ppm channel per multiplicity class.

    Returns a matrix of width ``len(classes) * len(grid)`` where channel ``k``
    spans columns ``[k*len(grid) : (k+1)*len(grid)]``. Each row is normalized
    jointly across all channels so the relative proton split across
    multiplicities is preserved.
    """

    grid_len = len(grid)
    class_index = {name: idx for idx, name in enumerate(classes)}
    matrix = np.zeros((len(peak_series), len(classes) * grid_len), dtype=np.float32)

    for row_idx, peaks in enumerate(peak_series):
        row = matrix[row_idx]

        for peak in _iter_peak_dicts(peaks):
            center = _safe_float(peak.get("delta"), _safe_float(peak.get("centroid")))
            weight = _safe_float(peak.get("nH"), 1.0)
            if center is None or weight is None:
                continue
            if center < grid[0] or center > grid[-1]:
                continue

            sigma = sigma_ppm
            if use_peak_width:
                physical = _peak_physical_sigma(peak, "h", width_scale)
                if physical is not None:
                    sigma = float(np.hypot(physical, sigma_ppm))

            channel = class_index[_peak_multiplicity_class(peak, classes)]
            sub = row[channel * grid_len : (channel + 1) * grid_len]
            _add_gaussian_peak(sub, grid, center, weight, sigma)

        total = row.sum()
        if total > 0:
            row /= total

    return matrix


def fit_shared_dictionary(
    df: pd.DataFrame,
    config: NMRDictionaryConfig,
) -> tuple[pd.DataFrame, dict]:
    """Learn one shared code per molecule and modality-specific H/C components.

    The NMF code is the feature vector. Its H slice reconstructs the proton
    peak fingerprint; its C slice reconstructs the carbon peak fingerprint.
    """

    h_grid = _make_grid(config.h_min_ppm, config.h_max_ppm, config.h_step_ppm)
    c_grid = _make_grid(config.c_min_ppm, config.c_max_ppm, config.c_step_ppm)

    if config.h_multiplicity_channels:
        h_matrix = build_h_multichannel_matrix(
            df["h_nmr_peaks"],
            h_grid,
            config.h_sigma_ppm,
            classes=config.h_multiplicity_classes,
            width_scale=config.h_width_scale,
            use_peak_width=config.use_peak_width,
        )
        h_channels = list(config.h_multiplicity_classes)
    else:
        h_matrix = build_soft_peak_matrix(
            df["h_nmr_peaks"],
            h_grid,
            config.h_sigma_ppm,
            modality="h",
            width_scale=config.h_width_scale,
            use_peak_width=config.use_peak_width,
        )
        h_channels = ["all"]
    c_matrix = build_soft_peak_matrix(
        df["c_nmr_peaks"],
        c_grid,
        config.c_sigma_ppm,
        modality="c",
        width_scale=config.c_width_scale,
        use_peak_width=config.use_peak_width,
    )

    h_transformed = _apply_transform(h_matrix, config.intensity_transform)
    c_transformed = _apply_transform(c_matrix, config.intensity_transform)

    x = np.concatenate(
        [
            config.h_modality_weight * h_transformed,
            config.c_modality_weight * c_transformed,
        ],
        axis=1,
    )

    nmf = NMF(
        n_components=config.n_components,
        init="nndsvda",
        random_state=config.random_state,
        max_iter=config.max_iter,
        solver="mu",
        beta_loss="kullback-leibler",
        alpha_W=0.0,
        alpha_H=0.01,
        l1_ratio=0.9
    )
    codes = nmf.fit_transform(x)
    weighted_components = nmf.components_

    h_width = h_matrix.shape[1]
    h_components = weighted_components[:, :h_width] / config.h_modality_weight
    c_components = weighted_components[:, h_width:] / config.c_modality_weight

    # NMF reconstructs in transformed space; map back to the original intensity
    # space so the reported error is comparable across transform choices.
    h_reconstruction = _invert_transform(codes @ h_components, config.intensity_transform)
    c_reconstruction = _invert_transform(codes @ c_components, config.intensity_transform)
    h_error = np.linalg.norm(h_matrix - h_reconstruction, axis=1)
    c_error = np.linalg.norm(c_matrix - c_reconstruction, axis=1)

    code_columns = [f"nmr_dict_code_{idx:02d}" for idx in range(config.n_components)]
    features = pd.DataFrame(codes, columns=code_columns, index=df.index)
    features["nmr_h_reconstruction_error"] = h_error
    features["nmr_c_reconstruction_error"] = c_error
    features["nmr_total_reconstruction_error"] = h_error + c_error

    id_columns = [col for col in ["smiles", "molecular_formula"] if col in df.columns]
    feature_df = pd.concat([df[id_columns], features], axis=1)

    model_artifact = {
        "config": asdict(config),
        "nmf_model": nmf,
        "h_grid_ppm": h_grid,
        "c_grid_ppm": c_grid,
        "h_components": h_components,
        "c_components": c_components,
        "h_channels": h_channels,
        "h_multiplicity_channels": config.h_multiplicity_channels,
        "feature_columns": code_columns,
    }

    return feature_df, model_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Learn shared dictionary features from 1H and 13C NMR peak lists.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/Users/jacknugent/Downloads/ids_nmr_10k_dictionary_features.pkl"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("/Users/jacknugent/Downloads/ids_nmr_10k_dictionary_model.pkl"),
    )
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--h-step-ppm", type=float, default=0.01)
    parser.add_argument("--h-sigma-ppm", type=float, default=0.015)
    parser.add_argument("--c-step-ppm", type=float, default=0.25)
    parser.add_argument("--c-sigma-ppm", type=float, default=0.25)
    parser.add_argument("--h-width-scale", type=float, default=0.25)
    parser.add_argument("--c-width-scale", type=float, default=1.0)
    parser.add_argument(
        "--intensity-transform",
        choices=VALID_TRANSFORMS,
        default="none",
        help="Elementwise transform before NMF to balance tall vs small peaks.",
    )
    parser.add_argument(
        "--no-peak-width",
        dest="use_peak_width",
        action="store_false",
        help="Disable data-driven per-peak widths and use a fixed sigma instead.",
    )
    parser.set_defaults(use_peak_width=True)
    parser.add_argument(
        "--h-multiplicity-channels",
        action="store_true",
        help="Split the proton fingerprint into one ppm channel per multiplicity class (s/d/t/q/m).",
    )
    parser.add_argument(
        "--h-multiplicity-classes",
        type=str,
        default="s,d,t,q,m",
        help="Comma-separated multiplicity classes; last one is the catch-all bucket.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = NMRDictionaryConfig(
        n_components=args.n_components,
        random_state=args.random_state,
        max_iter=args.max_iter,
        h_step_ppm=args.h_step_ppm,
        h_sigma_ppm=args.h_sigma_ppm,
        c_step_ppm=args.c_step_ppm,
        c_sigma_ppm=args.c_sigma_ppm,
        use_peak_width=args.use_peak_width,
        h_width_scale=args.h_width_scale,
        c_width_scale=args.c_width_scale,
        intensity_transform=args.intensity_transform,
        h_multiplicity_channels=args.h_multiplicity_channels,
        h_multiplicity_classes=tuple(
            c.strip().lower() for c in args.h_multiplicity_classes.split(",") if c.strip()
        ),
    )

    df = pd.read_pickle(args.input)
    feature_df, model_artifact = fit_shared_dictionary(df, config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    feature_df.to_pickle(args.output)
    with args.model_output.open("wb") as file:
        pickle.dump(model_artifact, file)

    print(f"Wrote features: {args.output}")
    print(f"Wrote model: {args.model_output}")
    print(f"Feature shape: {feature_df.shape}")
    print(
        "Mean reconstruction error: "
        f"H={feature_df['nmr_h_reconstruction_error'].mean():.4f}, "
        f"C={feature_df['nmr_c_reconstruction_error'].mean():.4f}"
    )


if __name__ == "__main__":
    main()
