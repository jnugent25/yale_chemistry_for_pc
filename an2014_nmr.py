"""An et al. (2014) 1H NMR QSDAR descriptors for logP.

Paper: Global Model for Octanol-Water Partition Coefficients from Proton
Nuclear Magnetic Resonance Spectra (DOI 10.1002/minf.201300172)

- 24 integration bins: 0.5 ppm wide over 0-12 ppm (500 MHz, CDCl3 in paper).
  Variables x0.5, x1, ... are proton counts in bins ending at that ppm edge.
- Broadness: three molecule-level terms b1, b2, b3 from broad proton
  resonances (width-at-half-height > ~75 Hz; range span is used as a proxy
  when only peak ranges are available).
- Final MLR (Eq. 6) uses 10 of 24 integration bins plus b1-b3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BINS = 24
PPM_MIN, PPM_MAX = 0.0, 12.0
BIN_WIDTH = (PPM_MAX - PPM_MIN) / N_BINS  # 0.5 ppm
BIN_EDGES = np.linspace(PPM_MIN, PPM_MAX, N_BINS + 1)
BIN_UPPER_EDGES = BIN_EDGES[1:]

# Eq. 6 integration terms (upper bin edges, ppm)
MLR_INTEGRATION_EDGES = [0.5, 1.0, 1.5, 2.0, 4.5, 5.0, 5.5, 6.5, 7.0, 7.5]
MLR_BROADNESS_EDGES = [1.0, 2.0, 3.0]

INTERCEPT = 0.390
MLR_X_COEFS = [0.229, 0.259, 0.234, -0.074, 0.516, 0.322, 0.407, 0.381, 0.476, 0.270]
MLR_B_COEFS = [-1.494, -2.198, -0.538]

# lets convert hz to ppm
PROTON_FREQ_MHZ = 500.0
WIDE_PEAK_MIN_HZ = 75.0
WIDE_PEAK_MIN_PPM = WIDE_PEAK_MIN_HZ / PROTON_FREQ_MHZ

# integrateion value
def integration_column(upper_edge: float) -> str:
    """Column for bin (upper_edge - 0.5, upper_edge] ppm."""
    edge = float(upper_edge)
    label = f"{edge:g}"
    return f"an14_x{label}"

# Bunch of naming stuff

def broadness_column(index: int) -> str:
    """b1, b2, b3 - molecule-level broadness terms from broad proton peaks."""
    return f"an14_b{index}"


def all_integration_columns() -> list[str]:
    return [integration_column(e) for e in BIN_UPPER_EDGES]


def all_broadness_columns() -> list[str]:
    return [broadness_column(i) for i in range(1, 4)]


def mlr_integration_columns() -> list[str]:
    return [integration_column(e) for e in MLR_INTEGRATION_EDGES]


def mlr_feature_columns() -> list[str]:
    """13 variables in paper Eq. 6 (10 integrations + 3 broadness)."""
    return mlr_integration_columns() + all_broadness_columns()


def full_descriptor_columns() -> list[str]:
    """Full initial QSDAR grid: 24 integrations + 3 broadness + total nH."""
    return all_integration_columns() + all_broadness_columns() + ["an14_total_nH"]


def _peak_delta(peak: dict) -> float | None:
    val = peak.get("delta", peak.get("centroid"))
    return None if val is None else float(val)


# getting the width of the peak in ppm
def _peak_width_ppm(peak: dict) -> float | None:
    """Width in ppm, preferring explicit widths and falling back to peak span."""
    for key in ("width_hz", "fwhm_hz", "widthAtHalfHeightHz"):
        val = peak.get(key)
        if val is not None:
            width = float(val) / PROTON_FREQ_MHZ
            return width if width > 0 else None

    for key in (
        "width (ppm)",
        "width_ppm",
        "fwhm_ppm",
        "widthAtHalfHeight",
        "width_at_half_height",
        "width",
    ):
        val = peak.get(key)
        if val is not None:
            width = float(val)
            if width > 1.0:
                width /= PROTON_FREQ_MHZ
            return width if width > 0 else None

    lo, hi = peak.get("rangeMin"), peak.get("rangeMax")
    if lo is None or hi is None:
        return None
    span = float(hi) - float(lo)
    return span if span > 0 else None

# bin index
def _bin_index(delta: float) -> int | None:
    if delta < PPM_MIN or delta >= PPM_MAX:
        return None
    return min(int((delta - PPM_MIN) / BIN_WIDTH), N_BINS - 1)

# an2014 feature creation (summing it all together)
def an2014_features_from_peaks(peaks) -> dict[str, float]:
    """Paper-style descriptors from a peak table (shift, nH, width metadata)."""
    integ = np.zeros(N_BINS, dtype=np.float64)
    broad_peak_widths: list[float] = []

    if peaks is None or (isinstance(peaks, float) and pd.isna(peaks)):
        peaks = []

    for peak in peaks:
        delta = _peak_delta(peak)
        nh = float(peak.get("nH", 1) or 1)
        if delta is None or nh <= 0:
            continue
        idx = _bin_index(delta)
        if idx is None:
            continue

        integ[idx] += nh
        width = _peak_width_ppm(peak)
        if width is not None and width >= WIDE_PEAK_MIN_PPM:
            broad_peak_widths.append(width)

    out: dict[str, float] = {}
    for i, upper in enumerate(BIN_UPPER_EDGES):
        out[integration_column(upper)] = float(integ[i])

    broad_peak_widths = sorted(broad_peak_widths, reverse=True)[:3]
    for j in range(1, 4):
        out[broadness_column(j)] = float(broad_peak_widths[j - 1]) if j <= len(broad_peak_widths) else 0.0

    out["an14_total_nH"] = float(integ.sum())
    return out

# using the an2014 features to predict logP
def predict_an2014_mlr(row: pd.Series) -> float:
    """Apply fixed Eq. 6 coefficients (no re-fitting)."""
    pred = INTERCEPT
    for edge, coef in zip(MLR_INTEGRATION_EDGES, MLR_X_COEFS):
        pred += coef * float(row[integration_column(edge)])
    for edge, coef in zip(MLR_BROADNESS_EDGES, MLR_B_COEFS):
        idx = MLR_BROADNESS_EDGES.index(edge) + 1
        pred += coef * float(row[broadness_column(idx)])
    return pred
