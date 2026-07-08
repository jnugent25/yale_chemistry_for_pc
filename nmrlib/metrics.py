"""Regression metrics shared by the notebooks and scripts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "medae": float(median_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }
