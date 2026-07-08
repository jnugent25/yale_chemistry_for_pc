"""Model templates, grid-search space, and the feature-set comparison loop.

These are the pieces feature_comparison.ipynb and ml_workflow.ipynb each kept
their own copies of. The CV comparison loop lives here as
``compare_feature_sets``; the notebooks keep only config and plotting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nmrlib.metrics import regression_metrics


def default_models(seed: int = 42) -> dict[str, Pipeline]:
    """Fixed-hyperparameter pipeline templates for quick CV comparisons.

    The HGB settings are the best params found by the ml_workflow grid search.
    """
    return {
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10_000, random_state=seed)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.07,
                max_depth=3,
                max_iter=200,
                min_samples_leaf=15,
                l2_regularization=2.0,
                random_state=seed,
            )),
        ]),
    }


def grid_search_space(n_features: int, seed: int = 42) -> tuple[Pipeline, list[dict]]:
    """The ml_workflow hyperparameter search: a scaler+model pipeline and a
    param grid spanning linear, PLS, ElasticNet, KNN, RF, and HGB models."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=10_000)),  # placeholder, swapped by the grid
    ])
    pls_components = [c for c in [2, 3, 5, 8] if c <= n_features]
    param_grid = [
        {"model": [LinearRegression()]},
        {"model": [PLSRegression()],
         "model__n_components": pls_components},
        {"model": [ElasticNet(max_iter=10_000, random_state=seed)],
         "model__alpha": [1e-3, 1e-2, 0.1],
         "model__l1_ratio": [0.2, 0.5, 0.8]},
        {"model": [KNeighborsRegressor(n_jobs=1)],
         "model__n_neighbors": [3, 5, 7, 11, 15],
         "model__weights": ["uniform", "distance"],
         "model__p": [1, 2]},
        {"model": [RandomForestRegressor(random_state=seed, n_jobs=1)],
         "model__n_estimators": [300],
         "model__max_depth": [4, 6, 8],
         "model__min_samples_leaf": [1, 3]},
        {"model": [HistGradientBoostingRegressor(random_state=seed)],
         "model__learning_rate": [0.03, 0.05, 0.07],
         "model__max_depth": [2, 3, 4],
         "model__max_iter": [150, 200],
         "model__min_samples_leaf": [10, 15, 20],
         "model__l2_regularization": [0.1, 0.5, 1.0, 2.0]},
    ]
    return pipe, param_grid


def compare_feature_sets(
    df: pd.DataFrame,
    target_col: str,
    feature_sets: dict[str, list[str]],
    models: dict[str, Pipeline] | None = None,
    n_splits: int = 10,
    test_frac: float = 0.15,
    seed: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """CV-compare every (feature set, model) pair on ``target_col``.

    For each feature set: drop rows with NaNs in target/features, hold out a
    test fraction, k-fold cross-validate each model template on the trainval
    part, then fit on all of trainval and score the held-out test set.
    """
    if models is None:
        models = default_models(seed)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = []

    for fs_name, fs_cols in feature_sets.items():
        model_df = df.dropna(subset=[target_col] + fs_cols)
        X = model_df[fs_cols].astype(float)
        y = model_df[target_col].astype(float)
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_frac, random_state=seed
        )
        if verbose:
            print(f"Feature Set: {fs_name:<28} | Samples: {len(model_df):<6} "
                  f"(TrainVal: {len(X_trainval)}, Test: {len(X_test)})")

        for model_name, template in models.items():
            pipe = clone(template)
            cv_res = cross_validate(
                pipe, X_trainval, y_trainval, cv=cv,
                scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
            )
            pipe.fit(X_trainval, y_trainval)
            test_metrics = regression_metrics(y_test, pipe.predict(X_test))
            results.append({
                "Feature Set": fs_name,
                "N Features": len(fs_cols),
                "Model": model_name,
                "CV RMSE": -cv_res["test_neg_root_mean_squared_error"].mean(),
                "CV MAE": -cv_res["test_neg_mean_absolute_error"].mean(),
                "CV R²": cv_res["test_r2"].mean(),
                "Test RMSE": test_metrics["rmse"],
                "Test MAE": test_metrics["mae"],
                "Test R²": test_metrics["r2"],
            })

    return pd.DataFrame(results)
