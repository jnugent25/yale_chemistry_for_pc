"""Shared library for the NMR property-prediction workflow.

Single source of truth for the pieces the notebooks kept re-implementing:

- ``nmrlib.data``     — dataset registry, path resolution, column-alias normalization
- ``nmrlib.features`` — feature-set column definitions and the featurization pipeline
- ``nmrlib.metrics``  — regression metrics
- ``nmrlib.models``   — model templates, grid-search space, feature-set comparison loop
"""

from nmrlib.data import DATASETS, load_dataset, resolve_dataset
from nmrlib.features import feature_sets, featurize, nmf_cols
from nmrlib.metrics import regression_metrics
from nmrlib.models import compare_feature_sets, default_models, grid_search_space
