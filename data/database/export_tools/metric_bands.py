"""Descriptive display bands for predicted binding-mode metrics."""

import numpy as np
import pandas as pd

SUPPORT_CUTOFFS = (0.10, 0.30)
SIMILARITY_CUTOFFS = (0.60, 0.80)
RECOVERY_CUTOFFS = (0.40, 0.70)


def _bands(values: pd.Series, cutoffs: tuple[float, float], missing: str) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    valid = values.notna() & np.isfinite(values)
    labels = pd.Series(missing, index=values.index, dtype="string")
    labels.loc[valid] = "Low"
    labels.loc[valid & (values >= cutoffs[0])] = "Medium"
    labels.loc[valid & (values >= cutoffs[1])] = "High"
    return labels


def add_metric_bands(modes: pd.DataFrame) -> pd.DataFrame:
    """Add three band columns while preserving the raw metrics and row order.

    Support uses each mode's n_members / num_specs; missing or invalid counts
    are unavailable. Missing reference comparisons get a neutral label.
    The same cutoffs apply to primary and alternate modes. These labels
    describe the metrics, rather than calibrated probabilities of correctness.
    """
    result = modes.copy()
    members = pd.to_numeric(modes["n_members"], errors="coerce")
    total = pd.to_numeric(modes["num_specs"], errors="coerce")
    support = members / total.where(total > 0)
    support = support.where(
        np.isfinite(members) & np.isfinite(total) & (members >= 0) & (members <= total)
    )
    result["support_band"] = _bands(support, SUPPORT_CUTOFFS, "Unavailable")
    result["homer_sim_band"] = _bands(
        modes["homer_sim"], SIMILARITY_CUTOFFS, "No reference"
    )
    result["recovery_band"] = _bands(
        modes["recovery"], RECOVERY_CUTOFFS, "No reference"
    )
    return result
