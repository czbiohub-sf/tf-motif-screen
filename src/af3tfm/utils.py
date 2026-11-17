"""
Utility functions.

Ben Iovino 2/13/26 Biohub
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

JASPAR_API = "https://jaspar.elixir.no/api/v1/matrix/"
_jaspar_cache: dict[str, str | None] = {}


def fetch_jaspar_motif(gene_symbol: str) -> str | None:
    """
    Query JASPAR for the latest vertebrate motif matching gene_symbol.
    Returns the matrix ID (e.g. 'MA0106.3') or None if not found.
    """
    key = gene_symbol.upper()
    if key in _jaspar_cache:
        return _jaspar_cache[key]
    try:
        resp = requests.get(
            JASPAR_API,
            params={
                "name": gene_symbol,
                "tax_group": "vertebrates",
                "version": "latest",
                "format": "json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data["results"][0]["matrix_id"] if data.get("count", 0) > 0 else None
    except Exception:
        result = None
    _jaspar_cache[key] = result
    return result


AMINO_ACIDS = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "SEC",
    "PYL",
    "MSE",
}


def get_id_from_symbol(symbol: str) -> str:
    """
    Retrieves the current Ensembl stable ID for a given gene symbol.

    Args:
        symbol (str): The gene symbol to look up.

    Returns:
        str: The corresponding Ensembl stable ID, or None if not found.
    """

    server = "https://rest.ensembl.org"
    ext = f"/lookup/symbol/danio_rerio/{symbol}?"

    try:
        r = requests.get(server + ext, headers={"Content-Type": "application/json"})
        r.raise_for_status()
        decoded = r.json()
        print(f"Successfully found ID for symbol '{symbol}'")
        return decoded.get("id")
    except requests.exceptions.HTTPError as err:
        print(f"HTTP Error for symbol '{symbol}': {err}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON response for symbol '{symbol}'.")
        return None


def deduplicate_gene_symbols(tf_df: pd.DataFrame) -> pd.DataFrame:
    """
    For any gene symbol that maps to multiple DBIDs, keeps only the row whose
    DBID matches the current primary Ensembl stable ID (via the Ensembl REST
    API). Rows with symbols that appear only once are left untouched.

    Args:
        tf_df (pd.DataFrame): TF dataframe with DBID and TF_Name columns.

    Returns:
        pd.DataFrame: Deduplicated dataframe.
    """

    multi_symbol_tfs = tf_df[tf_df.duplicated(subset=["TF_Name"], keep=False)]

    primary_dbids = {}
    for symbol in multi_symbol_tfs["TF_Name"].unique():
        primary_dbids[symbol] = get_id_from_symbol(symbol)

    for symbol, primary_dbid in primary_dbids.items():
        if primary_dbid is not None:
            rows_to_drop = tf_df[
                (tf_df["TF_Name"] == symbol) & (tf_df["DBID"] != primary_dbid)
            ].index
            tf_df = tf_df.drop(rows_to_drop)

    return tf_df


def set_plotting_style():
    """
    Set consistent matplotlib plotting style for publication-quality figures.

    Configures matplotlib to use:
    - Seaborn paper style
    - Specific font sizes for axes, titles, ticks
    - Sans-serif fonts with LaTeX support
    - SVG font embedding disabled for editability
    - Default figure size of 10x9 inches
    """

    sns.set_context("paper")

    plt.rc("axes", labelsize=12)
    plt.rc("axes", titlesize=12)
    plt.rc("xtick", labelsize=10)
    plt.rc("ytick", labelsize=10)
    plt.rc("legend", fontsize=10)
    plt.rc("xtick.major", pad=2)
    plt.rc("ytick.major", pad=2)
    plt.rc("figure", figsize=[10, 9])
    plt.rc("svg", fonttype="none")


def reverse_complement_pwm(pwm: np.ndarray) -> np.ndarray:
    """
    Reverse complement of a PWM.

    Args:
        pwm : np.ndarray of shape (L, 4)

    Returns:
        np.ndarray of shape (L, 4)
    """

    return pwm[::-1, [3, 2, 1, 0]].copy()


def _corr_matrix(pwm_a: np.ndarray, pwm_b: np.ndarray) -> np.ndarray:
    """
    Compute (len(pwm_a), len(pwm_b)) matrix of centered dot products.
    Each entry C[i, j] = dot(pwm_a[i] - 0.25, pwm_b[j] - 0.25).

    This is the per-position-pair numerator contribution for global Pearson
    correlation over the flattened overlap region, matching HOMER's default
    'correlation' score method.

    Args:
        pwm_a, pwm_b : np.ndarray of shape (L, 4)

    Returns:
        C : np.ndarray of shape (len(pwm_a), len(pwm_b))
    """
    return (pwm_a - 0.25) @ (pwm_b - 0.25).T


def _best_score_from_corr(
    corr_matrix: np.ndarray,
    pwm_a: np.ndarray,
    pwm_b: np.ndarray,
    min_overlap: int = 6,
) -> tuple[float, int]:
    """
    Find the best global Pearson correlation and offset between two PWMs.

    For each valid offset the numerator is the diagonal sum of corr_matrix
    (sum of centered dot products over overlapping positions). The denominator
    is sqrt(ssq_a * ssq_b) over the same overlap window, computed efficiently
    via cumulative sums. This matches HOMER's correlation score: a single
    Pearson over the flattened overlap region.

    Args:
        corr_matrix : np.ndarray  — output of _corr_matrix(pwm_a, pwm_b)
        pwm_a, pwm_b : np.ndarray of shape (L, 4)
        min_overlap : minimum overlap required (default 6, matching HOMER)

    Returns:
        best_score : float in [-1, 1]
        best_offset : int  — positive means pwm_b shifted right relative to pwm_a
    """
    la, lb = len(pwm_a), len(pwm_b)

    # Diagonal sums = numerator of global Pearson for each offset
    row_idx = np.arange(la)[:, None]
    col_idx = np.arange(lb)[None, :]
    diag_labels = (row_idx - col_idx).ravel() + (lb - 1)
    diag_sums = np.bincount(
        diag_labels, weights=corr_matrix.ravel(), minlength=la + lb - 1
    )

    offsets = np.arange(-(lb - 1), la)
    overlaps = np.minimum(la - np.maximum(offsets, 0), lb - np.maximum(-offsets, 0))
    valid = overlaps >= min_overlap

    if not valid.any():
        return 0.0, 0

    # Denominator: sqrt(sum_ssq_a * sum_ssq_b) over the overlap window,
    # computed via cumulative sums for all offsets at once.
    ssq_a = ((pwm_a - 0.25) ** 2).sum(axis=1)
    ssq_b = ((pwm_b - 0.25) ** 2).sum(axis=1)
    a_cum = np.concatenate([[0.0], np.cumsum(ssq_a)])
    b_cum = np.concatenate([[0.0], np.cumsum(ssq_b)])

    s_a = np.maximum(offsets, 0)
    s_b = np.maximum(-offsets, 0)
    denom = np.sqrt(
        (a_cum[s_a + overlaps] - a_cum[s_a]) * (b_cum[s_b + overlaps] - b_cum[s_b])
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(valid & (denom > 0), diag_sums / denom, -np.inf)

    best_idx = int(np.argmax(scores))
    return float(scores[best_idx]), int(offsets[best_idx])


def pwm_similarity(pwm_a: np.ndarray, pwm_b: np.ndarray) -> tuple[float, int, bool]:
    """
    Compute HOMER-style similarity between two PWMs. Tries all valid offsets and both orientations.

    Args:
        pwm_a, pwm_b : np.ndarray of shape (L, 4)
            Frequency matrices (rows should sum to ~1, columns = [A, C, G, T]).

    Returns:
        score : float in [-1, 1]
        best_offset : int
        reverse_complement : bool  — True if the best alignment used the RC of pwm_b.
    """
    corr_fwd = _corr_matrix(pwm_a, pwm_b)
    score_fwd, offset_fwd = _best_score_from_corr(corr_fwd, pwm_a, pwm_b)

    b_rc = reverse_complement_pwm(pwm_b)
    corr_rc = _corr_matrix(pwm_a, b_rc)
    score_rc, offset_rc = _best_score_from_corr(corr_rc, pwm_a, b_rc)

    if score_fwd >= score_rc:
        return score_fwd, offset_fwd, False

    return score_rc, offset_rc, True


def pairwise_similarity_matrix(pwms: list[np.ndarray]) -> np.ndarray:
    """
    Precomputes RC of each PWM and uses vectorized corr matrices.

    Args:
        pwms : list of np.ndarray, each of shape (L, 4)

    Returns:
        mat : np.ndarray of shape (n, n) where n = len(pwms
    """

    n = len(pwms)
    rcs = [reverse_complement_pwm(p) for p in pwms]
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            C_fwd = _corr_matrix(pwms[i], pwms[j])
            s, _ = _best_score_from_corr(C_fwd, pwms[i], pwms[j])
            C_rc = _corr_matrix(pwms[i], rcs[j])
            s_rc, _ = _best_score_from_corr(C_rc, pwms[i], rcs[j])
            mat[i, j] = mat[j, i] = max(s, s_rc)

    return mat
