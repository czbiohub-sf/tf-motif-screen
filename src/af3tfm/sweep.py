"""
Shared machinery for the specificity hyperparameter sweeps
(notebooks/03_building-ppms/03_sweep-na-mpnn.py and notebooks/03_building-ppms/03_sweep-consensus.py).

Both scripts run the same focused num_specs x cor_threshold mini-sweep over the
high-similarity TF set, scoring each cluster consensus against JASPAR. This
module carries the per-TF worker, the parallel runner, grid aggregation,
repeated K-fold cross-validation of the hyperparameter selection, and the
num_specs x cor heatmap. 03_sweep-na-mpnn.py wraps these to compare NA-MPNN
(num_inferences x temperature) sets; 03_sweep-consensus.py runs the detailed
focused sweep on a single, already-determined set.

Ben Iovino 8/13/26 CZ-Biohub
"""

import os
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from af3tfm.consensus import (
    _load_b_strand_ppm,
    cluster_consensus_pwm,
    cluster_upgma,
    pred_recovery_fraction,
)
from af3tfm.dataset import DATA_DIR
from af3tfm.motif import load_homer_multi_motif_file
from af3tfm.utils import pwm_similarity

DATA = Path(DATA_DIR)

# Focused mini-sweep cor axis, shared by both sweeps (0.40..0.90 inclusive).
COR_GRID = [round(0.40 + 0.01 * i, 2) for i in range(51)]

# Metrics cross-validated and selected over: significance in the top-1 and
# top-4 largest clusters.
CV_METRICS = [
    ("sig1", "top-1 cluster"),
    ("sig4", "top-4 clusters"),
]

# Confidence columns available for ranking predictions before consensus
RANK_COLS = ["ipTM", "dom_pLDDT", "dna_pLDDT"]
LOWER_IS_BETTER = {"iface_PAE", "iface_PAE_contact"}
RANK_COL_LABELS = {
    "ipTM": "ipTM",
    "dom_pLDDT": "DBD-pLDDT",
    "dna_pLDDT": "DNA-pLDDT",
    "iface_PAE": "iface-PAE",
    "iface_PAE_contact": "iface-PAE-contact",
}
DEFAULT_RANK_COMBO = ["ipTM", "dom_pLDDT"]


def rank_metric_combos(cols: list = None) -> list:
    """
    Every non-empty combination of confidence columns to rank predictions by.

    Args:
        cols: confidence columns to combine (defaults to RANK_COLS).

    Returns:
        list of column lists, one per non-empty subset (the three singletons, the
        three pairs, and the full triple for the default RANK_COLS).
    """

    cols = list(RANK_COLS if cols is None else cols)
    combos = []
    for k in range(1, len(cols) + 1):
        combos.extend(list(c) for c in combinations(cols, k))
    return combos


def rank_metric_label(combo: list) -> str:
    """Compact label for a rank-metric combo, e.g. 'ipTM_x_DBD-pLDDT'."""

    return "_x_".join(RANK_COL_LABELS.get(c, c) for c in combo)


def num_specs_grid(ns_min: int = 10, ns_max: int = 30) -> list:
    """Inclusive num_specs axis for the focused mini-sweep."""

    return list(range(ns_min, ns_max + 1))


def default_n_workers() -> int:
    """
    Worker count respecting a SLURM/cgroup CPU allocation (sched_getaffinity),
    so the pool doesn't oversubscribe a shared node; full core count off-cluster.
    """

    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def load_sweep_inputs() -> tuple:
    """
    Load the JASPAR motif set and the high-similarity TF table shared by both sweeps.

    Returns:
        results: DataFrame of TFs with Hom_Sim >= 95 and a usable JASPAR match
        jaspar_id_to_pwm: dict mapping JASPAR ID -> (L, 4) PWM
        jaspar_pwm_list: list of all JASPAR PWMs (the null for significance)
        job_dir: Path to data/jobs/hero
    """

    jaspar_path = DATA / "tools" / "homer" / "jaspar.homer.motif"
    jaspar_names, jaspar_pwms = load_homer_multi_motif_file(str(jaspar_path))
    jaspar_id_to_pwm = dict(zip(jaspar_names, jaspar_pwms))
    jaspar_pwm_list = list(jaspar_id_to_pwm.values())

    df = pd.read_csv(str(DATA / "sequences" / "animaltfdb" / "tfdb4_rep_tfs.csv"))
    high_sim = df[df["Hom_Sim"] >= 95].copy()
    results = (
        high_sim[
            high_sim["Hom_JASPAR"].notna()
            & high_sim["Hom_JASPAR"].isin(jaspar_id_to_pwm)
        ]
        .copy()
        .reset_index(drop=True)
    )

    job_dir = DATA / "jobs" / "hero"
    return results, jaspar_id_to_pwm, jaspar_pwm_list, job_dir


def process_one_tf_sweep(
    row_dict: dict,
    jaspar_id: str,
    ref_pwm: np.ndarray,
    spec_dir_str: str,
    job_dir_str: str,
    rank_cols: list,
    num_specs_grid: list,
    cor_grid: list,
    jaspar_pwm_list: list,
) -> list:
    """
    Run the full (num_specs x cor_threshold) grid for a single TF.

    Loads the top-max(num_specs_grid) structures once and, for each grid cell,
    clusters with cluster_upgma and scores each cluster consensus against JASPAR.
    The consensus->JASPAR scoring is memoized per member-set.

    Args:
        rank_cols: confidence columns (subset of RANK_COLS) whose rank-percentile
            product ranks predictions; the top num_specs are kept for consensus.

    Returns:
        list of per-cell dicts with keys:
            num_specs, cor_threshold, sig1, sig4, best_recovery
        where sig{n} is True if any of the TF's top-n largest clusters has p < 0.05,
        and best_recovery is the IC recovery of the lowest-p cluster within top-4
        (np.nan if the TF has no clusters).
    """

    max_num_specs = max(num_specs_grid)

    row = pd.Series(row_dict)
    tf_id = row["DBID"]
    dbid = row["DBID"].lower()

    spec_dir = Path(spec_dir_str)
    job_dir = Path(job_dir_str)

    indiv_dir = spec_dir / dbid / "specificity"
    if not indiv_dir.exists():
        return []

    # Only pull the confidence columns actually used for ranking (plus the keys)
    rank_cols_sql = ", ".join(rank_cols)
    jobdb = sqlite3.connect(str(job_dir / "jobdb.sqlite"))
    all_preds = pd.read_sql(
        f"SELECT motif, dna, {rank_cols_sql}, structure_path "
        "FROM runs WHERE protein = ?",
        jobdb,
        params=(dbid,),
    )
    jobdb.close()
    if all_preds.empty:
        return []

    # Drop predictions missing any rank column so the percentile ranking is well
    # defined (NaNs would otherwise sort unpredictably).
    all_preds = all_preds.dropna(subset=rank_cols)
    if all_preds.empty:
        return []

    all_preds["seed"] = all_preds["structure_path"].map(
        lambda sp: Path(sp).parent.parent.name.rsplit("_", 1)[-1]
    )
    # Rank by the rank-percentile product of the chosen confidence columns; a
    # single column reduces to ranking by that column alone
    combined = None
    for col in rank_cols:
        pct = all_preds[col].rank(pct=True, ascending=col not in LOWER_IS_BETTER)
        combined = pct if combined is None else combined * pct
    all_preds["combined"] = combined

    # Top max_num_specs by combined rank; a num_specs=N run uses those with rank < N.
    top_structs = [
        (r.motif, r.dna, r.seed)
        for r in all_preds.nlargest(max_num_specs, "combined").itertuples()
    ]

    # loaded[i] = (ppm, dna, rank_pos) where rank_pos is the nlargest position.
    loaded = []
    for rank_pos, (motif, dna, seed) in enumerate(top_structs):
        npz_path = indiv_dir / f"{dbid}_{motif}_{dna}_{seed}.npz"
        if not npz_path.exists():
            continue
        try:
            ppm_b = _load_b_strand_ppm(npz_path)
            loaded.append((ppm_b, dna, rank_pos))
        except Exception:
            continue

    if not loaded:
        return []

    # Memoize consensus->JASPAR scoring by frozenset of member global ranks.
    cache = {}

    def _score_member_set(member_entries):
        """member_entries: list of (ppm, align_off, align_rc, is_med, dna, rank_pos)."""
        key = frozenset(e[5] for e in member_entries)
        if key in cache:
            return cache[key]

        members = sorted(member_entries, key=lambda x: not x[3])  # medoid first
        consensus = cluster_consensus_pwm(
            [(ppm, off, rc) for ppm, off, rc, _, _, _ in members]
        )
        all_sims_c = np.array(
            [pwm_similarity(consensus, pwm)[0] for pwm in jaspar_pwm_list]
        )
        sim_c, off_c, rc_c = pwm_similarity(consensus, ref_pwm)
        p_c = float((all_sims_c >= sim_c).mean())
        recovery = pred_recovery_fraction(consensus, ref_pwm, off_c, rc_c)
        result = {"p_value": p_c, "recovery": recovery, "n_members": len(members)}
        cache[key] = result
        return result

    out = []
    for num_specs in num_specs_grid:
        subset = [e for e in loaded if e[2] < num_specs]
        if not subset:
            for cor_threshold in cor_grid:
                out.append(
                    {
                        "DBID": tf_id,
                        "num_specs": num_specs,
                        "cor_threshold": cor_threshold,
                        "sig1": False,
                        "sig4": False,
                        "best_recovery": np.nan,
                    }
                )
            continue

        # cluster_upgma indexes by position in this list; carry the global rank
        # so cached scoring keys are stable across num_specs.
        scored_with_null = [(ppm, dna, 0.0, 0.0, None) for ppm, dna, _ in subset]
        local_to_rank = {i: subset[i][2] for i in range(len(subset))}

        for cor_threshold in cor_grid:
            merged = cluster_upgma(scored_with_null, cor_threshold)

            clusters = {}
            for (
                orig_rank,
                merged_rank,
                ppm_b,
                dna,
                align_off,
                align_rc,
                is_med,
            ) in merged:
                rank_pos = local_to_rank[orig_rank - 1]
                clusters.setdefault(merged_rank, []).append(
                    (ppm_b, align_off, align_rc, is_med, dna, rank_pos)
                )

            # Size rank: 1 = most members (ties broken by appearance, like the notebook).
            ordered = sorted(clusters.items(), key=lambda x: -len(x[1]))
            cluster_stats = []  # (size_rank, p_value, recovery)
            for size_rank, (_, members) in enumerate(ordered, start=1):
                sc = _score_member_set(members)
                cluster_stats.append((size_rank, sc["p_value"], sc["recovery"]))

            top4 = [(sr, p, rec) for sr, p, rec in cluster_stats if sr <= 4]
            # Recovery of the lowest-p cluster within the top-4 largest
            best_recovery = min(top4, key=lambda x: x[1])[2] if top4 else np.nan

            out.append(
                {
                    "DBID": tf_id,
                    "num_specs": num_specs,
                    "cor_threshold": cor_threshold,
                    "sig1": any(p < 0.05 for sr, p, _ in top4 if sr <= 1),
                    "sig4": any(p < 0.05 for sr, p, _ in top4),
                    "best_recovery": best_recovery,
                }
            )

    return out


def run_focused_sweep(
    results: pd.DataFrame,
    jaspar_id_to_pwm: dict,
    jaspar_pwm_list: list,
    job_dir: Path,
    spec_sets: list,
    num_specs_grid: list,
    cor_grid: list,
    n_workers: int,
    rank_combos: list = None,
) -> list:
    """
    Run the focused num_specs x cor_threshold grid for every TF against every
    specificity set and every rank metric, in parallel across (set, rank, TF).

    Args:
        results: high-similarity TF table (from load_sweep_inputs).
        jaspar_id_to_pwm: JASPAR ID -> PWM map.
        jaspar_pwm_list: null list of all JASPAR PWMs.
        job_dir: data/jobs/hero (holds jobdb.sqlite).
        spec_sets: list of {"spec_dir": Path, "meta": dict}; meta columns (e.g.
            num_inferences / temperature) are merged into every record for that set.
        num_specs_grid, cor_grid: the mini-sweep axes.
        n_workers: process-pool size.
        rank_combos: list of confidence-column lists to rank predictions by (each
            a subset of RANK_COLS); every record gets a "rank_metric" label column.
            Defaults to [DEFAULT_RANK_COMBO], reproducing the historical ranking.

    Returns:
        Flat list of per-cell dicts, each carrying its set's meta columns and the
        rank_metric label of the ranking used.
    """

    if rank_combos is None:
        rank_combos = [DEFAULT_RANK_COMBO]

    flat = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for spec in spec_sets:
            for combo in rank_combos:
                for _, row in results.iterrows():
                    fut = executor.submit(
                        process_one_tf_sweep,
                        row.to_dict(),
                        row["Hom_JASPAR"],
                        jaspar_id_to_pwm[row["Hom_JASPAR"]],
                        str(spec["spec_dir"]),
                        str(job_dir),
                        list(combo),
                        num_specs_grid,
                        cor_grid,
                        jaspar_pwm_list,
                    )
                    futures[fut] = {
                        **spec["meta"],
                        "rank_metric": rank_metric_label(combo),
                    }
        done = 0
        total = len(futures)
        for future in as_completed(futures):
            meta = futures[future]
            for rec in future.result():
                rec.update(meta)
                flat.append(rec)
            done += 1
            if done % 50 == 0 or done == total:
                print(f"  processed {done}/{total} (set, TF) jobs", flush=True)

    return flat


def aggregate_grid(long_df: pd.DataFrame, hp_cols: list) -> pd.DataFrame:
    """
    Aggregate per-TF per-cell records into one row per hyperparameter cell.

    Args:
        long_df: per-TF per-cell records with sig1/sig4/best_recovery columns.
        hp_cols: hyperparameter columns to group by (2-D or 4-D).

    Returns:
        DataFrame with one row per cell: hp_cols, sig1/sig4 (TF counts),
        mean_recovery, n_tfs.
    """

    return (
        long_df.groupby(hp_cols)
        .agg(
            sig1=("sig1", "sum"),
            sig4=("sig4", "sum"),
            mean_recovery=("best_recovery", "mean"),
            n_tfs=("sig1", "size"),
        )
        .reset_index()
    )


def cell_accuracy(sub_df: pd.DataFrame, metric: str, hp_cols: list) -> pd.DataFrame:
    """
    Per-grid-cell accuracy over a subset of TFs.

    Args:
        sub_df: long-format records (one row per TF per grid cell) restricted to
            the TFs in this split.
        metric: boolean recovery column to average (e.g. "sig1" or "sig4").
        hp_cols: hyperparameter columns defining a cell.

    Returns:
        DataFrame with columns hp_cols + acc, mean_recovery, n_tfs where acc is
        the fraction of TFs in the split recovered at that cell.
    """

    return (
        sub_df.groupby(hp_cols)
        .agg(
            acc=(metric, "mean"),
            mean_recovery=("best_recovery", "mean"),
            n_tfs=(metric, "size"),
        )
        .reset_index()
    )


def select_best_cell(cell_df: pd.DataFrame, hp_cols: list) -> dict:
    """
    Pick the hyperparameter cell (over hp_cols) maximizing accuracy.

    Ties in accuracy are broken by higher mean IC recovery, then by the hp_cols
    in order (all ascending), i.e. prefer the cheaper/looser setting: lower
    num_specs, cor_threshold, num_inferences, and temperature.

    Returns:
        dict mapping each hp_cols name to the selected cell's value.
    """

    ranked = cell_df.sort_values(
        ["acc", "mean_recovery", *hp_cols],
        ascending=[False, False, *([True] * len(hp_cols))],
    )
    best = ranked.iloc[0]
    return {c: best[c] for c in hp_cols}


def cell_value(cell_df: pd.DataFrame, hp: dict, hp_cols: list) -> dict:
    """Look up a single cell's row by its hyperparameter values (hp_cols)."""

    m = np.ones(len(cell_df), dtype=bool)
    for col in hp_cols:
        vals = cell_df[col]
        m &= np.isclose(vals, hp[col]) if vals.dtype.kind == "f" else (vals == hp[col])
    return cell_df.loc[m].iloc[0].to_dict()


def run_cross_validation(
    long_df: pd.DataFrame,
    hp_cols: list,
    n_splits: int,
    n_repeats: int,
    seed: int,
    metrics: list,
) -> pd.DataFrame:
    """
    Repeated K-fold CV of the hyperparameter-selection procedure, over TFs.
    The selection step runs inside each fold, so the mean test accuracy is
    an unbiased estimate of the whole sweep-and-select procedure on unseen TFs.

    Args:
        long_df: per-TF per-cell records (must include a "DBID" column).
        hp_cols: hyperparameter columns selected over.
        n_splits: number of folds (5 => 80/20 splits).
        n_repeats: number of independent K-fold repeats.
        seed: base RNG seed; repeat r uses seed + r for reproducible splits.
        metrics: list of (column, label) tuples to cross-validate.

    Returns:
        Long DataFrame with one row per (metric, repeat, fold): metric, repeat,
        fold, the selected hp_cols values, train_acc, test_acc, n_train, n_test.
    """

    tfs = np.array(sorted(long_df["DBID"].unique()))

    records = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)
        fold_of = rng.permutation(len(tfs)) % n_splits  # round-robin after shuffle

        for metric, _ in metrics:
            for k in range(n_splits):
                test_tfs = set(tfs[fold_of == k])
                is_test = long_df["DBID"].isin(test_tfs)
                train_cells = cell_accuracy(long_df[~is_test], metric, hp_cols)
                test_cells = cell_accuracy(long_df[is_test], metric, hp_cols)

                hp = select_best_cell(train_cells, hp_cols)
                tr = cell_value(train_cells, hp, hp_cols)
                te = cell_value(test_cells, hp, hp_cols)

                records.append(
                    {
                        "metric": metric,
                        "repeat": repeat,
                        "fold": k,
                        **hp,
                        "train_acc": tr["acc"],
                        "test_acc": te["acc"],
                        "n_train": int(tr["n_tfs"]),
                        "n_test": int(te["n_tfs"]),
                    }
                )

    return pd.DataFrame(records)


def cv_accuracy_ci(sub: pd.DataFrame, conf: float = 0.95) -> dict:
    """
    Mean CV test accuracy for one metric, with a confidence interval.

    Folds within a repeat share training TFs and are not independent, so the
    interval is built from the per-repeat estimates (each the average over that
    repeat's K folds, i.e. one full pass over every TF) using a t interval with
    n_repeats - 1 degrees of freedom.

    Args:
        sub: rows of the run_cross_validation table for a single metric.
        conf: confidence level (0.95 => 95% interval).

    Returns:
        dict with mean, fold_std (spread over the individual repeat x fold evals),
        lo, hi, n_repeats, n_evals. lo/hi are np.nan with fewer than two repeats.
    """

    per_repeat = sub.groupby("repeat")["test_acc"].mean()
    mean = float(per_repeat.mean())
    n = len(per_repeat)
    lo = hi = np.nan
    if n > 1:
        sem = float(per_repeat.std(ddof=1) / np.sqrt(n))
        half = float(stats.t.ppf(0.5 + conf / 2, n - 1)) * sem
        lo, hi = mean - half, mean + half

    return {
        "mean": mean,
        "fold_std": float(sub["test_acc"].std(ddof=0)),
        "lo": lo,
        "hi": hi,
        "n_repeats": n,
        "n_evals": len(sub),
    }


def _fmt_hp(hp: dict, hp_cols: list) -> str:
    """
    Format a hyperparameter dict as 'col=val, ...'. Whole-number values (e.g.
    num_specs, num_inferences) print as ints even when carried as floats after a
    mixed-dtype row lookup; fractional values (cor_threshold, temperature) to 2 dp;
    non-numeric values (e.g. rank_metric) print verbatim.

    Args:
        hp: dict mapping hyperparameter names to values.
        hp_cols: hyperparameter columns to format (and order).

    Returns:
        str of the form 'col1=val1, col2=val2, ...
    """

    parts = []
    for c in hp_cols:
        try:
            fv = float(hp[c])
        except (TypeError, ValueError):
            parts.append(f"{c}={hp[c]}")
            continue
        parts.append(f"{c}={int(fv)}" if fv.is_integer() else f"{c}={fv:.2f}")
    return ", ".join(parts)


def report_best_cells(agg: pd.DataFrame, hp_cols: list) -> None:
    """Print the best-achieving cell for each metric as a quick read."""

    best_sig1 = agg.loc[agg["sig1"].idxmax()]
    best_sig4 = agg.loc[agg["sig4"].idxmax()]
    best_rec = agg.loc[agg["mean_recovery"].idxmax()]

    def _cell(r):
        return _fmt_hp({c: r[c] for c in hp_cols}, hp_cols)

    print(
        f"\nMax sig-in-top-1: {int(best_sig1['sig1'])}/{int(best_sig1['n_tfs'])} TFs "
        f"at {_cell(best_sig1)}"
    )
    print(
        f"Max sig-in-top-4: {int(best_sig4['sig4'])}/{int(best_sig4['n_tfs'])} TFs "
        f"at {_cell(best_sig4)}"
    )
    print(f"Max mean IC recovery: {best_rec['mean_recovery']:.3f} at {_cell(best_rec)}")


def report_cross_validation(
    long_df: pd.DataFrame,
    cv: pd.DataFrame,
    hp_cols: list,
    metrics: list,
    n_repeats: int,
    n_splits: int,
    conf: float = 0.95,
) -> None:
    """
    Print the CV summary: procedure test accuracy + CI, pick stability, and the
    full-data (deployment) recommendation, per metric.

    Args:
        long_df: per-TF per-cell records used for the full-data pick.
        cv: table returned by run_cross_validation.
        hp_cols: hyperparameter columns selected over.
        metrics: list of (column, label) tuples.
        n_repeats, n_splits: fold layout, for the header.
        conf: confidence level for the interval.
    """

    print(
        f"\n{n_repeats}x{n_splits}-fold cross-validation (80/20 train/test over TFs)."
    )
    for metric, label in metrics:
        # Full-data pick: the setting the procedure recommends for deployment.
        full_cells = cell_accuracy(long_df, metric, hp_cols)
        hp = select_best_cell(full_cells, hp_cols)
        full_acc = cell_value(full_cells, hp, hp_cols)["acc"]

        # CV estimate of the procedure over all repeat x fold evals.
        sub = cv[cv["metric"] == metric]
        ci = cv_accuracy_ci(sub, conf)

        # Pick stability: how often each hyperparameter cell was selected.
        pick_counts = sub.groupby(hp_cols).size().sort_values(ascending=False)

        def _key_to_hp(key):
            vals = key if isinstance(key, tuple) else (key,)
            return dict(zip(hp_cols, vals))

        top_picks = "; ".join(
            f"({_fmt_hp(_key_to_hp(k), hp_cols)})x{n}"
            for k, n in pick_counts.head(3).items()
        )
        print(f"\n  {metric} (significant in {label}):")
        print(
            f"    CV test acc (procedure)            = {ci['mean']:.3f} "
            f"+/- {ci['fold_std']:.3f} across folds"
            f"  (n={ci['n_evals']} evals)"
        )
        ci_label = f"{conf:.0%} CI over {ci['n_repeats']} repeats"
        print(f"    {ci_label:<35}= [{ci['lo']:.3f}, {ci['hi']:.3f}]")
        print(
            f"    most common picks                  : {top_picks}"
            f"  [{len(pick_counts)} distinct]"
        )
        print(
            f"    full-data pick (recommended)       : {_fmt_hp(hp, hp_cols)} "
            f"(full-data acc={full_acc:.3f})"
        )


def plot_heatmaps(agg: pd.DataFrame, out_svg: Path, title_suffix: str = "") -> None:
    """
    Draw the num_specs x cor_threshold sweep heatmaps from an aggregated grid table.

    Axes are derived from the table itself (sorted unique num_specs /
    cor_threshold), so this works identically whether agg comes from a live
    sweep or is read back from a results CSV. Pass a single NA-MPNN set's slice
    to get that set's num_specs x cor_threshold heatmaps.

    Args:
        agg: one row per (num_specs, cor_threshold) cell with columns sig1, sig4,
            mean_recovery. Panels for missing columns are skipped, so older result
            CSVs still plot. Must be a single NA-MPNN set (one num_inferences /
            temperature); pass a set slice for a 4-D grid.
        out_svg: path to write the figure to.
        title_suffix: appended to the suptitle (e.g. the NA-MPNN set label).

    Raises:
        ValueError: if agg has more than one row per (num_specs, cor_threshold),
            e.g. a broad-sweep (4-D) CSV was passed instead of a single-set slice.
    """

    # One row per cell is required to pivot; multiple rows means a 4-D grid
    # (e.g. the broad nampnn_sweep CSV) slipped in under a focused-sweep prefix.
    if agg.duplicated(subset=["num_specs", "cor_threshold"]).any():
        extra = [
            c
            for c in ("num_inferences", "temperature", "rank_metric")
            if c in agg.columns and agg[c].nunique() > 1
        ]
        hint = f" it spans multiple {' / '.join(extra)} values" if extra else ""
        raise ValueError(
            "plot_heatmaps needs one row per (num_specs, cor_threshold) but the "
            f"table has duplicates —{hint or ' it looks like a multi-set grid'}. "
            "Pass a single NA-MPNN-set slice, or regenerate a focused-sweep CSV "
            "by running 03_sweep-consensus.py."
        )

    # Rows = num_specs, cols = cor_threshold (sorted from the data).
    num_vals = sorted(agg["num_specs"].unique())
    cor_vals = sorted(agg["cor_threshold"].unique())

    def _grid(col):
        g = agg.pivot(index="num_specs", columns="cor_threshold", values=col)
        return g.reindex(index=num_vals, columns=cor_vals).values

    panels = [
        ("sig1", "TFs significant in top-1 cluster", "viridis"),
        ("sig4", "TFs significant in top-4 clusters", "viridis"),
        ("mean_recovery", "Mean IC recovery (best top-4 cluster)", "magma"),
    ]
    panels = [p for p in panels if p[0] in agg.columns]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 5.5))
    for ax, (col, title, cmap) in zip(np.atleast_1d(axes).flat, panels):
        grid = _grid(col)
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap)
        xticks = range(0, len(cor_vals), 5)
        yticks = range(0, len(num_vals), 5)
        ax.set_xticks(list(xticks))
        ax.set_xticklabels(
            [f"{cor_vals[i]:.2f}" for i in xticks], rotation=90, fontsize=6
        )
        ax.set_yticks(list(yticks))
        ax.set_yticklabels([num_vals[i] for i in yticks], fontsize=7)
        ax.set_xlabel("cor_threshold")
        ax.set_ylabel("num_specs")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    suptitle = "Focused consensus sweep " "(significant JASPAR matches & IC recovery)"
    if title_suffix:
        suptitle += f"\n{title_suffix}"
    plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_svg, bbox_inches="tight", dpi=150)
    plt.close(fig)
