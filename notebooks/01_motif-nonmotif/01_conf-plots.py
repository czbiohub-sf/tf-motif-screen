"""
Here we are analyzing a per-prediction confidence score between motif and
non-motif predictions for each protein to determine if the score can
effectively distinguish between TFs bound to motif-like sequences and
non-motif sequences.

Ben Iovino 7/14/26 Biohub
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.jobdb import Database
from af3tfm.tfactor import load_tfs
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "figures"
FIG_DIR.mkdir(exist_ok=True)

SCORE_MODE = "rank"
# "label"/"prefix" describe the separation score; "color_label" names the raw,
# interpretable per-TF quantity used to color the AUROC/AP lollipops
SCORE_CONFIG = {
    "iptm": {
        "label": "ipTM Score",
        "prefix": "",
        "color_label": "Average motif ipTM",
    },
    "rank": {
        "label": "√(ipTM x pLDDT rank)",
        "prefix": "rank_",
        "color_label": "Average motif ipTM x DBD pLDDT",
    },
    "iface_pae": {
        "label": "Interface PAE",
        "prefix": "ifacepae_",
        "color_label": "Average motif interface PAE",
    },
}
SCORE_LABEL = SCORE_CONFIG[SCORE_MODE]["label"]
PREFIX = SCORE_CONFIG[SCORE_MODE]["prefix"]
COLOR_LABEL = SCORE_CONFIG[SCORE_MODE]["color_label"]

# AF3/AlphaFold confidence palette
AF3_CMAP = LinearSegmentedColormap.from_list(
    "af3_confidence",
    ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"],  # very low -> very high
)
# For iface_pae lower is better, reverse color map
COLOR_CMAP = AF3_CMAP.reversed() if SCORE_MODE == "iface_pae" else AF3_CMAP

# Load TF information
dataset = DS["cisbp"]
tf_df = pd.read_csv(dataset.seq_csv)
tfs = load_tfs(tf_df, name="DBID")

# Read job database
run = "motif-nonmotif-genomic"
db = Database(
    job_path=f"{DATA_DIR}/jobs/{run}",
    dataset=dataset.name,
)

# Query database: pull ipTM, domain pLDDT, and DNA sequence per prediction
query = """
SELECT protein, motif, ipTM, dom_pLDDT, iface_PAE, dna
FROM runs
"""
runs_df = pd.read_sql(query, db.conn)
db.conn.close()

# Add boolean column for motif vs non-motif
runs_df["is_non"] = runs_df["motif"].str.contains("_neg")

# Add gene symbol column for plotting (merge on DBID)
tf_df["DBID"] = tf_df["DBID"].str.lower()
runs_df = runs_df.merge(
    tf_df[["DBID", "TF_Name"]], left_on="protein", right_on="DBID", how="left"
)
runs_df["tf"] = runs_df["TF_Name"]


def get_scores(tf_name: str) -> tuple[list, list]:
    """
    Return (motif_scores, nonmotif_scores) for a TF under the active SCORE_MODE.

    Args:
        tf_name (str): Base TF name (the "motif" column with any "-non" stripped)

    Returns:
        tuple[list, list]: (motif_scores, nonmotif_scores)
    """

    sub = runs_df[runs_df["tf"] == tf_name]
    if SCORE_MODE == "rank":
        score = np.sqrt(sub["ipTM"].rank(pct=True) * sub["dom_pLDDT"].rank(pct=True))
    elif SCORE_MODE == "iface_pae":
        # PAE is an error (lower = better), so negate for higher = better
        score = -sub["iface_PAE"]
    else:
        score = sub["ipTM"]
    motif_scores = score[~sub["is_non"]].tolist()
    nonmotif_scores = score[sub["is_non"]].tolist()
    return motif_scores, nonmotif_scores


def get_metric_scores(tf_name: str, col: str) -> tuple[list, list]:
    """
    Return (motif_scores, nonmotif_scores) of a single raw metric for a TF.

    Args:
        tf_name (str): Base TF name (the "motif" column with any "-non" stripped)
        col (str): Column to pull (e.g. "ipTM", "dom_pLDDT")

    Returns:
        tuple[list, list]: (motif_scores, nonmotif_scores)
    """

    sub = runs_df[runs_df["tf"] == tf_name]
    motif_scores = sub.loc[~sub["is_non"], col].tolist()
    nonmotif_scores = sub.loc[sub["is_non"], col].tolist()
    return motif_scores, nonmotif_scores


def get_color_value(tf_name: str) -> float:
    """
    Per-TF raw value used to color the AUROC/AP lollipops, matched to SCORE_MODE.

    Unlike the separation score (which may be a rank percentile), this returns
    the actual metric level averaged over the TF's motif structures so the color
    reflects where the values sit, not just their within-TF ranking.

    Args:
        tf_name (str): Base TF name (the "motif" column with any "-non" stripped)

    Returns:
        float: mean motif value under the active SCORE_MODE (nan if no structures)
    """

    if SCORE_MODE == "rank":
        # Raw ipTM x DBD pLDDT (pLDDT standardized to 0-1 to share ipTM's scale)
        iptm, _ = get_metric_scores(tf_name, "ipTM")
        plddt, _ = get_metric_scores(tf_name, "dom_pLDDT")
        return float(np.mean(np.asarray(iptm) * (np.asarray(plddt) / 100)))
    elif SCORE_MODE == "iface_pae":
        # Raw interface PAE in Angstroms (lower = better)
        pae, _ = get_metric_scores(tf_name, "iface_PAE")
        return float(np.mean(pae))
    else:  # iptm
        iptm, _ = get_metric_scores(tf_name, "ipTM")
        return float(np.mean(iptm))


# %%
def smooth_curve(
    x: np.ndarray, y: np.ndarray, n: int = 300
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate a step-like curve onto a fine grid for smooth plotting.

    ROC and precision-recall curves come back as step functions, so a plain
    line plot looks jagged. A monotone cubic (PCHIP) interpolant is fit over
    the unique x-values and evaluated on a dense grid; PCHIP avoids the
    overshoot a regular cubic spline would produce, keeping the curve within
    the data range (and thus within [0, 1] for these metrics).

    Args:
        x (np.ndarray): X-values (need not be sorted or unique)
        y (np.ndarray): Y-values matching x
        n (int): Number of points in the output grid

    Returns:
        tuple[np.ndarray, np.ndarray]: (x_smooth, y_smooth)
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Sort by x and collapse duplicate x-values (keep the max y at each x so
    # vertical steps map to the upper point) so PCHIP sees strictly increasing x
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    x_unique, idx = np.unique(x, return_index=True)
    y_unique = np.maximum.reduceat(y, idx)

    # Need at least 2 distinct points to interpolate
    if len(x_unique) < 2:
        return x, y

    x_smooth = np.linspace(x_unique[0], x_unique[-1], n)
    y_smooth = PchipInterpolator(x_unique, y_unique)(x_smooth)
    return x_smooth, y_smooth


def fast_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    AUROC via the Mann-Whitney U rank-sum identity (ties get average ranks).

    Equivalent to sklearn's roc_auc_score but far cheaper to call repeatedly,
    which matters for the bootstrap resampling below.

    Args:
        y_true (np.ndarray): binary labels (1 = motif, 0 = non-motif)
        y_scores (np.ndarray): scores aligned to y_true

    Returns:
        float: area under the ROC curve (nan if a class is empty)
    """

    y_true = np.asarray(y_true)
    ranks = stats.rankdata(y_scores)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def bootstrap_ci(
    y_true: list,
    y_scores: list,
    metric_fn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Stratified (within-class) bootstrap percentile CI for a ranking metric.

    Args:
        y_true (list): binary labels (1 = motif, 0 = non-motif)
        y_scores (list): scores aligned to y_true
        metric_fn: callable(y_true, y_scores) -> float (e.g. fast_auroc)
        n_boot (int): number of bootstrap resamples
        alpha (float): two-sided level (0.05 -> 95% CI)
        seed (int): RNG seed for reproducibility

    Returns:
        tuple[float, float]: (lower, upper) percentile CI bounds
    """

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate(
            [
                rng.choice(pos_idx, size=len(pos_idx), replace=True),
                rng.choice(neg_idx, size=len(neg_idx), replace=True),
            ]
        )
        boot[i] = metric_fn(y_true[idx], y_scores[idx])
    return (
        float(np.nanpercentile(boot, 100 * alpha / 2)),
        float(np.nanpercentile(boot, 100 * (1 - alpha / 2))),
    )


def plot_roc_curve(y_true: list, y_scores: list, tf_name: str, plot: bool) -> float:
    """
    Plot ROC curve and calculate AUC.

    Args:
        y_true (list): True labels
        y_scores (list): Predicted scores
        tf_name (str): Name of the transcription factor
        plot (bool): If True, plot the curve

    Returns:
        roc_auc (float): Area under the ROC curve
    """

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    if plot:
        fpr_s, tpr_s = smooth_curve(fpr, tpr)
        plt.plot(fpr_s, tpr_s, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle="--")  # Diagonal line for reference
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic for {tf_name.capitalize()}")
        plt.legend(loc="lower right")
        plt.savefig(
            f"{FIG_DIR}/{tf_name}_{PREFIX}roc_curve.svg", dpi=300, bbox_inches="tight"
        )
        plt.show()

    return roc_auc


def plot_prec_rec_curve(
    y_true: list, y_scores: list, tf_name: str, plot: bool
) -> float:
    """
    Plot precision-recall curve.

    Args:
        y_true (list): True labels
        y_scores (list): Predicted scores
        tf_name (str): Name of the transcription factor
        plot (bool): If True, plot the curve

    Returns:
        average_precision (float): Average precision score
    """

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    if plot:
        recall_s, precision_s = smooth_curve(recall, precision)
        plt.plot(
            recall_s, precision_s, label=f"Average precision = {average_precision:.2f}"
        )
        # No-skill baseline: a random classifier's precision is the positive-class
        # prevalence, so this horizontal line is the PR analog of the ROC diagonal
        baseline = sum(y_true) / len(y_true)
        plt.axhline(
            baseline,
            linestyle="--",
            color="gray",
            label=f"No skill = {baseline:.2f}",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall curve for {tf_name.capitalize()}")
        plt.legend(loc="lower right")
        plt.show()

    return average_precision


# %%
# Compare distribution of confidence scores for motif vs non-motif
auc_scores, prec_scores, results = [], [], []
for tf_name in tf_df["TF_Name"]:
    # Get confidence scores under the active SCORE_MODE
    motif_scores, nonmotif_scores = get_scores(tf_name)
    if len(motif_scores) == 0 or len(nonmotif_scores) == 0:
        continue

    # 30/150 motif/non-motif subset for average precision (1:5 ratio)
    motif_n, non_motif_n = 30, 150
    motif_sub = motif_scores[:motif_n]
    nonmotif_sub = nonmotif_scores[:non_motif_n]

    # AUROC on all sequences (motif=1, non-motif=0)
    y_true_roc = [1] * len(motif_scores) + [0] * len(nonmotif_scores)
    y_scores_roc = motif_scores + nonmotif_scores
    roc_auc = plot_roc_curve(y_true_roc, y_scores_roc, tf_name, plot=False)

    # Average precision on the 1:5 subset
    y_true_ap = [1] * len(motif_sub) + [0] * len(nonmotif_sub)
    y_scores_ap = motif_sub + nonmotif_sub
    avg_prec = plot_prec_rec_curve(y_true_ap, y_scores_ap, tf_name, plot=False)
    auc_scores.append(roc_auc)
    prec_scores.append(avg_prec)

    # Stratified bootstrap 95% CIs, each on its own pool (error bars)
    auroc_lo, auroc_hi = bootstrap_ci(y_true_roc, y_scores_roc, fast_auroc)
    ap_lo, ap_hi = bootstrap_ci(y_true_ap, y_scores_ap, average_precision_score)

    # Per-metric separation using ipTM alone and dom_pLDDT alone, matching the
    # pooling above: AUROC on all sequences, average precision on the subset.
    metric_seps = {}
    for col in ("ipTM", "dom_pLDDT"):
        m_scores, nm_scores = get_metric_scores(tf_name, col)
        roc = plot_roc_curve(
            [1] * len(m_scores) + [0] * len(nm_scores),
            m_scores + nm_scores,
            tf_name,
            plot=False,
        )
        ap = plot_prec_rec_curve(
            [1] * len(m_scores[:motif_n]) + [0] * len(nm_scores[:non_motif_n]),
            m_scores[:motif_n] + nm_scores[:non_motif_n],
            tf_name,
            plot=False,
        )
        metric_seps[col] = (roc, ap)

    # Significance test (Mann-Whitney U is the AUROC pairing, so use all scores)
    u_stat, p_value = stats.mannwhitneyu(
        motif_scores, nonmotif_scores, alternative="greater"
    )
    p_text = (
        f"Mann-Whitney U\np < {p_value:.2e}"
        if p_value < 0.001
        else f"Mann-Whitney U\np = {p_value:.4f}"
    )
    med_a = np.median(motif_scores)
    med_b = np.median(nonmotif_scores)

    # Raw per-TF value used only to color the lollipops; mirrors SCORE_MODE while
    # the separation above stays on the (possibly rank-based) SCORE_MODE score.
    avg_color = get_color_value(tf_name)

    results.append(
        {
            "TF_name": tf_name,
            "P_Value": p_value,
            "AUROC": roc_auc,
            "AUROC_lo": auroc_lo,
            "AUROC_hi": auroc_hi,
            "Avg_Precision": avg_prec,
            "Avg_Precision_lo": ap_lo,
            "Avg_Precision_hi": ap_hi,
            "ipTM_AUROC": metric_seps["ipTM"][0],
            "ipTM_AP": metric_seps["ipTM"][1],
            "pLDDT_AUROC": metric_seps["dom_pLDDT"][0],
            "pLDDT_AP": metric_seps["dom_pLDDT"][1],
            "Significant": p_value < 0.05,
            "Avg_ipTM": np.mean(motif_scores),
            "Avg_Color": avg_color,
        }
    )

    # Plot density curve with jitter if all scores are identical
    plt.figure(figsize=(8, 5))
    motif_scores_plot = motif_scores
    if len(set(motif_scores)) == 1:
        motif_scores_plot = [s + np.random.normal(0, 0.01) for s in motif_scores]
    sns.kdeplot(
        motif_scores_plot,
        label=f"Motif (n={len(motif_scores)})",
        fill=True,
        alpha=0.5,
        color="#738fc1",
    )
    sns.kdeplot(
        nonmotif_scores,
        label=f"Non-Motif (n={len(nonmotif_scores)})",
        fill=True,
        alpha=0.5,
        color="#d56c55",
    )
    plt.xlabel(SCORE_LABEL)
    plt.ylabel("Density")
    plt.title(f"{SCORE_LABEL} Distribution for {tf_name.capitalize()}")
    plt.legend()
    plt.text(
        0.05,
        0.95,
        p_text,
        transform=plt.gca().transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.legend()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # plt.savefig(
    #     f"{FIG_DIR}/{tf_name}_{PREFIX}distribution.svg", dpi=300, bbox_inches="tight"
    # )
    plt.show()

print(f"Average ROC AUC: {sum(auc_scores) / len(auc_scores):.3f}")
print(f"Average Precision: {sum(prec_scores) / len(prec_scores):.3f}")

# %%
# Motif vs non-motif separation per TF (AUROC + average precision)
df = pd.DataFrame(results)
df = df.sort_values("AUROC", ascending=True)
df = df.reset_index(drop=True)

# No-separation baselines: AUROC = 0.5, average precision = positive prevalence
auprc_ref = motif_n / (motif_n + non_motif_n)
METRICS = [
    ("AUROC", "AUROC (motif vs non-motif)", 0.5),
    ("Avg_Precision", "Average precision (motif vs non-motif)", auprc_ref),
]


# %%
def plot_lollipop(
    plot_df: pd.DataFrame,
    value_col: str,
    labels: list,
    title: str,
    save_path: str,
    xlabel: str,
    ref: float = 0.5,
    figsize: tuple = (6, 10),
    color_col: str = "Avg_Color",
    color_label: str = COLOR_LABEL,
):
    """
    Lollipop chart of a separation metric colored by a per-TF score.

    Args:
        plot_df (pd.DataFrame): Frame with `value_col` and `color_col` columns
        value_col (str): Column plotted on the x-axis (e.g. "AUROC", "Avg_Precision")
        labels (list): Y-axis tick labels, one per row of plot_df
        title (str): Plot title
        save_path (str): Path to save the figure
        xlabel (str): X-axis label
        ref (float): Reference line and lollipop baseline (no-separation value)
        figsize (tuple): Figure size
        color_col (str): Column used for the color scale (e.g. raw ipTM x pLDDT)
        color_label (str): Color bar label
    """

    norm = plt.Normalize(vmin=plot_df[color_col].min(), vmax=plot_df[color_col].max())
    cmap = COLOR_CMAP
    colors = cmap(norm(plot_df[color_col]))
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=figsize)

    # Lollipop chart (baseline at `ref`, the no-separation value)
    ax.hlines(
        y=plot_df.index,
        xmin=ref,
        xmax=plot_df[value_col],
        color=colors,
        alpha=0.6,
        linewidth=2,
    )
    # Bootstrap 95% CI error bars (drawn under the points if available)
    lo_col, hi_col = f"{value_col}_lo", f"{value_col}_hi"
    if lo_col in plot_df.columns and hi_col in plot_df.columns:
        xerr = np.vstack(
            [
                (plot_df[value_col] - plot_df[lo_col]).clip(lower=0),
                (plot_df[hi_col] - plot_df[value_col]).clip(lower=0),
            ]
        )
        ax.errorbar(
            plot_df[value_col],
            plot_df.index,
            xerr=xerr,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=2,
            alpha=0.5,
            zorder=2,
        )
    ax.scatter(
        plot_df[value_col],
        plot_df.index,
        color=colors,
        s=80,
        alpha=1,
        zorder=3,
        edgecolor="black",
    )
    ax.axvline(x=ref, color="black", linestyle="--", linewidth=2, alpha=0.5)
    ax.grid(True, axis="both", linestyle="-", alpha=0.3)

    # Names on the right
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_yticks(plot_df.index)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title, pad=20)

    # Color bar on the left
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, location="left", pad=0.03, aspect=50)
    cbar.set_label(color_label)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# Lollipop chart per TF, one per separation metric
for value_col, xlabel, ref in METRICS:
    d = df.sort_values(value_col, ascending=True).reset_index(drop=True)
    plot_lollipop(
        d,
        value_col=value_col,
        labels=d["TF_name"].str.capitalize(),
        title=f"Motif vs non-motif separation ({value_col})",
        save_path=f"{FIG_DIR}/lollipop_{PREFIX}{value_col.lower()}.svg",
        xlabel=xlabel,
        ref=ref,
    )

# %%
plt.rcdefaults()
set_plotting_style()

# %%
# Correlation between average motif score and each separation metric
label_tfs = {"sp2", "pax2a", "pax2b", "prkrira"}
fig, axes = plt.subplots(2, 1, figsize=(9, 7))
for ax, (value_col, xlabel, ref) in zip(axes, METRICS):
    x = df[value_col]
    # Raw per-TF score (ipTM x DBD pLDDT under rank mode), same quantity used to
    # color the AUROC lollipops — not the rank-based separation score.
    y = df["Avg_Color"]
    slope, intercept, r, p_value, std_err = stats.linregress(x, y)
    ax.scatter(x, y, color="#738fc1", linewidth=0.5, edgecolor="black", alpha=0.8, s=80)
    ax.axvline(x=ref, color="gray", linestyle="--", linewidth=0.8)

    # Regression line
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(
        x_line, slope * x_line + intercept, color="#ef4444", linestyle="--", linewidth=2
    )

    ax.text(
        0.05,
        0.95,
        f"Pearson r = {r:.2f}\np-value = {p_value:.2e}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Label specific TFs
    for _, row in df[df["TF_name"].isin(label_tfs)].iterrows():
        ax.annotate(
            row["TF_name"].capitalize(),
            xy=(row[value_col], row["Avg_Color"]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(COLOR_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/{PREFIX}correlation.svg", dpi=300)
plt.show()


# %%
def precision_vs_retained_curve(
    motif_scores: list,
    nonmotif_scores: list,
    grid: np.ndarray,
) -> np.ndarray:
    """
    Running precision vs fraction retained for one TF, on a common grid.

    Args:
        motif_scores (list): Confidence scores for this TF's motif structures
        nonmotif_scores (list): Confidence scores for this TF's non-motif structures
        grid (np.ndarray): Fraction-retained values to evaluate precision at

    Returns:
        np.ndarray: Precision at each grid point (nan if fewer than 2 structures)
    """

    scores = np.asarray(motif_scores + nonmotif_scores, dtype=float)
    labels = np.asarray([1] * len(motif_scores) + [0] * len(nonmotif_scores))
    n = len(scores)
    if n < 2:
        return np.full(grid.shape, np.nan)

    # Rank highest-confidence first, then running precision down the list
    order = np.argsort(-scores, kind="stable")
    scores_sorted = scores[order]
    tp = np.cumsum(labels[order])
    k = np.arange(1, n + 1)

    # Keep only tie-group ends (where the next score differs) so all tied
    # structures are counted together — no credit for within-tie ordering
    group_end = np.append(np.diff(scores_sorted) != 0, True)
    retained = k[group_end] / n
    precision = tp[group_end] / k[group_end]
    return np.interp(grid, retained, precision)


def plot_precision_vs_retained(
    tf_names: list,
    motif_n: int = 30,
    non_motif_n: int = 150,
    save_path: str = None,
):
    """
    Precision vs fraction of structures retained, ranked by confidence.

    Args:
        tf_names (list): TF names to include (matched against the "tf" column)
        motif_n (int): Motif sequences kept per TF
        non_motif_n (int): Non-motif sequences kept per TF
        save_path (str): If given, path to save the figure
    """

    # One precision-vs-retained curve per TF on a shared grid, per ranking
    grid = np.linspace(0.01, 1.0, 200)
    plt.figure(figsize=(5, 5))
    curves = []
    for tf_name in tf_names:
        motif_scores, nonmotif_scores = get_scores(tf_name)
        if len(motif_scores) == 0 or len(nonmotif_scores) == 0:
            continue
        curves.append(
            precision_vs_retained_curve(
                motif_scores[:motif_n], nonmotif_scores[:non_motif_n], grid
            )
        )
    curves = np.vstack(curves)
    median = np.nanmedian(curves, axis=0)
    q25 = np.nanpercentile(curves, 25, axis=0)
    q75 = np.nanpercentile(curves, 75, axis=0)
    plt.fill_between(grid, q25, q75, color="#738fc1", alpha=0.15, label="IQR")
    plt.plot(grid, median, color="#738fc1", linewidth=2, label="Median TF")

    baseline = motif_n / (motif_n + non_motif_n)
    plt.axhline(
        baseline,
        linestyle="--",
        color="gray",
        label=f"No skill = {baseline:.2f}",
    )
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.xlabel("Fraction of structures retained (ranked by confidence)")
    plt.ylabel("Precision (fraction motif)")
    plt.title("Precision vs fraction of structures retained")
    plt.legend(loc="upper right")
    ax = plt.gca()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


plot_precision_vs_retained(
    tf_df["TF_Name"],
    save_path=f"{FIG_DIR}/{PREFIX}precision_vs_retained.svg",
)


# %%
