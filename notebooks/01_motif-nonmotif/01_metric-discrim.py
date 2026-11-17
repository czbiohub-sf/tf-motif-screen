"""
Each of the 29 TFs was folded against (a) its real motif and (b) GC-matched
genomic k-mers that are NOT its motif ("_neg" jobs). The question here is:
which AF3 confidence metric in jobdb.sqlite best separates real-motif binding
from genomic non-motif binding, and does any combination of metrics do
better than the best single metric?

Every metric is ranked among the motif and non-motif structures of the same
TF, with lower interface PAE treated as higher confidence. Combinations use
the geometric mean of those percentile ranks; there are no fitted weights.
AUROC/AP are calculated per TF and summarized across TFs.

Ben Iovino 09/2026 Biohub
"""

# %%
import sqlite3
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "tables"
TAB_DIR.mkdir(exist_ok=True)

RUN = "motif-nonmotif-genomic"
DB_PATH = f"{DATA_DIR}/jobs/{RUN}/jobdb.sqlite"
# All numeric AF3 metrics stored in the runs table.
METRICS = [
    "ptm",
    "ipTM",
    "dom_pLDDT",
    "dna_pLDDT",
    "iface_PAE",
]
ORIENT = {m: -1.0 if m == "iface_PAE" else 1.0 for m in METRICS}
PIPELINE_COLS = ("ipTM", "dom_pLDDT")
COMBO_LABEL = "√(ipTM x DBD pLDDT ranks)"
CONFIDENCE_NOTE = (
    "Confidence levels use raw motif means; products use pLDDT / 100 "
    "and 1 / (1 + PAE)."
)

# %%
# Load runs and label real motif (positive) vs genomic non-motif (negative).
con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
df = pd.read_sql("SELECT * FROM runs", con)
con.close()

df["is_neg"] = df["motif"].str.contains("_neg", regex=False)
df["y"] = (~df["is_neg"]).astype(int)  # 1 = real motif, 0 = genomic non-motif

print(f"{len(df)} runs across {df['protein'].nunique()} TFs")
print(
    f"positives (real motif): {df['y'].sum()}   negatives (non-motif): {(1 - df['y']).sum()}"
)

# Drop metrics that are constant across the whole table (nothing to learn from).
usable = [m for m in METRICS if df[m].nunique() > 1]
dropped = sorted(set(METRICS) - set(usable))
if dropped:
    print(f"dropped constant metrics: {dropped}")
print(f"usable metrics ({len(usable)}): {usable}")
if not np.isfinite(df[usable].to_numpy()).all():
    raise ValueError("All confidence metrics must be finite for every prediction")


# %%
# Shared scores for every comparison, bar, and correlation.
def pctile(frame, cols):
    """Confidence percentiles among all predictions of the same TF; ties average."""
    oriented = frame[list(cols)].mul(pd.Series({col: ORIENT[col] for col in cols}))
    return oriented.groupby(frame["protein"], sort=False).rank(pct=True)


def rank_score(frame, cols):
    """Geometric mean of within-TF confidence percentiles, including both classes."""
    return pctile(frame, cols).prod(axis=1) ** (1 / len(cols))


def raw_confidence(frame, cols):
    """Absolute confidence for motif means, separate from the separation score.

    Single metrics retain their oriented raw units. Products use positive,
    higher-is-better factors: pLDDT / 100, 1 / (1 + PAE), or raw pTM/ipTM.
    In particular ipTM x DBD pLDDT matches the lollipop's raw confidence color.
    """
    if len(cols) == 1:
        return ORIENT[cols[0]] * frame[cols[0]]
    factors = frame[list(cols)].copy()
    for col in cols:
        if col.endswith("pLDDT"):
            factors[col] /= 100
        elif col == "iface_PAE":
            factors[col] = 1 / (1 + factors[col])
    return factors.prod(axis=1)


def evaluate_score(cols):
    """Per-TF performance and raw motif confidence, shared by all plot types."""
    frame = df.assign(_s=rank_score(df, cols), _level=raw_confidence(df, cols))
    records = []
    for protein, group in frame.groupby("protein"):
        if group["y"].nunique() < 2:
            continue
        records.append(
            {
                "protein": protein,
                "auroc": roc_auc_score(group["y"], group["_s"]),
                "ap": average_precision_score(group["y"], group["_s"]),
                "level": group.loc[group["y"] == 1, "_level"].mean(),
                "n_motif": int(group["y"].sum()),
                "n_nonmotif": int((1 - group["y"]).sum()),
            }
        )
    return pd.DataFrame(records)


score_results = {
    cols: evaluate_score(cols)
    for k in range(1, min(3, len(usable)) + 1)
    for cols in combinations(usable, k)
}
all_per_tf = pd.concat(
    [
        result.assign(metrics=" x ".join(cols), k=len(cols), ranking="within-TF")
        for cols, result in score_results.items()
    ],
    ignore_index=True,
)
all_per_tf.to_csv(f"{TAB_DIR}/per_tf_rank_discrimination.csv", index=False)


def per_tf_aucs(frame, score, label="y"):
    """Within-TF AUROC for every TF that has both classes (array)."""
    aucs = []
    for _, g in frame.groupby("protein"):
        if g[label].nunique() < 2:
            continue
        aucs.append(roc_auc_score(g[label], g[score]))
    return np.array(aucs)


def per_tf_auc(frame, score, label="y"):
    """Mean within-TF AUROC (only TFs that have both classes), + n TFs used."""
    aucs = per_tf_aucs(frame, score, label)
    return float(np.mean(aucs)), len(aucs)


def boot_mean_ci(vals, n_boot=2000, seed=0, alpha=0.05):
    """95% percentile-bootstrap CI for the mean, resampling over the entries."""
    vals = np.asarray(vals, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def disp(label):
    """Display name for a metric/combo string (data column dom_pLDDT -> dbd_pLDDT)."""
    return label.replace("dom_pLDDT", "dbd_pLDDT")


rows = []
for m in usable:
    s = rank_score(df, (m,))
    g_auc = roc_auc_score(df["y"], s)
    tf_auc = score_results[(m,)]["auroc"].mean()
    ap = average_precision_score(df["y"], s)
    rows.append(
        {
            "metric": m,
            "direction": "higher" if ORIENT[m] > 0 else "lower",
            "ranking": "within-TF",
            "global_auc": g_auc,
            "per_tf_auc": tf_auc,
            "avg_precision": ap,
        }
    )

single = (
    pd.DataFrame(rows).sort_values("per_tf_auc", ascending=False).reset_index(drop=True)
)
single.to_csv(f"{TAB_DIR}/single_metric_discrimination.csv", index=False)
print("\nSingle-metric discrimination (within-TF confidence ranks):")
print(single.to_string(index=False))


# %%
# Metric combinations -- geometric mean of within-TF confidence ranks.
def eval_combo(cols):
    """Return (pooled AUROC, mean per-TF AUROC, per-TF AUROCs) for one score."""
    g_auc = roc_auc_score(df["y"], rank_score(df, cols))
    vec = score_results[tuple(cols)]["auroc"].to_numpy()
    return g_auc, float(vec.mean()), vec


# Sweep every pair of usable metrics.
combo_rows = []
pairs = list(combinations(usable, 2))
print(f"\nEvaluating {len(pairs)} metric pairs using within-TF confidence ranks...")
for cols in pairs:
    g_auc, tf_auc, tf_vec = eval_combo(cols)
    lo, hi = boot_mean_ci(tf_vec)
    combo_rows.append(
        {
            "k": 2,
            "metrics": " x ".join(cols),
            "ranking": "within-TF",
            "global_auc": g_auc,
            "per_tf_auc": tf_auc,
            "per_tf_auc_lo": lo,
            "per_tf_auc_hi": hi,
        }
    )

combos = (
    pd.DataFrame(combo_rows)
    .sort_values("per_tf_auc", ascending=False)
    .reset_index(drop=True)
)
combos.to_csv(f"{TAB_DIR}/combination_discrimination.csv", index=False)

print("\nMetric pairs by mean within-TF AUROC:")
print(combos.head(15).to_string(index=False))

# %%
# Compare single metrics and pairs using exactly the same rank-scoring path.
# The legacy global_auc column denotes pooled AUROC, never global percentiles.
single_rank_rows = []
for m in usable:
    g_auc, tf_auc, tf_vec = eval_combo((m,))
    lo, hi = boot_mean_ci(tf_vec)
    single_rank_rows.append(
        {
            "metrics": m,
            "global_auc": g_auc,
            "per_tf_auc": tf_auc,
            "per_tf_auc_lo": lo,
            "per_tf_auc_hi": hi,
            "kind": "single",
            "ranking": "within-TF",
        }
    )
single_rank = pd.DataFrame(single_rank_rows)


# Highlight the pipeline's ipTM x DBD pLDDT pair once, without a duplicate row.
_combo_score = rank_score(df, PIPELINE_COLS)
_combo_vec = score_results[PIPELINE_COLS]["auroc"].to_numpy()
_combo_lo, _combo_hi = boot_mean_ci(_combo_vec)
composite = pd.DataFrame(
    [
        {
            "metrics": COMBO_LABEL,
            "global_auc": roc_auc_score(df["y"], _combo_score),
            "per_tf_auc": float(_combo_vec.mean()),
            "per_tf_auc_lo": _combo_lo,
            "per_tf_auc_hi": _combo_hi,
            "kind": "composite",
            "ranking": "within-TF",
        }
    ]
)

ranked = (
    pd.concat(
        [
            single_rank,
            combos[combos.metrics != " x ".join(PIPELINE_COLS)].assign(kind="pair"),
            composite,
        ],
        ignore_index=True,
    )
    .sort_values("per_tf_auc", ascending=True)
    .reset_index(drop=True)
)


ranked.to_csv(f"{TAB_DIR}/metric_rank_comparison.csv", index=False)

y = np.arange(len(ranked))
KIND_COLOR = {"single": "#dd8452", "pair": "#4c72b0", "composite": "#55a868"}
bar_colors = ranked["kind"].map(KIND_COLOR)
best_single = single_rank["per_tf_auc"].max()
best_single_name = single_rank.sort_values("per_tf_auc").iloc[-1]["metrics"]

xerr = np.vstack(
    [
        ranked["per_tf_auc"].values - ranked["per_tf_auc_lo"].values,
        ranked["per_tf_auc_hi"].values - ranked["per_tf_auc"].values,
    ]
)
fig, ax = plt.subplots(figsize=(8.5, max(5, 0.6 * len(ranked))))
ax.barh(
    y,
    ranked["per_tf_auc"],
    color=bar_colors,
    xerr=xerr,
    error_kw=dict(ecolor="0.25", lw=1.0, capsize=2),
)
ax.axvline(
    best_single,
    color="gray",
    ls=":",
    lw=1.2,
    label=f"best single metric: {disp(best_single_name)} ({best_single:.3f})",
)

ax.set_yticks(y)
ax.set_yticklabels(
    [disp(m) for m in ranked["metrics"]],
    rotation=45,
    ha="right",
    va="center",
    rotation_mode="anchor",
)
ax.set_xlim(0.5, max(ranked["per_tf_auc_hi"].max(), best_single) + 0.06)
ax.set_xlabel("Mean within-TF AUROC (motif vs non-motif)")
ax.set_title("Within-TF confidence ranking: single metrics and pairs")
handles = [
    plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR["single"]),
    plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR["pair"]),
    plt.Rectangle((0, 0), 1, 1, color=KIND_COLOR["composite"]),
]
ax.legend(handles, ["single metric", "metric pair", "pipeline pair"], loc="lower right")
fig.text(
    0.5,
    0.01,
    "Scores: geometric mean of within-TF percentiles. "
    "Error bars: 95% bootstrap confidence intervals across TFs.",
    ha="center",
    fontsize=7,
    color="0.35",
)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(f"{FIG_DIR}/metric_pair_discrimination.svg")
fig.savefig(f"{FIG_DIR}/metric_pair_discrimination.png", dpi=150)
plt.show()


# %%
# Within-TF confidence-rank combinations, including triples.
def score_composites():
    recs = []
    for cols, result in score_results.items():
        recs.append(
            {
                "kind": {1: "single", 2: "pair", 3: "triple"}[len(cols)],
                "expr": " x ".join(cols),
                "per_tf_auc": result["auroc"].mean(),
                "global_auc": roc_auc_score(df["y"], rank_score(df, cols)),
            }
        )
    out = (
        pd.DataFrame(recs)
        .sort_values("per_tf_auc", ascending=False)
        .reset_index(drop=True)
    )
    out.insert(0, "pctile", "per-TF")
    return out


composites = score_composites()
composites.to_csv(f"{TAB_DIR}/percentile_product_discrimination.csv", index=False)
print("\nTop 10 within-TF confidence-rank combinations (by per-TF AUROC):")
print(composites.head(10).to_string(index=False))


# %%
# Discrimination for the headline scorers -- per-TF trajectories + aggregate
SCORERS = usable

# Attach the same within-TF rank scores used for the comparison bars.
scored = df.copy()
for m in SCORERS:
    scored[f"_score::{m}"] = rank_score(df, (m,))
scored[f"_score::{COMBO_LABEL}"] = _combo_score

scorer_cols = {m: f"_score::{m}" for m in SCORERS}
scorer_cols[COMBO_LABEL] = f"_score::{COMBO_LABEL}"
scorer_order = SCORERS + [COMBO_LABEL]

# One shared row per (TF, scorer): AUROC, AP, and raw motif confidence level.
scorer_metrics = {m: (m,) for m in SCORERS}
scorer_metrics[COMBO_LABEL] = PIPELINE_COLS
per_tf = pd.concat(
    [score_results[cols].assign(scorer=name) for name, cols in scorer_metrics.items()],
    ignore_index=True,
)
per_tf.to_csv(f"{TAB_DIR}/per_tf_scorer_discrimination.csv", index=False)

# Aggregate views: pooled (all runs together) and mean across TFs (with CI).
agg_rows = []
for name, col in scorer_cols.items():
    tf_auroc = per_tf.loc[per_tf.scorer == name, "auroc"].values
    tf_ap = per_tf.loc[per_tf.scorer == name, "ap"].values
    auroc_lo, auroc_hi = boot_mean_ci(tf_auroc)
    ap_lo, ap_hi = boot_mean_ci(tf_ap)
    agg_rows.append(
        {
            "scorer": name,
            "pooled_auroc": roc_auc_score(scored["y"], scored[col]),
            "pooled_ap": average_precision_score(scored["y"], scored[col]),
            "mean_tf_auroc": tf_auroc.mean(),
            "mean_tf_auroc_lo": auroc_lo,
            "mean_tf_auroc_hi": auroc_hi,
            "mean_tf_ap": tf_ap.mean(),
            "mean_tf_ap_lo": ap_lo,
            "mean_tf_ap_hi": ap_hi,
        }
    )
agg = pd.DataFrame(agg_rows).set_index("scorer").reindex(scorer_order)
agg.to_csv(f"{TAB_DIR}/aggregate_scorer_discrimination.csv")

print("\nPer-TF discrimination (mean +/- sd across TFs):")
print(
    per_tf.groupby("scorer")[["auroc", "ap"]]
    .agg(["mean", "std"])
    .reindex(scorer_order)
    .to_string()
)
print("\nAggregate discrimination (pooled over all runs vs mean across TFs):")
print(agg.to_string())

# %%
# Check if metrics discriminate individual TFs differently (per-TF AUROC correlation)
auroc_wide = {}
for m in usable:
    s = rank_score(df, (m,))
    frame = df.assign(_s=s)
    per_tf_map = {}
    for tf, g in frame.groupby("protein"):
        if g["y"].nunique() < 2:
            continue
        per_tf_map[tf] = roc_auc_score(g["y"], g["_s"])
    auroc_wide[disp(m)] = pd.Series(per_tf_map)
auroc_wide = pd.DataFrame(auroc_wide)  # rows = TFs, cols = metrics
auroc_metrics = list(auroc_wide.columns)
auroc_wide.to_csv(f"{TAB_DIR}/per_tf_metric_auroc_wide.csv")
print(
    f"\nPer-TF AUROC table: {auroc_wide.shape[0]} TFs x {auroc_wide.shape[1]} metrics"
)

corr_mat = auroc_wide.corr(method="pearson")
corr_mat.to_csv(f"{TAB_DIR}/metric_auroc_correlation.csv")
print("\nPearson correlation between metrics' per-TF AUROC (discriminating ability):")
print(corr_mat.round(2).to_string())

# Heatmap.
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(corr_mat.values, vmin=-1, vmax=1, cmap="RdBu_r")
ax.set_xticks(np.arange(len(auroc_metrics)))
ax.set_yticks(np.arange(len(auroc_metrics)))
ax.set_xticklabels(auroc_metrics, rotation=45, ha="right")
ax.set_yticklabels(auroc_metrics)
for i in range(len(auroc_metrics)):
    for j in range(len(auroc_metrics)):
        v = corr_mat.values[i, j]
        ax.text(
            j,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            color="white" if abs(v) > 0.5 else "black",
            fontsize=9,
        )
ax.set_title("Do metrics discriminate the same TFs?\n(Pearson r of per-TF AUROC)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/metric_auroc_correlation_heatmap.svg")
fig.savefig(f"{FIG_DIR}/metric_auroc_correlation_heatmap.png", dpi=150)
plt.show()

# %%
# ipTM vs the other three metrics: one dot per TF, its within-TF AUROC under ipTM
# (x) vs the other metric (y). Chance lines at 0.5 on both axes and the identity
# diagonal; r annotated. Upper-right quadrant = TFs both metrics handle well;
# spread off the diagonal = the two metrics are complementary across TFs.
others = ["ptm", "dbd_pLDDT", "dna_pLDDT", "iface_PAE"]
# TFs to call out by name (Ensembl id -> label).
HIGHLIGHT = {
    "ensdarg00000031756": "mef2aa",
    "ensdarg00000042489": "prkrira",
    "ensdarg00000095896": "pou3f3b",
}
lo = float(np.nanmin(auroc_wide.values)) - 0.02
hi = 1.02
fig, axes = plt.subplots(2, 2, figsize=(9, 9))
for ax, mi in zip(axes.flat, others):
    ax.scatter(
        auroc_wide["ipTM"],
        auroc_wide[mi],
        s=32,
        color="#738fc1",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.8,
    )
    # Call out select TFs with a thicker stroke + a plain black label.
    for gid, name in HIGHLIGHT.items():
        if gid not in auroc_wide.index:
            continue
        xh, yh = auroc_wide.loc[gid, "ipTM"], auroc_wide.loc[gid, mi]
        ax.scatter(
            xh, yh, s=40, color="#738fc1", edgecolor="black", linewidth=1.8, zorder=5
        )
        ax.annotate(
            name,
            (xh, yh),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            color="black",
            zorder=6,
        )
    ax.axvline(0.5, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.axhline(0.5, color="k", ls="--", lw=0.7, alpha=0.5)
    ax.plot([lo, hi], [lo, hi], color="0.6", ls=":", lw=0.8)
    r = corr_mat.loc[mi, "ipTM"]
    ax.text(
        0.05,
        0.95,
        f"r={r:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("ipTM per-TF AUROC")
    ax.set_ylabel(f"{mi} per-TF AUROC")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
fig.suptitle("ipTM vs other metrics: per-TF AUROC agreement (one dot = one TF)")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/metric_auroc_scatter_grid.svg")
fig.savefig(f"{FIG_DIR}/metric_auroc_scatter_grid.png", dpi=150)
plt.show()


# %%
# Check if discrimination tracks the metric level (correlation, cf. conf_plots)
LEVEL_LABEL = {
    "ipTM": "mean motif ipTM",
    "dom_pLDDT": "mean motif dbd_pLDDT",
    "dna_pLDDT": "mean motif dna_pLDDT",
    "iface_PAE": "mean motif -iface_PAE (oriented)",
    "ptm": "mean motif pTM",
    COMBO_LABEL: "mean motif ipTM x (DBD pLDDT / 100)",
}

# Raw motif confidence is kept separate from rank-based discrimination.
per_tf_lvl = per_tf.copy()

# Pearson r between per-TF level and per-TF separation, for AUROC and AP.
corr_rows = []
for name in scorer_order:
    sub = per_tf_lvl[per_tf_lvl.scorer == name]
    for metric in ("auroc", "ap"):
        r, p = stats.pearsonr(sub["level"], sub[metric])
        corr_rows.append(
            {
                "scorer": name,
                "separation": metric,
                "pearson_r": r,
                "p_value": p,
                "ranking": "within-TF",
            }
        )
corr = pd.DataFrame(corr_rows)
corr.to_csv(f"{TAB_DIR}/level_vs_discrimination_correlation.csv", index=False)
print("\nCorrelation of per-TF metric level with per-TF discrimination:")
print(corr.to_string(index=False))

# %%
# Scatter + regression per scorer (rows) x separation metric (cols), cf. conf_plots.
prevalence = scored["y"].mean()  # positive fraction over the whole pool
SEP = [("auroc", "AUROC", 0.5), ("ap", "AP", prevalence)]
fig, axes = plt.subplots(
    len(scorer_order), len(SEP), figsize=(9, 3.2 * len(scorer_order)), squeeze=False
)
for i, name in enumerate(scorer_order):
    sub = per_tf_lvl[per_tf_lvl.scorer == name]
    for j, (metric, mlabel, ref) in enumerate(SEP):
        ax = axes[i, j]
        x, y = sub[metric].values, sub["level"].values
        slope, intercept, r, p, _ = stats.linregress(x, y)
        ax.scatter(
            x, y, color="#738fc1", edgecolor="black", linewidth=0.5, alpha=0.8, s=60
        )
        ax.axvline(ref, color="gray", linestyle="--", linewidth=0.8)
        xl = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            xl, slope * xl + intercept, color="#ef4444", linestyle="--", linewidth=2
        )
        ax.text(
            0.05,
            0.95,
            f"Pearson r = {r:.2f}\np = {p:.2e}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        ax.set_xlabel(f"{mlabel} (motif vs non-motif)")
        ax.set_ylabel(LEVEL_LABEL[name])
        ax.set_title(disp(name))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/level_vs_discrimination_correlation.svg")
fig.savefig(f"{FIG_DIR}/level_vs_discrimination_correlation.png", dpi=150)
plt.show()

# %%
# Raw motif confidence vs within-TF rank discrimination for every combination.
combo_cols = list(combinations(usable, 2)) + list(combinations(usable, 3))
combo_corr_rows = []
for cols in combo_cols:
    result = score_results[cols]
    for metric in ("auroc", "ap"):
        r, p = stats.pearsonr(result["level"], result[metric])
        combo_corr_rows.append(
            {
                "combo": " x ".join(disp(c) for c in cols),
                "k": len(cols),
                "separation": metric,
                "pearson_r": r,
                "p_value": p,
                "ranking": "within-TF",
            }
        )
combo_corr = pd.DataFrame(combo_corr_rows)
combo_corr.to_csv(
    f"{TAB_DIR}/level_vs_discrimination_correlation_combos.csv", index=False
)
print("\nRaw confidence vs within-TF rank discrimination for combinations:")
print(
    combo_corr[combo_corr.separation == "auroc"]
    .sort_values("pearson_r", ascending=False)
    .to_string(index=False)
)

# %%
# Compare raw-confidence correlations across all singles, pairs and triples.
# The highlighted pipeline pair replaces its duplicate among the combinations.
unified = pd.concat(
    [
        corr.rename(columns={"scorer": "name"}).assign(
            name=lambda d: d["name"].map(disp),
            kind=lambda d: np.where(
                d["name"] == disp(COMBO_LABEL), "composite", "single"
            ),
        ),
        (
            combo_corr[combo_corr.combo != " x ".join(disp(c) for c in PIPELINE_COLS)]
            .rename(columns={"combo": "name"})
            .assign(kind=lambda d: d["k"].map({2: "pair", 3: "triple"}))
        ),
    ],
    ignore_index=True,
)
unified.to_csv(f"{TAB_DIR}/all_metric_confidence_correlations.csv", index=False)
KIND_COLOR = {
    "single": "#dd8452",
    "composite": "#55a868",
    "pair": "#4c72b0",
    "triple": "#8172b3",
}

fig, axes = plt.subplots(1, 2, figsize=(13, max(6, 0.34 * unified["name"].nunique())))
for ax, (metric, mlabel) in zip(axes, [("auroc", "AUROC"), ("ap", "AP")]):
    sub = (
        unified[unified.separation == metric]
        .sort_values("pearson_r")
        .reset_index(drop=True)
    )
    ax.barh(np.arange(len(sub)), sub["pearson_r"], color=sub["kind"].map(KIND_COLOR))
    ax.axvline(0, color="k", lw=1, alpha=0.6)
    ax.set_yticks(np.arange(len(sub)))
    ax.set_yticklabels(sub["name"], fontsize=7)
    ax.set_xlabel(f"Pearson r (raw motif confidence vs per-TF {mlabel})")
    ax.set_title(f"Within-TF ranking: confidence vs {mlabel}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in KIND_COLOR.values()]
axes[1].legend(handles, list(KIND_COLOR), loc="lower right", fontsize=8)
fig.text(0.5, 0.01, CONFIDENCE_NOTE, ha="center", fontsize=7, color="0.35")
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(f"{FIG_DIR}/level_vs_discrimination_correlation_all.svg")
fig.savefig(f"{FIG_DIR}/level_vs_discrimination_correlation_all.png", dpi=150)
plt.show()

# %%
