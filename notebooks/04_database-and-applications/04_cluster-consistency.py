"""
Consistency checks on NA-MPNN predicted specificities from the clusterval job

Ben Iovino 8/6/26 CZ-Biohub
"""

# %%
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from af3tfm.consensus import (
    _load_b_strand_ppm,
    cluster_consensus_pwm,
    cluster_upgma,
    plot_aligned_logos,
)
from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import pwm_similarity, set_plotting_style

DATA = Path(DATA_DIR)
set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# HYPERPARAMETERS
num_specs = 45
cor_threshold = 0.77

JOB_DIR = DATA / "jobs" / "clusterval"
SPEC_DIR = JOB_DIR / "specificities"
MMSEQS_RESULTS_DIR = DATA / "tools" / "mmseqs" / "animaltfdb4" / "results"
VAL_TFS_CSV = DATA / "sequences" / "animaltfdb" / "tfdb4_val_tfs.csv"


# %%
def get_cluster_members(family: str, cluster_type: str, reps: list[str]) -> list[str]:
    """Returns all member IDs belonging to the given representative sequences.

    Args:
        family (str): TF family name (e.g. 'zf-C2H2', 'Homeobox').
        cluster_type (str): Clustering type, either 'mmseqs' or 'recog'.
        reps (list[str]): Representative sequence IDs to extract clusters for.

    Returns:
        list[str]: Member sequence IDs across all requested clusters.
    """

    if cluster_type == "recog":
        tsv_path = MMSEQS_RESULTS_DIR / f"{family}.recog_cluster.tsv"
    else:
        tsv_path = MMSEQS_RESULTS_DIR / f"{family}.fasta.mmseqs_cluster.tsv"

    rep_set = set(reps)
    members = []
    with open(tsv_path) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            if rep in rep_set:
                members.append(member)

    return members


def get_cluster_membership(
    family: str, cluster_type: str, reps: list[str]
) -> dict[str, list[str]]:
    """Like get_cluster_members but keeps which representative each member maps to.

    Args:
        family (str): TF family name (e.g. 'zf-C2H2', 'Homeobox').
        cluster_type (str): Clustering type, either 'mmseqs' or 'recog'.
        reps (list[str]): Representative sequence IDs to extract clusters for.

    Returns:
        dict[str, list[str]]: mapping of representative ID -> member IDs (rep included).
    """

    if cluster_type == "recog":
        tsv_path = MMSEQS_RESULTS_DIR / f"{family}.recog_cluster.tsv"
    else:
        tsv_path = MMSEQS_RESULTS_DIR / f"{family}.fasta.mmseqs_cluster.tsv"

    rep_set = set(reps)
    membership: dict[str, list[str]] = {}
    with open(tsv_path) as f:
        for line in f:
            rep, member = line.strip().split("\t")
            if rep in rep_set:
                membership.setdefault(rep, []).append(member)

    return membership


def load_top_ppms(
    con: sqlite3.Connection,
    dbid: str,
    num_specs: int,
    rep_filter: str | None = None,
    motif_filter: str | None = None,
) -> list[tuple]:
    """
    Load the top-num_specs B-strand PPMs for a TF, ranked by combined ipTM/dom_pLDDT.

    Args:
        con: open connection to the clusterval jobdb.
        dbid: lowercase protein DBID.
        num_specs: number of top-ranked structures to load.
        rep_filter: if set, keep only motifs whose rep-cluster (motif name with any
            '.similarN' suffix stripped) equals this value.
        motif_filter: if set, keep only runs whose raw motif name equals this value
            exactly (i.e. a single '.similarN' variant, not the whole rep-cluster).

    Returns:
        list of (ppm, dna) for the loaded structures.
    """

    indiv_dir = SPEC_DIR / dbid / "specificity"
    if not indiv_dir.exists():
        return []

    all_preds = pd.read_sql(
        "SELECT motif, dna, ipTM, dom_pLDDT, structure_path FROM runs WHERE protein = ?",
        con,
        params=(dbid,),
    )
    if all_preds.empty:
        return []

    # npz files are named {dbid}_{motif}_{dna}_{seed}.npz.
    all_preds["seed"] = all_preds["structure_path"].map(
        lambda sp: Path(sp).parent.parent.name.rsplit("_", 1)[-1]
    )
    all_preds["combined"] = all_preds["ipTM"].rank(pct=True) * all_preds[
        "dom_pLDDT"
    ].rank(pct=True)

    if rep_filter is not None:
        rep = all_preds["motif"].str.replace(r"\.similar\d+", "", regex=True)
        all_preds = all_preds[rep == rep_filter]

    if motif_filter is not None:
        all_preds = all_preds[all_preds["motif"] == motif_filter]

    top_structs = [
        (r.motif, r.dna, r.seed)
        for r in all_preds.nlargest(num_specs, "combined").itertuples()
    ]

    loaded = []
    for motif, dna, seed in top_structs:
        npz_path = indiv_dir / f"{dbid}_{motif}_{dna}_{seed}.npz"
        if not npz_path.exists():
            continue
        try:
            loaded.append((_load_b_strand_ppm(npz_path), dna))
        except Exception:
            continue

    return loaded


def build_representative_pwm(loaded: list[tuple], cor_threshold: float) -> tuple:
    """
    Collapse a set of PPMs into one representative PWM: the consensus of the largest UPGMA cluster.

    Args:
        loaded: list of (ppm, dna).
        cor_threshold: UPGMA merge threshold.

    Returns:
        (consensus_pwm, n_members_in_largest_cluster, n_loaded) or (None, 0, 0) if empty.
    """

    if not loaded:
        return None, 0, 0

    scored_with_null = [(ppm, dna, 0.0, 0.0, np.empty(0)) for ppm, dna in loaded]
    merged = cluster_upgma(scored_with_null, cor_threshold)

    clusters = {}
    for _orig_rank, cid, ppm, _dna, off, rc, is_med in merged:
        clusters.setdefault(cid, []).append((ppm, off, rc, is_med))

    largest = max(clusters.values(), key=len)
    largest.sort(
        key=lambda x: not x[3]
    )  # medoid first (seed for cluster_consensus_pwm)
    consensus = cluster_consensus_pwm([(ppm, off, rc) for ppm, off, rc, _ in largest])

    return consensus, len(largest), len(loaded)


def plot_similarity_to_rep_hist(
    records: list[dict],
    title: str,
    out_path: str | None = None,
    bins: np.ndarray | None = None,
    bin_width: float = 0.02,
):
    """
    Histogram of member PWM similarity to their cluster representative.

    Args:
        records: list of dicts with keys 'sim' and 'cluster'.
        title: figure title.
        out_path: path to save the plot, or None.
        bins: histogram bin edges; if None, uses width-`bin_width` bins spanning the
            lower end of the observed data up to 1
        bin_width: bin width used when `bins` is None.

    Returns:
        (mean, median) of the plotted similarities.
    """

    sims = np.array([r["sim"] for r in records])
    mean_sim = float(np.mean(sims))
    med_sim = float(np.median(sims))

    # Lower bound = the data minimum snapped down to the nearest bin edge
    lo = float(
        np.clip(np.floor(sims.min() / bin_width) * bin_width, 0.0, 1.0 - bin_width)
    )
    lo = min(lo, 0.8)
    if bins is None:
        bins = np.arange(lo, 1.0 + 1e-9, bin_width)

    clusters = sorted({r["cluster"] for r in records})
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(clusters))]
    data = [[r["sim"] for r in records if r["cluster"] == c] for c in clusters]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(
        data,
        bins=bins,
        stacked=True,
        color=colors,
        label=clusters,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.axvline(mean_sim, color="gray", ls="--", lw=1.2, label=f"mean = {mean_sim:.3f}")
    ax.axvline(
        med_sim, color="crimson", ls="-.", lw=1.2, label=f"median = {med_sim:.3f}"
    )
    ax.set_xlabel("HOMER pwm_similarity to cluster representative")
    ax.set_ylabel("# members")
    ax.set_xlim(lo, 1)
    ax.legend(fontsize=6, title="cluster" if len(clusters) > 1 else None)
    ax.set_title(
        f"{title}\nn={len(sims)} members, mean={mean_sim:.3f}, median={med_sim:.3f}"
    )
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()

    return mean_sim, med_sim


def plot_pwm_similarity_heatmap(
    pwms: list[np.ndarray],
    labels: list[str],
    title: str,
    out_path: str | None = None,
    annot_max: int = 25,
):
    """
    All-vs-all HOMER pwm_similarity heatmap for a set of PWMs.

    Args:
        pwms: list of PWMs to compare pairwise.
        labels: axis labels, one per PWM.
        title: figure title.
        out_path: path to save the plot, or None.
        annot_max: above this many PWMs, suppress annotations and tick labels.

    Returns:
        (similarity matrix in the plotted order, ordered labels).
    """

    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    n = len(pwms)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            sim, _, _ = pwm_similarity(pwms[i], pwms[j])
            mat[i, j] = mat[j, i] = float(sim)

    # Reorder by UPGMA on (1 - sim) so similar variants land in diagonal blocks.
    if n > 2:
        dist = np.clip(1.0 - mat, 0.0, None)
        np.fill_diagonal(dist, 0.0)
        order = leaves_list(linkage(squareform(dist, checks=False), method="average"))
    else:
        order = np.arange(n)

    mat = mat[np.ix_(order, order)]
    labels = [labels[i] for i in order]

    # Colour scale spans the off-diagonal range so structure is visible even when
    # all clusters are broadly similar.
    off_diag = mat[~np.eye(n, dtype=bool)]
    vmin = float(off_diag.min()) if off_diag.size else 0.0
    annotate = n <= annot_max

    side = (1.1 * n + 2) if annotate else min(0.12 * n + 2, 14)
    fig, ax = plt.subplots(figsize=(side + 1.5, side))
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=1.0)
    if annotate:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if mat[i, j] < (vmin + 1.0) / 2 else "black",
                )
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"{n} motif variants (UPGMA-ordered)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="HOMER pwm_similarity")
    ax.set_title(title)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()

    return mat, labels


# %%
# Name/family lookup for the validation TFs
val_df = pd.read_csv(str(VAL_TFS_CSV))
val_df["dbid"] = val_df["DBID"].str.lower()
name_of = dict(zip(val_df["dbid"], val_df["TF_Name"]))
fam_of = dict(zip(val_df["dbid"], val_df["Family"]))


# %%
# TF-cluster consistency
# For each sequence cluster (as defined in get_val_seqs.py), build one representative PWM per TF
# and compare them. High off-diagonal similarity => sequence-clustered TFs give consistent predictions.

# %%
# Clusters exactly as written by get_val_seqs.py (family, cluster_type, seq_reps)
CLUSTERS = [
    ("zf-C2H2", "recog", ["ENSDARG00000037421"]),
    ("zf-C2H2", "recog", ["ENSDARG00000015554"]),
    ("Homeobox", "recog", ["ENSDARG00000010630"]),
    ("Homeobox", "recog", ["ENSDARG00000042032"]),
    ("Fork_head", "mmseqs", ["ENSDARG00000093129"]),
    ("bHLH", "mmseqs", ["ENSDARG00000041689", "ENSDARG00000029830"]),
]

con = sqlite3.connect(str(JOB_DIR / "jobdb.sqlite"))


_pwm_cache: dict[str, np.ndarray | None] = {}


def representative_pwm(dbid: str):
    """Build (and cache) the representative PWM for a TF, or None if it has no PPMs."""

    if dbid not in _pwm_cache:
        if (SPEC_DIR / dbid / "specificity").exists():
            loaded = load_top_ppms(con, dbid, num_specs)
            _pwm_cache[dbid], _, _ = build_representative_pwm(loaded, cor_threshold)
        else:
            _pwm_cache[dbid] = None

    return _pwm_cache[dbid]


# One entry per cluster (cluster_tag) so each cluster gets its own bar plot
cluster_hist: dict[
    str, list[dict]
] = {}  # cluster_tag -> member-vs-representative sim records
for family, cluster_type, reps in CLUSTERS:
    membership = get_cluster_membership(family, cluster_type, reps)
    n_members = sum(len(v) for v in membership.values())

    # The recog clustering runs per animalTFDB "family" file, but the Homeobox file
    # lumps together true homeodomain and POU-domain TFs (POU carries a homeodomain).
    # Group/plot by each cluster's actual member family so Homeobox and Pou land in
    # separate plots; fall back to the clustering family if members aren't in the table.
    mem_fams = [
        fam_of[m.lower()]
        for mem_ids in membership.values()
        for m in mem_ids
        if m.lower() in fam_of
    ]
    plot_family = pd.Series(mem_fams).mode().iat[0] if mem_fams else family
    cluster_tag = f"{plot_family}_{'_'.join(r[-6:] for r in reps)}"

    logo_pwms, logo_labels, cluster_sims = [], [], []
    for rep, mem_ids in membership.items():
        rep_l = rep.lower()
        rep_pwm = representative_pwm(rep_l)
        if rep_pwm is None:
            print(f"rep {name_of.get(rep_l, rep_l)}: no PPMs, sub-cluster skipped")
            continue
        logo_pwms.append(rep_pwm)
        logo_labels.append(f"{name_of.get(rep_l, rep_l)} (rep)")

        for m in mem_ids:
            m_l = m.lower()
            if m_l == rep_l:
                continue
            m_pwm = representative_pwm(m_l)
            if m_pwm is None:
                continue
            sim, _, _ = pwm_similarity(m_pwm, rep_pwm)
            cluster_hist.setdefault(cluster_tag, []).append(
                {"sim": float(sim), "cluster": rep[-6:]}
            )
            cluster_sims.append(float(sim))
            logo_pwms.append(m_pwm)
            logo_labels.append(name_of.get(m_l, m_l))

    if not cluster_sims:
        print("no member-vs-representative comparisons, skipping")
        continue

    # Stack of representative logos (rep first, cap at 8 for readability)
    plot_aligned_logos(
        [(pwm, lab, "") for pwm, lab in zip(logo_pwms[:8], logo_labels[:8])],
        cluster_tag,
        f"representative PWM per TF (first {min(8, len(logo_pwms))})",
        out_path=f"{FIG_DIR}/{cluster_tag}_logos.svg",
    )

# One histogram per cluster: member similarity to the cluster representative
# (separate plots for each cluster, so the two zf-C2H2 and two Homeobox clusters
# each get their own bar plot rather than being stacked into one per-family figure).
for cluster_tag, records in cluster_hist.items():
    plot_similarity_to_rep_hist(
        records,
        f"Similarity to cluster representative — {cluster_tag}",
        out_path=f"{FIG_DIR}/{cluster_tag}_hist.svg",
    )


# %%
# ### Within-cluster .similar consistency (primary motif rep-cluster)
# Finer-grained than the across-rep-cluster comparison above: within usf1's primary motif
# rep-cluster, build one representative PWM per individual `.similarN` variant and compare
# them all-vs-all. Tight diagonal blocks => the predicted PWM is robust to which specific
# similar-motif DNA scaffolded the prediction.

# %%
usf1_runs = pd.read_sql(
    "SELECT motif FROM runs WHERE protein = ?", con, params=("ensdarg00000041689",)
)
usf1_runs["rep"] = usf1_runs["motif"].str.replace(r"\.similar\d+", "", regex=True)
rep_counts = usf1_runs["rep"].value_counts()
primary_rep = rep_counts.index[0]
variants = sorted(usf1_runs.loc[usf1_runs["rep"] == primary_rep, "motif"].unique())
print(f"\n{primary_rep}: {len(variants)} .similar variants -> per-variant PWMs")

var_pwms, var_labels = [], []
for variant in variants:
    loaded = load_top_ppms(con, "ensdarg00000041689", num_specs, motif_filter=variant)
    consensus, _, _ = build_representative_pwm(loaded, cor_threshold)
    if consensus is None:
        continue
    var_pwms.append(consensus)
    # Strip the shared rep prefix so labels read "rep"/".similarN"
    var_labels.append(variant.replace(primary_rep, "") or "(base)")

var_mat, _ = plot_pwm_similarity_heatmap(
    var_pwms,
    var_labels,
    f"usf1 within-{primary_rep} .similar consistency ({len(var_pwms)} variants)",
    out_path=f"{FIG_DIR}/usf1_similar_sim_heatmap.svg",
)
_off = var_mat[~np.eye(len(var_mat), dtype=bool)]
print(
    f"  within-{primary_rep} pairwise similarity: "
    f"mean={_off.mean():.3f}, median={np.median(_off):.3f}, min={_off.min():.3f}"
)

con.close()

# %%
