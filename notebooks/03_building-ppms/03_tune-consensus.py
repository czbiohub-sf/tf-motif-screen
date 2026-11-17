"""
Compare NA-MPNN consensus specificities to JASPAR PWMs for TFs with high similarity (>= 95%) to a
human ortholog. Structures ranked by combined ipTM/dom_pLDDT score; PPMs averaged into cluster
consensuses. Clustering uses UPGMA (average linkage) with RSAT/JASPAR cor >= N as merge criterion.

Ben Iovino 5/15/26 Biohub
"""

# %%
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from af3tfm.consensus import (
    _load_b_strand_ppm,
    cluster_consensus_pwm,
    cluster_upgma,
    plot_aligned_logos,
    pred_recovery_fraction,
)
from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.motif import load_homer_multi_motif_file
from af3tfm.utils import pwm_similarity, set_plotting_style

DATA = Path(DATA_DIR)
set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "03_building-ppms" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "03_building-ppms" / "tables"
TAB_DIR.mkdir(exist_ok=True)

# AlphaFold confidence palette (very low -> very high), matching 01_conf-plots.py
AF3_CMAP = LinearSegmentedColormap.from_list(
    "af3_confidence",
    ["#FF7D45", "#FFDB13", "#65CBF3", "#0053D6"],
)

# HYPERPARAMETERS
num_specs = 47  # Number of NA-MPNN specificities to consider per TF (top-ranked by combined ipTM/dom_pLDDT)
cor_threshold = 0.73  # Minimum correlation for UPGMA cluster merging (min overlap hardcoded to be 6 in pwm_similarity)
num_show = (
    5  # Number of top-ranked PWMs / largest clusters to draw per aligned-logo figure
)
only_dbid = "ensdarg00000016526"  # Restrict the run to one TF (figure generation); None = all TFs


def plot_tf_null_clusters(
    cluster_score: tuple,
    tf_name: str,
    jaspar_id: str,
):
    """
    One plot per TF: null KDE + JASPAR score for the largest cluster's consensus.

    Args:
        cluster_score: (merged_rank, n_members, sim_c, all_sims_c) for the largest cluster
        tf_name: str, TF name for title
        jaspar_id: str, JASPAR ID for title
    """

    merged_rank, n_members, sim_c, null_sims_c = cluster_score
    color = "#738fc1"

    fig, ax = plt.subplots(figsize=(5, 5))
    xs = np.linspace(null_sims_c.min(), null_sims_c.max(), 300)
    ys = stats.gaussian_kde(null_sims_c)(xs)
    ax.plot(xs, ys, color=color, lw=1.2, alpha=0.7)
    ax.fill_between(xs, ys, alpha=0.15, color=color)
    ax.axvline(
        sim_c,
        color=color,
        lw=1.5,
        label=f"Cluster {merged_rank} (n={n_members}, sim={sim_c:.3f})",
    )

    ax.set_xlabel("HOMER similarity")
    ax.set_ylabel("Density")
    ax.set_title(f"{tf_name} — JASPAR {jaspar_id}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    safe_name = tf_name.replace("/", "_").replace(" ", "_")
    plt.savefig(
        f"{FIG_DIR}/{safe_name}_{jaspar_id}_null_clusters.svg",
        bbox_inches="tight",
        dpi=150,
    )
    plt.show()


def plot_iptm_vs_plddt(
    all_preds: pd.DataFrame,
    tf_name: str,
    dbid: str,
):
    """
    Plots ipTM vs. domain pLDDT for all NA-MPNN predictions of a given TF, highlighting the top 5% by combined rank.

    Args:
        all_preds: DataFrame with columns "ipTM", "dom_pLDDT", and "combined" (rank product of ipTM and dom_pLDDT)
        tf_name: str, TF name for title
        dbid: str, DBID for title and potential filename
    """

    top5 = all_preds["combined"] >= all_preds["combined"].quantile(0.95)
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.scatter(all_preds["ipTM"], all_preds["dom_pLDDT"], alpha=0.5, color="#738fc1")
    ax.scatter(
        all_preds.loc[top5, "ipTM"],
        all_preds.loc[top5, "dom_pLDDT"],
        color="#d56c55",
        s=80,
        edgecolor="black",
    )
    ax.set_xlabel("ipTM")
    ax.set_ylabel("Domain pLDDT")
    ax.set_title(f"{tf_name} ipTM vs. Domain pLDDT")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{dbid}_iptm_vs_dom_plddt.svg", bbox_inches="tight", dpi=150)
    plt.show()


# %%
# Load JASPAR motifs from the HOMER-format file
jaspar_path = DATA / "tools" / "homer" / "jaspar.homer.motif"
jaspar_names, jaspar_pwms = load_homer_multi_motif_file(str(jaspar_path))
jaspar_id_to_pwm = dict(zip(jaspar_names, jaspar_pwms))
print(f"Loaded {len(jaspar_id_to_pwm)} JASPAR motifs")

# Load TF table and filter for Hom_Sim >= 95
csv_path = DATA / "sequences" / "animaltfdb" / "tfdb4_rep_tfs.csv"
df = pd.read_csv(str(csv_path))
high_sim = df[df["Hom_Sim"] >= 95].copy()
print(f"TFs with Hom_Sim >= 95%: {len(high_sim)}")

job_dir = DATA / "jobs" / "hero"
spec_dir = job_dir / "specificities"
jobdb = sqlite3.connect(str(job_dir / "jobdb.sqlite"))
jaspar_pwm_list = list(jaspar_id_to_pwm.values())

results = (
    high_sim[
        high_sim["Hom_JASPAR"].notna() & high_sim["Hom_JASPAR"].isin(jaspar_id_to_pwm)
    ]
    .copy()
    .reset_index(drop=True)
)

print(f"TFs with JASPAR match: {len(results)}")


# %%
def _process_one_tf(
    row_dict: dict,
    jaspar_id: str,
    ref_pwm: np.ndarray,
    spec_dir_str: str,
    job_dir_str: str,
    num_specs: int,
    cor_threshold: float,
    jaspar_pwm_list: list,
) -> tuple:
    import matplotlib.pyplot as plt

    plt.switch_backend("agg")

    row = pd.Series(row_dict)
    dbid = row["DBID"].lower()
    if only_dbid and dbid != only_dbid:
        return [], []

    spec_dir = Path(spec_dir_str)
    job_dir = Path(job_dir_str)
    jobdb = sqlite3.connect(str(job_dir / "jobdb.sqlite"))

    top_rows_tf = []
    cluster_rows_tf = []

    indiv_dir = spec_dir / dbid / "specificity"
    if not indiv_dir.exists():
        jobdb.close()
        return [], []
    all_preds = pd.read_sql(
        "SELECT motif, dna, ipTM, dom_pLDDT, structure_path FROM runs WHERE protein = ?",
        jobdb,
        params=(dbid,),
    )
    # Seed is the trailing integer of the fold output dir (see run_spec.get_all_structs);
    # npz files are named {dbid}_{motif}_{dna}_{seed}.npz.
    all_preds["seed"] = all_preds["structure_path"].map(
        lambda sp: Path(sp).parent.parent.name.rsplit("_", 1)[-1]
    )
    all_preds["combined"] = all_preds["ipTM"].rank(pct=True) * all_preds[
        "dom_pLDDT"
    ].rank(pct=True)
    # plot_iptm_vs_plddt(all_preds, row["TF_Name"], dbid)
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
            ppm_b = _load_b_strand_ppm(npz_path)
            loaded.append((ppm_b, dna))
        except Exception:
            continue

    if not loaded:
        jobdb.close()
        return [], []

    # Pre-compute JASPAR null for each PPM (HOMER sim)
    scored_with_null = []
    for ppm_b, dna in loaded:
        all_sims = np.array([pwm_similarity(ppm_b, pwm)[0] for pwm in jaspar_pwm_list])
        scored_with_null.append((ppm_b, dna, all_sims.mean(), all_sims.std(), all_sims))
    merged = cluster_upgma(scored_with_null, cor_threshold)

    # cluster_upgma IDs clusters by member index; re-rank by N_Members descending (1 = largest)
    size_rank_map = {
        old_rank: new_rank
        for new_rank, (old_rank, _) in enumerate(
            Counter(m[1] for m in merged).most_common(), start=1
        )
    }

    # Score individuals; collect members per cluster for consensus computation
    clusters = {}
    top_info = []
    for orig_rank, merged_rank, ppm_b, dna, align_off, align_rc, is_med in merged:
        sim, off, rc = pwm_similarity(ppm_b, ref_pwm)
        _, _, null_mean, null_std, null_sims = scored_with_null[orig_rank - 1]
        z_score = (sim - null_mean) / null_std if null_std > 0 else 0.0
        p_value = float((null_sims >= sim).mean())
        recovery = pred_recovery_fraction(ppm_b, ref_pwm, off, rc)
        top_rows_tf.append(
            {
                "DBID": row["DBID"],
                "TF_Name": row["TF_Name"],
                "Hom_JASPAR": jaspar_id,
                "Score_Rank": orig_rank,
                "Merged_Rank": size_rank_map[merged_rank],
                "DNA": dna,
                "HOMER_Sim": round(sim, 4),
                "Z_Score": round(z_score, 3),
                "P_Value": round(p_value, 5),
                "Recovery": round(recovery, 4),
            }
        )

        top_info.append(
            (orig_rank, ppm_b, dna, size_rank_map[merged_rank], sim, p_value)
        )

        clusters.setdefault(merged_rank, []).append(
            (ppm_b, align_off, align_rc, is_med, dna)
        )

    plot_aligned_logos(
        [
            (
                ppm,
                f"top-{r}  {dna}  cluster {c}",
                f"(HOMER={s:.3f}, p={p:.4f})",
            )
            for r, ppm, dna, c, s, p in top_info[:num_show]
        ],
        row["TF_Name"],
        f"top {min(num_show, len(top_info))} PWMs by AF3 confidence",
        out_path=f"{FIG_DIR}/{dbid}_top_pwms_aligned.svg",
    )

    # Compute consensus PWM per cluster and score against JASPAR
    tf_cluster_scores = []
    consensus_info = []
    for merged_rank, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        size_rank = size_rank_map[merged_rank]
        members.sort(key=lambda x: not x[3])  # medoid first
        consensus = cluster_consensus_pwm(
            [(ppm, off, rc) for ppm, off, rc, _, _ in members]
        )

        all_sims_c = np.array(
            [pwm_similarity(consensus, pwm)[0] for pwm in jaspar_pwm_list]
        )
        sim_c, off_c, rc_c = pwm_similarity(consensus, ref_pwm)
        null_mean_c = all_sims_c.mean()
        null_std_c = all_sims_c.std()
        p_c = float((all_sims_c >= sim_c).mean())
        z_c = (sim_c - null_mean_c) / null_std_c if null_std_c > 0 else 0.0

        recovery = pred_recovery_fraction(consensus, ref_pwm, off_c, rc_c)
        cluster_rows_tf.append(
            {
                "DBID": row["DBID"],
                "TF_Name": row["TF_Name"],
                "Hom_JASPAR": jaspar_id,
                "Merged_Rank": size_rank,
                "N_Members": len(members),
                "HOMER_Sim": round(sim_c, 4),
                "Z_Score": round(z_c, 3),
                "P_Value": round(p_c, 5),
                "Recovery": round(recovery, 4),
            }
        )
        tf_cluster_scores.append((size_rank, len(members), sim_c, all_sims_c))

        consensus_info.append(
            (
                consensus,
                f"cluster {size_rank}  (n={len(members)})",
                f"(HOMER={sim_c:.3f}, z={z_c:.2f}, p={p_c:.4f}, IC={recovery:.3f})",
            )
        )

    plot_aligned_logos(
        consensus_info[:num_show],
        row["TF_Name"],
        f"top {min(num_show, len(consensus_info))} cluster consensus PWMs",
        ref_panel=(ref_pwm, f"JASPAR {jaspar_id}"),
        out_path=f"{FIG_DIR}/{dbid}_cluster_consensus_aligned_{jaspar_id}.svg",
    )

    # plot_tf_null_clusters(tf_cluster_scores[0], row["TF_Name"], jaspar_id)

    jobdb.close()
    return top_rows_tf, cluster_rows_tf


def _family_ranking_one_tf(
    row_dict: dict,
    spec_dir_str: str,
    job_dir_str: str,
    num_specs: int,
    cor_threshold: float,
    jaspar_pwm_list: list,
) -> tuple:
    """
    Lightweight per-TF ranking for the family metric: rebuild the deployed prediction
    (largest UPGMA cluster consensus) and score it against every JASPAR motif, returning
    the full similarity vector. Skips the per-member null (only item[0] feeds clustering),
    so this is much cheaper than _process_one_tf while giving the identical consensus.

    Returns:
        (dbid, sims) where sims is a float32 vector aligned to jaspar_pwm_list, or (dbid, None).
    """

    row = pd.Series(row_dict)
    dbid = row["DBID"].lower()
    spec_dir = Path(spec_dir_str)
    job_dir = Path(job_dir_str)

    indiv_dir = spec_dir / dbid / "specificity"
    if not indiv_dir.exists():
        return dbid, None

    jobdb = sqlite3.connect(str(job_dir / "jobdb.sqlite"))
    all_preds = pd.read_sql(
        "SELECT motif, dna, ipTM, dom_pLDDT, structure_path FROM runs WHERE protein = ?",
        jobdb,
        params=(dbid,),
    )
    jobdb.close()
    if all_preds.empty:
        return dbid, None

    all_preds["seed"] = all_preds["structure_path"].map(
        lambda sp: Path(sp).parent.parent.name.rsplit("_", 1)[-1]
    )
    all_preds["combined"] = all_preds["ipTM"].rank(pct=True) * all_preds[
        "dom_pLDDT"
    ].rank(pct=True)
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
            loaded.append(_load_b_strand_ppm(npz_path))
        except Exception:
            continue
    if not loaded:
        return dbid, None

    # cluster_upgma only reads item[0]; the null slots are unused, so pass placeholders.
    scored = [(ppm, "", 0.0, 0.0, None) for ppm in loaded]
    merged = cluster_upgma(scored, cor_threshold)

    clusters = {}
    for _, merged_rank, ppm_b, _, align_off, align_rc, is_med in merged:
        clusters.setdefault(merged_rank, []).append(
            (ppm_b, align_off, align_rc, is_med)
        )

    big = max(clusters.values(), key=len)
    big.sort(key=lambda x: not x[3])  # medoid first
    consensus = cluster_consensus_pwm([(ppm, off, rc) for ppm, off, rc, _ in big])

    sims = np.array(
        [pwm_similarity(consensus, pwm)[0] for pwm in jaspar_pwm_list], dtype=np.float32
    )
    return dbid, sims


top_rows = []
cluster_rows = []

with ProcessPoolExecutor() as executor:
    futures = {
        executor.submit(
            _process_one_tf,
            row.to_dict(),
            row["Hom_JASPAR"],
            jaspar_id_to_pwm[row["Hom_JASPAR"]],
            str(spec_dir),
            str(job_dir),
            num_specs,
            cor_threshold,
            jaspar_pwm_list,
        ): row["DBID"]
        for _, row in results.iterrows()
        if not only_dbid or row["DBID"].lower() == only_dbid
    }
    for future in as_completed(futures):
        tf_top, tf_cluster = future.result()
        top_rows.extend(tf_top)
        cluster_rows.extend(tf_cluster)

top_df = pd.DataFrame(top_rows)
# top_df.to_csv(
#    f"{TAB_DIR}/ind_ppm_jaspar_{num_specs}_{cor_threshold}_new.csv", index=False
# )

cluster_df = pd.DataFrame(cluster_rows)
# cluster_df.to_csv(
#    f"{TAB_DIR}/cluster_ppm_jaspar_{num_specs}_{cor_threshold}_new.csv", index=False
# )


# %%
cluster_df = pd.read_csv(
    f"{TAB_DIR}/cluster_ppm_jaspar_{num_specs}_{cor_threshold}_new.csv"
)
top_df = pd.read_csv(f"{TAB_DIR}/ind_ppm_jaspar_{num_specs}_{cor_threshold}_new.csv")
if not top_df.empty:
    max_rank = int(top_df["Score_Rank"].max())
    # Merged_Rank is already the cluster size rank (1 = most members)
    max_size_rank = int(cluster_df["Merged_Rank"].max()) if not cluster_df.empty else 0
    total_tfs = top_df["DBID"].nunique()

    display_ranks = 5

    cumulative_original = []
    for n in range(1, min(max_rank, display_ranks) + 1):
        subset = top_df[top_df["Score_Rank"] <= n]
        cumulative_original.append(subset[subset["P_Value"] < 0.05]["DBID"].nunique())

    cumulative_merged = []
    for n in range(1, min(max_size_rank, display_ranks) + 1):
        subset = cluster_df[cluster_df["Merged_Rank"] <= n]
        cumulative_merged.append(subset[subset["P_Value"] < 0.05]["DBID"].nunique())

    def _draw_saturation(ax, x, counts, total, color, xlabel, title, alpha=1.0):
        x = np.array(x)
        bars = ax.bar(
            x, counts, color=color, edgecolor="white", linewidth=0.5, alpha=alpha
        )
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count),
                ha="center",
                va="bottom",
            )
        ax.axhline(
            total,
            color="gray",
            lw=1,
            ls="--",
            alpha=0.7,
            label=f"Total TFs (n={total})",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("TFs with significant JASPAR match")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_ylim(0, total + total * 0.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(5, 4), gridspec_kw={"width_ratios": [1, 1]}
    )
    _draw_saturation(
        ax1,
        x=list(range(1, len(cumulative_original) + 1)),
        counts=cumulative_original,
        total=total_tfs,
        color="grey",
        xlabel="Combined score rank (n)",
        title="PWM by AF3 confidence",
    )
    _draw_saturation(
        ax2,
        x=list(range(1, len(cumulative_merged) + 1)),
        counts=cumulative_merged,
        total=total_tfs,
        color="orange",
        alpha=0.7,
        xlabel="Cluster size rank (1=most members)",
        title="PWM by UPGMA consensus",
    )
    plt.suptitle("Saturation of significantly similar PWMs vs JASPAR")
    plt.tight_layout()
    plt.savefig(
        f"{FIG_DIR}/saturation_{num_specs}_{cor_threshold}_new.svg",
        bbox_inches="tight",
        dpi=150,
    )
    plt.show()

# %%
if not cluster_df.empty:
    # Per-TF: the largest cluster (one row per DBID)
    plot_df = cluster_df.loc[
        cluster_df["Merged_Rank"] == 1,
        ["DBID", "TF_Name", "HOMER_Sim", "Recovery", "P_Value"],
    ].copy()
    plot_df["homer_sig"] = plot_df["P_Value"] < 0.05
    plot_df["recovery_pct"] = plot_df["Recovery"] * 100

    labeled_tfs = [
        "tbx1",
        "satb1b",
        "sox2",
        "jund",
        "nfyba",
        "si:ch211-154o6.2",
        "znf384l",
    ]

    fig, ax = plt.subplots(figsize=(5, 5))
    is_labeled_all = plot_df["TF_Name"].isin(labeled_tfs)

    # Draw non-significant first so significant points lay on top;
    for sig, color, label, zorder in [
        (False, "#aaaaaa", "p ≥ 0.05", 3),
        (True, "#d6604d", "p < 0.05", 4),
    ]:
        sub = plot_df[(plot_df["homer_sig"] == sig) & ~is_labeled_all]
        ax.scatter(
            sub["HOMER_Sim"],
            sub["recovery_pct"],
            color=color,
            s=50,
            zorder=zorder,
            label=label,
        )

    # Draw labeled TFs on top of all other dots, colored by significance.
    labeled_df = plot_df[is_labeled_all]
    ax.scatter(
        labeled_df["HOMER_Sim"],
        labeled_df["recovery_pct"],
        color=labeled_df["homer_sig"]
        .map({False: "#aaaaaa", True: "#d6604d"})
        .to_numpy(),
        s=50,
        zorder=5,
        edgecolors="black",
        linewidths=1.5,
    )
    for _, r in labeled_df.iterrows():
        ax.annotate(
            r["TF_Name"],
            xy=(r["HOMER_Sim"], r["recovery_pct"]),
            xytext=(4, 2),
            textcoords="offset points",
        )
    ax.set_xlabel("Similarity (HOMER)")
    ax.set_ylabel("Information content recovery (%)")
    ax.legend(loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(
        f"{FIG_DIR}/icrecovery_{num_specs}_{cor_threshold}_new.svg",
        bbox_inches="tight",
        dpi=150,
    )
    plt.show()

# %%
# Same per-TF scatter but colored by the AF3 confidence score we compute for a TF
if not cluster_df.empty:
    plot_df = cluster_df.loc[
        cluster_df["Merged_Rank"] == 1,
        ["DBID", "TF_Name", "HOMER_Sim", "Recovery"],
    ].copy()
    plot_df["recovery_pct"] = plot_df["Recovery"] * 100
    plot_df["dbid_l"] = plot_df["DBID"].str.lower()

    # Per-TF top-N
    dbids = set(plot_df["dbid_l"])
    all_runs = pd.read_sql("SELECT protein, ipTM, dom_pLDDT FROM runs", jobdb)
    all_runs = all_runs[all_runs["protein"].isin(dbids)]
    conf_rows = []
    for prot, g in all_runs.groupby("protein", sort=False):
        combined = g["ipTM"].rank(pct=True) * g["dom_pLDDT"].rank(pct=True)
        top = g.loc[combined.nlargest(num_specs).index]
        conf_rows.append(
            {"dbid": prot, "conf_raw": (top["ipTM"] * top["dom_pLDDT"] / 100).mean()}
        )
    conf = pd.DataFrame(conf_rows)
    plot_df = plot_df.merge(
        conf[["dbid", "conf_raw"]], left_on="dbid_l", right_on="dbid", how="left"
    )
    # Color by the raw ipTM x DBD pLDDT, but clip the colorbar to the observed range so
    # the palette isn't wasted on 0-0.7 (empty)
    cmin, cmax = plot_df["conf_raw"].min(), plot_df["conf_raw"].max()

    labeled_tfs = [
        "tbx1",
        "satb1b",
        "sox2",
        "jund",
        "nfyba",
        "si:ch211-154o6.2",
        "znf384l",
    ]
    is_labeled = plot_df["TF_Name"].isin(labeled_tfs)

    fig, ax = plt.subplots(figsize=(3.4, 5))
    sc = ax.scatter(
        plot_df["HOMER_Sim"],
        plot_df["recovery_pct"],
        c=plot_df["conf_raw"],
        cmap=AF3_CMAP,
        vmin=cmin,
        vmax=cmax,
        s=32,
        zorder=3,
    )
    # Labeled TFs on top with a black edge, colored by the same confidence scale.
    lab = plot_df[is_labeled]
    ax.scatter(
        lab["HOMER_Sim"],
        lab["recovery_pct"],
        c=lab["conf_raw"],
        cmap=AF3_CMAP,
        vmin=cmin,
        vmax=cmax,
        s=40,
        zorder=5,
        edgecolors="black",
        linewidths=1.5,
    )
    for _, r in lab.iterrows():
        tf_name = r["TF_Name"]
        align_right = tf_name in {"tbx1", "jund", "sox2"}
        label_y = 8 if tf_name == "si:ch211-154o6.2" else 2
        ax.annotate(
            tf_name[:1].upper() + tf_name[1:],
            xy=(r["HOMER_Sim"], r["recovery_pct"]),
            xytext=(-4 if align_right else 4, label_y),
            textcoords="offset points",
            ha="right" if align_right else "left",
            fontsize=8,
        )
    cbar = fig.colorbar(
        sc, ax=ax, orientation="vertical", pad=0.04, fraction=0.045, aspect=30
    )
    cbar.set_label("Confidence (mean ipTM × DBD pLDDT)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_xlabel("Similarity (HOMER)")
    ax.set_ylabel("Information content recovery (%)")

    # How well confidence (the color) tracks each axis, annotated in-panel.
    rho_h = stats.spearmanr(plot_df["conf_raw"], plot_df["HOMER_Sim"]).statistic
    rho_ic = stats.spearmanr(plot_df["conf_raw"], plot_df["recovery_pct"]).statistic
    ax.text(
        0.03,
        0.97,
        f"conf ~ HOMER: ρ={rho_h:.2f}\nconf ~ IC rec.: ρ={rho_ic:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="#cccccc", alpha=0.8),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for extension in ("svg", "png"):
        plt.savefig(
            f"{FIG_DIR}/icrecovery_conf_{num_specs}_{cor_threshold}_new.{extension}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()

# %%
# Cluster size vs HOMER similarity for the deployed (largest) cluster per TF
if not cluster_df.empty:
    sz = cluster_df.loc[
        cluster_df["Merged_Rank"] == 1, ["N_Members", "HOMER_Sim", "P_Value"]
    ].copy()
    sz["sig"] = sz["P_Value"] < 0.05
    rho, p_rho = stats.spearmanr(sz["N_Members"], sz["HOMER_Sim"])
    lr = stats.linregress(sz["N_Members"].to_numpy(), sz["HOMER_Sim"].to_numpy())

    fig, ax = plt.subplots(figsize=(3.4, 5))
    for sig, color, label in [
        (False, "#aaaaaa", "p ≥ 0.05"),
        (True, "#d6604d", "p < 0.05"),
    ]:
        s = sz[sz["sig"] == sig]
        ax.scatter(
            s["N_Members"],
            s["HOMER_Sim"],
            color=color,
            s=32,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.4,
            zorder=4 if sig else 3,
            label=label,
        )
    xs = np.linspace(sz["N_Members"].min(), sz["N_Members"].max(), 100)
    ax.plot(
        xs,
        lr.intercept + lr.slope * xs,
        color="#333333",
        lw=1.5,
        ls="--",
        zorder=5,
        label=f"Spearman ρ={rho:.2f}\np={p_rho:.1e}",
    )
    ax.set_xlabel("Deployed cluster size\n(n members)")
    ax.set_ylabel("Similarity (HOMER)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for extension in ("svg", "png"):
        plt.savefig(
            f"{FIG_DIR}/clustersize_vs_homer_{num_specs}_{cor_threshold}_new.{extension}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()


# %%
# IC recovery by native complex type
COMPLEX_TYPE = {
    # Monomer: one DNA-contacting chain reads the whole site
    "zf-C2H2": "Monomer",
    "Homeobox": "Monomer",
    "Fork_head": "Monomer",
    "HMG": "Monomer",
    "ETS": "Monomer",
    "PAX": "Monomer",
    "Pou": "Monomer",
    "T-box": "Monomer",
    "CUT": "Monomer",
    "TF_Otx": "Monomer",
    "SAND": "Monomer",
    "zf-BED": "Monomer",
    "MYB": "Monomer",
    "zf-GATA": "Monomer",
    "ZBTB": "Monomer",
    "TEA": "Monomer",
    "CSL": "Monomer",
    "Runt": "Monomer",
    "SF-like": "Monomer",
    "NGFIB-like": "Monomer",
    "IRF": "Monomer",
    # Homodimer: two identical chains on a (pseudo)palindrome
    "bHLH": "Homodimer",
    "TF_bZIP": "Homodimer",
    "ESR-like": "Homodimer",
    "RXR-like": "Homodimer",
    "SRF": "Homodimer",
    "STAT": "Homodimer",
    "CTF_NFI": "Homodimer",
    "GCNF-like": "Homodimer",
    "COE": "Homodimer",
    "RFX": "Homodimer",
    "DM": "Homodimer",
    "Nrf1": "Homodimer",
    # Heterodimer: two different chains
    "THR-like": "Heterodimer",
    "E2F": "Heterodimer",
    # Larger complex: trimer / tetramer
    "HSF": "Larger complex",
    "P53": "Larger complex",
    "NF-YB": "Larger complex",
    "NF-YC": "Larger complex",
    "MH1": "Larger complex",
}
# Homo- and heterodimers collapse to one "Dimer" bucket for the plot
DIMER_MERGE = {"Homodimer": "Dimer", "Heterodimer": "Dimer"}
# Individual native assemblies can differ from their family defaults.
COMPLEX_TYPE_OVERRIDES = {
    "satb1b": "Larger complex",  # SATB1 tetramer
    "ptf1a": "Larger complex",  # PTF1A / E protein / RBPJ or RBPJL trimer
    "foxp1a": "Dimer",
    "foxp4": "Dimer",
    "bcl6aa": "Dimer",
    "bcl6ab": "Dimer",
    "patz1": "Dimer",
    "ikzf1": "Dimer",
    # GABPA / GABPB heterodimer on one ETS site; paired sites can bind a tetramer.
    "gabpa": "Dimer",
    "rorab": "Monomer",
    "e2f8": "Monomer",  # Both DNA-binding domains are in the same chain.
}
CTYPE_ORDER = ["Monomer", "Dimer", "Larger complex"]
CTYPE_COLORS = {
    "Monomer": "#4c78a8",
    "Dimer": "#f58518",
    "Larger complex": "#72b7b2",
}

if not cluster_df.empty:
    fam = pd.read_csv(
        DATA / "sequences" / "animaltfdb" / "tfdb4_rep_tfs.csv",
        usecols=["DBID", "Family"],
    )
    ctype_df = cluster_df.loc[cluster_df["Merged_Rank"] == 1].merge(
        fam, on="DBID", how="left"
    )
    ctype_df["ctype"] = ctype_df["Family"].map(COMPLEX_TYPE).replace(DIMER_MERGE)
    ctype_df["ctype"] = (
        ctype_df["TF_Name"]
        .str.lower()
        .map(COMPLEX_TYPE_OVERRIDES)
        .fillna(ctype_df["ctype"])
    )
    ctype_df["recovery_pct"] = ctype_df["Recovery"] * 100

    # Graded IC recovery per group
    groups = [
        ctype_df.loc[ctype_df["ctype"] == g, "recovery_pct"].to_numpy()
        for g in CTYPE_ORDER
    ]
    kw_p = stats.kruskal(*[g for g in groups if len(g)]).pvalue
    mono = ctype_df.loc[ctype_df["ctype"] == "Monomer", "recovery_pct"]
    dim = ctype_df.loc[ctype_df["ctype"] == "Dimer", "recovery_pct"]
    mwu_p = stats.mannwhitneyu(mono, dim, alternative="greater").pvalue

    # Keep this panel narrow enough to sit beside the other recovery plots.
    fig, ax = plt.subplots(figsize=(3.4, 5))
    bp = ax.boxplot(
        groups,
        positions=range(len(CTYPE_ORDER)),
        widths=0.6,
        showfliers=False,
        medianprops=dict(color="black", lw=1.5),
        patch_artist=True,
    )
    for patch, g in zip(bp["boxes"], CTYPE_ORDER):
        patch.set_facecolor(CTYPE_COLORS[g])
        patch.set_alpha(0.35)
    labeled_tfs = ["tbx1", "jund", "nfyba"]
    rng = np.random.default_rng(0)
    for i, (g, vals) in enumerate(zip(CTYPE_ORDER, groups)):
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            color=CTYPE_COLORS[g],
            s=18,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.4,
            zorder=3,
        )
        ax.text(i, 103, f"n={len(vals)}", ha="center", va="bottom", fontsize=9)

        # Outline and name the highlighted TFs at their jittered positions
        names = ctype_df.loc[ctype_df["ctype"] == g, "TF_Name"].to_numpy()
        is_lab = np.isin(names, labeled_tfs)
        ax.scatter(
            i + jitter[is_lab],
            vals[is_lab],
            color=CTYPE_COLORS[g],
            s=28,
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )
        for x, y, name in zip(i + jitter[is_lab], vals[is_lab], names[is_lab]):
            ax.annotate(
                name,
                xy=(x, y),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xticks(range(len(CTYPE_ORDER)))
    ax.set_xticklabels(["Monomer", "Dimer", "Larger\ncomplex"], fontsize=9)
    ax.set_ylabel("Information content recovery (%)")
    ax.set_ylim(0, 110)
    ax.set_title(
        "IC recovery by\nnative complex type\n"
        "Monomer > dimer (MWU):\n"
        f"p={mwu_p:.1e}; KW: p={kw_p:.1e}",
        fontsize=10,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for extension in ("svg", "png"):
        plt.savefig(
            f"{FIG_DIR}/recovery_by_complextype_{num_specs}_{cor_threshold}_new.{extension}",
            bbox_inches="tight",
            dpi=300,
        )
    plt.show()


# %%
# Baseline without NA-MPNN: per TF, pick the input library motif whose complexes have the
# highest mean AF3 confidence, and score that motif's library PWM against JASPAR
def _library_null_one(name: str, lib_pwm: np.ndarray, jaspar_pwm_list: list) -> tuple:
    """
    HOMER sim of one library motif against every JASPAR motif (TF-independent null).
    """

    return name, np.array(
        [pwm_similarity(lib_pwm, pwm)[0] for pwm in jaspar_pwm_list], dtype=np.float32
    )


# 312 non-redundant HOMER motifs the hero DNA was sampled from (motifN = jobdb motif column)
lib_dir = DATA / "tools" / "homer" / "jaspar_clustered" / "homerResults"
lib_pwms, lib_hdrs = {}, {}
for p in lib_dir.glob("motif*.motif"):
    if not p.stem.removeprefix("motif").isdigit():  # skip motifNRV.motif
        continue
    lines = p.read_text().strip().splitlines()
    lib_hdrs[p.stem] = lines[0]
    lib_pwms[p.stem] = np.array([list(map(float, ln.split())) for ln in lines[1:]])

# Null is TF-independent, so compute once and cache
lib_null_path = Path("../library_null_jaspar.npz")
if lib_null_path.exists():
    z = np.load(lib_null_path)
    lib_null = dict(zip(z["names"], z["null"]))
else:
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_library_null_one, m, pwm, jaspar_pwm_list)
            for m, pwm in lib_pwms.items()
        ]
        lib_null = dict(f.result() for f in as_completed(futures))
    np.savez(
        lib_null_path,
        names=list(lib_null),
        null=np.stack(list(lib_null.values())),
    )

dbids = results["DBID"].str.lower().tolist()
runs = pd.read_sql(
    f"SELECT protein, motif, ipTM, dom_pLDDT FROM runs WHERE protein IN ({','.join('?' * len(dbids))})",
    jobdb,
    params=dbids,
)
runs["combined"] = runs.groupby("protein")["ipTM"].rank(pct=True) * runs.groupby(
    "protein"
)["dom_pLDDT"].rank(pct=True)
per_motif = (
    runs.groupby(["protein", "motif"])
    .agg(ipTM=("ipTM", "mean"), combined=("combined", "mean"))
    .reset_index()
)

lib_rows = []
for _, row in results.iterrows():
    dbid = row["DBID"].lower()
    ref_pwm = jaspar_id_to_pwm[row["Hom_JASPAR"]]
    g = per_motif[per_motif["protein"] == dbid]
    if g.empty:
        continue
    scores = {}
    for m in g["motif"]:
        sim = pwm_similarity(lib_pwms[m], ref_pwm)[0]
        scores[m] = (sim, float((lib_null[m] >= sim).mean()))
    rec = {"DBID": row["DBID"], "TF_Name": row["TF_Name"], "Family": row["Family"]}
    for key in ["ipTM", "combined"]:
        order = g.sort_values(key, ascending=False)["motif"].tolist()
        rec[f"{key}_motif"] = order[0]
        rec[f"{key}_sim"], rec[f"{key}_p"] = scores[order[0]]
        rec[f"{key}_sig_top3"] = any(scores[m][1] < 0.05 for m in order[:3])
        rec[f"{key}_sig_top5"] = any(scores[m][1] < 0.05 for m in order[:5])
    best = max(scores, key=lambda m: scores[m][0])
    rec["best_ipTM_rank"] = (
        g.sort_values("ipTM", ascending=False)["motif"].tolist().index(best) + 1
    )
    rec["n_lib_sig"] = sum(p < 0.05 for _, p in scores.values())
    lib_rows.append(rec)
lib_df = pd.DataFrame(lib_rows)

# Compare against the deployed NA-MPNN consensus (largest cluster, and best of top-k clusters)
lib_df["nampnn_p"] = lib_df["DBID"].map(
    cluster_df.loc[cluster_df["Merged_Rank"] == 1].set_index("DBID")["P_Value"]
)
lib_df["nampnn_sim"] = lib_df["DBID"].map(
    cluster_df.loc[cluster_df["Merged_Rank"] == 1].set_index("DBID")["HOMER_Sim"]
)
for k in [3, 5]:
    lib_df[f"nampnn_sig_top{k}"] = lib_df["DBID"].map(
        cluster_df[cluster_df["Merged_Rank"] <= k].groupby("DBID")["P_Value"].min()
        < 0.05
    )
lib_df = lib_df[lib_df["nampnn_p"].notna()]

n = len(lib_df)
lib_sig = lib_df["ipTM_p"] < 0.05
nampnn_sig = lib_df["nampnn_p"] < 0.05
print(f"TFs: {n}")
print(
    f"top-1 sig: library ipTM {lib_sig.sum()}, library combined "
    f"{(lib_df['combined_p'] < 0.05).sum()}, NA-MPNN {nampnn_sig.sum()}, "
    f"random ~{lib_df['n_lib_sig'].sum() / len(lib_pwms):.0f}"
)
for k in [3, 5]:
    print(
        f"top-{k} sig: library ipTM {lib_df[f'ipTM_sig_top{k}'].sum()}, library combined "
        f"{lib_df[f'combined_sig_top{k}'].sum()}, NA-MPNN {lib_df[f'nampnn_sig_top{k}'].sum()}"
    )
lib_only, nampnn_only = (lib_sig & ~nampnn_sig).sum(), (~lib_sig & nampnn_sig).sum()
print(
    f"McNemar: library-only {lib_only}, NA-MPNN-only {nampnn_only}, "
    f"p={stats.binomtest(lib_only, lib_only + nampnn_only).pvalue:.2g}"
)
print(
    f"median HOMER sim: library {lib_df['ipTM_sim'].median():.3f}, NA-MPNN "
    f"{lib_df['nampnn_sim'].median():.3f} (Wilcoxon "
    f"p={stats.wilcoxon(lib_df['ipTM_sim'], lib_df['nampnn_sim']).pvalue:.2g})"
)
print(
    f"median ipTM rank of best-matching library motif: "
    f"{lib_df['best_ipTM_rank'].median():.0f} / {len(lib_pwms)}"
)
fam_tab = lib_df.groupby("Family").agg(
    n=("DBID", "size"),
    library=("ipTM_p", lambda x: (x < 0.05).sum()),
    nampnn=("nampnn_p", lambda x: (x < 0.05).sum()),
)
print(fam_tab[fam_tab["n"] >= 5].sort_values("n", ascending=False).to_string())

# %%
