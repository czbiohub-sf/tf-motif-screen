"""
Compare the stock individual-protein PWMs to the co-folded heterodimer PWM for
myod1 (ensdarg00000030110) + tcf3a (ensdarg00000005915):
  - myod1 Mono : hero runs for myod1 individually
  - tcf3a Mono : hero runs for tcf3a individually
  - Heterodimer: co-folded myod1+tcf3a runs (dimers-myod1-tcf3a)

Ben Iovino 8/18/26 Biohub
"""

# %%
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

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

FIG_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "tables"
TAB_DIR.mkdir(exist_ok=True)

COND_COLORS = {
    "myod1 Mono": "#4393c3",
    "tcf3a Mono": "#d6604d",
    "Heterodimer": "#4dac26",
}

PROTEIN_A = "ensdarg00000030110"  # myod1
PROTEIN_B = "ensdarg00000005915"  # tcf3a
TF_A = "myod1"
TF_B = "tcf3a"
JASPAR_A = "MA0499.3"  # myod1
JASPAR_B = "MA0522.4"  # tcf3a
NUM_SPECS = 24  # Top-N specificities per condition
COR_THRESHOLD = 0.60  # UPGMA cluster-merge correlation threshold


# %%
# Load runs and select top-N specificities per condition
def _select_top_specs(runs: pd.DataFrame, num_specs: int) -> pd.DataFrame:
    """Add the npz seed and combined rank-percentile score, return the top num_specs runs.

    Args:
        runs: DataFrame of runs for a single condition (protein)
        num_specs: number of top specificities to select

    Returns:
        DataFrame of the top num_specs runs, with added "seed" and "combined" columns.
    """
    runs = runs.copy()
    runs["seed"] = runs["structure_path"].map(
        lambda sp: Path(sp).parent.parent.name.rsplit("_", 1)[-1]
    )
    runs["combined"] = runs["ipTM"].rank(pct=True) * runs["dom_pLDDT"].rank(pct=True)
    return runs.nlargest(num_specs, "combined").reset_index(drop=True)


conn_hero = sqlite3.connect(str(DATA / "jobs" / "hero" / "jobdb.sqlite"))
conn_hetero = sqlite3.connect(
    str(DATA / "jobs" / "dimers-myod1-tcf3a" / "jobdb.sqlite")
)

myod1_runs = pd.read_sql(
    "SELECT * FROM runs WHERE protein = ?", conn_hero, params=(PROTEIN_A,)
)
tcf3a_runs = pd.read_sql(
    "SELECT * FROM runs WHERE protein = ?", conn_hero, params=(PROTEIN_B,)
)
hetero_runs = pd.read_sql(
    "SELECT * FROM runs WHERE protein = ?", conn_hetero, params=(PROTEIN_A,)
)
conn_hero.close()
conn_hetero.close()

myod1_top = _select_top_specs(myod1_runs, NUM_SPECS)
tcf3a_top = _select_top_specs(tcf3a_runs, NUM_SPECS)
hetero_top = _select_top_specs(hetero_runs, NUM_SPECS)

cond_top = {
    "myod1 Mono": myod1_top,
    "tcf3a Mono": tcf3a_top,
    "Heterodimer": hetero_top,
}

print(f"myod1 runs:       {len(myod1_runs):>5} -> top {len(myod1_top)}")
print(f"tcf3a runs:       {len(tcf3a_runs):>5} -> top {len(tcf3a_top)}")
print(f"heterodimer runs: {len(hetero_runs):>5} -> top {len(hetero_top)}")

metrics = ["ipTM", "dom_pLDDT", "dna_pLDDT"]
metric_labels = {"ipTM": "ipTM", "dom_pLDDT": "Domain pLDDT", "dna_pLDDT": "DNA pLDDT"}

print(f"\n=== Confidence of top-{NUM_SPECS} specificities ===")
print(f"\n{'Metric':<15} {'myod1 mean':>11} {'tcf3a mean':>11} {'Hetero mean':>12}")
print("-" * 55)
for m in metrics:
    print(
        f"{m:<15} {myod1_top[m].mean():>11.3f} {tcf3a_top[m].mean():>11.3f} {hetero_top[m].mean():>12.3f}"
    )


# %%
# Load B-strand PPMs for the top-N specificities of each condition
def _load_condition_ppms(top_df: pd.DataFrame, spec_dir: Path, name_fn) -> list:
    """Return list of (ppm, dna) for the selected runs whose npz files exist.

    Args:
        top_df: DataFrame of the top runs for a condition
        spec_dir: directory containing the npz files for the condition
        name_fn: function to generate the npz filename from a row of top_df

    Returns:
        List of tuples (ppm, dna) for the runs whose npz files were successfully loaded
    """
    loaded = []
    for r in top_df.itertuples():
        npz_path = spec_dir / name_fn(r)
        if not npz_path.exists():
            print(f"  Missing npz: {npz_path.name}")
            continue
        try:
            loaded.append((_load_b_strand_ppm(npz_path), r.dna))
        except Exception as e:
            print(f"  Failed {npz_path.name}: {e}")
    return loaded


myod1_spec_dir = DATA / "jobs" / "hero" / "specificities" / PROTEIN_A / "specificity"
tcf3a_spec_dir = DATA / "jobs" / "hero" / "specificities" / PROTEIN_B / "specificity"
hetero_spec_dir = (
    DATA / "jobs" / "dimers-myod1-tcf3a" / "specificities" / PROTEIN_A / "specificity"
)

myod1_loaded = _load_condition_ppms(
    myod1_top,
    myod1_spec_dir,
    lambda r: f"{PROTEIN_A}_{r.motif}_{r.dna}_{r.seed}.npz",
)
tcf3a_loaded = _load_condition_ppms(
    tcf3a_top,
    tcf3a_spec_dir,
    lambda r: f"{PROTEIN_B}_{r.motif}_{r.dna}_{r.seed}.npz",
)
hetero_loaded = _load_condition_ppms(
    hetero_top,
    hetero_spec_dir,
    lambda r: f"{PROTEIN_A}_{PROTEIN_B}_{r.dna}_{r.seed}.npz",
)

cond_loaded = {
    "myod1 Mono": myod1_loaded,
    "tcf3a Mono": tcf3a_loaded,
    "Heterodimer": hetero_loaded,
}

print(f"\nLoaded myod1 PPMs:       {len(myod1_loaded)}")
print(f"Loaded tcf3a PPMs:       {len(tcf3a_loaded)}")
print(f"Loaded heterodimer PPMs: {len(hetero_loaded)}")


# %%
# Load JASPAR references
jaspar_path = DATA / "tools" / "homer" / "jaspar.homer.motif"
jaspar_names, jaspar_pwms = load_homer_multi_motif_file(str(jaspar_path))
jaspar_id_to_pwm = dict(zip(jaspar_names, jaspar_pwms))
jaspar_pwm_list = list(jaspar_id_to_pwm.values())

ref_pwm_a = jaspar_id_to_pwm[JASPAR_A]  # myod1
ref_pwm_b = jaspar_id_to_pwm[JASPAR_B]  # tcf3a


# %%
# Run UPGMA + JASPAR scoring for all conditions vs both refs
def run_pwm_analysis(
    loaded, label, ref_pwm, jaspar_id
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run UPGMA clustering and JASPAR scoring for a condition's loaded PPMs.

    Args:
        loaded: list of (ppm, dna) tuples for the condition's top specificities
        label: string label for the condition (e.g., "myod1 Mono")
        ref_pwm: JASPAR reference PWM to score against
        jaspar_id: JASPAR ID string for the reference (e.g., "MA0499.3")

    Returns:
        indiv_df: DataFrame of individual PPM scores and metrics
        cluster_df: DataFrame of cluster consensus scores and metrics
        consensus_ppms: dict mapping cluster size rank to (consensus PPM, number of members)
    """
    scored_with_null = []
    for ppm, dna in loaded:
        all_sims = np.array([pwm_similarity(ppm, pwm)[0] for pwm in jaspar_pwm_list])
        scored_with_null.append((ppm, dna, all_sims.mean(), all_sims.std(), all_sims))

    merged = cluster_upgma(scored_with_null, COR_THRESHOLD)

    indiv_rows, clusters = [], {}
    for orig_rank, merged_rank, ppm, dna, align_off, align_rc, is_med in merged:
        sim, off, rc = pwm_similarity(ppm, ref_pwm)
        _, _, null_mean, null_std, null_sims = scored_with_null[orig_rank - 1]
        z = (sim - null_mean) / null_std if null_std > 0 else 0.0
        p = float((null_sims >= sim).mean())
        rec = pred_recovery_fraction(ppm, ref_pwm, off, rc)
        indiv_rows.append(
            {
                "Condition": label,
                "JASPAR_Ref": jaspar_id,
                "Rank": orig_rank,
                "DNA": dna,
                "HOMER_Sim": round(sim, 4),
                "Z_Score": round(z, 3),
                "P_Value": round(p, 5),
                "Recovery": round(rec, 4),
                "Cluster": merged_rank,
                "Is_Medoid": is_med,
            }
        )
        clusters.setdefault(merged_rank, []).append(
            (ppm, align_off, align_rc, is_med, dna)
        )

    size_rank_map = {
        old: new
        for new, (old, _) in enumerate(
            sorted(clusters.items(), key=lambda x: -len(x[1])), start=1
        )
    }

    cluster_rows, consensus_ppms = [], {}
    for merged_rank, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        sr = size_rank_map[merged_rank]
        members.sort(key=lambda x: not x[3])
        consensus = cluster_consensus_pwm(
            [(ppm, off, rc) for ppm, off, rc, _, _ in members]
        )
        consensus_ppms[sr] = (consensus, len(members))

        all_sims_c = np.array(
            [pwm_similarity(consensus, pwm)[0] for pwm in jaspar_pwm_list]
        )
        sim_c, off_c, rc_c = pwm_similarity(consensus, ref_pwm)
        null_mean_c = all_sims_c.mean()
        null_std_c = all_sims_c.std()
        p_c = float((all_sims_c >= sim_c).mean())
        z_c = (sim_c - null_mean_c) / null_std_c if null_std_c > 0 else 0.0
        rec_c = pred_recovery_fraction(consensus, ref_pwm, off_c, rc_c)
        cluster_rows.append(
            {
                "Condition": label,
                "JASPAR_Ref": jaspar_id,
                "Cluster_Size_Rank": sr,
                "N_Members": len(members),
                "HOMER_Sim": round(sim_c, 4),
                "Z_Score": round(z_c, 3),
                "P_Value": round(p_c, 5),
                "Recovery": round(rec_c, 4),
            }
        )

    return pd.DataFrame(indiv_rows), pd.DataFrame(cluster_rows), consensus_ppms


# Score each condition against both JASPAR references
analyses = {}  # (label, jaspar_id) -> (indiv_df, cluster_df, consensus_ppms)

for cond_label, loaded in cond_loaded.items():
    for jaspar_id, ref_pwm in [(JASPAR_A, ref_pwm_a), (JASPAR_B, ref_pwm_b)]:
        print(f"\nPWM analysis: {cond_label} vs {jaspar_id}")
        ind, clust, cons = run_pwm_analysis(loaded, cond_label, ref_pwm, jaspar_id)
        analyses[(cond_label, jaspar_id)] = (ind, clust, cons)


# %%
# Save CSVs
all_indiv = pd.concat([v[0] for v in analyses.values()], ignore_index=True)
all_clusters = pd.concat([v[1] for v in analyses.values()], ignore_index=True)

all_indiv.to_csv(f"{TAB_DIR}/myod1_tcf3a_hetero_indiv.csv", index=False)
all_clusters.to_csv(f"{TAB_DIR}/myod1_tcf3a_hetero_clusters.csv", index=False)


# %%
# Consensus logo plots (one figure per JASPAR reference)
TOP_N_LOGOS = 3  # show only the top-N largest-cluster consensus PWMs per condition
for jaspar_id, ref_pwm, tf_name in [
    (JASPAR_A, ref_pwm_a, TF_A),
    (JASPAR_B, ref_pwm_b, TF_B),
]:
    panels = []
    for cond_label in ["myod1 Mono", "tcf3a Mono", "Heterodimer"]:
        _, _, consensus_ppms = analyses[(cond_label, jaspar_id)]
        for rank in sorted(consensus_ppms.keys())[:TOP_N_LOGOS]:
            cons, n_mem = consensus_ppms[rank]
            sim, off, rc = pwm_similarity(cons, ref_pwm)
            rec = pred_recovery_fraction(cons, ref_pwm, off, rc)
            all_sims_c = np.array(
                [pwm_similarity(cons, pwm)[0] for pwm in jaspar_pwm_list]
            )
            p_val = float((all_sims_c >= sim).mean())
            label_str = f"{cond_label} cluster {rank} (n={n_mem})"
            metrics = f"HOMER={sim:.3f}, p={p_val:.4f}, IC_rec={rec:.3f}"
            panels.append((cons, label_str, metrics))

    out_name = f"{FIG_DIR}/myod1_tcf3a_{jaspar_id.replace('.', '_')}.svg"
    plot_aligned_logos(
        panels,
        tf_name=f"{TF_A}/{TF_B}",
        subtitle=f"consensus PWMs vs JASPAR {jaspar_id} ({tf_name})",
        ref_panel=(ref_pwm, f"JASPAR {jaspar_id} (reference)"),
        out_path=out_name,
    )

# %%
