"""
Compare monomer vs heterotetramer AF3 predictions for the tp63/tp73 complex:
  - tp63 Mono / tp73 Mono : hero runs for each TF individually
  - Tetramer : co-folded tp63+tp73 heterotetramer runs (tetramer-tp63-tp73)

The tetramer job (`tetramer-tp63-tp73`) is a HETERO-tetramer: chains A/B are tp63
(ensdarg00000113627), chains C/D are tp73 (ensdarg00000017953), and chains E/F are
the two DNA strands. The DNA is a doubled palindromic half-site (e.g. AACGTT·AACGTT).
We extract the forward-strand PWM exactly as the dimer notebooks do.

Both monomer baselines come from the `hero` job. As in the homodimer notebooks, the
tetramer DNA set does NOT overlap the monomer (hero) DNA sets, so we compare PWM
*distributions* (clusters + JASPAR scoring), not matched per-sequence pairs. The
tetramer PWMs are scored against BOTH JASPAR references (MA0525.2 tp63, MA0861.2 tp73)
since the complex contains both TFs.

Ben Iovino 6/26/26 Biohub
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

# Heterotetramer job + the two TFs it contains.
TET_JOB = "tetramer-tp63-tp73"
TET_SPEC_PROTEIN = "ensdarg00000113627"  # specificities subdir / jobdb protein column

# Each monomer baseline: hero protein id + its JASPAR reference.
MONOMERS = [
    {"key": "tp63", "protein": "ensdarg00000113627", "jaspar": "MA0525.2"},
    {"key": "tp73", "protein": "ensdarg00000017953", "jaspar": "MA0861.2"},
]

COR_THRESHOLD = 0.60  # UPGMA cluster-merge correlation threshold
TOP_N_MONO = (
    24  # keep only the top-N monomer specificity DNAs (ranked by ipTM×dom_pLDDT)
)
TOP_N_LOGOS = 3  # show only the top-N largest-cluster consensus PWMs per condition
OUT_PREFIX = "tetramer_tp63_tp73"


# %%
# Load runs and select the monomer/tetramer specificity DNA sets
conn_hero = sqlite3.connect(str(DATA / "jobs" / "hero" / "jobdb.sqlite"))
conn_tet = sqlite3.connect(str(DATA / "jobs" / TET_JOB / "jobdb.sqlite"))

tet_runs = pd.read_sql(
    "SELECT * FROM runs WHERE protein = ?", conn_tet, params=(TET_SPEC_PROTEIN,)
)
tet_spec_dir = (
    DATA / "jobs" / TET_JOB / "specificities" / TET_SPEC_PROTEIN / "specificity"
)

# Per-monomer: hero runs, the DNAs that have specificity PPMs, and a top-N subset
# ranked by the product of percentile ranks of ipTM and dom_pLDDT.
mono_info = {}
for m in MONOMERS:
    prot = m["protein"]
    hero_runs = pd.read_sql(
        "SELECT * FROM runs WHERE protein = ?", conn_hero, params=(prot,)
    )
    spec_dir = DATA / "jobs" / "hero" / "specificities" / prot / "specificity"
    spec_npz_dnas = {f.stem.split("_")[-2] for f in spec_dir.glob(f"{prot}_*.npz")}

    mono_rank = (
        hero_runs[hero_runs["dna"].isin(spec_npz_dnas)]
        .groupby("dna")[["ipTM", "dom_pLDDT"]]
        .mean()
    )
    mono_rank["score"] = mono_rank["ipTM"].rank(pct=True) * mono_rank["dom_pLDDT"].rank(
        pct=True
    )
    keep_dnas = set(
        mono_rank.sort_values("score", ascending=False).head(TOP_N_MONO).index
    )
    subset = hero_runs[hero_runs["dna"].isin(keep_dnas)].copy()

    mono_info[m["key"]] = {
        **m,
        "spec_dir": spec_dir,
        "keep_dnas": keep_dnas,
        "subset": subset,
    }
    print(
        f"[{m['key']}] {prot}: hero runs={len(hero_runs)}, "
        f"specificity DNAs avail={len(spec_npz_dnas)}, kept top {TOP_N_MONO}, "
        f"subset runs={len(subset)}"
    )

conn_hero.close()
conn_tet.close()
print(f"\nTetramer ({TET_JOB}) runs: {len(tet_runs)}")

# Confidence for monomers vs. tetramer (disjoint DNA sets -> distribution comparison)
cond_runs = {
    "tp63 monomer": mono_info["tp63"]["subset"],
    "tp73 monomer": mono_info["tp73"]["subset"],
    "Tetramer": tet_runs,
}
metrics = ["ipTM", "dom_pLDDT", "dna_pLDDT"]
print("\n=== Confidence Metric Summary (mean / median) ===")
header = f"{'Metric':<14}" + "".join(f"{c:>26}" for c in cond_runs)
print(header)
print("-" * len(header))
for metric in metrics:
    cells = "".join(
        f"{df[metric].mean():>13.3f}{df[metric].median():>13.3f}"
        for df in cond_runs.values()
    )
    print(f"{metric:<14}{cells}")


# %%
# Load B-strand PPMs (one per DNA) for each condition
def _load_condition_ppms(spec_dir: Path, protein: str, keep_dnas: set | None) -> list:
    """Return list of (ppm, dna) for the condition's specificity npz files.

    Naming: {protein}_{motif}_{dna}_{seed}.npz (the motif token itself may contain
    underscores), so DNA is always the second-to-last token. Only the first npz
    encountered per DNA is kept.

    Args:
        spec_dir: directory containing the condition's specificity npz files
        protein: protein id prefix used to glob the npz files
        keep_dnas: set of DNA tokens to keep, or None to keep all

    Returns:
        List of tuples (ppm, dna) for the loaded npz files.
    """
    loaded, seen = [], set()
    for npz_path in spec_dir.glob(f"{protein}_*.npz"):
        dna = npz_path.stem.split("_")[-2]
        if dna in seen or (keep_dnas is not None and dna not in keep_dnas):
            continue
        try:
            loaded.append((_load_b_strand_ppm(npz_path), dna))
            seen.add(dna)
        except Exception as e:
            print(f"  Failed {npz_path.name}: {e}")
    return loaded


tet_loaded = _load_condition_ppms(tet_spec_dir, TET_SPEC_PROTEIN, None)
print(f"\nLoaded tetramer PPMs: {len(tet_loaded)}")
for key, info in mono_info.items():
    info["loaded"] = _load_condition_ppms(
        info["spec_dir"], info["protein"], info["keep_dnas"]
    )
    print(f"Loaded {key} monomer PPMs: {len(info['loaded'])}")


# %%
# Load JASPAR references
jaspar_path = DATA / "tools" / "homer" / "jaspar.homer.motif"
jaspar_names, jaspar_pwms = load_homer_multi_motif_file(str(jaspar_path))
jaspar_id_to_pwm = dict(zip(jaspar_names, jaspar_pwms))
jaspar_pwm_list = list(jaspar_id_to_pwm.values())
print(f"\nLoaded {len(jaspar_id_to_pwm)} JASPAR motifs")
for m in MONOMERS:
    print(f"  {m['key']} ref={m['jaspar']} L={len(jaspar_id_to_pwm[m['jaspar']])}")


# %%
# Run UPGMA + JASPAR scoring for a condition against one reference
def run_pwm_analysis(
    loaded, label, ref_pwm, ref_id
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run UPGMA clustering and JASPAR scoring for a condition's loaded PPMs.

    Args:
        loaded: list of (ppm, dna) tuples for the condition's specificities
        label: string label for the condition (e.g., "Tetramer")
        ref_pwm: reference JASPAR PWM to score against
        ref_id: JASPAR id of the reference (for table columns)

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
                "JASPAR_Ref": ref_id,
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
                "JASPAR_Ref": ref_id,
                "Cluster_Size_Rank": sr,
                "N_Members": len(members),
                "HOMER_Sim": round(sim_c, 4),
                "Z_Score": round(z_c, 3),
                "P_Value": round(p_c, 5),
                "Recovery": round(rec_c, 4),
            }
        )

    return pd.DataFrame(indiv_rows), pd.DataFrame(cluster_rows), consensus_ppms


# Each monomer is scored against its own ref; the tetramer is scored against BOTH.
# Keyed by (label, ref_id).
analyses = {}
for m in MONOMERS:
    label = f"{m['key']} monomer"
    print(f"\nPWM analysis: {label} vs {m['jaspar']}")
    analyses[(label, m["jaspar"])] = run_pwm_analysis(
        mono_info[m["key"]]["loaded"], label, jaspar_id_to_pwm[m["jaspar"]], m["jaspar"]
    )
for m in MONOMERS:
    print(f"\nPWM analysis: Tetramer vs {m['jaspar']} ({m['key']})")
    analyses[("Tetramer", m["jaspar"])] = run_pwm_analysis(
        tet_loaded, "Tetramer", jaspar_id_to_pwm[m["jaspar"]], m["jaspar"]
    )


# %%
# Save CSVs
all_indiv = pd.concat([a[0] for a in analyses.values()], ignore_index=True)
all_clusters = pd.concat([a[1] for a in analyses.values()], ignore_index=True)

print("\n=== Cluster Consensus Scores ===")
print(all_clusters.to_string(index=False))

all_indiv.to_csv(f"{TAB_DIR}/{OUT_PREFIX}_monomer_tetramer_indiv.csv", index=False)
all_clusters.to_csv(
    f"{TAB_DIR}/{OUT_PREFIX}_monomer_tetramer_clusters.csv", index=False
)


# %%
# Consensus logo plots (monomer vs tetramer clusters vs JASPAR reference), one per ref
for m in MONOMERS:
    ref_id = m["jaspar"]
    ref_pwm = jaspar_id_to_pwm[ref_id]

    panels = []
    for cond_label in (f"{m['key']} monomer", "Tetramer"):
        _, _, consensus_ppms = analyses[(cond_label, ref_id)]
        for rank in sorted(consensus_ppms.keys())[:TOP_N_LOGOS]:
            cons, n_mem = consensus_ppms[rank]
            sim, off, rc = pwm_similarity(cons, ref_pwm)
            rec = pred_recovery_fraction(cons, ref_pwm, off, rc)
            all_sims_c = np.array(
                [pwm_similarity(cons, pwm)[0] for pwm in jaspar_pwm_list]
            )
            p_val = float((all_sims_c >= sim).mean())
            label_str = f"{cond_label} cluster {rank} (n={n_mem})"
            metrics_str = f"HOMER={sim:.3f}, p={p_val:.4f}, IC_rec={rec:.3f}"
            panels.append((cons, label_str, metrics_str))

    plot_aligned_logos(
        panels,
        tf_name=m["key"],
        subtitle=f"monomer vs heterotetramer consensus PWMs vs JASPAR {ref_id}",
        ref_panel=(ref_pwm, f"JASPAR {ref_id} (reference)"),
        out_path=f"{FIG_DIR}/{OUT_PREFIX}_pwm_comparison_{m['key']}.svg",
    )

# %%
