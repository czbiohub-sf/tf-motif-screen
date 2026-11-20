"""
Build and package the complete ZebraMotif data release.

By default, build the scientific tables from existing research predictions and
write a release under data/database/export/<version>/. Use --from-existing to package
the current database without rebuilding motifs, or --check to verify a release.
The release contains raw/ with tables, structures, MEME and metadata TSV files,
release documents and notices; a complete dataset ZIP; primary/all motif ZIPs;
and checksums of both extracted files and archives.

Produces working scientific files under data/database/export/database/:
    pwm_modes.parquet  -- one row per binding mode (primary + alternates) for every
                          representative TF that has a usable specificity prediction.
    tf_to_rep.parquet  -- one row per queryable zebrafish DBID (full set + reps),
                          resolving each to the representative whose PWM the app shows.
    structures/        -- gzipped AF3 .cif of the most-confident structure (ipTM × DBD-pLDDT
                          rank product) of each TF's primary mode, referenced from
                          pwm_modes.structure_path.
    interface_residues.parquet -- one row per DNA-contacting (interface) residue for every
                          representative TF, keyed by DBID. residue_num matches the author
                          numbering in the referenced structures/ .cif, so the app can
                          select and colour each residue in the 3D viewer.
    dbd_domains.parquet -- all annotated domains and DNA-placement flags for the
                          exported TFs; filter DNA-Binding for structure highlighting.
    confidence_metrics.parquet -- protein-level confidence distributions for all
                          predictions and the same top 47 used for motif construction.
    zebramotif_primary_modes.meme -- primary probability matrices, labeled with
                          all query symbols resolving to each representative.
    zebramotif_all_modes.meme -- primary and alternate probability matrices with
                          the same labels, derived from the final query lookup.
    SCHEMA.md          -- documentation of the tables for the app repo.


Ben Iovino 6/17/26 Biohub
"""

# %%
import argparse
import gzip
import shutil
import sqlite3
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from af3tfm.dataset import DATA_DIR
from export_tools.confidence_export import export_confidence_metrics
from export_tools.database_export import (
    DOMAIN_COLUMNS,
    ZFIN_REFERENCE,
    add_lookup_annotations,
    add_sequence_types,
    consolidate_database_annotations,
    load_zfin_ids,
    reconcile_representatives,
    refresh_database_metadata,
    select_domain_annotations,
    sequence_type,
    write_domain_annotations,
    write_meme_exports,
)
from export_tools.metric_bands import add_metric_bands
from export_tools.release_export import export_release, load_config

EXPORT_ROOT = Path(__file__).resolve().parent
ROOT = EXPORT_ROOT.parents[1]
DATA = Path(DATA_DIR)
DEFAULT_DATABASE = EXPORT_ROOT / "export/database"
DOMAIN_COLS = DOMAIN_COLUMNS

# Operating point from the consensus sweep (see module docstring)
NUM_SPECS = 47
COR = 0.73
N_ALTERNATES = 3
REFINED_COR = 0.72

# The most-confident structure of each TF's primary mode
STRUCT_SUBDIR = "structures"

# Per-(TF, residue) interface table produced by scripts/get_contacts.py
INTERFACE_SRC = "interface_residues.parquet"
INTERFACE_COLS = [
    "DBID",
    "residue_num",
    "resname",
    "contact_freq",
    "frac_base_contact",
    "contact_mode",
    "med_min_dist_base",
    "med_plddt",
]


def build_domain_annotations(src_path: Path, keep_dbids: set) -> pd.DataFrame:
    """Keep all domain annotations so the app can explain an absent DBD highlight.

    Missing rows for a modeled TF mean no domains were annotated. Present rows
    with no DNA-Binding flag mean no domains were confidently placed with DNA.
    Require the source and complete boolean flags rather than treating missing
    annotation data as a negative result.
    """
    return select_domain_annotations(pd.read_csv(src_path), keep_dbids)


def export_domain_annotations(src_path: Path, out_dir: Path, keep_dbids: set):
    """Export full annotations and DNA-placement flags in one Parquet table."""
    domains = build_domain_annotations(src_path, keep_dbids)
    dbd = domains.loc[domains["DNA-Binding"]]
    write_domain_annotations(domains, out_dir)
    print(
        f"domain_annotations: {len(domains)} domains | "
        f"{len(keep_dbids - set(domains['DBID']))} TFs with no domains annotated | "
        f"{len(set(domains['DBID']) - set(dbd['DBID']))} TFs with no domains "
        "confidently placed with DNA"
    )


# Member to representative mapping
def build_tf_to_rep(
    rep_df: pd.DataFrame, full_df: pd.DataFrame, modeled_dbids: set | None = None
) -> pd.DataFrame:
    """
    Resolve every queryable zebrafish DBID to the representative whose PWM the app shows.

    Args:
        rep_df: representative TF table (tfdb4_rep_tfs.csv)
        full_df: full zebrafish TF table (tfdb4_tfs.csv)
        modeled_dbids: when supplied, resolve only to representatives with exported
            primary modes; retain all input DBIDs as queryable entries.

    Returns:
        DataFrame with columns DBID, TF_Name, Family, sequence_type, Rep_DBID,
        is_representative, resolution, zfin_id, in_original_catalog.
    """

    rep_df = add_sequence_types(rep_df)
    full_df = add_sequence_types(full_df)
    res_dir = DATA / "tools" / "mmseqs" / "animaltfdb4" / "results"
    reps = set(rep_df["DBID"]) if modeled_dbids is None else set(modeled_dbids)

    # Prefer recognition-code clustering where it exists for a family stem.
    recog_stems = {
        f.name.split(".recog")[0] for f in res_dir.glob("*.recog_cluster.tsv")
    }
    member_to_cluster_rep = {}
    for tsv in sorted(res_dir.glob("*_cluster.tsv")):
        if ".recog_cluster" in tsv.name:
            pass
        elif ".mmseqs_cluster" in tsv.name:
            if tsv.name.split(".mmseqs")[0] in recog_stems:
                continue  # superseded by recog clustering
        else:
            continue
        for line in tsv.read_text().splitlines():
            rep, member = line.split("\t")
            member_to_cluster_rep[member] = rep

    # TF_Name lookups (full set + reps), and gene symbol -> surviving rep DBID.
    dbid_to_name = dict(zip(full_df["DBID"], full_df["TF_Name"]))
    dbid_to_name.update(dict(zip(rep_df["DBID"], rep_df["TF_Name"])))
    name_to_rep = (
        rep_df.loc[rep_df["DBID"].isin(reps)]
        .groupby("TF_Name")["DBID"]
        .first()
        .to_dict()
    )

    def resolve(dbid):
        if dbid in reps:
            return dbid, "self"
        cluster_rep = member_to_cluster_rep.get(dbid)
        if cluster_rep in reps:
            return cluster_rep, "cluster"
        if cluster_rep is not None:
            rep_name = dbid_to_name.get(cluster_rep)
            if rep_name in name_to_rep:
                return name_to_rep[rep_name], "symbol_redirect"
        own_name = dbid_to_name.get(dbid)
        if own_name in name_to_rep:
            return name_to_rep[own_name], "symbol_redirect"
        return None, "unresolved"

    # Union of full set and reps (reps may carry suffixed DBIDs absent from the full set).
    family_lookup = dict(zip(full_df["DBID"], full_df["Family"]))
    family_lookup.update(dict(zip(rep_df["DBID"], rep_df["Family"])))
    all_dbids = list(dict.fromkeys(list(full_df["DBID"]) + list(rep_df["DBID"])))

    rows = []
    for dbid in all_dbids:
        rep_dbid, resolution = resolve(dbid)
        rows.append(
            {
                "DBID": dbid,
                "TF_Name": dbid_to_name.get(dbid),
                "Family": family_lookup.get(dbid),
                "sequence_type": sequence_type(dbid),
                "Rep_DBID": rep_dbid,
                "is_representative": dbid in reps,
                "resolution": resolution,
            }
        )
    zfin_ids = load_zfin_ids(DATA / ZFIN_REFERENCE)
    return add_lookup_annotations(pd.DataFrame(rows), full_df, zfin_ids)


# Interface residue export
def build_interface_residues(src_path: Path, keep_dbids: set) -> pd.DataFrame:
    """
    Select the interface residues to expose for the app's structure viewer.

    Args:
        src_path: path to interface_residues.parquet from the shard pipeline
        keep_dbids: DBIDs present in pwm_modes (upper-case), the only ones with a structure

    Returns:
        DataFrame with columns INTERFACE_COLS (empty with those columns if src is missing).
    """

    if not src_path.exists():
        print(f"WARNING: {src_path} not found; skipping interface_residues export")
        return pd.DataFrame(columns=INTERFACE_COLS)

    df = pd.read_parquet(src_path)
    df = df[df["interface"]].copy()
    df["DBID"] = df["dbid"].str.upper()
    df = df[df["DBID"].isin(keep_dbids)]
    for col in ("contact_freq", "frac_base_contact", "med_min_dist_base", "med_plddt"):
        df[col] = df[col].round(4)
    return df[INTERFACE_COLS].reset_index(drop=True)


# Per-TF binding-mode extraction
def _cluster_consensuses(loaded: list, cor_threshold: float) -> list:
    """
    Cluster a set of PPMs and return each cluster's consensus, ordered by member count.

    Args:
        loaded: list of (ppm, dna) for the structures to cluster, most-confident first
        cor_threshold: UPGMA merge correlation cutoff

    Returns:
        list of (n_members, consensus_ppm, top_member_idx), largest cluster first
        (equal-size ties broken by rank product of confidence). top_member_idx is the
        0-based position into `loaded` of the cluster's most-confident member (the one
        with the smallest input rank), letting the caller map back to its source structure.
    """

    from af3tfm.consensus import cluster_consensus_pwm, cluster_upgma

    # cluster_upgma only reads item[0] (the ppm); null fields are unused here.
    scored_with_null = [(ppm, dna, 0.0, 0.0, None) for ppm, dna in loaded]
    merged = cluster_upgma(scored_with_null, cor_threshold)

    clusters = {}
    for orig_rank, merged_rank, ppm_b, dna, align_off, align_rc, is_med in merged:
        clusters.setdefault(merged_rank, []).append(
            (ppm_b, align_off, align_rc, is_med, orig_rank)
        )

    scored = []
    for members in clusters.values():
        rank_product = float(np.prod([m[4] for m in members]))
        # orig_rank is the 1-based input position; the smallest is the most confident.
        top_member_idx = min(m[4] for m in members) - 1
        members = sorted(members, key=lambda x: not x[3])  # medoid first
        consensus = cluster_consensus_pwm(
            [(ppm, off, rc) for ppm, off, rc, _, _ in members]
        )
        scored.append((len(members), rank_product, consensus, top_member_idx))

    # Largest clusters first; equal-size ties broken by smaller (more confident)
    # rank product of member confidence ranks.
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(n, consensus, top_idx) for n, _, consensus, top_idx in scored]


def _score_vs_jaspar(
    consensus: np.ndarray, ref_pwm: np.ndarray, jaspar_pwm_list: list
) -> dict:
    """
    HOMER similarity, empirical p-value and IC recovery of a consensus vs its cognate JASPAR.

    Args:
        consensus: (L, 4) consensus PPM
        ref_pwm: (L_ref, 4) cognate JASPAR PPM
        jaspar_pwm_list: all JASPAR PPMs, forming the empirical null

    Returns:
        dict with homer_sim, z_score, p_value, recovery.
    """

    from af3tfm.consensus import pred_recovery_fraction
    from af3tfm.utils import pwm_similarity

    all_sims = np.array([pwm_similarity(consensus, p)[0] for p in jaspar_pwm_list])
    sim, off, rc = pwm_similarity(consensus, ref_pwm)
    null_mean, null_std = all_sims.mean(), all_sims.std()
    return {
        "homer_sim": round(float(sim), 4),
        "z_score": round(float((sim - null_mean) / null_std), 3)
        if null_std > 0
        else 0.0,
        "p_value": round(float((all_sims >= sim).mean()), 5),
        "recovery": round(
            float(pred_recovery_fraction(consensus, ref_pwm, off, rc)), 4
        ),
    }


def process_one_tf(
    row_dict: dict,
    jaspar_id,
    ref_pwm,
    hom_sim,
    spec_dir_str: str,
    job_dir_str: str,
    jaspar_pwm_list: list,
    struct_out_dir_str: str,
) -> list:
    """
    Build the binding-mode rows for a single representative TF.

    Args:
        row_dict: a tfdb4_rep_tfs.csv row as a dict (DBID, TF_Name, Family, ...)
        jaspar_id: cognate JASPAR id, or None if the TF has no usable homolog
        ref_pwm: cognate JASPAR PPM, or None
        hom_sim: Hom_Sim homology percentage for the TF (may be NaN)
        spec_dir_str: path to the specificities directory
        job_dir_str: path to the job directory holding jobdb.sqlite
        jaspar_pwm_list: all JASPAR PPMs (empirical null)
        struct_out_dir_str: directory to copy the primary mode's top structure into

    Returns:
        list of per-mode row dicts (empty if the TF has no usable specificities/clusters).
    """

    from af3tfm.consensus import _load_b_strand_ppm
    from af3tfm.utils import pwm_similarity

    dbid = row_dict["DBID"].lower()
    spec_dir = Path(spec_dir_str)
    job_dir = Path(job_dir_str)

    indiv_dir = spec_dir / dbid / "specificity"
    if not indiv_dir.exists():
        return []

    jobdb = sqlite3.connect(str(job_dir / "jobdb.sqlite"))
    all_preds = pd.read_sql(
        "SELECT motif, dna, ipTM, dom_pLDDT, dna_pLDDT, structure_path FROM runs WHERE protein = ?",
        jobdb,
        params=(dbid,),
    )
    jobdb.close()
    if all_preds.empty:
        return []

    # Seed is the trailing integer of the fold output dir (see run_spec.get_all_structs).
    all_preds["seed"] = all_preds["structure_path"].map(
        lambda sp: Path(sp).parent.parent.name.rsplit("_", 1)[-1]
    )
    all_preds["combined"] = all_preds["ipTM"].rank(pct=True) * all_preds[
        "dom_pLDDT"
    ].rank(pct=True)

    # Load the top NUM_SPECS once, carrying the combined-rank position so the primary
    # mode's top member can be mapped back to its source .cif via structure_path.
    top_structs = [
        (r.motif, r.dna, r.seed, r.structure_path)
        for r in all_preds.nlargest(NUM_SPECS, "combined").itertuples()
    ]
    loaded = []  # (ppm, dna, rank_pos)
    for rank_pos, (motif, dna, seed, _sp) in enumerate(top_structs):
        npz_path = indiv_dir / f"{dbid}_{motif}_{dna}_{seed}.npz"
        if not npz_path.exists():
            continue
        try:
            loaded.append((_load_b_strand_ppm(npz_path), dna, rank_pos))
        except Exception:
            continue
    if not loaded:
        return []

    cluster_in = [(ppm, dna) for ppm, dna, rp in loaded]
    # rank_pos parallel to cluster_in, to map a cluster member back to top_structs.
    cluster_meta = [rp for ppm, dna, rp in loaded]
    clusters = _cluster_consensuses(cluster_in, COR)
    if not clusters:
        return []  # no PWM -> nothing to export for this TF

    has_jaspar = jaspar_id is not None and ref_pwm is not None
    hom_sim_val = float(hom_sim) if pd.notna(hom_sim) else None

    # ZM<digits of the Ensembl DBID>.<n>: .1 for the primary, .2-.4 for the alternates.
    zm_base = "ZM" + "".join(c for c in row_dict["DBID"] if c.isdigit())

    def base_row(tier, mode_rank, n_members, consensus):
        row = {
            "motif_id": f"{zm_base}.{1 if tier == 'primary' else mode_rank + 1}",
            "DBID": row_dict["DBID"],
            "TF_Name": row_dict["TF_Name"],
            "Family": row_dict["Family"],
            "sequence_type": sequence_type(row_dict["DBID"]),
            "tier": tier,
            "mode_rank": mode_rank,
            "n_members": n_members,
            "num_specs": NUM_SPECS,
            "cor_threshold": COR,
            "pwm": consensus.astype(float).tolist(),
            "pwm_length": int(consensus.shape[0]),
            "structure_path": None,
            "corr_to_primary": None,
            "offset_to_primary": None,
            "rc_to_primary": None,
            "relationship": None,
            "hom_jaspar": jaspar_id,
            "hom_sim": hom_sim_val,
            "homer_sim": None,
            "z_score": None,
            "p_value": None,
            "recovery": None,
        }
        if has_jaspar:
            row.update(_score_vs_jaspar(consensus, ref_pwm, jaspar_pwm_list))
        return row

    rows = []

    # Primary: the single largest cluster.
    n_members_p, primary_ppm, top_idx_p = clusters[0]
    primary_row = base_row("primary", 1, n_members_p, primary_ppm)
    primary_row["relationship"] = "primary"

    # Copy the primary mode's most-confident structure (gzipped) and reference it.
    src_cif = Path(top_structs[cluster_meta[top_idx_p]][3])
    if src_cif.exists():
        rel_path = f"{STRUCT_SUBDIR}/{primary_row['motif_id']}.cif.gz"
        out_cif = Path(struct_out_dir_str) / f"{primary_row['motif_id']}.cif.gz"
        out_cif.write_bytes(gzip.compress(src_cif.read_bytes(), 9, mtime=0))
        primary_row["structure_path"] = rel_path

    rows.append(primary_row)

    # Alternates: the next N clusters by member count, reconciled to the primary.
    for mode_rank, (n_members_a, alt_ppm, _) in enumerate(
        clusters[1 : 1 + N_ALTERNATES], start=1
    ):
        corr, off, rc = pwm_similarity(primary_ppm, alt_ppm)
        row = base_row("alternate", mode_rank, n_members_a, alt_ppm)
        row["corr_to_primary"] = round(float(corr), 4)
        row["offset_to_primary"] = int(off)
        row["rc_to_primary"] = bool(rc)
        row["relationship"] = "primary_refined" if corr >= REFINED_COR else "distinct"
        rows.append(row)

    return rows


SCHEMA_DOC = """# PPM database schema

The dataset contains predicted DNA-binding probability matrices (PPMs), protein
structures, and annotations for zebrafish transcription factors. The `pwm` fields
contain (L, 4) probability matrices with columns A,C,G,T. Each row is one DNA
position; the values are base probabilities rather than log-odds weights.

Release-specific counts and operating settings are recorded in `README.md`,
`release_metadata.json`, and `PROVENANCE.json`. The table values are authoritative:
`num_specs = 47`, `cor_threshold = 0.73`, and the `primary_refined` similarity
cutoff is 0.72. Paths below are relative to the extracted dataset directory.

## Reading the data

```python
import numpy as np
import pandas as pd

modes = pd.read_parquet("pwm_modes.parquet", engine="pyarrow")
tf2rep = pd.read_parquet("tf_to_rep.parquet", engine="pyarrow").set_index("DBID")

def lookup(dbid):
    rep = tf2rep.loc[dbid, "Rep_DBID"]          # resolve to the rep that has a PPM
    if pd.isna(rep):
        return None                              # unresolved query
    tf_modes = modes[modes["DBID"] == rep].sort_values("motif_id")
    return tf_modes

def probability_matrix(mode):
    # Stack Arrow's nested arrays into a numeric (L, 4) matrix.
    return np.stack(mode["pwm"]).astype(float)
```

## pwm_modes.parquet — one row per binding mode

| column | type | notes |
|---|---|---|
| `motif_id` | str | stable motif id `ZM<digits of DBID>.<n>`: `.1` primary, `.2`-`.4` alternates |
| `DBID` | str | representative Ensembl id |
| `TF_Name` | str | gene symbol |
| `Family` | str | TF family (AnimalTFDB4) |
| `sequence_type` | str | `full_length` or `domain`, describing the protein sequence modeled for this representative |
| `tier` | str | `primary` or `alternate` |
| `mode_rank` | int | primary = 1; alternates 1..3 by member count (ties broken by rank product of confidence) |
| `n_members` | int | cluster support |
| `num_specs` | int | target number of specificities clustered (47) |
| `cor_threshold` | float | UPGMA merge cutoff (0.73) |
| `pwm` | list[list[float]] | (L, 4) probability matrix, columns A,C,G,T |
| `pwm_length` | int | L |
| `structure_path` | str | path relative to the extracted dataset directory to the gzipped AF3 `.cif` of the primary mode's most-confident structure (ipTM × DBD-pLDDT rank product); set on the primary row only, null otherwise / when the source `.cif` is missing |
| `corr_to_primary` | float | alternate vs primary similarity (null for primary) |
| `offset_to_primary` | int | alignment offset of alternate vs primary (null for primary) |
| `rc_to_primary` | bool | alternate reverse-complemented vs primary (null for primary) |
| `relationship` | str | `primary`, `primary_refined` (corr >= 0.72), or `distinct` |
| `hom_jaspar` | str | cognate JASPAR id (null if no homolog) |
| `hom_sim` | float | protein homology % to the JASPAR ortholog (null if none) |
| `homer_sim` | float | consensus-vs-cognate HOMER similarity (null if no homolog) |
| `z_score` | float | z score against the JASPAR motif null distribution (null if no homolog) |
| `p_value` | float | empirical p vs the JASPAR null (null if no homolog) |
| `recovery` | float | JASPAR-IC-weighted recovery fraction (null if no homolog) |
| `support_band` | str | `Low`, `Medium`, `High`, or `Unavailable`; based on `n_members / num_specs` |
| `homer_sim_band` | str | `Low`, `Medium`, `High`, or `No reference`; based on `homer_sim` |
| `recovery_band` | str | `Low`, `Medium`, `High`, or `No reference`; based on `recovery` |

### Score bands

The same descriptive bands apply to primary and alternate modes. They summarize
cluster support and agreement with reference motifs. They are not calibrated
probabilities of biological correctness.

| metric | Low | Medium | High |
|---|---|---|---|
| Support (`n_members / num_specs`) | < 0.10 | >= 0.10 and < 0.30 | >= 0.30 |
| Motif similarity (`homer_sim`) | < 0.60 | >= 0.60 and < 0.80 | >= 0.80 |
| IC recovery (`recovery`) | < 0.40 | >= 0.40 and < 0.70 | >= 0.70 |

Missing reference metrics have the neutral label `No reference`; zero scores are
valid `Low` values. Missing, nonfinite, or invalid support counts (including a zero
denominator) are `Unavailable`. With `num_specs = 47`, support bands correspond to
1-4, 5-14, and 15-47 cluster members.

## tf_to_rep.parquet — one row per queryable zebrafish DBID

| column | type | notes |
|---|---|---|
| `DBID` | str | any zebrafish TF (representative or not) |
| `TF_Name` | str | gene symbol |
| `Family` | str | TF family |
| `sequence_type` | str | `full_length` for gene entries or `domain` for separately modeled domains (DBIDs ending in `_1`, `_2`, etc.) |
| `Rep_DBID` | str | modeled representative assigned to this query (null if unresolved) |
| `is_representative` | bool | whether this DBID has its own computed PWM |
| `resolution` | str | `self`, `cluster`, `symbol_redirect`, or `unresolved` |
| `zfin_id` | str or null | ZFIN gene ID for this exact query DBID; missing or domain-specific mappings are null |
| `in_original_catalog` | bool | whether this exact DBID occurs in the original AnimalTFDB4 gene catalog; domain entries are false |

Every query with an exported primary mode resolves to its own DBID, with
`is_representative = True` and `resolution = "self"`. This takes precedence over
cluster and symbol redirects. `cluster` assigns a sequence-cluster representative;
`symbol_redirect` resolves through a shared gene symbol. `unresolved` indicates
that no representative motif is available.

Every gene ID in the original AnimalTFDB4 catalog resolves to an exported
representative motif in this release. Gene entries have
`sequence_type = "full_length"`; separately modeled domains have `"domain"`
and `in_original_catalog = False`. Domain DBIDs append `_1`, `_2`, or `_3` to the
parent gene ID. ZFIN IDs refer to the exact query DBID; missing mappings and
separately modeled domains have null values.

## Release metadata and collection downloads

`release_metadata.json` records versions, counts, methods, citation, and data-use
terms. Unrecorded reference versions and dates are null. `PROVENANCE.json` records
source identities and coverage, including domain entries and missing annotations.

`zebramotif_primary_modes.meme` contains primary modes, and
`zebramotif_all_modes.meme` contains primary and alternate modes. Each matrix
matches its `pwm_modes.parquet` row, with `motif_id` as its identifier. The
matching `.metadata.tsv` files contain query-to-mode mappings, sequence types,
and raw metrics. Matrices use A,C,G,T columns, six decimal places, and uniform
background frequencies. `nsites` is cluster support (`n_members`), not a count of
experimentally measured binding sites. The MEME alternate name lists the sorted,
deduplicated query gene symbols assigned to the representative, separated by
commas; a missing symbol group falls back to the mode's `TF_Name`.

Collection ZIPs contain one MEME file per mode, metadata, citation, terms, and
checksums. The full dataset ZIP also contains Parquet tables and structures.

## dbd_domains.parquet — all annotated domains for modeled TFs

All annotated domains for the modeled representatives, with AF3-derived
DNA-placement flags. Join through `Rep_DBID` to obtain domains for the modeled
protein assigned to a query. Coordinates refer to that protein's sequence.

| column | type | notes |
|---|---|---|
| `DBID` | str | modeled representative Ensembl id |
| `TF_Name` | str | gene symbol |
| `accession` | str | domain accession from the collapsed InterPro annotations |
| `description` | str | domain description |
| `start`, `end` | int | inclusive, 1-based protein sequence coordinates |
| `DNA-Binding` | bool | domain selected by the AF3 protein-DNA placement analysis |

No rows for a representative means no domain annotations are available. Rows
with no true `DNA-Binding` flags mean domains are annotated but none were
confidently placed with DNA. The flags describe placement evidence, separately
from local fold confidence. False flags are retained to distinguish these cases.

## interface_residues.parquet — one row per DNA-contacting residue

DNA-contact residues pooled over each TF's AF3 structures, filtered to reliable,
confident contacts. `residue_num` is the author residue number in the corresponding
mmCIF structure. These annotations summarize the protein across binding modes;
join on the representative `DBID`, rather than `motif_id`.

| column | type | notes |
|---|---|---|
| `DBID` | str | representative Ensembl id (joins `pwm_modes` / `tf_to_rep`) |
| `residue_num` | int | author residue number in the `structures/` `.cif` |
| `resname` | str | 3-letter amino acid |
| `contact_freq` | float | fraction of the TF's structures in which the residue contacts DNA (0–1) |
| `frac_base_contact` | float | fraction of its contacts that are base- (vs backbone-) directed |
| `contact_mode` | str | contact classification: `base`, `backbone`, or `mixed` |
| `med_min_dist_base` | float | median distance (Å) to the nearest nucleobase |
| `med_plddt` | float | median side-chain pLDDT (placement confidence) |

## structures/ — most-confident structure per primary mode

One gzipped AF3 mmCIF per representative, named `<motif_id>.cif.gz`,
holding the primary mode's most-confident structure (the primary cluster member with the
highest ipTM × DBD-pLDDT rank product). pLDDT is in the CIF B-factor column.
Referenced by `structure_path`; decompress relative to the extracted dataset
directory:

```python
import gzip
from pathlib import Path

DATA_DIR = Path(".")  # directory containing pwm_modes.parquet and structures/

def structure_text(rel_path):  # rel_path = modes row's structure_path
    with gzip.open(DATA_DIR / rel_path, "rt") as handle:
        return handle.read()
```

## confidence_metrics.parquet — per-protein distribution of AF3 confidence metrics

One row per (`DBID`, `subset`, `metric`) summarizes the distribution of a
confidence metric over a protein's AF3 predictions. The `all` subset describes
the full screen; `top47` uses the predictions selected for motif construction.

| column | type | notes |
|---|---|---|
| `DBID` | str | protein Ensembl id (joins `pwm_modes` / `tf_to_rep`) |
| `subset` | str | `all` = every prediction; `top47` = up to 47 predictions selected by ipTM × DBD-pLDDT percentile-rank product (fewer when fewer predictions are available) |
| `metric` | str | one of ipTM, iface_PAE, iface_PAE_contact, iface_contact_prob, n_contacted_bases, dna_pLDDT, dom_pLDDT, prot_pTM, fraction_disordered, ptm, ranking_score, has_clash |
| `category` | str | `tf` (fold), `interface` (protein-DNA), or `global` |
| `higher_is_better` | bool | whether higher values indicate greater confidence (False for PAE / disorder / clash) |
| `n` | int | non-null predictions in the subset |
| `mean`, `std` | float | distribution mean and standard deviation |
| `p0`..`p100` | float | percentiles 0,5,10,25,50,75,90,95,100 |

Some modeled representatives have no confidence rows; their exact DBIDs and
names are listed in `PROVENANCE.json` under
`coverage.proteins_without_confidence`. Missing rows mean unavailable confidence.

These summaries pool predictions across binding modes. Individual cluster
members and per-mode confidence estimates are not provided in this table.
Prediction support for each mode is its own `n_members` value.

The metric key `dom_pLDDT` denotes DBD-pLDDT, the mean pLDDT over the DNA-binding
domain. An all-zero distribution means the estimate was unavailable because the
protein lacked an annotated DNA-binding domain.
See the [AF3 confidence documentation](https://github.com/google-deepmind/alphafold3/blob/main/docs/output.md#confidence-metrics)
for interpretation of the underlying structure-confidence scores.
"""


def _write_modes(modes: pd.DataFrame, path: Path):
    """Write the updated parquet atomically, preserving existing file permissions."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".parquet", delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        modes.to_parquet(temporary, index=False)
        if path.exists():
            shutil.copymode(path, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def build_database(out_dir):
    """Build the scientific database from the research predictions and annotations."""
    out_dir.mkdir(parents=True, exist_ok=True)
    domains_path = DATA / "sequences" / "animaltfdb" / "tfdb4_rep_doms.csv"
    previous_structures = set()
    if (out_dir / "pwm_modes.parquet").is_file():
        previous_structures = set(
            pd.read_parquet(out_dir / "pwm_modes.parquet", columns=["structure_path"])[
                "structure_path"
            ].dropna()
        )
    struct_out_dir = out_dir / STRUCT_SUBDIR
    struct_out_dir.mkdir(exist_ok=True)

    from af3tfm.motif import load_homer_multi_motif_file

    # JASPAR motifs (empirical null + cognate references)
    jaspar_path = DATA / "tools" / "homer" / "jaspar.homer.motif"
    jaspar_names, jaspar_pwms = load_homer_multi_motif_file(str(jaspar_path))
    jaspar_id_to_pwm = dict(zip(jaspar_names, jaspar_pwms))
    jaspar_pwm_list = list(jaspar_id_to_pwm.values())

    # Representative and full TF tables from AnimalTFDB4
    rep_df = add_sequence_types(
        pd.read_csv(DATA / "sequences" / "animaltfdb" / "tfdb4_rep_tfs.csv")
    )
    full_df = add_sequence_types(
        pd.read_csv(DATA / "sequences" / "animaltfdb" / "tfdb4_tfs.csv")
    )

    # per-TF binding modes
    job_dir = DATA / "jobs" / "hero"
    spec_dir = job_dir / "specificities"

    n_with_homolog = rep_df["Hom_JASPAR"].isin(jaspar_id_to_pwm).sum()
    print(f"Reps with a JASPAR ortholog in the HOMER file: {n_with_homolog}")

    all_rows = []
    with ProcessPoolExecutor() as executor:
        futures = {}
        for _, row in rep_df.iterrows():
            jid = row["Hom_JASPAR"]
            has_hom = pd.notna(jid) and jid in jaspar_id_to_pwm
            futures[
                executor.submit(
                    process_one_tf,
                    row.to_dict(),
                    jid if has_hom else None,
                    jaspar_id_to_pwm[jid] if has_hom else None,
                    row["Hom_Sim"],
                    str(spec_dir),
                    str(job_dir),
                    jaspar_pwm_list,
                    str(struct_out_dir),
                )
            ] = row["DBID"]

        done = 0
        for future in as_completed(futures):
            all_rows.extend(future.result())
            done += 1
            if done % 100 == 0 or done == len(futures):
                print(f"  processed {done}/{len(futures)} reps", flush=True)

    modes_df = add_metric_bands(pd.DataFrame(all_rows))
    # Resolve only after extraction: attempted representatives may have no usable PWM.
    modeled_dbids = set(modes_df.loc[modes_df["tier"].eq("primary"), "DBID"])
    tf_to_rep = reconcile_representatives(
        build_tf_to_rep(rep_df, full_df, modeled_dbids), modes_df
    )
    _write_modes(modes_df, out_dir / "pwm_modes.parquet")
    tf_to_rep.to_parquet(out_dir / "tf_to_rep.parquet", index=False)
    write_meme_exports(modes_df, tf_to_rep, out_dir)
    res_counts = tf_to_rep["resolution"].value_counts().to_dict()
    print(f"tf_to_rep: {len(tf_to_rep)} rows | resolution {res_counts}")

    n_primary = (modes_df["tier"] == "primary").sum()
    n_validated = modes_df.loc[modes_df["homer_sim"].notna(), "DBID"].nunique()
    n_structs = modes_df["structure_path"].notna().sum()
    print(
        f"pwm_modes: {len(modes_df)} rows | {n_primary} TFs with a primary PWM | "
        f"{n_validated} TFs with JASPAR validation | {n_structs} primary structures copied"
    )

    # interface residues (per-TF DNA contacts) for the exported reps
    iface_df = build_interface_residues(job_dir / INTERFACE_SRC, set(modes_df["DBID"]))
    iface_df.to_parquet(out_dir / "interface_residues.parquet", index=False)
    print(
        f"interface_residues: {len(iface_df)} residues | "
        f"{iface_df['DBID'].nunique()} TFs with interface data"
    )

    export_domain_annotations(domains_path, out_dir, set(modes_df["DBID"]))
    (out_dir / "SCHEMA.md").write_text(SCHEMA_DOC)
    for relative in previous_structures - set(modes_df["structure_path"].dropna()):
        path = out_dir / relative
        if not path.resolve().is_relative_to(struct_out_dir.resolve()):
            raise ValueError(f"Structure path outside the export: {relative}")
        path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--from-existing",
        action="store_true",
        help="Package the existing scientific database without rebuilding predictions or changing its cluster assignments.",
    )
    modes.add_argument(
        "--lookup-only",
        action="store_true",
        help="Repair lookup self-resolution, then export the complete release.",
    )
    modes.add_argument(
        "--metadata-only",
        action="store_true",
        help="Refresh lookup and reference annotations, then export the complete release.",
    )
    modes.add_argument(
        "--domains-only",
        action="store_true",
        help="Refresh domain annotations, then export the complete release.",
    )
    modes.add_argument(
        "--bands-only",
        action="store_true",
        help="Refresh display bands, then export the complete release.",
    )
    modes.add_argument(
        "--confidence-only",
        action="store_true",
        help="Rebuild confidence summaries, then export the complete release.",
    )
    parser.add_argument(
        "--database-dir",
        type=Path,
        default=DEFAULT_DATABASE,
        help="Scientific database input/output directory (default: data/database/export/database).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Complete release directory (default: data/database/export/<version>).",
    )
    parser.add_argument(
        "--release-metadata",
        type=Path,
        default=EXPORT_ROOT / "release/metadata.json",
        help="Dataset identity, methods, citation, and source configuration.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Read-only verification of all raw files, documents, ZIPs, and checksums against the current database and sources.",
    )
    args = parser.parse_args()
    if args.check and any(
        [
            args.lookup_only,
            args.metadata_only,
            args.domains_only,
            args.bands_only,
            args.confidence_only,
        ]
    ):
        parser.error("--check cannot be combined with a database refresh")
    try:
        load_config(args.release_metadata)
        out_dir = args.database_dir.resolve()
        domains_path = DATA / "sequences/animaltfdb/tfdb4_rep_doms.csv"
        if args.from_existing or args.check:
            pass
        elif args.lookup_only:
            print(refresh_database_metadata(out_dir))
        elif args.metadata_only:
            catalog = pd.read_csv(DATA / "sequences/animaltfdb/tfdb4_tfs.csv")
            zfin_ids = load_zfin_ids(DATA / ZFIN_REFERENCE)
            domains = pd.read_csv(domains_path)
            summary = refresh_database_metadata(out_dir)
            summary.update(
                consolidate_database_annotations(out_dir, catalog, zfin_ids, domains)
            )
            print(summary)
        elif args.domains_only:
            dbids = set(
                pd.read_parquet(out_dir / "pwm_modes.parquet", columns=["DBID"])["DBID"]
            )
            export_domain_annotations(domains_path, out_dir, dbids)
        elif args.bands_only:
            path = out_dir / "pwm_modes.parquet"
            _write_modes(add_metric_bands(pd.read_parquet(path)), path)
        else:
            if not args.confidence_only:
                build_database(out_dir)
            dbids = set(
                pd.read_parquet(out_dir / "pwm_modes.parquet", columns=["DBID"])["DBID"]
            )
            export_confidence_metrics(
                DATA / "jobs/hero/jobdb.sqlite", out_dir, dbids, NUM_SPECS
            )
        export_release(
            ROOT,
            out_dir,
            args.output_dir,
            schema=SCHEMA_DOC,
            metadata_path=args.release_metadata,
            check=args.check,
        )
    except (ValueError, FileNotFoundError) as error:
        parser.exit(1, f"Export failed: {error}\n")


if __name__ == "__main__":
    main()
