"""
Report the ZFIN ortholog of each zebrafish Homeobox TF and its recognition-code similarity
The homeodomain is aligned to PF00046 and residues at canonical positions 47, 50, 51, 52, 54
form a 5-char recognition code (shared DNA-binding specificity); the similarity reported is
the percent of those 5 residues that match the ortholog's code, so a partial match is reported
at its true value (4/5 -> 80%). The ortholog is reported whether or not it carries a JASPAR motif.

Ben Iovino  4/22/26  Biohub
"""

# %%
import time
from pathlib import Path

import pandas as pd
import pyhmmer

from af3tfm.dataset import DATA_DIR
from af3tfm.homeobox import (
    CANONICAL_P51,
    CANONICAL_P51_MS,
    HD_POSITIONS,
    HMM_PATH,
    align_sequences,
    extract_hd_sequences,
    extract_recognition_codes,
    load_hmm,
    recognition_code_str,
)
from af3tfm.utils import _jaspar_cache, fetch_jaspar_motif


def find_hd_via_hmmsearch(
    hmm: pyhmmer.plan7.HMM, tfs_df: pd.DataFrame
) -> dict[str, str]:
    """
    Find HD boundaries in full-length sequences via hmmsearch (no doms file needed).

    Args:
        hmm: pre-loaded HMM for the homeodomain (e.g. PF00046)
        tfs_df: DataFrame of TFs with columns "DBID" and "Sequence"

    Returns:
        dict: mapping unique instance IDs (DBID or DBID__hdN) to homeodomain sequences
    """

    alphabet = pyhmmer.easel.Alphabet.amino()
    seq_map = dict(zip(tfs_df["DBID"], tfs_df["Sequence"]))
    digital_seqs = [
        pyhmmer.easel.TextSequence(name=dbid.encode(), sequence=seq).digitize(alphabet)
        for dbid, seq in seq_map.items()
    ]
    sequences: dict[str, str] = {}
    instance_counts: dict[str, int] = {}

    # Run hmmsearch and extract domain sequences for each significant hit
    for hits in pyhmmer.hmmer.hmmsearch([hmm], digital_seqs):
        for hit in hits:
            if not hit.included:
                continue
            name = hit.name.decode() if isinstance(hit.name, bytes) else hit.name
            full_seq = seq_map.get(name, "")
            for domain in hit.domains:
                if not domain.included:
                    continue
                dom_seq = full_seq[domain.env_from - 1 : domain.env_to]
                n = instance_counts.get(name, 0)
                key = name if n == 0 else f"{name}__hd{n + 1}"
                instance_counts[name] = n + 1
                sequences[key] = dom_seq

    return sequences


def hd_similarity(codes_a: dict[int, str], codes_b: dict[int, str]) -> float:
    """
    Fraction of HD positions that match. A gapped position on either side cannot match, so
    it counts against the score rather than voiding it.

    Args:
        codes_a: dict mapping HD position to residue for sequence A
        codes_b: dict mapping HD position to residue for sequence B

    Returns:
        float: similarity score between 0 and 1
    """

    matches = 0
    for pos in HD_POSITIONS:
        a = codes_a.get(pos, "-")
        b = codes_b.get(pos, "-")
        if a in ("-", ".") or b in ("-", "."):
            continue
        if a == b:
            matches += 1

    return matches / len(HD_POSITIONS)


# %%
# Load data
zf_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
zf_hd = zf_df[zf_df["Family"] == "Homeobox"].copy().reset_index(drop=True)
print(f"Zebrafish Homeobox TFs: {len(zf_hd)}")

doms_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_doms.csv")
for col in doms_df.select_dtypes(include="object").columns:
    doms_df[col] = doms_df[col].str.strip()
if doms_df["DBID"].str.contains("\t", na=False).any():
    doms_df["DBID"] = doms_df["DBID"].str.split("\t").str[1]

human_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/human_tfs.csv")
human_hd = human_df[human_df["Family"] == "Homeobox"].copy().reset_index(drop=True)
print(f"Human Homeobox TFs: {len(human_hd)}")


# %%
# Extract homeodomain sequences and align to PF00046
hmm = load_hmm(HMM_PATH)

print("Zebrafish — extracting HD from domain annotations:")
zf_hd_seqs = extract_hd_sequences(zf_hd, doms_df)

print("Human — finding HD via hmmsearch:")
human_hd_seqs = find_hd_via_hmmsearch(hmm, human_hd)

print("Aligning all HD sequences to PF00046:")
all_hd_seqs = {**zf_hd_seqs, **human_hd_seqs}
aligned_seqs, match_cols = align_sequences(hmm, all_hd_seqs)
rec_codes = extract_recognition_codes(aligned_seqs, match_cols)

zf_codes = {k: rec_codes[k] for k in zf_hd_seqs if k in rec_codes}
human_codes = {k: rec_codes[k] for k in human_hd_seqs if k in rec_codes}
print(
    f"Recognition codes extracted: {len(zf_codes)} zebrafish, {len(human_codes)} human"
)


# %%
# Collapse multi-instance proteins: use first HD instance per protein
human_name_map = dict(zip(human_hd["DBID"], human_hd["TF_Name"]))
human_protein_codes: dict[str, dict[int, str]] = {}
for iid, code in human_codes.items():
    protein = iid.split("__hd")[0]
    if protein not in human_protein_codes:
        human_protein_codes[protein] = code

zf_name_map = dict(zip(zf_hd["DBID"], zf_hd["TF_Name"]))
zf_protein_codes: dict[str, dict[int, str]] = {}
for iid, code in zf_codes.items():
    protein = iid.split("__hd")[0]
    if protein not in zf_protein_codes:
        zf_protein_codes[protein] = code

# HGNC symbol (upper) -> recognition code, for scoring a candidate
human_codes_by_sym: dict[str, dict[int, str]] = {}
for h_dbid, h_code in human_protein_codes.items():
    sym = str(human_name_map.get(h_dbid, "")).upper()
    if sym and sym not in human_codes_by_sym:
        human_codes_by_sym[sym] = h_code


# %%
# ZFIN ortholog mappings (same source as hom_results.py). DBID == ensdarg.
ZFIN_DIR = Path(f"{DATA_DIR}/tools/blast/zfin")
orthos_df = pd.read_csv(ZFIN_DIR / "ensdarg_human_orthos.csv")

# ensdarg (== DBID) -> list of human ortholog symbols
dbid_orthologs = {
    row.ensdarg: [s for s in str(row.human_symbols).split("|") if s]
    for row in orthos_df.itertuples()
}


# %%
# Resolve JASPAR motifs for every ZFIN ortholog of a zebrafish TF. Prime the cache from the
# reps csv first (absent on the first run) so re-runs only hit the JASPAR API for unseen genes.
_seed = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_tfs.csv")
if (
    "Hom_Gene" in _seed.columns
):  # absent on the first run, before any Hom_* columns exist
    for _, _row in _seed.dropna(subset=["Hom_Gene"]).iterrows():
        _jaspar_cache.setdefault(
            str(_row["Hom_Gene"]).upper(),
            _row["Hom_JASPAR"] if pd.notna(_row["Hom_JASPAR"]) else None,
        )

_needed: set[str] = set()
for _dbid in zf_protein_codes:
    _needed.update(o.upper() for o in dbid_orthologs.get(_dbid, []))
_to_fetch = [s for s in sorted(_needed) if s not in _jaspar_cache]
print(
    f"Resolving JASPAR for {len(_to_fetch)} unseen genes (of {len(_needed)} candidates)"
)
for j, sym in enumerate(_to_fetch):
    fetch_jaspar_motif(sym)
    if (j + 1) % 50 == 0:
        print(f"  {j + 1}/{len(_to_fetch)}")
    time.sleep(0.05)  # rate limit


def _jaspar(sym: str) -> str | None:
    """JASPAR matrix id for a gene symbol, or None (from the primed/fetched cache)."""
    return _jaspar_cache.get(str(sym).upper())


def _hd_score(zf_code: dict, sym: str) -> float | None:
    """
    Recognition-code similarity between a zebrafish homeodomain and a human gene's
    homeodomain, or None if the gene is not a scorable human Homeobox.
    """
    h_code = human_codes_by_sym.get(str(sym).upper())
    if not h_code:
        return None
    return hd_similarity(zf_code, h_code)


# %%
# Report the ZFIN ortholog and its recognition-code similarity. A motif-bearing ortholog wins
# over a motif-less one (only a motif is usable downstream); ties go to the higher similarity,
# then to the gene symbol so the choice is deterministic.
rows = []
for protein, zf_code in zf_protein_codes.items():
    code_str = recognition_code_str(zf_code)
    is_atypical = zf_code.get(CANONICAL_P51_MS, "-") != CANONICAL_P51
    orths = dbid_orthologs.get(protein, [])
    cands = [(o, _hd_score(zf_code, o), _jaspar(o)) for o in orths]

    chosen = None  # (gene, recog_sim, jaspar)
    if cands:
        chosen = max(
            cands, key=lambda c: (bool(c[2]), c[1] is not None, c[1] or 0, c[0])
        )

    rows.append(
        {
            "DBID": protein,
            "TF_Name": zf_name_map.get(protein, protein),
            "hd_code": code_str,
            "is_atypical": is_atypical,
            "Hom_Gene": chosen[0] if chosen else None,
            "Hom_Sim": round(chosen[1] * 100, 1)
            if chosen and chosen[1] is not None
            else None,
            "Hom_JASPAR": chosen[2] if chosen else None,
        }
    )

results_df = pd.DataFrame(rows)
n_orth = int(results_df["Hom_Gene"].notna().sum())
n_motif = int(results_df["Hom_JASPAR"].notna().sum())
n_scored = int(results_df["Hom_Sim"].notna().sum())
print(f"\nZebrafish Homeobox with a ZFIN ortholog: {n_orth}/{len(zf_protein_codes)}")
print(f"  ortholog carries a JASPAR motif: {n_motif}")
print(f"  recognition code scoreable: {n_scored} (rest are not human Homeobox)")
print("\nReported orthologs, lowest similarity first:")
print(
    results_df[results_df["Hom_Gene"].notna()][
        ["TF_Name", "hd_code", "Hom_Gene", "Hom_Sim", "Hom_JASPAR"]
    ]
    .sort_values("Hom_Sim", ascending=True)
    .to_string(index=False)
)


# %%
# Update reps csv (ZFIN-anchored Homeobox assignments). Same Hom_* schema hom_results.py
# writes for the single-DBD families.
rep_tfs_path = Path(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_tfs.csv")
rep_tfs_df = pd.read_csv(rep_tfs_path)

hom_cols = ["Hom_Sim", "Hom_Gene", "Hom_JASPAR"]
for col in hom_cols:
    if col not in rep_tfs_df.columns:
        rep_tfs_df[col] = None
rep_tfs_df = rep_tfs_df.drop(
    columns=["Hom_Source"], errors="ignore"
)  # paralog-era column

hom_map = results_df.set_index("DBID")[hom_cols]
mask = rep_tfs_df["DBID"].isin(hom_map.index)
for col in hom_cols:
    rep_tfs_df.loc[mask, col] = rep_tfs_df.loc[mask, "DBID"].map(hom_map[col])

rep_tfs_df.to_csv(rep_tfs_path, index=False)
print(f"Updated {int(mask.sum())} rows in {rep_tfs_path.name}")

# %%
