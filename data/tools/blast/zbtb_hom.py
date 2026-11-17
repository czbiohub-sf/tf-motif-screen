"""
Report the ZFIN ortholog of each zebrafish ZBTB TF and its recognition-code similarity
Specificity scoring is restricted to the fingers AF3/interproscan annotation marks as DNA-binding.
Those fingers are slid along the human ortholog's full array and the best register is scored.

Ben Iovino  5/19/26 Biohub
"""

# %%
import time
from pathlib import Path

import pandas as pd

from af3tfm.dataset import DATA_DIR, DS
from af3tfm.utils import _jaspar_cache, fetch_jaspar_motif
from af3tfm.zfc2h2 import (
    binding_finger_similarity,
    dna_binding_finger_indices,
    extract_zf_codes,
    extract_zf_codes_indexed,
)

# %%
# Load data. Use the reps dataset: it is what we write back to, and its doms table is the
# only one carrying the DNA-Binding call per finger.
dataset = DS["tfdb4_rep"]
zf_df = pd.read_csv(dataset.seq_csv)
zf_df = zf_df[zf_df["Family"] == "ZBTB"].copy().reset_index(drop=True)
print(f"Zebrafish ZBTB TFs: {len(zf_df)}")

doms_df = pd.read_csv(dataset.dom_csv)
binding_ranges: dict[str, list[tuple[int, int]]] = {}
for _r in doms_df[
    (doms_df["DNA-Binding"]) & doms_df["description"].str.contains("C2H2", na=False)
].itertuples():
    binding_ranges.setdefault(_r.DBID, []).append((_r.start, _r.end))
_annotated = set(binding_ranges) & set(zf_df["DBID"])
print(f"  with DNA-binding finger annotations: {len(_annotated)}")

# Score against any human TF with a detectable finger array, not just the ones AnimalTFDB
# files under ZBTB: a ZFIN ortholog is the ortholog whatever family label it carries.
human_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/human_tfs.csv")
print(f"Human TFs: {len(human_df)}")


# %%
# Extract recognition codes. Zebrafish: only the DNA-binding fingers, keeping each one's
# index in the full array so the spacing of non-adjacent binding fingers is preserved.
# Human: the full array, which is the search space the binding fingers are slid along.
zf_codes: dict[str, tuple] = {}
for _, row in zf_df.iterrows():
    fingers = extract_zf_codes_indexed(row["Sequence"])
    idx = dna_binding_finger_indices(
        row["Sequence"], binding_ranges.get(row["DBID"], [])
    )
    # keep unannotated TFs so their ortholog is still reported, just with no similarity
    codes = [fingers[i][2] for i in idx] if idx else None
    zf_codes[row["DBID"]] = (row["TF_Name"], codes, idx, len(fingers))

# HGNC symbol (upper) -> full recognition code array, for scoring a candidate human gene
human_codes_by_sym: dict[str, list] = {}
for _, _row in human_df.iterrows():
    _c = extract_zf_codes(_row["Sequence"])
    if _c:
        human_codes_by_sym.setdefault(str(_row["TF_Name"]).upper(), _c)

n_scoreable = sum(1 for v in zf_codes.values() if v[1])
print(f"Zebrafish with ≥1 DNA-binding ZF: {n_scoreable}/{len(zf_codes)}")
print(f"Human with ≥1 ZF detected:        {len(human_codes_by_sym)}")


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
for _dbid in zf_codes:
    _needed.update(o.upper() for o in dbid_orthologs.get(_dbid, []))
_to_fetch = [s for s in sorted(_needed) if s not in _jaspar_cache]
print(
    f"Resolving JASPAR for {len(_to_fetch)} unseen genes (of {len(_needed)} candidates)"
)
for sym in _to_fetch:
    fetch_jaspar_motif(sym)
    time.sleep(0.05)  # rate limit


def _jaspar(sym: str) -> str | None:
    """JASPAR matrix id for a gene symbol, or None (from the primed/fetched cache)."""
    return _jaspar_cache.get(str(sym).upper())


def _array_score(zf_c: list[str] | None, offsets: list[int], sym: str) -> float | None:
    """
    Percent of matching DNA-contacting residues (0-100) between a zebrafish TF's DNA-binding
    fingers and the best-matching stretch of a human gene's array. None when the zebrafish TF
    has no annotated DNA-binding finger to score, or the human gene has no detectable array.
    """
    h_c = human_codes_by_sym.get(str(sym).upper())
    if not zf_c or not h_c:
        return None
    sim = binding_finger_similarity(zf_c, offsets, h_c)
    return None if sim is None else round(sim * 100, 1)


# %%
# Report the ZFIN ortholog and its recognition-code similarity. A motif-bearing ortholog wins
# over a motif-less one (only a motif is usable downstream); ties go to the higher similarity.
rows = []
for zf_dbid, (zf_name, zf_c, offsets, n_total) in zf_codes.items():
    orths = dbid_orthologs.get(zf_dbid, [])
    cands = [(o, _array_score(zf_c, offsets, o), _jaspar(o)) for o in orths]

    chosen = None  # (gene, score, jaspar)
    if cands:
        chosen = max(cands, key=lambda c: (bool(c[2]), c[1] is not None, c[1] or 0))

    rows.append(
        {
            "DBID": zf_dbid,
            "TF_Name": zf_name,
            "n_zf": len(zf_c) if zf_c else 0,  # DNA-binding fingers, the ones scored
            "n_total_zf": n_total,  # fingers detected in the full sequence
            "Hom_Gene": chosen[0] if chosen else None,
            "Hom_Sim": chosen[1] if chosen else None,
            "Hom_JASPAR": chosen[2] if chosen else None,
        }
    )

results_df = pd.DataFrame(rows)
n_orth = int(results_df["Hom_Gene"].notna().sum())
n_motif = int(results_df["Hom_JASPAR"].notna().sum())
n_scored = int(results_df["Hom_Sim"].notna().sum())
n_unann = int(((results_df["Hom_Gene"].notna()) & (results_df["n_zf"] == 0)).sum())
print(f"Zebrafish ZBTB with a ZFIN ortholog: {n_orth}/{len(zf_codes)}")
print(f"  ortholog carries a JASPAR motif: {n_motif}")
print(f"  similarity scoreable: {n_scored}")
print(
    f"  null similarity: {n_unann} lack a DNA-binding finger annotation, "
    f"{n_orth - n_scored - n_unann} have an ortholog with no detectable finger array"
)
print("\nReported orthologs, lowest similarity first:")
print(
    results_df[results_df["Hom_Gene"].notna()][
        ["TF_Name", "n_zf", "n_total_zf", "Hom_Gene", "Hom_Sim", "Hom_JASPAR"]
    ]
    .sort_values("Hom_Sim", ascending=True)
    .to_string(index=False)
)


# %%
# Update reps csv (ZFIN-anchored ZBTB assignments). Same Hom_* schema hom_results.py writes
# for the single-DBD families.
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
