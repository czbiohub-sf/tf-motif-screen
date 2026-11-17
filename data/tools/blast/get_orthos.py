"""
Build a zebrafish (ENSDARG) -> human ortholog symbol mapping from ZFIN's curated orthology,
reduced to the TFs in our dataset, and flag which orthologs have a JASPAR motif.

Ben Iovino  7/23/26 Biohub
"""

# %%
import time
import requests
import pandas as pd
from pathlib import Path
from af3tfm.dataset import DATA_DIR
from af3tfm.utils import fetch_jaspar_motif, _jaspar_cache

ZFIN_DIR = Path(f"{DATA_DIR}/tools/blast/zfin")
ZFIN_FILES = ["human_orthos.txt", "ensembl_1_to_1.txt"]
ZFIN_BASE_URL = "https://zfin.org/downloads"


# %%
def download_zfin_files():
    """
    Download the ZFIN orthology files.
    """

    ZFIN_DIR.mkdir(parents=True, exist_ok=True)
    for fname in ZFIN_FILES:
        dest = ZFIN_DIR / fname
        url = f"{ZFIN_BASE_URL}/{fname}"
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        dest.write_bytes(resp.content)


download_zfin_files()


# %%
# ensembl_1_to_1.txt: ZFIN gene ID, SO term, zebrafish symbol, ENSDARG ID
ensembl = pd.read_csv(
    ZFIN_DIR / "ensembl_1_to_1.txt",
    sep="\t",
    header=None,
    names=["zfin_id", "so_term", "zf_symbol", "ensdarg"],
    dtype=str,
)

# human_orthos.txt: ZFIN gene ID, zf symbol, zf name, human symbol, human name,
# OMIM ID, human NCBI gene ID, human HGNC ID, evidence, pub, ... (extra pub cols)
orthos = pd.read_csv(
    ZFIN_DIR / "human_orthos.txt",
    sep="\t",
    header=None,
    usecols=[0, 1, 3, 6, 7],
    names=["zfin_id", "zf_symbol", "human_symbol", "human_ncbi_id", "human_hgnc_id"],
    dtype=str,
)

# Collapse the pub-level duplication in human_orthos down to unique ortholog assertions.
orthos = orthos.drop_duplicates(subset=["zfin_id", "human_symbol"]).reset_index(
    drop=True
)

# Join ENSDARG onto the ortholog assertions via ZFIN gene ID.
merged = ensembl[["zfin_id", "zf_symbol", "ensdarg"]].merge(
    orthos[["zfin_id", "human_symbol", "human_ncbi_id", "human_hgnc_id"]],
    on="zfin_id",
    how="inner",
)


# %%
# Aggregate all human symbols per ENSDARG
def _agg(series: pd.Series) -> str:
    """
    Aggregate a series of strings into a single string with unique values separated by '|'.

    Args:
        series (pd.Series): A pandas Series containing strings.

    Returns:
        str: A single string with unique values separated by '|'.
    """
    return "|".join(sorted(set(series.dropna())))


mapping = (
    merged.groupby("ensdarg")
    .agg(
        zf_symbol=("zf_symbol", _agg),
        zfin_id=("zfin_id", _agg),
        human_symbols=("human_symbol", _agg),
        human_ncbi_ids=("human_ncbi_id", _agg),
        human_hgnc_ids=("human_hgnc_id", _agg),
    )
    .reset_index()
)
mapping["n_human_orthos"] = mapping["human_symbols"].str.count(r"\|").add(1)
mapping = mapping.sort_values("ensdarg").reset_index(drop=True)


# %%
# Reduce the mapping to the TFs in our dataset (matched on ENSDARG == DBID).
tfs = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_tfs.csv")
dataset_dbids = set(tfs["DBID"])
mapping = mapping[mapping["ensdarg"].isin(dataset_dbids)].reset_index(drop=True)
print(f"dataset TFs: {len(dataset_dbids)}; with a ZFIN human ortholog: {len(mapping)}")


# %%
# AnimalTFDB4 lists some genes under more than one ensembl ID
# Copy each ortholog onto the other IDs for the same zebrafish symbol
sym_to_dbids: dict[str, list[str]] = {}
for dbid, name in zip(tfs["DBID"], tfs["TF_Name"]):
    sym_to_dbids.setdefault(str(name).lower(), []).append(dbid)

known = set(mapping["ensdarg"])
aliases = []
for rec in mapping.to_dict("records"):
    for sym in str(rec["zf_symbol"]).split("|"):
        for dbid in sym_to_dbids.get(sym.lower(), []):
            if dbid in known:
                continue
            known.add(dbid)
            aliases.append({**rec, "ensdarg": dbid})

if aliases:
    mapping = pd.concat([mapping, pd.DataFrame(aliases)], ignore_index=True)
    mapping = mapping.sort_values("ensdarg").reset_index(drop=True)
print(
    f"  alias rows for duplicate-ENSDARG genes: {len(aliases)}; total: {len(mapping)}"
)


# %%
# Query JASPAR once for every unique human ortholog symbol.
unique_symbols = sorted(
    {s for syms in mapping["human_symbols"] for s in str(syms).split("|") if s}
)
print(f"Unique human ortholog symbols to check: {len(unique_symbols)}")
for j, sym in enumerate(unique_symbols):
    fetch_jaspar_motif(sym)
    time.sleep(0.05)  # rate limit
    if (j + 1) % 500 == 0:
        print(f"  checked {j + 1}/{len(unique_symbols)}")


# %%
# Add a jaspar_ids column: matrix IDs for the orthologs that have one (pipe-joined,
# empty string if none of the row's orthologs are in JASPAR).
def _jaspar_ids(syms: str) -> str:
    """
    Look up each pipe-joined human symbol in the JASPAR cache and return the matching
    matrix IDs, pipe-joined ('' if none of the orthologs have a JASPAR motif).

    Args:
        syms (str): Pipe-joined human ortholog symbols.

    Returns:
        str: Pipe-joined JASPAR matrix IDs.
    """
    ids = [_jaspar_cache.get(s.upper()) for s in str(syms).split("|") if s]
    return "|".join(i for i in ids if i)


mapping["jaspar_ids"] = mapping["human_symbols"].apply(_jaspar_ids)

n_with = int((mapping["jaspar_ids"] != "").sum())
print(f"ENSDARG whose ZFIN ortholog has a JASPAR motif: {n_with}/{len(mapping)}")


# %%
# Output to csv
out_path = ZFIN_DIR / "ensdarg_human_orthos.csv"
mapping.to_csv(out_path, index=False)
print(f"Wrote {out_path}: {len(mapping)} ENSDARG -> human ortholog rows")

# %%
