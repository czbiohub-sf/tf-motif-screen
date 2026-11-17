"""
Reusable core for protein-DNA interface extraction.

Holds the single-structure contact parser (identical to the interactive
notebooks/02_screen-design/02_interface-contacts.py) plus the pieces needed to run it at scale:
a robust in-memory parser, the per-(TF, residue) aggregation, and a
structure-count balancer for sharding a large TF screen across SLURM tasks.

Every interface metric is per-TF (pooled over that TF's structures), so as
long as a TF's structures are never split across shards the aggregation is
final per shard -- the only cross-shard step is concatenation.

Ben Iovino 7/2026 Biohub
"""

import heapq
import json
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from scipy.spatial import cKDTree

from af3tfm.utils import AMINO_ACIDS

CONTACT_CUTOFF = 5.0
# The three base-reader gates below are calibrated in
# notebooks/02_screen-design/02_sweep-interface-params.py against the zf-C2H2 recognition code (756
# TFs, 8072 known base positions): F1 0.921 / precision 0.968 at this triple,
# and they transfer to homeodomains (F1 0.787 vs 0.792 at that fold's own
# optimum). FRAC_BASE_MIN is the only gate that trades precision for recall;
# MIN_CONTACT and PLDDT_MIN leave precision pinned near 0.97, so these sit on a
# flat ridge rather than at a sharp optimum. Previously 0.3 / 0.4 / 70 (F1 0.870
# on zf-C2H2, 0.758 on homeodomains).
MIN_CONTACT = 0.1  # fraction of structures in which a residue must contact DNA
FRAC_BASE_MIN = 0.3  # base-directed fraction for a base-reader (vs mixed) call
PLDDT_MIN = 50.0  # absolute side-chain pLDDT for a confidently placed residue

MAINCHAIN_ATOMS = {"N", "CA", "C", "O"}
DNA_BACKBONE_ATOMS = {
    "P",
    "OP1",
    "OP2",
    "O5'",
    "C5'",
    "C4'",
    "O4'",
    "C3'",
    "O3'",
    "C2'",
    "O2'",
    "C1'",
}
NUCLEIC_RESIDUES = {"DA": "A", "DC": "C", "DG": "G", "DT": "T"}

# aggregated per-(TF, residue) schema, used for empty-shard parquet consistency
RES_COLUMNS = [
    "dbid",
    "residue_num",
    "resname",
    "contact_freq",
    "frac_base_contact",
    "med_min_dist",
    "med_min_dist_base",
    "med_plddt",
    "confident",
    "interface",
    "contact_mode",
]


def residue_contacts(cif_path: str, dbid: str, is_non: bool) -> list[dict]:
    """
    Extract every protein-residue -> DNA contact (<= CONTACT_CUTOFF) in one
    structure, annotated with distance, contact type, read base + position,
    and side-chain pLDDT.

    Args:
        cif_path (str): Path to the AF3 model CIF (its sibling
            *_confidences.json holds atom_plddts, in CIF atom order).
        dbid (str): TF identifier (Ensembl DBID) to tag rows with.
        is_non (bool): True for a non-motif (scrambled) DNA input.

    Returns:
        list[dict]: one row per contacting residue (empty if unparseable).
    """

    p = Path(cif_path)
    # Derive the confidences file from the model filename, not the parent dir:
    # AF3 names siblings by a shared prefix (<prefix>_model.cif /
    # <prefix>_confidences.json), but the parent dir is sometimes the prefix and
    # sometimes a timestamped name -- so <dir>_confidences.json misses ~1% of jobs.
    conf_path = p.with_name(p.name.replace("_model.cif", "_confidences.json"))
    if not conf_path.exists():
        return []
    try:
        structure = MMCIFParser(QUIET=True).get_structure("s", cif_path)
    except Exception:
        return []

    with open(conf_path) as f:
        conf = json.load(f)
    atom_plddts = np.array(conf["atom_plddts"])

    # Walk residues in CIF order == token order (1 token/residue here).
    prot_coords, prot_res_local = [], []
    dna_coords, dna_nt_local, dna_is_base = [], [], []
    residues = []
    atom_i = 0
    for model in structure:
        for chain in model:
            for res in chain:
                token = len(residues)
                resname = res.get_resname().strip()
                is_prot = resname in AMINO_ACIDS
                is_dna = resname in NUCLEIC_RESIDUES
                sc_plddt = []
                for atom in res:
                    pl = atom_plddts[atom_i]
                    atom_i += 1
                    name = atom.get_name()
                    if is_prot and name not in MAINCHAIN_ATOMS:
                        prot_coords.append(atom.coord)
                        prot_res_local.append(token)
                        sc_plddt.append(pl)
                    elif is_dna:
                        dna_coords.append(atom.coord)
                        dna_nt_local.append(token)
                        dna_is_base.append(name not in DNA_BACKBONE_ATOMS)
                residues.append(
                    {
                        "resname": resname,
                        "resnum": res.get_id()[1],
                        "chain": chain.id,
                        "base": NUCLEIC_RESIDUES.get(resname),
                        "plddt": float(np.mean(sc_plddt)) if sc_plddt else np.nan,
                    }
                )

    if atom_i != len(atom_plddts):
        return []
    if not prot_coords or not dna_coords:
        return []

    prot_coords = np.array(prot_coords)
    prot_res_local = np.array(prot_res_local)
    dna_coords = np.array(dna_coords)
    dna_nt_local = np.array(dna_nt_local)
    dna_is_base = np.array(dna_is_base)

    tree_all = cKDTree(dna_coords)
    d_all, i_all = tree_all.query(prot_coords, k=1)
    base_mask = dna_is_base
    tree_base = cKDTree(dna_coords[base_mask])
    base_nt = dna_nt_local[base_mask]
    d_base, i_base = tree_base.query(prot_coords, k=1)

    job_name = p.parent.parent.name
    rows = []
    for token in np.unique(prot_res_local):
        m = prot_res_local == token
        if not m.any():
            continue
        j = np.argmin(d_all[m])
        min_dist = float(d_all[m][j])
        if min_dist > CONTACT_CUTOFF:
            continue
        is_base_contact = bool(dna_is_base[i_all[m][j]])
        jb = np.argmin(d_base[m])
        min_dist_base = float(d_base[m][jb])
        read_nt = int(base_nt[i_base[m][jb]])

        res = residues[token]
        read = residues[read_nt]
        rows.append(
            {
                "dbid": dbid,
                "job_name": job_name,
                "is_non": is_non,
                "residue_num": res["resnum"],
                "resname": res["resname"],
                "min_dist": round(min_dist, 3),
                "min_dist_base": round(min_dist_base, 3),
                "contact_type": "base" if is_base_contact else "backbone",
                "base": read["base"],
                "read_pos": f"{read['chain']}{read['resnum']}",
                "plddt": round(res["plddt"], 2),
            }
        )
    return rows


def _worker(args: tuple) -> list[dict]:
    """ProcessPool wrapper: (structure_path, dbid, is_non) -> contact rows."""
    sp, dbid, is_non = args
    if not Path(sp).exists():
        return []
    return residue_contacts(sp, dbid, is_non)


def parse_contacts(tasks, max_workers=16, chunk_size=2000, timeout=60, log=print):
    """
    Robustly parse structures -> contact-row dicts, accumulated in memory.

    Same crash-resilient strategy as the interactive extract_contacts: each
    chunk runs in its own short-lived ProcessPoolExecutor, so a worker that
    dies (segfault / OOM) or a structure that hangs wedges at most one chunk,
    never the whole shard. Per-structure results are retrieved with a timeout;
    stragglers and errors are skipped and counted. Rows stay in memory (bounded
    by the shard size) rather than streaming to disk -- at scale, crash
    recovery is provided by re-running the failed SLURM array element.

    Args:
        tasks (list[tuple]): (structure_path, dbid, is_non) per structure.
        max_workers (int): pool size per chunk.
        chunk_size (int): structures per short-lived pool.
        timeout (float): seconds to wait on a single structure before skip.
        log (callable | None): progress sink (default print; None to silence).

    Returns:
        tuple[list[dict], int, int]: (rows, n_ok, n_skipped).
    """
    rows, n_ok, n_skip = [], 0, 0
    for start in range(0, len(tasks), chunk_size):
        chunk = tasks[start : start + chunk_size]
        pool = ProcessPoolExecutor(max_workers=max_workers)
        futs = [pool.submit(_worker, t) for t in chunk]
        for fut, task in zip(futs, chunk):
            try:
                rows.extend(fut.result(timeout=timeout))
                n_ok += 1
            except FuturesTimeout:
                n_skip += 1
                if log:
                    log(f"  skip (timeout): {task[0]}")
            except Exception as e:  # BrokenProcessPool, unpickling, etc.
                n_skip += 1
                if log:
                    log(f"  skip ({type(e).__name__}): {task[0]}")
        # never block on a wedged/hung worker; drop the pool and move on
        pool.shutdown(wait=False, cancel_futures=True)
        if log:
            log(
                f"  {min(start + chunk_size, len(tasks))}/{len(tasks)} structures "
                f"({n_ok} ok, {n_skip} skipped)"
            )
    return rows, n_ok, n_skip


def aggregate_interface(df) -> pd.DataFrame:
    """
    Collapse per-structure contact rows to per-(TF, residue) interface metrics.

    Every metric is per-TF (pooled over that TF's structures), so this is
    correct as long as all of a TF's structures are present in `df` -- which
    by-TF sharding guarantees. Logic is identical to the interactive notebook,
    factored out so the notebook and the shard pipeline agree exactly.

    Args:
        df (pd.DataFrame): contact rows from residue_contacts (one or many TFs).

    Returns:
        pd.DataFrame: per-(TF, residue) rows with interface flags and mode.
    """
    if df.empty:
        return pd.DataFrame(columns=RES_COLUMNS)

    df = df.copy()
    n_struct = df.groupby("dbid")["job_name"].nunique()

    recs = []
    for (dbid, rnum, rname), g in df.groupby(["dbid", "residue_num", "resname"]):
        recs.append(
            {
                "dbid": dbid,
                "residue_num": rnum,
                "resname": rname,
                # fraction of structures in which the residue contacts DNA
                # (dedup across protein chains, so a homo-oligomer copy is not
                # double-counted); bounded 0-1, unlike a per-chain contact count
                "contact_freq": g["job_name"].nunique() / n_struct[dbid],
                # base- vs backbone-directed engagement
                "frac_base_contact": (g["contact_type"] == "base").mean(),
                "med_min_dist": g["min_dist"].median(),  # to any DNA atom
                "med_min_dist_base": g[
                    "min_dist_base"
                ].median(),  # to nearest nucleobase
                # confidence: absolute side-chain pLDDT
                "med_plddt": g["plddt"].median(),
            }
        )
    res_df = pd.DataFrame(recs)

    # --- Interface flags ---
    # pLDDT is an absolute bar (0-100), not a within-TF percentile, so that a
    # uniformly confident DBD does not have half its residues thrown out.
    res_df["confident"] = res_df["med_plddt"] >= PLDDT_MIN
    # interface residue = reliably AND confidently contacts DNA
    res_df["interface"] = (res_df["contact_freq"] >= MIN_CONTACT) & res_df["confident"]
    # descriptive contact mode of an interface residue
    res_df["contact_mode"] = np.select(
        [
            res_df["frac_base_contact"] >= FRAC_BASE_MIN,
            res_df["frac_base_contact"] <= 0.2,
        ],
        ["base", "backbone"],
        default="mixed",
    )
    return res_df


def balance_shards(tf_sizes: dict, n_shards: int) -> list[list[str]]:
    """
    Partition TFs into n_shards balancing total structure count per shard.

    Greedy longest-processing-time (LPT): assign each TF, largest first, to the
    currently-lightest shard. Whole TFs are never split, so every TF's
    structures land in a single shard -- required for the per-TF aggregation.
    Balancing by structure count (not TF count) keeps shard wall-times even
    when the per-TF structure count is skewed.

    The order is fully deterministic (size descending, TF name as tie-break),
    so independent callers with the same inputs produce the identical partition
    -- this lets each SLURM array element recompute its own shard without a
    shared plan file.

    Args:
        tf_sizes (dict[str, int]): structures per TF (dbid -> count).
        n_shards (int): desired number of shards (clamped to [1, #TFs]).

    Returns:
        list[list[str]]: TF (dbid) lists, one per shard (empty shards dropped).
    """
    n_shards = max(1, min(n_shards, len(tf_sizes)))
    shards = [[] for _ in range(n_shards)]
    heap = [(0, i) for i in range(n_shards)]  # (load, shard_idx)
    heapq.heapify(heap)
    for tf, size in sorted(tf_sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        load, i = heapq.heappop(heap)
        shards[i].append(tf)
        heapq.heappush(heap, (load + size, i))
    return [s for s in shards if s]


def run_shard(tasks, out_path, max_workers=16, chunk_size=2000, timeout=60, log=print):
    """
    Parse one shard's structures and write its per-(TF, residue) aggregate.

    Args:
        tasks (list[tuple]): (structure_path, dbid, is_non) for this shard's TFs.
        out_path (str | Path): parquet path for the aggregated res_df.
        max_workers, chunk_size, timeout: forwarded to parse_contacts.
        log (callable | None): progress sink.

    Returns:
        pd.DataFrame: the aggregated res_df that was written.
    """
    out_path = Path(out_path)
    rows, n_ok, n_skip = parse_contacts(
        tasks,
        max_workers=max_workers,
        chunk_size=chunk_size,
        timeout=timeout,
        log=log,
    )
    if log:
        log(f"parsed {n_ok} ok / {n_skip} skipped -> {len(rows)} contact rows")
    res_df = aggregate_interface(pd.DataFrame(rows))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    res_df.to_parquet(tmp, index=False)
    os.replace(tmp, out_path)  # atomic; a partial shard is never left in place
    if log:
        log(f"wrote {out_path} ({len(res_df)} TF-residue rows)")
    return res_df
