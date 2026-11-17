"""
Predicted protein-DNA interface residues for every TF in a structure panel.

For each (TF, residue) we ask: does the residue reliably and confidently contact
DNA across the panel? An interface residue is one that is within CONTACT_CUTOFF
of DNA in >= MIN_CONTACT of structures AND is confidently placed (absolute
side-chain pLDDT >= PLDDT_MIN). Each interface residue is annotated with its
contact mode (base / backbone / mixed), contact frequency, readout distances,
and confidence.

Everything here is LABEL-FREE and length-agnostic (metrics pooled over all
structures, no motif/non split, no DNA-position dependence), so the identical
pipeline runs on a real variable-length TF screen. Sequence-specificity /
register analysis is intentionally NOT done here.

Outputs (under data/jobs/<run>/):
  residue_contacts_all.csv  - one row per residue-contact per structure
  interface_residues.csv    - per-(TF, residue) aggregate with interface flag

Ben Iovino 7/2026 Biohub
"""

# %%
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from scipy.spatial import cKDTree

from af3tfm.contacts import (
    CONTACT_CUTOFF,
    FRAC_BASE_MIN,
    MIN_CONTACT,
    PLDDT_MIN,
)
from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.jobdb import Database
from af3tfm.utils import AMINO_ACIDS, set_plotting_style

warnings.filterwarnings("ignore")
set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# The cutoff and the three base-reader gates are imported, not redefined: this
# notebook and the sharded pipeline must agree exactly, and local copies here
# silently shadowed af3tfm.contacts when the gates were recalibrated. See
# notebooks/02_screen-design/02_sweep-interface-params.py for how the gates were chosen.
run = "hero"
dataset = DS["tfdb4_rep"]
# Set to a single TF_Name to parse/aggregate/plot just that TF (fast, its own
# per-TF cache + suffixed outputs, so the full-panel files are untouched);
# None runs the whole panel.
SINGLE_TF = "sp3a"

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


# %%
def residue_contacts(cif_path: str, dbid: str, is_non: bool) -> list[dict]:
    """
    Extract every protein-residue -> DNA contact (<= CONTACT_CUTOFF) in one
    structure, annotated with distance, contact type, read base + position, and
    side-chain pLDDT.

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

    # Walk residues in CIF order; atom_plddts is in that same atom order.
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


# %%
def extract_contacts(tasks, out_csv, max_workers=32, chunk_size=1000, timeout=60):
    """
    Parse every structure -> DNA contact rows, robustly and incrementally.

    Structures are processed in chunks, each in its own short-lived
    ProcessPoolExecutor. A worker that dies (segfault / OOM) or a structure
    that hangs can therefore wedge at most one chunk, never the whole run,
    and each completed chunk is appended to disk immediately so a partial run
    is never lost. Per-structure results are retrieved with a timeout;
    stragglers and errors are skipped and counted rather than propagated.

    Args:
        tasks (list[tuple]): (structure_path, dbid, is_non) per structure.
        out_csv (Path): final CSV; written only after all chunks complete.
            Intermediate rows accumulate in a sibling ".partial.csv" that is
            promoted to out_csv on success (so a partial file is never
            mistaken for a finished cache).
        max_workers (int): pool size per chunk.
        chunk_size (int): structures per short-lived pool.
        timeout (float): seconds to wait on a single structure before skip.

    Returns:
        pd.DataFrame: all collected contact rows.
    """
    partial = out_csv.with_suffix(".partial.csv")
    if partial.exists():
        partial.unlink()
    n_ok = n_skip = 0
    wrote_header = False
    for start in range(0, len(tasks), chunk_size):
        chunk = tasks[start : start + chunk_size]
        rows = []
        pool = ProcessPoolExecutor(max_workers=max_workers)
        futs = [pool.submit(_worker, t) for t in chunk]
        for fut, task in zip(futs, chunk):
            try:
                rows.extend(fut.result(timeout=timeout))
                n_ok += 1
            except FuturesTimeout:
                n_skip += 1
                print(f"  skip (timeout): {task[0]}")
            except Exception as e:  # BrokenProcessPool, unpickling, etc.
                n_skip += 1
                print(f"  skip ({type(e).__name__}): {task[0]}")
        # never block on a wedged/hung worker; drop the pool and move on
        pool.shutdown(wait=False, cancel_futures=True)
        if rows:
            pd.DataFrame(rows).to_csv(
                partial, mode="a", header=not wrote_header, index=False
            )
            wrote_header = True
        print(
            f"  {min(start + chunk_size, len(tasks))}/{len(tasks)} structures "
            f"({n_ok} ok, {n_skip} skipped)"
        )
    if not partial.exists():  # no structure yielded any contact rows
        print(f"WARNING: no contact rows produced ({n_ok} ok, {n_skip} skipped)")
        pd.DataFrame().to_csv(out_csv, index=False)
        return pd.DataFrame()
    os.replace(partial, out_csv)  # promote to final cache only on completion
    return pd.read_csv(out_csv)


# %%
# --- Extract all structures (cached, parallel, crash-resilient) ---
# Guarded by __main__ so `import interface_contacts` never spawns the pool;
# in a Jupyter/interactive window __name__ == "__main__", so cells still run.
if __name__ == "__main__":
    # Restrict to one TF's structures when SINGLE_TF is set. The jobdb keys
    # proteins by lowercased DBID, so resolve TF_Name -> DBID and lowercase it.
    if SINGLE_TF:
        _tf = pd.read_csv(dataset.seq_csv)
        _name2dbid = {
            n.lower(): str(d).lower() for n, d in zip(_tf["TF_Name"], _tf["DBID"])
        }
        dbid = _name2dbid[SINGLE_TF.lower()]
        contacts_csv = Path(f"{DATA_DIR}/jobs/{run}/residue_contacts_{SINGLE_TF}.csv")
        where_sql, where_params = "WHERE protein = ?", (dbid,)
    else:
        contacts_csv = Path(f"{DATA_DIR}/jobs/{run}/residue_contacts_all.csv")
        where_sql, where_params = "", ()

    if contacts_csv.exists():
        df = pd.read_csv(contacts_csv)
    else:
        db = Database(job_path=f"{DATA_DIR}/jobs/{run}", dataset=dataset.name)
        runs = db.cursor.execute(
            f"SELECT structure_path, protein, motif FROM runs {where_sql}",
            where_params,
        ).fetchall()
        db.conn.close()
        tasks = [(sp, protein, motif.endswith("-non")) for sp, protein, motif in runs]
        print(
            f"Parsing {len(tasks)} structures across {len({t[1] for t in tasks})} TFs..."
        )
        df = extract_contacts(tasks, contacts_csv, max_workers=32)

    print(
        f"{len(df)} contacts | {df['dbid'].nunique()} TFs | "
        f"{df['job_name'].nunique()} structures"
    )


# %%
# --- Aggregate per (TF, residue): interface metrics ---
# All interface metrics are LABEL-FREE and length-agnostic (pooled over every
# structure), so the same aggregation runs on a real variable-length TF screen.
if __name__ == "__main__":
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

    # TF name for readability. Built from the full tfdb4 table first, then
    # overlaid with the run's own dataset so its names win on overlap: the hero
    # jobdb is a superset of the representative subset, so names for the extra
    # TFs (klf1 among them) live only in the full table and a dataset-only map
    # leaves them NaN -- which then reads as "no residues for TF" downstream.
    full_df = pd.read_csv(DS["tfdb4"].seq_csv)
    tf_df = pd.read_csv(dataset.seq_csv)
    name_map = {str(d).upper(): n for n, d in zip(full_df["TF_Name"], full_df["DBID"])}
    name_map.update(
        {str(d).upper(): n for n, d in zip(tf_df["TF_Name"], tf_df["DBID"])}
    )
    res_df["tf"] = res_df["dbid"].str.upper().map(name_map)

    suffix = f"_{SINGLE_TF}" if SINGLE_TF else ""
    out_csv = Path(f"{DATA_DIR}/jobs/{run}/interface_residues{suffix}.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(res_df)} TF-residue rows)")


# %%
# --- Per-TF interface summary ---
if __name__ == "__main__":
    iface = res_df[res_df["interface"]]
    summary = (
        iface.groupby("tf")
        .agg(
            n_interface=("interface", "sum"),
            n_base=("contact_mode", lambda s: (s == "base").sum()),
            n_backbone=("contact_mode", lambda s: (s == "backbone").sum()),
            n_mixed=("contact_mode", lambda s: (s == "mixed").sum()),
        )
        .reindex(sorted(res_df["tf"].dropna().unique()))
        .fillna(0)
        .astype(int)
        .sort_values("n_interface", ascending=False)
    )
    print("\nInterface residues per TF (by contact mode):")
    print(summary.to_string())

    print("\nInterface residues per TF:")
    cols = [
        "resname",
        "residue_num",
        "contact_mode",
        "contact_freq",
        "frac_base_contact",
        "med_min_dist",
        "med_min_dist_base",
        "med_plddt",
    ]
    for tf, grp in iface.sort_values(
        ["tf", "contact_freq"], ascending=[True, False]
    ).groupby("tf"):
        print(f"\n  {tf} ({len(grp)} interface residues):")
        print(grp[cols].round(2).to_string(index=False))


# %%
# --- Contact geometry along the sequence, vs domain architecture ---
# A 1D "interface map", drawn vertically (sequence running top to bottom): each
# *interface* residue (reliable + confident DNA contact) is a stem whose length
# is its contact frequency (contacts per structure) and whose colour is its
# contact mode (base / backbone / mixed). Transient contacts are dropped -- only
# the residues we call interface are drawn. The InterPro domain architecture
# sits in a track alongside it; a domain
# is highlighted when it actually carries the contacts (>= MIN_DOM_HITS
# interface residues), rather than by a precomputed flag, so tandem zinc-finger
# arrays and bipartite folds (POU) light up honestly. Runs per-TF, or as a
# cross-family panel to contrast geometry across folds.

AA3TO1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}
MODE_COLORS = {"base": "#738fc1", "mixed": "#c8b6d6", "backbone": "#e0a860"}
DOM_CONTACT = "#c0504d"  # domain that carries the DNA contacts
DOM_OTHER = "#cfcfcf"  # other annotated (non-contacting) domain
MIN_DOM_HITS = 2  # interface residues in a domain to call it contacting
DBD_PAD = 25  # residues of flank shown either side of the DBD


def _short_dom(desc):
    """Compact label for an InterPro description so it fits its box; long
    descriptions (esp. tandem zinc fingers) otherwise overflow and collide."""
    rules = [
        ("c2h2", "C2H2"),
        ("gata", "GATA"),
        ("helix-loop-helix", "bHLH"),
        ("basic-leucine zipper", "bZIP"),
        ("leucine zipper", "LZ"),
        ("fork head", "Forkhead"),
        ("ets", "ETS"),
        ("pou-specific", "POU-specific"),
        ("pou domain", "POU C-term"),
        ("engrailed", "engrailed C-term"),
        ("kix", "KIX-binding"),
        ("transactivation", "transactivation"),
        ("t-box", "T-box"),
        ("homeodomain", "Homeodomain"),
        ("pointed", "Pointed"),
        ("early growth response", "EGR"),
        ("transcription regulator myc", "Myc N-term"),
        ("ga-binding protein", "GABPa N-term"),
    ]
    low = desc.lower()
    for key, short in rules:
        if key in low:
            return short
    return desc[:18] + "..." if len(desc) > 20 else desc


def _tf_interface(tf_name, res_df):
    """Interface residues of one TF, deduped to a single (deterministic) gene
    model -- a TF name can map to >1 near-identical DBID in the panel."""
    sub = res_df[res_df["tf"] == tf_name]
    if sub.empty:
        raise ValueError(f"no residues for TF {tf_name!r}")
    if sub["dbid"].nunique() > 1:
        sub = sub[sub["dbid"] == sorted(sub["dbid"].unique())[0]]
    return sub[sub["interface"]].copy()


def _dbd_window(iface, dom, seq_len, pad=DBD_PAD):
    """
    Residue-axis window around the DNA-binding domain(s) +/- pad residues.

    The DBD is data-driven (same rule as the domain track): any InterPro
    domain carrying >= MIN_DOM_HITS interface residues. Spans all such domains
    so bipartite folds (POU) and tandem zinc-finger arrays stay whole. Falls
    back to the interface-residue span, then the whole chain, when no domain
    qualifies. Clamped to [0, seq_len].
    """
    inums = iface["residue_num"].to_numpy()
    starts, ends = [], []
    for _, d in dom.iterrows():
        hits = int(((inums >= d["start"]) & (inums <= d["end"])).sum())
        if hits >= MIN_DOM_HITS:
            starts.append(d["start"])
            ends.append(d["end"])
    if starts:
        lo, hi = min(starts), max(ends)
    elif len(inums):
        lo, hi = int(inums.min()), int(inums.max())
    else:
        lo, hi = 0, seq_len
    return max(0, lo - pad), min(seq_len, hi + pad)


def _draw_geometry(ax, dax, iface, dom, seq_len, label="all"):
    """
    Render one TF's interface map into a (stem-axis, domain-axis) pair.

    The map runs vertically: the sequence is the y-axis (N-term at top), each
    interface residue a horizontal stem growing rightwards, and contact
    frequency the x-axis along the bottom.

    Args:
        ax (plt.Axes): axis for the contact stems.
        dax (plt.Axes): thin axis to its left for the domain track.
        iface (pd.DataFrame): interface residues (from _tf_interface).
        dom (pd.DataFrame): InterPro domains (description, start, end) for
            this TF.
        seq_len (int): protein length, for the shared y-axis.
        label ("all" | "base" | None): which interface residues to annotate
            with their identity ("base" also includes mixed-mode readers).
    """
    xmax = float(iface["contact_freq"].max()) if len(iface) else 1.0
    y0, y1 = _dbd_window(iface, dom, seq_len)  # zoom to the DBD +/- DBD_PAD

    # interface stems, coloured by contact mode
    for _, r in iface.iterrows():
        c = MODE_COLORS.get(r["contact_mode"], "gray")
        ax.hlines(r["residue_num"], 0, r["contact_freq"], color=c, lw=2.2, zorder=3)
        ax.plot(
            r["contact_freq"],
            r["residue_num"],
            "o",
            color=c,
            ms=6,
            mec="black",
            mew=0.5,
            zorder=4,
        )

    # annotate residues in a non-overlapping column tied to each stem by a thin
    # leader line (greedy top-to-bottom minimum gap keeps dense clusters legible)
    to_label = iface
    if label == "base":
        to_label = iface[iface["contact_mode"].isin(["base", "mixed"])]
    elif label is None:
        to_label = iface.iloc[0:0]
    x_lab = xmax * 1.14
    min_gap = (y1 - y0) * 0.014
    prev_y = -1e9
    for _, r in to_label.sort_values("residue_num").iterrows():
        ly = max(r["residue_num"], prev_y + min_gap)
        prev_y = ly
        lab = f"{AA3TO1.get(r['resname'], r['resname'])}{r['residue_num']}"
        ax.plot(
            [r["contact_freq"], x_lab * 0.99],
            [r["residue_num"], ly],
            color="0.6",
            lw=0.4,
            zorder=1,
        )
        ax.text(x_lab, ly, lab, ha="left", va="center", fontsize=5.5)

    ax.set_xlim(0, xmax * 1.55)
    ax.set_ylim(y1, y0)  # N-term at the top
    ax.tick_params(labelleft=False)
    ax.spines[["top", "right"]].set_visible(False)

    # domain track: highlight domains that carry the contacts (data-driven)
    inums = iface["residue_num"].to_numpy()
    for _, d in dom.iterrows():
        hits = int(((inums >= d["start"]) & (inums <= d["end"])).sum())
        color = DOM_CONTACT if hits >= MIN_DOM_HITS else DOM_OTHER
        dax.add_patch(
            plt.Rectangle(
                (0.15, d["start"]),
                0.7,
                d["end"] - d["start"],
                facecolor=color,
                edgecolor="black",
                lw=0.5,
                zorder=2,
            )
        )
        dax.text(
            0.5,
            (d["start"] + d["end"]) / 2,
            _short_dom(d["description"]),
            ha="center",
            va="center",
            rotation=90,
            fontsize=6,
            color="white" if color == DOM_CONTACT else "black",
            zorder=3,
        )
    dax.set_xlim(0, 1)
    dax.set_xticks([])
    dax.set_ylim(y1, y0)
    dax.spines[["top", "right", "bottom"]].set_visible(False)


def _mode_legend(ax):
    """Shared legend: contact modes."""
    from matplotlib.patches import Patch

    handles = [
        Patch(color=MODE_COLORS[m], label=f"{m} contact")
        for m in ("base", "mixed", "backbone")
    ]
    ax.legend(handles=handles, fontsize=7, loc="lower right", frameon=False)


def _seq_len(iface, dom, seqlen_map, tf_name):
    """Protein length for the residue axis: explicit map, else furthest
    annotation."""
    L = (seqlen_map or {}).get(tf_name)
    if L is None:
        L = (
            int(max(iface["residue_num"].max(), dom["end"].max() if len(dom) else 0))
            + 5
        )
    return L


def plot_contact_geometry(tf_name, res_df, doms_df, seqlen_map=None, save=None):
    """One TF: interface map vs domain architecture (see _draw_geometry)."""
    iface = _tf_interface(tf_name, res_df)
    dom = doms_df[doms_df["TF_Name"] == tf_name]
    seq_len = _seq_len(iface, dom, seqlen_map, tf_name)

    fig = plt.figure(figsize=(4.5, 11))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 5], wspace=0.08)
    dax = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharey=dax)
    _draw_geometry(ax, dax, iface, dom, seq_len, label="all")
    ax.set_xlabel("contact frequency\n(fraction of structures)")
    ax.set_title(f"{tf_name}: predicted DNA-contact\ngeometry vs domain architecture")
    dax.set_ylabel("residue position")
    _mode_legend(ax)
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    return fig


# %%
# --- Example: a single TF (klf1, a tandem C2H2 array) ---
if __name__ == "__main__":
    doms_df = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/old_tfdb4_rep_doms.csv")
    # same full-set fallback as the name map, so TFs outside the rep subset get
    # a real length instead of the furthest-annotation guess in _seq_len
    seqlen_map = dict(zip(full_df["TF_Name"], full_df["Sequence"].str.len()))
    seqlen_map.update(zip(tf_df["TF_Name"], tf_df["Sequence"].str.len()))
    tf_to_plot = SINGLE_TF or "sp3a"
    plot_contact_geometry(
        tf_to_plot,
        res_df,
        doms_df,
        seqlen_map,
        save=f"{FIG_DIR}/contact_geometry_{tf_to_plot}.svg",
    )
    plt.show()

# %%
