# %%
import io
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import molview as mv
import numpy as np
import pandas as pd
from Bio.PDB import PDBIO, MMCIFParser, Polypeptide, Select

from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.jobdb import Database
from af3tfm.tfactor import load_tfs
from af3tfm.utils import set_plotting_style

parser = MMCIFParser()
set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Adjacent DNA-binding domains separated by this many residues or fewer (e.g. the
# linkers of a C2H2 zinc-finger array) are treated as a single region.
DBD_MERGE_GAP = 20


class ProteinRangesWithDNA(Select):
    """Keep DNA/ligand plus protein residues falling inside any of `ranges`."""

    def __init__(self, ranges):
        self.ranges = ranges

    def accept_residue(self, residue):
        # Filter out solvent molecules
        if residue.id[0] == "W" or residue.resname in ["HOH", "WAT"]:
            return 0

        # Check if the residue is a standard Amino Acid (Protein)
        if Polypeptide.is_aa(residue, standard=True):
            resid = residue.id[1]
            if any(start <= resid <= end for start, end in self.ranges):
                return 1
            else:
                return 0

        # If it's NOT an amino acid and NOT water, it's likely DNA/RNA/Ligand.
        return 1


def get_structure(name, filepath, ranges):
    structure = parser.get_structure(name, filepath)
    outfile = io.StringIO()
    io_saver = PDBIO()
    io_saver.set_structure(structure)
    io_saver.save(outfile, select=ProteinRangesWithDNA(ranges))
    subset_pdb_block = outfile.getvalue()
    outfile.close()

    return subset_pdb_block


# Load TF information
dataset = DS["cisbp"]
tf_df = pd.read_csv(dataset.seq_csv)
doms_df = pd.read_csv(dataset.dom_csv)
tfs = load_tfs(tf_df, name="DBID")

# Read job database
run = "motif-nonmotif-genomic"
db = Database(
    job_path=f"{DATA_DIR}/jobs/{run}",
    dataset=dataset.name,
)

# Query database to get ipTM scores and collect by motif.
_NEG_RE = re.compile(r"^(?P<base>.+)_neg\d+$")

query = """
SELECT structure_path, motif, ipTM, dom_pLDDT, dna_pLDDT, iface_PAE
FROM runs
"""
motif_iptm = defaultdict(list)
for structure_path, motif, iptm, dom_plddt, dna_plddt, iface_pae in db.cursor.execute(
    query
).fetchall():
    neg = _NEG_RE.match(motif)
    key = f"{neg.group('base')}-non" if neg else motif
    motif_iptm[key].append((iptm, structure_path, dom_plddt, dna_plddt, iface_pae))

db.conn.close()


# %%
_MEDIAN_COLS = ["ipTM", "structure_path", "dom_pLDDT", "dna_pLDDT", "iface_PAE"]


def _get_motif_median_path(tf_name):
    """Return (motif_path, motif_iptm, motif_iface_pae) at the median rank for a TF's motif-bound structure."""
    motif_df = pd.DataFrame(motif_iptm[tf_name], columns=_MEDIAN_COLS)
    motif_df = motif_df.sort_values(
        by=["ipTM", "dom_pLDDT"], ascending=[False, False]
    ).reset_index(drop=True)
    mid = len(motif_df) // 2
    return (
        motif_df.loc[mid, "structure_path"],
        motif_df.loc[mid, "ipTM"],
        motif_df.loc[mid, "iface_PAE"],
    )


def _get_non_motif_median_path(tf_name):
    """Return (non_motif_path, non_motif_iptm, non_motif_iface_pae) at the median rank for a TF's non-motif-bound structure."""
    non_motif_df = pd.DataFrame(motif_iptm[f"{tf_name}-non"], columns=_MEDIAN_COLS)
    non_motif_df = non_motif_df.sort_values(
        by=["ipTM", "dom_pLDDT"], ascending=[False, False]
    ).reset_index(drop=True)
    mid = len(non_motif_df) // 2
    return (
        non_motif_df.loc[mid, "structure_path"],
        non_motif_df.loc[mid, "ipTM"],
        non_motif_df.loc[mid, "iface_PAE"],
    )


def _get_median_paths(tf_name):
    """Return median-rank motif and non-motif (path, ipTM, iface_PAE) values."""
    motif_path, motif_iptm_val, motif_ipae_val = _get_motif_median_path(tf_name)
    non_motif_path, non_motif_iptm_val, non_motif_ipae_val = _get_non_motif_median_path(
        tf_name
    )
    print(motif_path)
    print(non_motif_path)
    return (
        motif_path,
        motif_iptm_val,
        motif_ipae_val,
        non_motif_path,
        non_motif_iptm_val,
        non_motif_ipae_val,
    )


def _get_dbd_ranges(tf_name, merge_gap=DBD_MERGE_GAP):
    """Return a TF's DNA-binding domain ranges as sorted (start, end) tuples.

    TFs such as zinc-finger arrays carry several DBD annotations. Ranges
    separated by `merge_gap` residues or fewer are merged into one region so
    that short inter-domain linkers stay with the domains they connect.
    """
    tf_doms = doms_df[(doms_df["TF_Name"] == tf_name) & (doms_df["DNA-Binding"])]
    ranges = sorted(
        (int(row.start), int(row.end)) for row in tf_doms.itertuples(index=False)
    )

    merged = []
    for start, end in ranges:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    return merged


def get_median_structures(tf_name):
    """Return median-rank motif and non-motif (pdb, ipTM, iface_PAE) for a TF."""
    dbd_ranges = _get_dbd_ranges(tf_name)

    (
        motif_path,
        motif_iptm_val,
        motif_ipae_val,
        non_motif_path,
        non_motif_iptm_val,
        non_motif_ipae_val,
    ) = _get_median_paths(tf_name)

    motif_struct = get_structure(tf_name, motif_path, dbd_ranges)
    non_motif_struct = get_structure(tf_name, non_motif_path, dbd_ranges)

    return (
        motif_struct,
        motif_iptm_val,
        motif_ipae_val,
        non_motif_struct,
        non_motif_iptm_val,
        non_motif_ipae_val,
    )


def _get_plddt(structure_path):
    """Return pLDDT scores from a CIF file."""
    p = MMCIFParser()
    p.get_structure("model", structure_path)
    return [float(x) for x in p._mmcif_dict["_ma_qa_metric_local.metric_value"]]


def _get_dna_start_idx(structure_path):
    """Return the index in the pLDDT array where the first non-protein chain begins (i.e. DNA)."""
    p = MMCIFParser()
    p.get_structure("model", structure_path)
    chain_ids = p._mmcif_dict.get("_ma_qa_metric_local.label_asym_id", [])
    if not chain_ids:
        return None
    first_chain = chain_ids[0]
    for i, chain_id in enumerate(chain_ids):
        if chain_id != first_chain:
            return i
    return None


def plot_tf_comparison(tf_name):
    """Plot median motif-bound vs non-motif-bound structures for a TF side by side."""
    (
        motif_struct,
        motif_iptm_val,
        motif_ipae_val,
        non_motif_struct,
        non_motif_iptm_val,
        non_motif_ipae_val,
    ) = get_median_structures(tf_name)

    v = mv.view(viewergrid=(1, 2), width=1000, height=600, panel=True, resolution=20)
    v.addModel(
        motif_struct,
        viewer=(0, 0),
        name=f"{tf_name.capitalize()} motif (iptm={motif_iptm_val:.2f}, iPAE={motif_ipae_val:.2f})",
    )
    v.addModel(
        non_motif_struct,
        viewer=(0, 1),
        name=f"{tf_name.capitalize()} non-motif (iptm={non_motif_iptm_val:.2f}, iPAE={non_motif_ipae_val:.2f})",
    )
    v.setColorMode("plddt")
    v.setIllustrativeStyle(True)
    v.removeSolvent(True)
    v.show()


def plot_plddt_comparison(tf_name):
    """Plot pLDDT for the median motif vs non-motif structure of a TF.

    Green dashed lines mark the boundaries of every DBD the TF carries; a blue
    dashed line marks where the DNA chain begins in the pLDDT array. If there is
    a large gap between the last DBD and DNA, a broken axis is used to skip it
    (starting 50 positions after that DBD's end).
    """
    motif_path, _, _, non_motif_path, _, _ = _get_median_paths(tf_name)

    dbd_ranges = _get_dbd_ranges(tf_name)
    print(dbd_ranges)

    plddt_motif = _get_plddt(motif_path)
    plddt_non = _get_plddt(non_motif_path)

    # Average across every DBD position, so multiple DBDs contribute jointly
    dbd_idx = sorted({i for start, end in dbd_ranges for i in range(start - 1, end)})
    avg_dbd_plddt_motif = np.mean([plddt_motif[i] for i in dbd_idx])
    avg_dbd_plddt_non = np.mean([plddt_non[i] for i in dbd_idx])

    # Average pLDDT across the DNA chain (everything from its own chain start),
    # computed per structure since motif and non-motif DNA can differ in length
    avg_dna_plddt_motif = np.mean(plddt_motif[_get_dna_start_idx(motif_path) :])
    avg_dna_plddt_non = np.mean(plddt_non[_get_dna_start_idx(non_motif_path) :])

    first_start, last_end = dbd_ranges[0][0], dbd_ranges[-1][1]
    trim = max(0, first_start - 10)
    dna_start = _get_dna_start_idx(motif_path)

    break_start = last_end + 50
    break_end = dna_start
    use_break = break_start < break_end

    x_motif = np.arange(trim, len(plddt_motif))
    x_non = np.arange(trim, len(plddt_non))

    def _draw_on(ax):
        ax.plot(
            x_motif,
            plddt_motif[trim:],
            label=(
                f"{tf_name.capitalize()} motif "
                f"(DBD pLDDT={avg_dbd_plddt_motif:.2f}, "
                f"DNA pLDDT={avg_dna_plddt_motif:.2f})"
            ),
            color="#738fc1",
        )
        ax.plot(
            x_non,
            plddt_non[trim:],
            label=(
                f"{tf_name.capitalize()} non-motif "
                f"(DBD pLDDT={avg_dbd_plddt_non:.2f}, "
                f"DNA pLDDT={avg_dna_plddt_non:.2f})"
            ),
            color="#d56c55",
        )
        for dom_start, dom_end in dbd_ranges:
            ax.axvline(dom_start, color="#6fc276", linestyle="--")
            ax.axvline(dom_end, color="#6fc276", linestyle="--")
            ax.axvspan(dom_start, dom_end, color="#6fc276", alpha=0.1)
        ax.axvline(dna_start, color="#4c72b0", linestyle="--")
        ax.axvspan(dna_start, len(plddt_motif), color="#4c72b0", alpha=0.1)

    def _label_dbds(ax, y):
        """Label each DBD region; numbered when the TF has more than one."""
        for i, (dom_start, dom_end) in enumerate(dbd_ranges, start=1):
            label = "DNA-Binding Domain" if len(dbd_ranges) == 1 else f"DBD {i}"
            ax.text(
                (dom_start + dom_end) / 2,
                y,
                label,
                color="#4a8c52",
                ha="center",
                va="bottom",
                clip_on=False,
            )

    if use_break:
        left_range = break_start - trim
        right_range = len(plddt_motif) - break_end
        fig, (ax_l, ax_r) = plt.subplots(
            1,
            2,
            sharey=True,
            figsize=(8, 5),
            gridspec_kw={"width_ratios": [left_range, right_range], "wspace": 0.05},
        )
        _draw_on(ax_l)
        _draw_on(ax_r)

        ax_l.set_xlim(trim, break_start)
        ax_r.set_xlim(break_end, len(plddt_motif))

        # Hide inner spines and left-axis ticks on right panel
        ax_l.spines["right"].set_visible(False)
        ax_l.spines["top"].set_visible(False)
        ax_r.spines["left"].set_visible(False)
        ax_r.spines["top"].set_visible(False)
        ax_r.tick_params(left=False)

        # Diagonal break marks at the bottom edge only (~45 degrees)
        kw = dict(
            marker=[(-1, -1), (1, 1)],
            markersize=10,
            linestyle="none",
            color="k",
            mec="k",
            mew=1,
            clip_on=False,
        )
        ax_l.plot([1], [0], transform=ax_l.transAxes, **kw)
        ax_r.plot([0], [0], transform=ax_r.transAxes, **kw)

        # Region labels on the axis where they're visible
        _label_dbds(ax_l, 25)
        ax_r.text(
            (dna_start + len(plddt_motif)) / 2,
            25,
            "DNA",
            color="#4c72b0",
            ha="center",
            va="bottom",
            clip_on=False,
        )

        ax_l.set_ylabel("pLDDT")
        ax_l.legend()
        fig.supxlabel("Position (Residues and Bases)")
        fig.suptitle(f"pLDDT scores for {tf_name.capitalize()} structures", y=1.05)
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        _draw_on(ax)
        _label_dbds(ax, 45)
        ax.text(
            (dna_start + len(plddt_motif)) / 2,
            45,
            "DNA",
            color="#4c72b0",
            ha="center",
            va="bottom",
            clip_on=False,
        )
        ax.set_xlim(trim, len(plddt_motif))
        ax.set_xlabel("Position (Residues and Bases)")
        ax.set_ylabel("pLDDT")
        ax.set_title(f"pLDDT scores for {tf_name.capitalize()} structures", y=1.05)
        ax.spines["top"].set_visible(False)
        ax.legend()

    plt.tight_layout()
    plt.savefig(
        f"{FIG_DIR}/{tf_name}_plddt_comparison.svg", dpi=300, bbox_inches="tight"
    )
    plt.show()


# %%
plot_tf_comparison("sp2")
plot_plddt_comparison("sp2")

# %%
plot_tf_comparison("pax2a")
plot_plddt_comparison("pax2a")

# %%
plot_tf_comparison("pax2b")
plot_plddt_comparison("pax2b")

# %%
plot_tf_comparison("prkrira")
plot_plddt_comparison("prkrira")

# %%
