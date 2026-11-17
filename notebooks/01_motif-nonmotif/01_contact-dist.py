"""
Here we ask whether a TF's motif vs non-motif separability (AUROC, computed as
in 01_conf-plots.py) relates to how much its protein-DNA contacts tighten when
the correct (motif) sequence is bound.

Ben Iovino 7/15/26 Biohub
"""

# %%
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.metrics import auc, roc_curve

from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.jobdb import Database
from af3tfm.utils import AMINO_ACIDS, set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "01_motif-nonmotif" / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Interface = residues within this many angstroms of DNA in either condition
CONTACT_CUTOFF = 5

# Load TF information
dataset = DS["cisbp"]
tf_df = pd.read_csv(dataset.seq_csv)

# Read job database
run = "motif-nonmotif-genomic"
db = Database(job_path=f"{DATA_DIR}/jobs/{run}", dataset=dataset.name)
runs_df = pd.read_sql("SELECT protein, motif, ipTM, dom_pLDDT FROM runs", db.conn)
db.conn.close()

# Genomic controls have a trailing "_negNNN" in their motif label.
runs_df["is_non"] = runs_df["motif"].str.contains("_neg")
runs_df["tf"] = runs_df["motif"].str.replace(r"_neg\d+$", "", regex=True)
print(
    f"Run: {run}; {len(runs_df)} predictions "
    f"({(~runs_df['is_non']).sum()} motif, {runs_df['is_non'].sum()} non-motif)",
    flush=True,
)


def get_scores(tf_name: str) -> tuple[list, list]:
    """
    Return (motif_scores, nonmotif_scores) for a TF under the rank score.

    Args:
        tf_name (str): Base TF name with any "_negNNN" suffix stripped

    Returns:
        tuple[list, list]: (motif_scores, nonmotif_scores)
    """

    sub = runs_df[runs_df["tf"] == tf_name]
    score = np.sqrt(sub["ipTM"].rank(pct=True) * sub["dom_pLDDT"].rank(pct=True))
    motif_scores = score[~sub["is_non"]].tolist()
    nonmotif_scores = score[sub["is_non"]].tolist()
    return motif_scores, nonmotif_scores


# %%
# Per-TF separation metric: AUROC on all sequences (motif=1, non-motif=0),
sep_rows = []
for tf_name in tf_df["TF_Name"]:
    motif_scores, nonmotif_scores = get_scores(tf_name)
    if len(motif_scores) == 0 or len(nonmotif_scores) == 0:
        continue

    fpr, tpr, _ = roc_curve(
        [1] * len(motif_scores) + [0] * len(nonmotif_scores),
        motif_scores + nonmotif_scores,
    )
    sep_rows.append({"TF_Name": tf_name, "AUROC": auc(fpr, tpr)})

sep_df = pd.DataFrame(sep_rows)


# %%
# Get per-residue mean nearest-DNA distance in motif vs non-motif structures
BACKBONE_ATOMS = {  # Nucleobase atoms only (exclude phosphate + sugar backbone)
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

# Exclude main-chain backbone atoms; keep CB and all side-chain atoms
MAINCHAIN_ATOMS = {"N", "CA", "C", "O"}
NUCLEIC_RESIDUES = {"DA", "DC", "DG", "DT"}


def residue_min_dists(cif_path: str) -> dict:
    """
    Nearest-nucleobase distance for every protein residue in one structure.

    Args:
        cif_path (str): Path to the AF3 model CIF

    Returns:
        dict: {residue_num: (resname, min_dist)} — uncensored distance from the
              residue's closest side-chain atom to the closest nucleobase atom
              (empty dict if the structure can't be parsed or lacks protein/DNA)
    """

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("s", cif_path)
    except Exception:
        return {}

    prot_coords, prot_resnums, resnum_to_resname = [], [], {}
    nuc_coords = []

    for model in structure:
        for chain in model:
            for residue in chain:
                res_name = residue.get_resname().strip()
                if res_name in AMINO_ACIDS:
                    res_num = residue.get_id()[1]
                    resnum_to_resname[res_num] = res_name
                    for atom in residue:
                        if atom.get_name() not in MAINCHAIN_ATOMS:
                            prot_coords.append(atom.coord)
                            prot_resnums.append(res_num)
                elif res_name in NUCLEIC_RESIDUES:
                    for atom in residue:
                        if atom.get_name() not in BACKBONE_ATOMS:
                            nuc_coords.append(atom.coord)

    if not prot_coords or not nuc_coords:
        return {}

    prot_arr = np.array(prot_coords)
    prot_resnums = np.array(prot_resnums)

    # Nearest nucleobase atom for each protein atom, then min per residue
    tree = cKDTree(np.array(nuc_coords))
    nearest, _ = tree.query(prot_arr, k=1)

    out = {}
    for res_num in np.unique(prot_resnums):
        d = float(nearest[prot_resnums == res_num].min())
        out[int(res_num)] = (resnum_to_resname[int(res_num)], d)
    return out


def process_tf(args: tuple) -> list[dict]:
    """
    Accumulate per-residue mean nearest-DNA distance for one TF, split into
    motif and non-motif predictions.

    Args:
        args: (dbid, predictions) where predictions is a list of
              (struct_path, is_non) tuples for that TF

    Returns:
        List of dicts, one per residue, with keys: dbid, residue_num, resname,
        mean_dist_motif, mean_dist_non, n_motif, n_non
    """

    dbid, predictions = args

    # residue -> [sum_dist, count] for motif and non-motif separately
    motif_acc = defaultdict(lambda: [0.0, 0])
    non_acc = defaultdict(lambda: [0.0, 0])
    resnames = {}

    for struct_path, is_non in predictions:
        if not Path(struct_path).exists():
            continue
        dists = residue_min_dists(struct_path)
        acc = non_acc if is_non else motif_acc
        for res_num, (resname, d) in dists.items():
            acc[res_num][0] += d
            acc[res_num][1] += 1
            resnames[res_num] = resname

    rows = []
    for res_num in sorted(set(motif_acc) | set(non_acc)):
        m_sum, m_n = motif_acc[res_num]
        n_sum, n_n = non_acc[res_num]
        rows.append(
            {
                "dbid": dbid,
                "residue_num": res_num,
                "resname": resnames.get(res_num, "UNK"),
                "mean_dist_motif": round(m_sum / m_n, 3) if m_n else np.nan,
                "mean_dist_non": round(n_sum / n_n, 3) if n_n else np.nan,
                "n_motif": m_n,
                "n_non": n_n,
            }
        )
    return rows


def build_tightening() -> pd.DataFrame:
    """
    Parse every prediction and return the per-(dbid, residue) distance table.

    Returns:
        pd.DataFrame with columns dbid, residue_num, resname, mean_dist_motif,
        mean_dist_non, n_motif, n_non
    """

    db = Database(job_path=f"{DATA_DIR}/jobs/{run}", dataset=dataset.name)
    rows_db = db.cursor.execute(
        "SELECT protein, motif, structure_path FROM runs"
    ).fetchall()
    db.conn.close()

    # Group predictions by TF, tagging motif vs non-motif from the motif column
    tf_jobs = defaultdict(list)
    for protein, motif, struct_path in rows_db:
        tf_jobs[protein].append((struct_path, "_neg" in motif))
    print(f"Extracting {len(tf_jobs)} TFs, {len(rows_db)} predictions...")

    all_rows = []
    with ProcessPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(process_tf, a) for a in tf_jobs.items()]
        for completed, future in enumerate(as_completed(futures), start=1):
            all_rows.extend(future.result())
            print(f"Processed {completed}/{len(futures)} TFs", flush=True)
    return pd.DataFrame(all_rows)


tightening_csv = Path(f"{DATA_DIR}/jobs/{run}/contact_tightening.csv")
if tightening_csv.exists():
    tight = pd.read_csv(tightening_csv)
else:
    tight = build_tightening()
    tight.to_csv(tightening_csv, index=False)

tight["delta"] = tight["mean_dist_non"] - tight["mean_dist_motif"]

# Interface residues: close to DNA in either condition (symmetric selection so
# delta is not conditioned on the motif distance)
iface = tight[tight[["mean_dist_motif", "mean_dist_non"]].min(axis=1) <= CONTACT_CUTOFF]
contact_df = (
    iface.groupby("dbid")
    .agg(tightening=("delta", "mean"), n_iface=("delta", "size"))
    .reset_index()
)

# Map dbid -> TF_Name to join onto the separation metrics. Build this from the
# job db (protein -> base motif name) rather than tf_df's DBID column, which can
# drift out of sync with the run predictions (e.g. sp2's Ensembl ID changed).
dbid_to_name = dict(zip(runs_df["protein"].str.upper(), runs_df["tf"]))
contact_df["TF_Name"] = contact_df["dbid"].str.upper().map(dbid_to_name)

df = sep_df.merge(contact_df, on="TF_Name", how="inner")
df.to_csv(DATA_DIR / "jobs" / run / "contact_dist_corr.csv", index=False)
print(f"{len(df)} TFs with AUROC and tightening")
print(
    df[["TF_Name", "AUROC", "tightening", "n_iface"]]
    .round(3)
    .sort_values("tightening")
    .to_string()
)


# %%
# Correlate AUROC against contact tightening
x = df["AUROC"].to_numpy(dtype=float)
y = df["tightening"].to_numpy(dtype=float)
slope, intercept, r, p_value, std_err = stats.linregress(x, y)
print(f"Pearson r = {r:.3f}, p = {p_value:.2e} (n={len(x)})")

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(x, y, color="#738fc1", linewidth=0.5, edgecolor="black", alpha=0.8, s=80)
ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8)
ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)

# Label a few TFs of interest (outliers and a well-behaved reference)
label_tfs = ["prkrira", "sp2", "foxo1a", "gata3"]
for _, row in df[df["TF_Name"].isin(label_tfs)].iterrows():
    ax.annotate(
        row["TF_Name"],
        xy=(row["AUROC"], row["tightening"]),
        xytext=(5, 4),
        textcoords="offset points",
    )

# Regression line
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(
    x_line, slope * x_line + intercept, color="#ef4444", linestyle="--", linewidth=2
)

ax.text(
    0.05,
    0.95,
    f"Pearson r = {r:.2f}\np-value = {p_value:.2e}",
    transform=ax.transAxes,
    fontsize=9,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

ax.set_xlabel("AUROC (motif vs non-motif)")
ax.set_ylabel("Interface contact distance (Å, non-motif − motif)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/contact_dist_corr.svg", dpi=300)
plt.show()

# %%
