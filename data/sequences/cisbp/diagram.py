import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from af3tfm.dataset import DS
from af3tfm.utils import set_plotting_style

set_plotting_style()

dataset = DS["cisbp"]
_df = pd.read_csv(dataset.seq_csv)
_msa_map = dict(zip(_df["TF_Name"], _df["MSA_Depth"]))
_pid_map = dict(zip(_df["TF_Name"], _df["Template_PID"]))
_dbid_map = dict(zip(_df["TF_Name"], _df["DBID"].str.lower()))

_templates_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(dataset.seq_csv))),
    "jobs/motif-nonmotif/templates",
)


def _has_dna_template(tf_name):
    dbid = _dbid_map.get(tf_name)
    if not dbid:
        return False
    d = os.path.join(_templates_dir, dbid)
    return os.path.isdir(d) and len(os.listdir(d)) > 0


data = [
    ("Homeodomain", ["pou3f3b", "si:dkey-43p13.5", "hoxd13a", "vsx2", "dharma"], []),
    ("Zinc finger (C2H2)", [], ["sp2", "klf3"]),
    ("Zinc finger (GATA)", [], ["gata3"]),
    ("Zinc finger (CXXC)", ["kdm2aa", "kdm2ba"], []),
    ("Zinc finger (PHD)", ["phf21ab"], []),
    ("Zinc finger (NHR)", ["nr2f2"], []),
    ("T-box", [], ["tbx1", "tbx2a"]),
    ("Paired box", ["pax2b"], ["pax2a"]),
    ("HMG box", [], ["sox2"]),
    ("bHLH", [], ["myod1"]),
    ("bZIP", [], ["jund"]),
    ("Ets", [], ["erg"]),
    ("Forkhead", [], ["foxo1a"]),
    ("MADS-box", [], ["mef2aa"]),
    ("STAT", [], ["stat3"]),
    ("HSF", ["hsf2"], []),
    ("THAP", ["prkrira", "prkrirb"], []),
    ("NDT80", ["myrf"], []),
    ("CXC/CRC", ["lin54"], []),
    ("ARID", ["arid6"], []),
]

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_xlim(0, 15.5)
ax.set_ylim(-1, len(data) + 2)
ax.axis("off")

# Column positions
col_family = 0.5
col_cisbp = 3.5
col_jaspar = 6.0
col_msa = 8.5
col_pid = 11.0
col_dna = 13.5
dot_spacing = 0.35
dot_size = 60

# Header
ax.text(col_family, len(data) + 1, "DBD Family", va="center")
ax.text(col_cisbp, len(data) + 1.2, "CisBP", va="center", ha="center")
ax.text(col_cisbp, len(data) + 0.55, "(PBM)", ha="center")
ax.text(col_jaspar, len(data) + 1.2, "JASPAR", va="center", ha="center")
ax.text(col_jaspar, len(data) + 0.55, "(human ortholog)", va="center", ha="center")
ax.text(col_msa, len(data) + 1.0, "MSA Depth", va="center", ha="center")
ax.text(col_pid, len(data) + 1.2, "Template PID", va="center", ha="center")
ax.text(col_pid, len(data) + 0.55, "(DNA-Binding Domain)", ha="center", va="center")
ax.text(col_dna, len(data) + 1.2, "Template", va="center", ha="center")
ax.text(col_dna, len(data) + 0.55, "with DNA", ha="center", va="center")
ax.plot([0, 15.5], [len(data) + 0.25, len(data) + 0.25], linewidth=1)

# Data rows
for i, (family, cisbp, jaspar) in enumerate(data):
    y = len(data) - i - 0.5

    # Alternating background
    if i % 2 == 0:
        ax.add_patch(
            mpatches.Rectangle(
                (0, y - 0.4), 15.5, 0.8, facecolor="#f8fafc", edgecolor="none", zorder=0
            )
        )

    # Family name
    ax.text(col_family, y, family, va="center")

    # CisBP dots
    if cisbp:
        start_x = col_cisbp - (len(cisbp) - 1) * dot_spacing / 2
        x_coords = [start_x + j * dot_spacing for j in range(len(cisbp))]
        ax.scatter(
            x_coords,
            [y] * len(cisbp),
            s=dot_size,
            color="#334155",
            edgecolors="none",
            zorder=3,
        )
    else:
        ax.text(col_cisbp, y, "—", va="center", ha="center", color="lightgray")

    # JASPAR dots
    if jaspar:
        start_x = col_jaspar - (len(jaspar) - 1) * dot_spacing / 2
        x_coords = [start_x + j * dot_spacing for j in range(len(jaspar))]
        ax.scatter(
            x_coords,
            [y] * len(jaspar),
            s=dot_size,
            color="#334155",
            edgecolors="none",
            zorder=3,
        )
    else:
        ax.text(col_jaspar, y, "—", va="center", ha="center", color="lightgray")

    # Avg MSA depth
    all_tfs = cisbp + jaspar
    depths = [_msa_map[tf] for tf in all_tfs if tf in _msa_map]
    if depths:
        avg = sum(depths) / len(depths)
        rounded = round(avg / 1000)
        if avg > 30000:
            label = f"High ({rounded}K)"
        elif avg >= 10000:
            label = f"Medium ({rounded}K)"
        else:
            label = f"Low ({rounded}K)"
        ax.text(col_msa, y, label, va="center", ha="center")
    else:
        ax.text(col_msa, y, "—", va="center", ha="center", color="lightgray")

    # Avg Template PID
    pids = [_pid_map[tf] for tf in all_tfs if tf in _pid_map]
    if pids:
        avg_pid = sum(pids) / len(pids)
        rounded_pid = round(avg_pid)
        if avg_pid >= 80:
            label_pid = f"High ({rounded_pid}%)"
        else:
            label_pid = f"Low ({rounded_pid}%)"
        ax.text(col_pid, y, label_pid, va="center", ha="center")
    else:
        ax.text(col_pid, y, "—", va="center", ha="center", color="lightgray")

    # Template with DNA
    has_dna = any(_has_dna_template(tf) for tf in all_tfs)
    dna_label = "Yes" if has_dna else "No"
    dna_color = "#334155" if has_dna else "lightgray"
    ax.text(col_dna, y, dna_label, va="center", ha="center", color=dna_color)


plt.tight_layout()
plt.savefig("dbd_family_matrix.svg", bbox_inches="tight", dpi=300)
plt.show()
