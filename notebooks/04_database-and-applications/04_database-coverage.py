"""
Compare TF and DNA-binding-domain (DBD) coverage across databases.

Ben Iovino 6/26/26 Biohub
"""

# %%
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
from matplotlib_venn import venn3, venn3_circles

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.utils import set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "04_database-and-applications" / "tables"
TAB_DIR.mkdir(exist_ok=True)


# %%
# Canonical DBD class -> {interpro[], animaltfdb[], jaspar_class[] or jaspar_family[], cisbp[]}
CROSSWALK_YAML = "../dbd_crosswalk.yaml"
with open(CROSSWALK_YAML) as f:
    CROSSWALK = yaml.safe_load(f)

# Build native-term to canonical lookups for each resource.
animaltfdb2canon, jaspar_class2canon, jaspar_fam2canon, cisbp2canon = {}, {}, {}, {}
for canon, d in CROSSWALK.items():
    for x in d.get("animaltfdb", []):
        animaltfdb2canon[x] = canon
    for x in d.get("jaspar_class", []):
        jaspar_class2canon[x] = canon
    for x in d.get("jaspar_family", []):
        jaspar_fam2canon[x] = canon
    for x in d.get("cisbp", []):
        cisbp2canon[x] = canon

CANON_ORDER = list(CROSSWALK.keys())
UNCLASSIFIED = "Other / unclassified"

# ZebraMotif blue (light and dark), JASPAR green, CisBP gold.
PALETTE = {
    "Direct": "#2a7a2a",  # experimental gold standard
    "Inferred": "#d9a441",  # cross-species homology transfer
    "Predicted": "#2c6fbb",  # in-silico structure, protein-specific
    "Transferred": "#9ec6e8",  # predicted PWM carried rep -> cluster member
}
RES = ["zebramotif_zf", "jaspar_vert", "cisbp_zf"]  # also the venn's A, B, C order
BAR_ORDER = ["zebramotif_zf", "jaspar_vert", "cisbp_zf"]  # top -> bottom in each group
RES_LABELS = {
    "zebramotif_zf": "ZebraMotif (zf)",
    "jaspar_vert": "JASPAR (vert)",
    "cisbp_zf": "CisBP (zf)",
}
RES_COLORS = {
    "zebramotif_zf": PALETTE["Predicted"],
    "jaspar_vert": PALETTE["Direct"],
    "cisbp_zf": PALETTE["Inferred"],
}
XLIM = (0.7, 1300)
DROP_CLASSES = ["zf-CXXC"]  # cofactors in AnimalTFDB4, not TFs


def to_canon(families: pd.Series) -> pd.Series:
    """Map AnimalTFDB family labels onto canonical DBD classes.

    Args:
        families (pd.Series): native AnimalTFDB family labels

    Returns:
        pd.Series: canonical class per row, unmapped folded to UNCLASSIFIED
    """
    return families.map(animaltfdb2canon).fillna(UNCLASSIFIED)


def label_bars(ax, ys: Iterable, values: Iterable, color: str) -> None:
    """Write each bar's count just past its tip.

    Args:
        ax: axes to annotate
        ys (iterable): bar centers
        values (iterable): bar tip positions, i.e. the numbers to print
        color (str): text color
    """
    for yi, v in zip(ys, values):
        if v > 0:
            ax.text(
                v * 1.12,
                yi,
                f"{int(v)}",
                va="center",
                ha="left",
                fontsize=8,
                color=color,
            )


# %%
# ZebraMotif predicted coverage: representatives with their own structure-derived PWM.
modes = pd.read_parquet(f"{DATA_DIR}/database/pwm_modes.parquet")
modes = modes[~modes["DBID"].astype(str).str.contains(r"_\d+$")]
zebramotif = modes.drop_duplicates("DBID")[["DBID", "Family"]].copy()
zebramotif["canon"] = to_canon(zebramotif["Family"])
zebramotif_counts = zebramotif.groupby("canon")["DBID"].nunique()
reps_with_pwm = set(zebramotif["DBID"].astype(str))
n_predicted = len(reps_with_pwm)
print(
    f"zebramotif: {n_predicted} predicted TFs across "
    f"{zebramotif_counts[zebramotif_counts > 0].size} canonical classes"
)

# %%
# ZebraMotif transferred coverage: cluster members that inherit their rep's PWM,
# counted only when Rep_DBID actually has a PWM in pwm_modes. Exclude members that
# already have their own predicted PWM (a handful are non-rep members of another
# cluster yet still carry a mode of their own) so they aren't double-counted as both
# predicted and transferred — matches the Venn's union of the two sets.
rep_map = pd.read_parquet(f"{DATA_DIR}/database/tf_to_rep.parquet")
rep_map = rep_map[~rep_map["DBID"].astype(str).str.contains(r"_\d+$")]
transferred = rep_map[
    (~rep_map["is_representative"])
    & rep_map["Rep_DBID"].astype(str).isin(reps_with_pwm)
    & ~rep_map["DBID"].astype(str).isin(reps_with_pwm)
].copy()
transferred["canon"] = to_canon(transferred["Family"])
transferred_counts = transferred.groupby("canon")["DBID"].nunique()
n_transferred = int(transferred_counts.sum())
print(
    f"transferred: {n_transferred} member TFs across "
    f"{transferred_counts[transferred_counts > 0].size} canonical classes"
)

# %%
# JASPAR 2026 CORE
JASPAR_TSV = "../jaspar_core.tsv"  # downloaded from mencius.uio.no JASPAR_metadata/2026
jas = pd.read_csv(JASPAR_TSV, sep="\t")
jas = jas[jas["tax_group"] == "vertebrates"].copy()

# dedupe to one row per TF lineage (base_id collapses matrix versions)
jas = jas.drop_duplicates("base_id")

# Classes deliberately keyed on the finer `family` label instead; they are not misses.
split_classes = {
    "Tryptophan cluster factors",
    "Fork head/winged helix factors",
    "High-mobility group (HMG) domain factors",
    "Other C4 zinc finger-type factors",
}
jas["base_class"] = (
    jas["class"].str.split("::").str[0]
)  # take primary of composite classes
jas["canon"] = jas.apply(
    lambda r: jaspar_fam2canon.get(r["family"])
    or jaspar_class2canon.get(r["base_class"], UNCLASSIFIED),
    axis=1,
)
# unique TFs = unique uniprot (fallback to name when uniprot missing)
jas["tf_key"] = jas["uniprot_ids"].fillna(jas["name"])
jaspar_counts = jas.groupby("canon")["tf_key"].nunique()
print(
    f"jaspar(vert): {jas['tf_key'].nunique()} TFs across {jaspar_counts[jaspar_counts>0].size} canonical classes"
)

# %%
# CisBP — Danio TFs with a motif, filtered from the whole-DB bulk dump
cis = pd.read_csv(f"{DATA_DIR}/sequences/cisbp/TF_Information.txt", sep="\t")
cis = cis.rename(
    columns={"DBID": "Gene_ID", "Family_Name": "Family", "TF_Status": "Status"}
)
cis["canon"] = cis["Family"].map(cisbp2canon).fillna(UNCLASSIFIED)

# Resolve CisBP ids to the current zebrafish ENSDARG and collapse to one row per gene
# up front, so every downstream count (evidence chart, coverage table, overlap Venn)
# reports the same post-merge TF number
sym2cur = {}
for _fn in ["tfdb4_tfs.csv", "tfdb4_rep_tfs.csv", "tfdb4_val_tfs.csv"]:
    _d = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/{_fn}")
    for _dbid, _name in zip(_d["DBID"].astype(str), _d["TF_Name"].astype(str)):
        sym2cur.setdefault(_name.lower(), _dbid)
_cur_universe = set(sym2cur.values())
_cis_native = {str(g): str(n).lower() for g, n in zip(cis["Gene_ID"], cis["TF_Name"])}

# CisBP ids outside the AnimalTFDB4 universe that Ensembl REST /lookup/id no longer
# resolves (checked 9/22/26). Mostly unnamed clone genes whose symbols can't be remapped,
# so they would otherwise survive as stale ids and inflate the CisBP-only region.
RETIRED_CISBP_ENS = {
    "ENSDARG00000018913",  # hmbox1b
    "ENSDARG00000091086",  # alx3
    "ENSDARG00000091603",  # CABZ01066696.1 (bHLH)
    "ENSDARG00000098299",  # FQ377654.1 (CxxC)
    "ENSDARG00000098472",  # CR936540.2 (Ets)
    "ENSDARG00000100408",  # FQ377628.1 (Sox)
    "ENSDARG00000103592",  # CABZ01102124.1 (Runt)
    "ENSDARG00000104156",  # CABZ01017151.1 (Homeodomain)
    "ENSDARG00000104196",  # FQ311895.1 (Nuclear receptor)
    "ENSDARG00000104578",  # CABZ01081359.1 (CxxC)
}


def _resolve_cisbp_ens(gene_id: str, tf_name: str) -> str | None:
    """Current ENSDARG for a CisBP row: keep the id if still current, else map via symbol."""
    gene_id = str(gene_id)
    if not gene_id.startswith("ENSDARG") or gene_id in RETIRED_CISBP_ENS:
        return None
    if gene_id in _cur_universe:
        return gene_id
    name = str(tf_name).lower()
    remapped = sym2cur.get(name)
    if remapped is None:
        return gene_id  # unmappable symbol: keep the stale id
    # Refuse the remap if the target is natively a different CisBP gene (name conflict);
    # keep this row's own id so distinct TFs aren't collapsed. Legit stale duplicates
    # (same name under current + retired id) still merge.
    if _cis_native.get(remapped, name) != name:
        return gene_id
    return remapped


cis["ens"] = [_resolve_cisbp_ens(g, n) for g, n in zip(cis["Gene_ID"], cis["TF_Name"])]
_n_unresolved = cis["ens"].isna().sum()
cis = cis.dropna(subset=["ens"]).drop_duplicates(subset=["ens"])
print(
    f"cisbp: dropped {_n_unresolved} non-ENSDARG or retired-id row(s), "
    "merged to one row per current id"
)

cisbp_counts = cis.groupby("canon")["ens"].nunique()
# D is motif from a direct experiment on the zebrafish TF, I is inferred by homology from another species
cisbp_status = cis.groupby("Status")["ens"].nunique()
print(
    f"cisbp(zf): {cis['ens'].nunique()} TFs across {cisbp_counts[cisbp_counts>0].size} canonical classes"
)

# %%
# Unified coverage table
cov = (
    pd.DataFrame(
        {
            "zebramotif_zf": zebramotif_counts,
            "jaspar_vert": jaspar_counts,
            "cisbp_zf": cisbp_counts,
        }
    )
    .reindex(CANON_ORDER)
    .fillna(0)
    .astype(int)
)
# attach an InterPro anchor for reference
cov["interpro"] = [";".join(CROSSWALK[c]["interpro"]) for c in cov.index]
cov["n_resources"] = (cov[RES] > 0).sum(axis=1)
cov = cov.sort_values("zebramotif_zf", ascending=False)
print(cov.to_string())

# %%
# Coverage summary across individual TFs
present = cov[RES] > 0
print("Canonical DBD classes present per resource:")
print(present.sum())
print("\nClasses unique to zebramotif (absent in JASPAR-vert AND CisBP-zf):")
print(
    cov.index[
        present["zebramotif_zf"] & ~present["jaspar_vert"] & ~present["cisbp_zf"]
    ].tolist()
)
print("\nClasses in JASPAR/CisBP but absent in zebramotif:")
print(
    cov.index[
        ~present["zebramotif_zf"] & (present["jaspar_vert"] | present["cisbp_zf"])
    ].tolist()
)
print("\nClasses shared by all three:")
print(cov.index[present.all(axis=1)].tolist())

# %%
# Top N classes by total coverage; everything below folds into one "Remaining" row.
N = 15
by_total = (
    cov.drop(index=DROP_CLASSES, errors="ignore")[RES]
    .assign(total=lambda d: d.sum(axis=1))
    .sort_values("total", ascending=False)
)
top, rest = by_total.head(N)[RES], by_total.iloc[N:][RES]
REST_LABEL = f"Remaining ({len(rest)} classes)"
bars = pd.concat([top, rest.sum().to_frame(REST_LABEL).T]) if len(rest) else top
bars = bars.iloc[::-1]  # reverse so the largest sits at the top of the barh

# Transferred members stack on top of the (dark-blue) ZebraMotif predicted bar, on the
# same rows, with the tail classes aggregated into the Remaining row.
bars_t = transferred_counts.reindex(top.index).fillna(0)
if len(rest):
    bars_t[REST_LABEL] = transferred_counts.reindex(rest.index).fillna(0).sum()
bars_t = bars_t.reindex(bars.index).fillna(0)

y = np.arange(len(bars))
h = 0.26

# Plot three horizontal bar charts, one per resource, stacked vertically within each DBD class.
fig_dbd, ax = plt.subplots(figsize=(8, 9))
for i, r in enumerate(BAR_ORDER):
    offset = (1 - i) * h  # +h is the top of the group, -h the bottom
    base = bars[r].values
    ax.barh(y + offset, base, height=h, color=RES_COLORS[r], label=RES_LABELS[r])
    # light-blue extension on ZebraMotif = TFs whose PWM is transferred rep -> member
    extra = bars_t.values if r == "zebramotif_zf" else 0
    if r == "zebramotif_zf":
        ax.barh(y + offset, extra, height=h, left=base, color=PALETTE["Transferred"])
    label_bars(ax, y + offset, base + extra, RES_COLORS[r])
ax.set_yticks(y)
ax.set_yticklabels(bars.index, rotation=45, va="center")
# nudge the rotated labels down so they sit on their tick marks, not above
_dy = ScaledTranslation(0, -6 / 72, fig_dbd.dpi_scale_trans)
for _lbl in ax.get_yticklabels():
    _lbl.set_transform(_lbl.get_transform() + _dy)
ax.set_ylim(-0.6, len(bars) - 0.4)
ax.set_xscale("log")
ax.set_xlim(*XLIM)
ax.set_xlabel("unique TFs per class (log scale)")
ax.set_title(f"DBD coverage — top {N} classes by total")
# legend follows BAR_ORDER; the ZebraMotif entry pairs its predicted and transferred
# swatches into one handle
ax.legend(
    [
        (Patch(facecolor=RES_COLORS[r]), Patch(facecolor=PALETTE["Transferred"]))
        if r == "zebramotif_zf"
        else Patch(facecolor=RES_COLORS[r])
        for r in BAR_ORDER
    ],
    [RES_LABELS[r] for r in BAR_ORDER],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    frameon=False,
    loc="center right",
    handlelength=2.4,
)
ax.grid(axis="x", which="major", color="0.92", lw=0.7)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
fig_dbd.tight_layout()
fig_dbd.savefig(f"{FIG_DIR}/dbd_coverage.svg")

# %%
# TF motif coverage by evidence level. Only the levels a resource actually supplies are
# listed per row; the rest fill with 0. Direct (JASPAR) = unique vertebrate TFs with an
# experimental motif; n_predicted / n_transferred come from the ZebraMotif blocks above.
EVIDENCE_ORDER = ["Direct", "Inferred", "Predicted", "Transferred"]
# x-axis order is independent of the DBD plot's BAR_ORDER: ZebraMotif goes last here.
evidence = (
    pd.DataFrame(
        {
            "JASPAR\n(vertebrate)": {"Direct": jas["tf_key"].nunique()},
            "CisBP\n(zebrafish)": {
                "Direct": cisbp_status.get("D", 0),
                "Inferred": cisbp_status.get("I", 0),
            },
            "ZebraMotif\n(zebrafish)": {
                "Predicted": n_predicted,
                "Transferred": n_transferred,
            },
        }
    )
    .T.reindex(columns=EVIDENCE_ORDER)
    .fillna(0)
    .astype(int)
)

fig_evidence, ax = plt.subplots(figsize=(3, 5))
bottom = np.zeros(len(evidence))
x = np.arange(len(evidence))
for level in EVIDENCE_ORDER:
    vals = evidence[level].values
    ax.bar(
        x,
        vals,
        bottom=bottom,
        color=PALETTE[level],
        label=level,
        width=0.4,
        edgecolor="white",
        linewidth=0.5,
    )
    for xi, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0:
            ax.text(xi, b + v / 2, f"{v:,}", ha="center", va="center", color="white")
    bottom += vals

totals = evidence.sum(axis=1).values
for xi, t in enumerate(totals):
    ax.text(xi, t + max(totals) * 0.015, f"{t:,}", ha="center", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(evidence.index)
ax.set_ylabel("transcription factors covered")
ax.set_title("TF coverage by evidence level")
ax.set_ylim(0, max(totals) * 1.1)
ax.legend(title="evidence", frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig_evidence.tight_layout()
fig_evidence.savefig(f"{FIG_DIR}/tf_coverage_by_evidence.svg")

# %%
# Three-way coverage overlap, all in zebrafish ensembl ID space
zm_covered = reps_with_pwm | set(transferred["DBID"].astype(str))
cis_ens = set(cis["ens"])

# ZFIN curated orthology, already reduced to the TFs in tfdb4_tfs.csv. One row per
# ENSDARG; jaspar_ids is empty when none of that gene's human orthologs are in JASPAR.
zfin = pd.read_csv(f"{DATA_DIR}/tools/blast/zfin/ensdarg_human_orthos.csv").astype(
    {"ensdarg": str}
)
jaspar_ortho = set(zfin.loc[zfin["jaspar_ids"].fillna("") != "", "ensdarg"])
print(
    f"zfin: {zfin['ensdarg'].nunique()} TFs with a curated human ortholog, "
    f"{len(jaspar_ortho)} whose ortholog has a JASPAR motif"
)

# The 7 regions, for the log only (venn3 derives its areas from the sets directly)
regions = {
    "ZM": zm_covered - jaspar_ortho - cis_ens,
    "JAS": jaspar_ortho - zm_covered - cis_ens,
    "CIS": cis_ens - zm_covered - jaspar_ortho,
    "ZM&JAS": (zm_covered & jaspar_ortho) - cis_ens,
    "ZM&CIS": (zm_covered & cis_ens) - jaspar_ortho,
    "CIS&JAS": (cis_ens & jaspar_ortho) - zm_covered,
    "all": zm_covered & jaspar_ortho & cis_ens,
}
print(
    f"venn3 sizes: ZebraMotif {len(zm_covered)}, JASPAR-ortho {len(jaspar_ortho)}, "
    f"CisBP {len(cis_ens)} | regions "
    + " ".join(f"{k}={len(v)}" for k, v in regions.items())
)

# Area-proportional Venn - A=ZebraMotif, B=JASPAR, C=CisBP.
fig_venn, ax = plt.subplots(figsize=(7.5, 7))
_subsets = [zm_covered, jaspar_ortho, cis_ens]
venn3(
    _subsets,
    set_labels=(
        f"ZebraMotif\n(n = {len(zm_covered):,})",
        f"JASPAR ortholog\n(n = {len(jaspar_ortho):,})",
        f"CisBP\n(n = {len(cis_ens):,})",
    ),
    set_colors=tuple(RES_COLORS[r] for r in RES),
    alpha=0.5,
    subset_label_formatter=lambda n: f"{n:,}",
    ax=ax,
)
venn3_circles(_subsets, lw=0.6, color="0.5", ax=ax)
ax.set_title("Zebrafish TF motif coverage overlap")
fig_venn.tight_layout()
fig_venn.savefig(f"{FIG_DIR}/coverage_venn.svg")

# %%
# TFs ZebraMotif misses that the external DBs cover, organized by canonical DBD class
dbid2sym, dbid2fam = {}, {}
for _fn in ["tfdb4_tfs.csv", "tfdb4_rep_tfs.csv", "tfdb4_val_tfs.csv"]:
    _d = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/{_fn}")
    for _dbid, _name, _fam in zip(
        _d["DBID"].astype(str), _d["TF_Name"].astype(str), _d["Family"].astype(str)
    ):
        dbid2sym.setdefault(_dbid, _name)
        dbid2fam.setdefault(_dbid, _fam)
_ens2canon_cis = dict(zip(cis["ens"], cis["canon"]))
_ens2cisname = dict(zip(cis["ens"], cis["TF_Name"].astype(str)))


def _canon_of(ens: str) -> str:
    """Canonical DBD class for an ENSDARG: CisBP's mapping if present, else AnimalTFDB's."""
    if ens in _ens2canon_cis:
        return _ens2canon_cis[ens]
    return animaltfdb2canon.get(dbid2fam.get(ens, ""), UNCLASSIFIED)


def _summarize(ens_set: set, name: str) -> pd.DataFrame:
    """Build a symbol/canon/source table for a set of ENSDARG ids and print it by class."""
    rows = [
        {
            "ensdarg": e,
            "symbol": dbid2sym.get(e, _ens2cisname.get(e, "?")),
            "canon": _canon_of(e),
            "in_cisbp": e in cis_ens,
            "in_jaspar": e in jaspar_ortho,
        }
        for e in ens_set
    ]
    df = pd.DataFrame(rows).sort_values(["canon", "symbol"]).reset_index(drop=True)
    print(f"\n=== {name}: {len(df)} TFs ===")
    for canon, g in df.groupby("canon", sort=True):
        print(
            f"\n[{canon}]  (n={len(g)})\n  "
            + ", ".join(sorted(g["symbol"], key=str.lower))
        )
    return df


either_not_zm = (cis_ens | jaspar_ortho) - zm_covered
df_either = _summarize(
    either_not_zm, "Outside ZebraMotif, covered by EITHER CisBP or JASPAR"
)
df_either.to_csv(f"{TAB_DIR}/coverage_gap_either.csv", index=False)

# %%
