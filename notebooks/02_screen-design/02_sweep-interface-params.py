"""
Sweep the 02_interface-contacts.py parameters for true base-calling, then test the
chosen gates on a held-out fold.

02_interface-contacts.py calls a residue a base-reader when three gates pass on the
contacts af3tfm.contacts collected:
    contact_freq       >= MIN_CONTACT
    med_plddt          >= PLDDT_MIN
    frac_base_contact  >= FRAC_BASE_MIN
This asks which mix of (MIN_CONTACT, FRAC_BASE_MIN, PLDDT_MIN) best recovers the
residues we KNOW read bases.

Outputs (under notebooks/ and data/jobs/hero/):
  zfc2h2_contacts.parquet            - cached zf-C2H2 binding-finger contact rows
  homeobox_contacts.parquet          - cached homeodomain contact rows
  interface_param_sweep.csv          - one row per parameter combo with all metrics
  interface_param_sweep.svg          - F1 / MCC vs each gate, one panel per gate
  interface_param_sweep_homeobox.csv - the same grid scored on homeodomains
  interface_param_sweep_homeobox.svg - where predicted base-readers land in the domain

Ben Iovino 8/4/2026 Biohub
"""

# %%
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from af3tfm import homeobox as hb
from af3tfm.contacts import CONTACT_CUTOFF
from af3tfm.dataset import DATA_DIR, DS, PROJECT_ROOT
from af3tfm.utils import set_plotting_style
from af3tfm.zfc2h2 import ZF_RE, dna_binding_finger_indices

warnings.filterwarnings("ignore")
set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "tables"
TAB_DIR.mkdir(exist_ok=True)

run = "hero"
dataset = DS["tfdb4_rep"]
CONTACTS_CSV = Path(f"{DATA_DIR}/jobs/{run}/residue_contacts_all.csv")

MIN_CONTACTS = [0.1, 0.2, 0.3, 0.4, 0.5]
FRAC_BASE_MINS = [0.2, 0.3, 0.4, 0.5, 0.6]
PLDDT_MINS = [50.0, 60.0, 70.0, 80.0]
# (MIN_CONTACT, FRAC_BASE_MIN, PLDDT_MIN)
# FRAC_BASE_MIN is the only gate that moves the precision/recall balance here;
# MIN_CONTACT and PLDDT_MIN leave precision pinned at ~0.968 with flat F1
RECOMMENDED = (0.1, 0.3, 50.0)
DEFAULTS = (0.3, 0.4, 70.0)  # current 02_interface-contacts.py settings

# regex-group index -> recognition-helix label for the 4 base-contacting
# positions (helix -1/+2/+3/+6). Group 4 (helix +5, bystander) is absent: it is
# captured in the code but does not read a base.
BASE_GROUPS = {1: "m1", 2: "p2", 3: "p3", 5: "p6"}

HD_LABELS = {46: "p47", 49: "p50", 50: "p51", 51: "p52", 53: "p54"}  # canonical
# Two corrections to homeodomain recognition code:
#  - p52 is a specificity-classification position, NOT a base contact: across all
#    42 TFs its residues sit >5 A from DNA (median contact_freq 0.01, frac_base
#    0.00) at high pLDDT (~87), it points away from the groove. So the base-
#    contacting subset is the other four helix positions.
#  - the N-terminal arm (Arg residues at domain offset < NARM_OFFSET) makes real
#    minor-groove base contacts to the TAAT core that the code does NOT list, so
#    it is excluded from the scored universe rather than counted as a negative;
#    scoring it would penalise correct calls as false positives.
# Precision on homeodomains is therefore a lower bound, and the FP-location cell
# checks whether the "extra" calls really do cluster in the N-terminal arm.
CONTACTING_LABELS = {"p47", "p50", "p51", "p54"}
NARM_OFFSET = 10


# %%
def load_contacts(prefix, universe):
    """
    Cached contact rows for one fold, restricted to its scored universe.

    Args:
        prefix (str): cache basename under data/jobs/<run>/ --
            "<prefix>_contacts.parquet" plus a "<prefix>_nstruct.parquet"
            sidecar (parquet does not persist DataFrame.attrs).
        universe (dict[str, set[int]]): dbid(lower) -> scored residue numbers.

    Returns:
        (cdf, n_struct): contact rows carrying a lowercased "dbl" column, and
        dbid(lower) -> number of structures the TF appears in.
    """
    cache = Path(f"{DATA_DIR}/jobs/{run}/{prefix}_contacts.parquet")
    nstruct_cache = cache.with_name(f"{prefix}_nstruct.parquet")
    if cache.exists():
        cdf = pd.read_parquet(cache)
        n_struct = (
            pd.read_parquet(nstruct_cache).set_index("dbid")["n_struct"].to_dict()
        )
    else:
        cols = [
            "dbid",
            "job_name",
            "residue_num",
            "resname",
            "min_dist",
            "contact_type",
            "plddt",
        ]
        struct_seen = {d: set() for d in universe}  # dbid -> {job_name}
        kept, n = [], 0
        for chunk in pd.read_csv(CONTACTS_CSV, usecols=cols, chunksize=2_000_000):
            chunk["dbl"] = chunk["dbid"].astype(str).str.lower()
            chunk = chunk[chunk["dbl"].isin(universe)]
            for d, g in chunk.groupby("dbl"):
                struct_seen[d].update(g["job_name"].unique())
            mask = [
                rn in universe[d] for d, rn in zip(chunk["dbl"], chunk["residue_num"])
            ]
            kept.append(chunk[mask])
            n += len(chunk)
            print(f"  scanned {n:,} {prefix} rows, kept {sum(len(k) for k in kept):,}")
        cdf = pd.concat(kept, ignore_index=True)
        n_struct = {d: len(s) for d, s in struct_seen.items() if s}
        cdf.to_parquet(cache, index=False)
        pd.DataFrame(
            {"dbid": list(n_struct), "n_struct": list(n_struct.values())}
        ).to_parquet(nstruct_cache, index=False)
    if "dbl" not in cdf.columns:
        cdf["dbl"] = cdf["dbid"].astype(str).str.lower()
    print(f"cached {len(cdf):,} {prefix} contact rows, {cdf['dbl'].nunique()} TFs")
    return cdf, n_struct


# No gate changes the per-contact rows, so collapse to a per-(TF, residue) table
# of (contact_freq, frac_base_contact, med_plddt) exactly once; MIN_CONTACT /
# FRAC_BASE_MIN / PLDDT_MIN are then vectorized thresholds over that small table.
def aggregate_metrics(cdf, n_struct):
    """Per-(dbid, residue_num) interface metrics over the cached contact rows."""
    g = (
        cdf.groupby(["dbl", "residue_num"])
        .agg(
            contact_freq=("job_name", "nunique"),
            frac_base_contact=("contact_type", lambda s: (s == "base").mean()),
            med_plddt=("plddt", "median"),
        )
        .reset_index()
    )
    g["contact_freq"] /= g["dbl"].map(n_struct)
    return g


def align_truth(g, pos_lookup):
    """
    Ground-truth arrays aligned to the rows of `g`.

    Args:
        g (pd.DataFrame): per-residue metric table from aggregate_metrics.
        pos_lookup (dict): (dbid(lower), residue_num) -> position label.

    Returns:
        (is_pos, labels): boolean positive mask, and the position label of each
        row (None for negatives).
    """
    labels = [pos_lookup.get((d, rn)) for d, rn in zip(g["dbl"], g["residue_num"])]
    return (
        np.array([lab is not None for lab in labels]),
        np.array(labels, dtype=object),
    )


def predict(g, mc, fb, pl):
    """
    Base-reader calls over `g` at one gate triple (boolean array).

    Args:
        g (pd.DataFrame): per-residue metric table from aggregate_metrics.
        mc (float): MIN_CONTACT threshold.
        fb (float): FRAC_BASE_MIN threshold.
        pl (float): PLDDT_MIN threshold.

    Returns:
        (np.ndarray): boolean array of the same length as `g`, True for residues
        that pass all three gates.
    """
    return (
        (g["contact_freq"] >= mc)
        & (g["med_plddt"] >= pl)
        & (g["frac_base_contact"] >= fb)
    ).to_numpy()


def score(g, is_pos, n_pos_total, mc, fb, pl, labels=None, recov_keys=()):
    """
    Confusion matrix + P/R/F1/MCC for one gate combo (returns a dict).

    Args:
        g (pd.DataFrame): per-residue metric table from aggregate_metrics.
        is_pos (np.ndarray): boolean array of the same length as `g`, True for
            residues that are ground-truth positives.
        n_pos_total (int): total number of ground-truth positives (the
            denominator for recall).
        mc (float): MIN_CONTACT threshold.
        fb (float): FRAC_BASE_MIN threshold.
        pl (float): PLDDT_MIN threshold.
        labels (np.ndarray): position label of each row in `g` (None for
            negatives), used to count per-position recovery.

    Returns:
        (dict): the gate triple, confusion matrix counts, precision, recall,
        F1, MCC, and per-position recovery counts if `labels` is provided.
    """
    pred = predict(g, mc, fb, pl)
    tp = int((pred & is_pos).sum())
    fp = int((pred & ~is_pos).sum())
    fn = n_pos_total - tp
    tn = int((~pred & ~is_pos).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F1 = 2 * P * R / (P + R) if P + R else 0.0
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    out = {
        "MIN_CONTACT": mc,
        "FRAC_BASE_MIN": fb,
        "PLDDT_MIN": pl,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": P,
        "recall": R,
        "f1": F1,
        "mcc": mcc,
    }
    if labels is not None:
        out.update(
            {f"recov_{k}": int((pred & (labels == k)).sum()) for k in recov_keys}
        )
    return out


def run_sweep(g, is_pos, n_pos_total, labels=None, recov_keys=()):
    """
    Score every (MIN_CONTACT, FRAC_BASE_MIN, PLDDT_MIN) combo in the grid.

    Args:
        g (pd.DataFrame): per-residue metric table from aggregate_metrics.
        is_pos (np.ndarray): boolean array of the same length as `g`, True for
            residues that are ground-truth positives.
        n_pos_total (int): total number of ground-truth positives (the
            denominator for recall).
        labels (np.ndarray): position label of each row in `g` (None for
            negatives), used to count per-position recovery.

    Returns:
        (pd.DataFrame): one row per gate triple, with confusion matrix counts,
        precision, recall, F1, MCC, and per-position recovery counts if
        `labels` is provided.
    """
    sweep = pd.DataFrame(
        [
            score(g, is_pos, n_pos_total, mc, fb, pl, labels, recov_keys)
            for mc, fb, pl in product(MIN_CONTACTS, FRAC_BASE_MINS, PLDDT_MINS)
        ]
    )
    sweep.insert(0, "CONTACT_CUTOFF", CONTACT_CUTOFF)
    return sweep


def combo(sweep, mc, fb, pl):
    """
    The grid row at one gate triple.

    Args:
        sweep (pd.DataFrame): the full grid, one row per gate triple.
        mc (float): MIN_CONTACT threshold.
        fb (float): FRAC_BASE_MIN threshold.
        pl (float): PLDDT_MIN threshold.

    Returns:
        (pd.Series): the row of `sweep` at the requested gate triple.
    """
    m = (
        (sweep["MIN_CONTACT"] == mc)
        & (sweep["FRAC_BASE_MIN"] == fb)
        & (sweep["PLDDT_MIN"] == pl)
    )
    return sweep[m].iloc[0]


def label_totals(pos_lookup, keys):
    """
    Ground-truth count of each position label (the recovery denominators).

    Args:
        pos_lookup (dict): (dbid(lower), residue_num) -> position label.
        keys (iterable): position labels to count.

    Returns:
        (dict): position label -> number of ground-truth positives with that
        label.
    """
    return {k: sum(1 for lab in pos_lookup.values() if lab == k) for k in keys}


# %%
# Ground truth: per-TF base positions + scored universe for zf-C2H2 proteins
def base_contacting_positions(sequence, ranges, pad=2):
    """
    Ground-truth base-contacting residues of a zf-C2H2 protein, restricted to
    its DNA-binding fingers.

    Args:
        sequence (str): full protein sequence.
        ranges (list[tuple[int, int]]): 1-based inclusive DNA-binding spans.
        pad (int): residues of slack when testing finger/range overlap.

    Returns:
        (positives, universe):
          positives (dict[int, str]): residue_num -> position label
              ("m1"/"p2"/"p3"/"p6") for the base readers of the binding fingers.
          universe (set[int]): every residue number inside a binding finger's
              span -- the set over which base-vs-not is scored.
    """
    matches = list(ZF_RE.finditer(sequence))  # left-to-right == N->C order
    binding = set(dna_binding_finger_indices(sequence, ranges, pad=pad))
    positives, universe = {}, set()
    for i, m in enumerate(matches):
        if i not in binding:
            continue
        universe.update(range(m.start() + 1, m.end() + 1))
        for g, label in BASE_GROUPS.items():
            positives[m.start(g) + 1] = label
    return positives, universe


if __name__ == "__main__":
    tfs = pd.read_csv(dataset.seq_csv)
    doms = pd.read_csv(f"{DATA_DIR}/sequences/animaltfdb/tfdb4_rep_doms.csv")
    dbd = doms[doms["DNA-Binding"] == True]  # noqa: E712
    dbd_ranges = (
        dbd.groupby("DBID").apply(lambda g: list(zip(g["start"], g["end"]))).to_dict()
    )

    zf_positives, zf_universe = {}, {}  # dbid(lower) -> positives / universe
    for _, r in tfs[tfs["Family"] == "zf-C2H2"].iterrows():
        ranges = dbd_ranges.get(r["DBID"], [])
        if not ranges:
            continue
        pos, univ = base_contacting_positions(r["Sequence"], ranges)
        if pos:
            zf_positives[str(r["DBID"]).lower()] = pos
            zf_universe[str(r["DBID"]).lower()] = univ
    print(f"zf-C2H2 TFs with a scored universe: {len(zf_universe)}")
    print(
        f"total positives: {sum(len(p) for p in zf_positives.values())}, "
        f"total universe residues: {sum(len(u) for u in zf_universe.values())}"
    )


# %%
if __name__ == "__main__":
    zf_cdf, zf_nstruct = load_contacts("zfc2h2", zf_universe)

    zf_pos_lookup = {
        (d, rn): lab for d, p in zf_positives.items() for rn, lab in p.items()
    }
    zf_npos = len(zf_pos_lookup)
    zf_g = aggregate_metrics(zf_cdf, zf_nstruct)
    zf_is_pos, zf_labels = align_truth(zf_g, zf_pos_lookup)

    zf_sweep = run_sweep(
        zf_g, zf_is_pos, zf_npos, zf_labels, tuple(BASE_GROUPS.values())
    )
    zf_sweep.to_csv(TAB_DIR / "interface_param_sweep.csv", index=False)
    print(f"swept {len(zf_sweep)} combos -> {TAB_DIR / 'interface_param_sweep.csv'}")


# %%
# Report: F1-optimal, chosen operating points, per-position recovery
if __name__ == "__main__":
    best = zf_sweep.sort_values("f1", ascending=False).iloc[0]
    print("\nF1-optimal combo:")
    print(
        best[
            [
                "MIN_CONTACT",
                "FRAC_BASE_MIN",
                "PLDDT_MIN",
                "precision",
                "recall",
                "f1",
                "mcc",
                "tp",
                "fp",
                "fn",
            ]
        ].to_string()
    )

    n_by = label_totals(zf_pos_lookup, BASE_GROUPS.values())
    print("\nPer-position recovery at F1-optimal (recovered / total):")
    for lab in BASE_GROUPS.values():
        print(
            f"  {lab}: {int(best[f'recov_{lab}'])} / {n_by[lab]} "
            f"({best[f'recov_{lab}'] / n_by[lab]:.2f})"
        )


# %%
# F1 / MCC vs each gate, one panel per gate
PARAMS = ["MIN_CONTACT", "FRAC_BASE_MIN", "PLDDT_MIN"]
METRICS = [("f1", "F1", "#738fc1"), ("mcc", "MCC", "#c0504d")]
PARAM_LABELS = {
    "MIN_CONTACT": "Min. fraction of structures\ncontacting DNA  (MIN_CONTACT)",
    "FRAC_BASE_MIN": "Min. fraction of DNA contacts\nthat are base-directed  (FRAC_BASE_MIN)",
    "PLDDT_MIN": "Min. median side-chain pLDDT\n(PLDDT_MIN)",
}


def param_profile(sweep, param, metric, hold):
    """
    One gate's metric profile with the other two gates pinned.

    Args:
        sweep (pd.DataFrame): the full grid, one row per gate triple.
        param (str): the gate to vary (a column of `sweep`).
        metric (str): metric column to read ("f1" / "mcc").
        hold (dict): gate -> fixed value for the other two gates.

    Returns:
        (x, y): swept values of `param` and the metric at each, ascending in x.
    """
    others = [p for p in PARAMS if p != param]
    m = np.logical_and.reduce([(sweep[o] == hold[o]).values for o in others])
    s = sweep[m].sort_values(param)
    return s[param].values, s[metric].values


if __name__ == "__main__":
    hold = dict(zip(PARAMS, RECOMMENDED))

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.5), sharey=True)
    for i, (ax, param) in enumerate(zip(axes, PARAMS)):
        for col, label, color in METRICS:
            x, y = param_profile(zf_sweep, param, col, hold)
            ax.plot(
                x,
                y,
                "-o",
                color=color,
                ms=4.5,
                lw=1.6,
                zorder=3,
                label=label if i == 0 else None,
            )
        ax.axvline(
            hold[param],
            color="0.35",
            ls="--",
            lw=1,
            zorder=2,
            label="recommended" if i == 0 else None,
        )
        ax.set_xlabel(PARAM_LABELS[param], fontsize=9)
        ax.set_xticks(sorted(zf_sweep[param].unique()))
        ax.margins(x=0.09)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("score")
    axes[0].set_ylim(0.65, 0.95)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(
        "zf-C2H2 base-calling: F1 / MCC vs each gate "
        f"({len(zf_universe)} TFs, {zf_npos} base positions)",
        y=1.02,
    )
    fig.savefig(FIG_DIR / "interface_param_sweep.svg", dpi=300, bbox_inches="tight")
    plt.show()

    for param in PARAMS:
        x, f1 = param_profile(zf_sweep, param, "f1", hold)
        _, mcc = param_profile(zf_sweep, param, "mcc", hold)
        print(f"\n{param} (other two gates at RECOMMENDED):")
        print(
            pd.DataFrame({param: x, "f1": f1.round(3), "mcc": mcc.round(3)}).to_string(
                index=False
            )
        )


# %%
# Held-out fold, homeodomains
def hd_ground_truth(tfs, doms):
    """Return (positives, universe, offset) keyed by lowercased DBID.

    Args:
        tfs (pd.DataFrame): TF sequences, must have "DBID" and "Sequence".
        doms (pd.DataFrame): domain annotations, must have "DBID", "accession",
            "start", and "end".

    Returns:
        positives[dbl] : {residue_num -> position label}
        universe[dbl]  : set of residue numbers inside any homeodomain span
        offset[dbl]    : {residue_num -> 0-based offset within its homeodomain}
    """
    seq_map = dict(zip(tfs["DBID"], tfs["Sequence"]))
    fam = set(tfs["DBID"])
    hd_doms = doms[(doms["accession"] == hb.IPR_ACCESSION) & (doms["DBID"].isin(fam))]

    seqs, meta, cnt = {}, {}, {}
    for _, r in hd_doms.iterrows():
        pid = r["DBID"]
        if pid not in seq_map:
            continue
        n = cnt.get(pid, 0)
        key = pid if n == 0 else f"{pid}__hd{n + 1}"
        cnt[pid] = n + 1
        seqs[key] = seq_map[pid][int(r["start"]) - 1 : int(r["end"])]
        meta[key] = (pid, int(r["start"]), int(r["end"]))

    hmm = hb.load_hmm(hb.HMM_PATH)
    aln, match_cols = hb.align_sequences(hmm, seqs)

    def resnum(aseq, col, start):
        if col >= len(aseq) or not aseq[col].isalpha():
            return None
        return start + sum(1 for c in aseq[: col + 1] if c.isalpha()) - 1

    positives, universe, offset = {}, {}, {}
    for key, a in aln.items():
        pid, start, end = meta[key]
        dbl = str(pid).lower()
        u = universe.setdefault(dbl, set())
        o = offset.setdefault(dbl, {})
        for rn in range(start, end + 1):
            u.add(rn)
            o[rn] = rn - start
        p = positives.setdefault(dbl, {})
        for pos, lab in HD_LABELS.items():
            rn = resnum(a, match_cols[pos - 1], start)
            if rn is not None:
                p[rn] = lab
    return positives, universe, offset


if __name__ == "__main__":
    # every TF carrying a homeodomain (IPR001356), not just the Homeobox family:
    # Pou / CUT / PAX / TF_Otx also have one, and the recognition-helix ground
    # truth applies identically to all of them.
    ipr_dbids = set(doms.loc[doms["accession"] == hb.IPR_ACCESSION, "DBID"])
    hd_tfs = tfs[tfs["DBID"].isin(ipr_dbids)].copy()
    fam_map = {str(d).lower(): f for d, f in zip(tfs["DBID"], tfs["Family"])}
    hd_positives, hd_universe, offset = hd_ground_truth(hd_tfs, doms)
    off_lookup = {(d, rn): o for d, om in offset.items() for rn, o in om.items()}

    # raw ground truth: all five code positions, full domain (naive, for contrast)
    hd_pos_raw = {
        (d, rn): lab for d, p in hd_positives.items() for rn, lab in p.items()
    }
    # corrected: base-contacting positions only, N-terminal arm excluded (see the
    # CONTACTING_LABELS / NARM_OFFSET comment above)
    hd_pos_lookup = {
        k: lab
        for k, lab in hd_pos_raw.items()
        if lab in CONTACTING_LABELS and off_lookup[k] >= NARM_OFFSET
    }
    hd_npos = len(hd_pos_lookup)
    print(f"homeodomain TFs (IPR001356, any family): {len(hd_universe)}")
    print("  by family:", hd_tfs["Family"].value_counts().to_dict())
    print(
        f"base positions -- raw 5-code: {len(hd_pos_raw)} | "
        f"corrected (4 contacting, N-arm excluded): {hd_npos}"
    )
    print(f"universe residues: {sum(len(u) for u in hd_universe.values())}")


# %%
# Cache homeodomain contact rows, then sweep the same grid
if __name__ == "__main__":
    hd_cdf, hd_nstruct = load_contacts("homeobox", hd_universe)

    hd_g = aggregate_metrics(hd_cdf, hd_nstruct)
    hd_g["off"] = [
        off_lookup.get((d, rn), -1) for d, rn in zip(hd_g["dbl"], hd_g["residue_num"])
    ]
    # scored universe excludes the N-terminal arm (unlabelled real base contacts)
    hd_scored = hd_g[hd_g["off"] >= NARM_OFFSET].reset_index(drop=True)
    hd_is_pos, _ = align_truth(hd_scored, hd_pos_lookup)

    hd_sweep = run_sweep(hd_scored, hd_is_pos, hd_npos)
    hd_sweep.to_csv(TAB_DIR / "interface_param_sweep_homeobox.csv", index=False)
    print(
        f"swept {len(hd_sweep)} combos -> "
        f"{TAB_DIR / 'interface_param_sweep_homeobox.csv'}"
    )


# %%
# Generalization test: apply the zf-C2H2-calibrated point, compare
if __name__ == "__main__":
    raw_is_pos, raw_labels = align_truth(hd_g, hd_pos_raw)  # naive GT, full domain
    hb_raw = score(hd_g, raw_is_pos, len(hd_pos_raw), *RECOMMENDED)
    hb_reco = combo(hd_sweep, *RECOMMENDED)  # corrected GT (hd_scored + hd_pos_lookup)
    hb_dflt = combo(hd_sweep, *DEFAULTS)
    hb_best = hd_sweep.sort_values("f1", ascending=False).iloc[0]

    print(f"\n{'':<26}{'precision':>10}{'recall':>9}{'f1':>7}{'mcc':>7}")

    def line(tag, r):
        print(
            f"{tag:<26}{r['precision']:>10.3f}{r['recall']:>9.3f}"
            f"{r['f1']:>7.3f}{r['mcc']:>7.3f}"
        )

    line("homeodomain raw (5-code)", hb_raw)
    line("homeodomain @ recommended", hb_reco)
    line("homeodomain F1-optimal", hb_best)
    print(
        "(homeodomain rows use corrected GT -- 4 contacting positions, N-arm "
        "excluded -- except 'raw')"
    )
    print(
        "homeodomain F1-optimal gates (MIN/FRAC/PLDDT): "
        f"{hb_best['MIN_CONTACT']}/{hb_best['FRAC_BASE_MIN']}/{hb_best['PLDDT_MIN']}"
    )

    # per-family breakdown at the recommended point (corrected GT)
    print("\nPer-family @ recommended (corrected GT):")
    for f in ["Homeobox", "Pou", "CUT", "PAX", "TF_Otx"]:
        fd = {d for d in hd_universe if fam_map.get(d) == f}
        if not fd:
            continue
        gg = hd_scored[hd_scored["dbl"].isin(fd)]
        pl = {k: v for k, v in hd_pos_lookup.items() if k[0] in fd}
        ip, _ = align_truth(gg, pl)
        r = score(gg, ip, len(pl), *RECOMMENDED)
        print(
            f"  {f:<9} nTF={len(fd):<3} P={r['precision']:.3f} "
            f"R={r['recall']:.3f} F1={r['f1']:.3f} (npos={len(pl)})"
        )

    # per-position recovery over ALL five code positions on the full domain, so
    # p52's total failure (it never contacts a base) is visible as the evidence
    # motivating its removal from the corrected ground truth.
    pred_raw = predict(hd_g, *RECOMMENDED)
    n_by = label_totals(hd_pos_raw, HD_LABELS.values())
    print("\nPer-position recovery @ recommended (recovered / total):")
    for lab in HD_LABELS.values():
        rec = int((pred_raw & (raw_labels == lab)).sum())
        print(
            f"  {lab}: {rec} / {n_by[lab]} ({rec / n_by[lab]:.2f})"
            if n_by[lab]
            else f"  {lab}: n/a"
        )


# %%
# Looking where false positives land in homeodomains
if __name__ == "__main__":
    fp_off = hd_g["off"].to_numpy()[pred_raw & ~raw_is_pos]
    fp_off = fp_off[fp_off >= 0]
    n_arm = int((fp_off < NARM_OFFSET).sum())
    print(f"\nfalse positives at recommended (raw GT): {len(fp_off)}")
    print(
        f"  in N-terminal arm (offset < {NARM_OFFSET}): {n_arm} ({n_arm / len(fp_off):.2f})"
    )
    print(f"  offset quartiles: {np.percentile(fp_off, [25, 50, 75]).round(1)}")

    fig, ax = plt.subplots(figsize=(6, 4))
    pos_off = np.array([off_lookup[k] for k in hd_pos_raw])
    ax.hist(
        pos_off,
        bins=range(0, 65, 3),
        alpha=0.6,
        color="#738fc1",
        label="true base positions",
    )
    ax.hist(
        fp_off,
        bins=range(0, 65, 3),
        alpha=0.6,
        color="#c0504d",
        label="false positives",
    )
    ax.axvspan(0, NARM_OFFSET, color="0.85", zorder=0, label="N-terminal arm")
    ax.set_xlabel("residue offset within homeodomain")
    ax.set_ylabel("count")
    ax.set_title("Homeodomains: where predicted base-readers fall")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(
        FIG_DIR / "interface_param_sweep_homeobox.svg", dpi=300, bbox_inches="tight"
    )
    plt.show()

# %%
