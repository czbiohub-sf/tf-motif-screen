"""
Baselines for the AF3/NA-MPNN recovery claim.

Four baselines:
    - random assignment from the 312-motif library
        (data/tools/homer/jaspar_clustered/all_motifs)
    - assigning each TF the highest-IC library motif matching its DBD family.
    - top-confidence matching: each TF's individual AF3/NA-MPNN PWMs ranked by the
        combined AF3 confidence score (Score_Rank) instead of consensus cluster size.
    - oracle (ceiling): the library motif most similar to each TF's reference JASPAR
        motif. It uses the reference to choose, so it is an upper bound, not a method.

Ben Iovino 8/27/26 Biohub
"""

# %%
import glob
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.motif import load_homer_multi_motif_file
from af3tfm.utils import pwm_similarity, set_plotting_style

DATA = Path(DATA_DIR)
set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "03_building-ppms" / "figures"
FIG_DIR.mkdir(exist_ok=True)
TAB_DIR = PROJECT_ROOT / "notebooks" / "03_building-ppms" / "tables"
TAB_DIR.mkdir(exist_ok=True)

# HYPERPARAMETERS (match 03_tune-consensus.py so the pipeline column is directly comparable)
num_specs = 47
cor_threshold = 0.73
P_SIG = 0.05  # significance threshold on the JASPAR-null p-value
N_MAX = 5  # largest "top-n" budget on the saturation curves

LIB_DIR = DATA / "tools" / "homer" / "jaspar_clustered" / "all_motifs"

# %%
# Load the JASPAR null set
jaspar_path = DATA / "tools" / "homer" / "jaspar.homer.motif"
jaspar_names, jaspar_pwms = load_homer_multi_motif_file(str(jaspar_path))
jaspar_id_to_idx = {name: i for i, name in enumerate(jaspar_names)}
print(f"Loaded {len(jaspar_pwms)} JASPAR motifs (null set)")


# Load the 312-motif library: PWM, BestGuess identity, information content
def parse_library_motif(fp: str) -> tuple:
    """Return (named_gene | None, best_guess_MA_id | None, ppm (L,4), total IC).

    - gene is the leading token of the named guess (e.g. POU4F3, GATA6)
    - ma is the JASPAR MA id used as the family-assignment fallback when the named
      gene is unmapped.
    """
    gene, ma, rows = None, None, []
    for line in open(fp):
        if line.startswith(">"):
            m = re.search(r"BestGuess:(MA\d+\.\d+)", line)  # JASPAR guess
            ma = m.group(1) if m else None
            if ma is None:  # fall back to any MA id in the header
                m = re.search(r"(MA\d+\.\d+)", line)
                ma = m.group(1) if m else None
            segs = line.split("BestGuess:")  # last segment == the named gene guess
            if len(segs) >= 2:
                tok = re.split(r"[/(,\t]", segs[-1])[0].strip()
                gene = tok or None
        else:
            rows.append([float(x) for x in line.split()])
    ppm = np.array(rows)
    ic = (ppm * np.log2(ppm / 0.25 + 1e-9)).clip(min=0).sum()
    return gene, ma, ppm, float(ic)


lib_files = sorted(glob.glob(str(LIB_DIR / "*.motif")))
lib_gene, lib_ma, lib_pwms, lib_ic = zip(*(parse_library_motif(f) for f in lib_files))
lib_gene, lib_ma, lib_pwms, lib_ic = (
    list(lib_gene),
    list(lib_ma),
    list(lib_pwms),
    np.array(lib_ic),
)
n_lib = len(lib_pwms)
print(
    f"Loaded {n_lib} library motifs "
    f"({sum(m is not None for m in lib_ma)} with a BestGuess JASPAR id)"
)

# %%
# DBD family for each library motif: BestGuess identity -> animalTFDB family
base2fam, name2fam = {}, {}
for fn in ["tfdb4_rep_tfs.csv", "tfdb4_tfs.csv", "tfdb4_val_tfs.csv"]:
    d = pd.read_csv(DATA / "sequences" / "animaltfdb" / fn)
    if {"Hom_JASPAR", "Family"} <= set(d.columns):
        for _, r in d.dropna(subset=["Hom_JASPAR"]).iterrows():
            base2fam.setdefault(r["Hom_JASPAR"].split(".")[0], r["Family"])
    for _, r in d.dropna(subset=["TF_Name", "Family"]).iterrows():
        name2fam.setdefault(str(r["TF_Name"]).lower(), r["Family"])


def library_family(gene: str | None, ma: str | None) -> str | None:
    """animalTFDB family for a library motif: named-gene guess first, MA id fallback."""
    if gene and gene.lower() in name2fam:
        return name2fam[gene.lower()]
    if ma:
        return base2fam.get(ma.split(".")[0])
    return None


lib_fam = [library_family(g, m) for g, m in zip(lib_gene, lib_ma)]
print(
    f"Library motifs with an assigned family: "
    f"{sum(f is not None for f in lib_fam)}/{n_lib}"
)

# %%
# TF set: exactly the TFs benchmarked by the pipeline (from its cluster CSV)
cluster_df = pd.read_csv(
    f"{TAB_DIR}/cluster_ppm_jaspar_{num_specs}_{cor_threshold}_new.csv"
)
tf_meta = (
    cluster_df[["DBID", "TF_Name", "Hom_JASPAR"]]
    .drop_duplicates("DBID")
    .reset_index(drop=True)
)
fam_map = pd.read_csv(
    DATA / "sequences" / "animaltfdb" / "tfdb4_rep_tfs.csv"
).set_index("DBID")["Family"]
tf_meta["Family"] = tf_meta["DBID"].map(fam_map)
# Every reference must be present in the null set (it is, by construction).
tf_meta = tf_meta[tf_meta["Hom_JASPAR"].isin(jaspar_id_to_idx)].reset_index(drop=True)
n_tf = len(tf_meta)
print(f"Benchmarked TFs: {n_tf}")


# %%
# Similarity matrix S[i, j] = HOMER sim(library motif i, JASPAR motif j)
def _row_sims(args):
    i, ppm = args
    return i, np.array([pwm_similarity(ppm, jp)[0] for jp in jaspar_pwms])


S = np.zeros((n_lib, len(jaspar_pwms)))
with ProcessPoolExecutor() as ex:
    for fut in as_completed(
        [ex.submit(_row_sims, (i, p)) for i, p in enumerate(lib_pwms)]
    ):
        i, row = fut.result()
        S[i] = row
print(f"Similarity matrix computed: {S.shape}")

# %%
# Per-(TF, library motif) significance from the matrix
ref_idx = tf_meta["Hom_JASPAR"].map(jaspar_id_to_idx).to_numpy()  # (n_tf,)
sim_to_ref = S[:, ref_idx]  # (n_lib, n_tf)
# p-value: mean over the 1912-null of (null >= sim), computed columnwise per TF.
pval = np.empty((n_lib, n_tf))
for t in range(n_tf):
    pval[:, t] = (S >= sim_to_ref[:, t][:, None]).mean(axis=1)
sig = (
    pval < P_SIG
)  # (n_lib, n_tf) boolean: is library motif i a significant hit for TF t?

# %%
# Pipeline (reference): per-TF significance of its top-s clusters
pipe_by_tf = cluster_df[cluster_df["P_Value"] < P_SIG]


def pipe_tops_hits(s):
    """Boolean array: is any of the pipeline's top-s clusters significant for TF t?"""
    hit = set(pipe_by_tf[pipe_by_tf["Merged_Rank"] <= s]["DBID"])
    return tf_meta["DBID"].isin(hit).to_numpy()


# Baseline (a): random library motif
# Exact expectation (no sampling): a uniform random draw is significant for TF t
# with probability k_t / n_lib, where k_t = number of significant library motifs.
k = sig.sum(axis=0)  # (n_tf,) significant library motifs per TF


def p_any_in_draws(k_t, m, shots):
    """P(>=1 significant) in `shots` distinct draws (without replacement) from m motifs,
    k_t of them significant: 1 - C(m-k_t, shots)/C(m, shots)."""
    if k_t == 0:
        return 0.0
    if k_t > m - shots:
        return 1.0
    return 1.0 - comb(m - k_t, shots) / comb(m, shots)


def rand_tops_prob(s):
    """Per-TF probability that s random library draws include a significant hit."""
    return np.array([p_any_in_draws(int(kt), n_lib, s) for kt in k])


# Baseline (b): highest-IC family-matched motif
# Caveat (assumption of homology): a library motif is a HOMER de-novo motif whose
# only identity is its BestGuess, a similarity match to a JASPAR motif. Its family
# label is therefore similarity-derived, which makes (b) a generous baseline.
lib_fam_arr = np.array([f if f is not None else "" for f in lib_fam])

# IC-ordered family candidate lists per TF, for any top-s.
fam_order = []
for fam in tf_meta["Family"]:
    cand = np.where(lib_fam_arr == fam)[0]  # library motifs in this TF's family
    fam_order.append(cand[np.argsort(-lib_ic[cand])])  # highest IC first


def fam_tops_hits(order_list, s):
    """Boolean array: is any of the top-s IC family motifs a significant hit for TF t?"""
    out = np.zeros(n_tf, dtype=bool)
    for t in range(n_tf):
        top = order_list[t][:s]
        if top.size:
            out[t] = sig[top, t].any()
    return out


# %%
# McNemar paired significance tests
def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact-binomial McNemar p-value on discordant counts (b, c).

    Under H0 each of the b+c discordant TFs is an independent coin flip, so
    b ~ Binomial(b + c, 0.5); the exact two-sided p-value is 2 * P(X <= min(b, c)).
    Preferred over the chi-square approximation when b + c is small.
    """
    n = b + c
    if n == 0:
        return 1.0
    lo = min(b, c)
    tail = sum(comb(n, i) for i in range(lo + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


def chi2_cc_p(b: int, c: int) -> float:
    """Asymptotic McNemar p-value with Yates continuity correction."""
    n = b + c
    if n == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / n
    return float(chi2.sf(stat, 1))


def mcnemar_2x2(p, o):
    """(both, pipe_only[b], other_only[c], neither) for boolean per-TF arrays p, o."""
    return (
        int((p & o).sum()),
        int((p & ~o).sum()),
        int((~p & o).sum()),
        int((~p & ~o).sum()),
    )


# Shot-budget sweep (top-s, s=1..4): pipeline vs family / random
def mcnemar_expected(p_bool, q):
    """Expected (both, pipe_only, other_only, neither) for deterministic p vs Bernoulli(q)."""
    p = p_bool.astype(float)
    return (
        float((p * q).sum()),
        float((p * (1 - q)).sum()),
        float(((1 - p) * q).sum()),
        float(((1 - p) * (1 - q)).sum()),
    )


SWEEP_SHOTS = [1, 2, 3, 4]
sweep_rows = []
for s in SWEEP_SHOTS:
    pipe_s = pipe_tops_hits(s)
    fam_s = fam_tops_hits(fam_order, s)
    q_s = rand_tops_prob(s)

    both, b, c, none = mcnemar_2x2(pipe_s, fam_s)
    sweep_rows.append(
        {
            "shot": s,
            "comparison": "pipeline vs family",
            "both": float(both),
            "pipeline_only": float(b),
            "other_only": float(c),
            "neither": float(none),
            "exact_p": round(mcnemar_exact(b, c), 5),
            "chi2_cc_p": round(chi2_cc_p(b, c), 5),
        }
    )
    both, b, c, none = mcnemar_expected(pipe_s, q_s)  # random: expected discordants
    sweep_rows.append(
        {
            "shot": s,
            "comparison": "pipeline vs random (exp.)",
            "both": both,
            "pipeline_only": b,
            "other_only": c,
            "neither": none,
            "exact_p": round(mcnemar_exact(int(round(b)), int(round(c))), 5),
            "chi2_cc_p": round(chi2_cc_p(b, c), 5),
        }
    )

sweep_df = pd.DataFrame(sweep_rows)
sweep_df.to_csv(
    f"{TAB_DIR}/baseline_mcnemar_sweep_{num_specs}_{cor_threshold}.csv", index=False
)
print("\nShot-budget sweep (matched top-s): pipeline vs family / random")
print("  random = EXPECTED discordants from Bernoulli q_t; exact p on rounded counts")
print(
    f"{'shot':>4s}{'comparison':>28s}{'both':>7s}{'pipe+':>7s}"
    f"{'other+':>8s}{'none':>7s}{'exact p':>10s}{'chi2cc p':>10s}"
)
print("-" * 81)
for r in sweep_rows:
    print(
        f"{r['shot']:>4d}{r['comparison']:>28s}{r['both']:>7.1f}"
        f"{r['pipeline_only']:>7.1f}{r['other_only']:>8.1f}{r['neither']:>7.1f}"
        f"{r['exact_p']:>10.4g}{r['chi2_cc_p']:>10.4g}"
    )


# %%
# Saturation curves: cumulative TFs recovered vs budget (top-n), one line per method.
# Every method is scored on the same TF set (tf_meta) so the curves share a denominator.
def pct(x):
    return f"{x:5.1f}  ({100 * x / n_tf:4.1f}%)"


ns = np.arange(1, N_MAX + 1)
tf_set = set(tf_meta["DBID"])

# AF3 confidence: individual PWMs ranked by the combined AF3 score (Score_Rank).
ind_df = pd.read_csv(f"{TAB_DIR}/ind_ppm_jaspar_{num_specs}_{cor_threshold}_new.csv")
ind_df = ind_df[ind_df["DBID"].isin(tf_set)]

# (1) AF3/NA-MPNN consensus: top-n clusters by size (Merged_Rank).
consensus_cum = [int(pipe_tops_hits(n).sum()) for n in ns]
# (2) AF3 confidence: top-n individual PWMs by AF3 score.
af3_cum = [
    ind_df[(ind_df["Score_Rank"] <= n) & (ind_df["P_Value"] < P_SIG)]["DBID"].nunique()
    for n in ns
]
# (3) family highest-IC: any of the top-n IC-ordered family motifs is significant.
family_cum = [int(fam_tops_hits(fam_order, n).sum()) for n in ns]
# (4) random library: expected TFs recovered in n distinct draws (closed form).
random_cum = [rand_tops_prob(n).sum() for n in ns]

# (5) ceiling (oracle): the library motif MOST similar to each TF's reference JASPAR
# motif. This uses the reference identity to pick the assignment, so it is the best
# any library-selection method could do -- an upper bound, not an achievable method.
ceil_order = np.argsort(
    -sim_to_ref, axis=0
)  # (n_lib, n_tf), best reference-match first


def ceil_tops_hits(s):
    """Is any of the top-s most-reference-similar library motifs significant for TF t?"""
    top = ceil_order[:s, :]  # (s, n_tf)
    return sig[top, np.arange(n_tf)].any(axis=0)  # (n_tf,)


ceiling_cum = [int(ceil_tops_hits(n).sum()) for n in ns]
print(
    f"\nLibrary ceiling (most reference-similar motif): "
    f"top-1 = {pct(ceiling_cum[0])}, top-{N_MAX} = {pct(ceiling_cum[-1])}"
)

curves = [
    ("ceiling (best library match)", ceiling_cum, "#333333", "*"),
    ("AF3/NA-MPNN consensus", consensus_cum, "#738fc1", "o"),
    ("AF3 confidence", af3_cum, "#8e7cc3", "s"),
    ("(b) family highest-IC", family_cum, "#d56c55", "^"),
    ("(a) random library", random_cum, "#aaaaaa", "D"),
]

fig, ax = plt.subplots(figsize=(5, 6))
for label, cum, color, marker in curves:
    ls = "--" if label.startswith("ceiling") else "-"
    ax.plot(ns, cum, marker=marker, color=color, label=label, lw=1.8, ms=7, ls=ls)
ax.axhline(n_tf, color="gray", lw=1, ls="--", alpha=0.7, label=f"Total TFs (n={n_tf})")
ax.set_xlabel("Budget (top-n assignments)")
ax.set_ylabel("TFs with significant JASPAR match")
ax.set_xticks(ns)
ax.set_ylim(0, n_tf * 1.1)
ax.set_title(f"Saturation vs baselines (n={n_tf}, p<{P_SIG})")
ax.legend(frameon=False, loc="lower right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(
    f"{FIG_DIR}/baseline_saturation_{num_specs}_{cor_threshold}.svg",
    bbox_inches="tight",
    dpi=150,
)
plt.show()

# %%
# Saturation curve values at each budget: TFs recovered (and % of n_tf) per method.
print(f"\nSaturation curves  (n = {n_tf} TFs, significant = JASPAR-null p < {P_SIG})")
print(f"{'top-n':>5s}" + "".join(f"{label:>32s}" for label, *_ in curves))
for i, n in enumerate(ns):
    print(f"{n:>5d}" + "".join(f"{pct(cum[i]):>32s}" for _, cum, *_ in curves))

# %%
