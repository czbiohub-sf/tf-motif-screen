"""Generate GC-matched genomic non-motif sequences per TF, with a core filter.

For each TF we read its cognate PWM from motifs/<motif_id>.txt (the TF -> motif
assignment comes from the motifs.txt manifest built by get_motifs.py), sample
positives from it to fix the GC target, then draw genomic k-mers of the
motif width that are (a) below the PWM's p<PVAL full-width log-odds threshold
and (b) free of the motif's high-IC core seed.

The reject rule is a  contiguous-seed match: build the motif's high-IC
consensus core (contiguous span of IC >= IC_CORE bits), enumerate every
length-MINSEED window of it plus their reverse-complements, and reject any
candidate k-mer that contains any of those seeds as a substring.

Ben Iovino  08/27/26    CZ-Biohub
"""

import csv
import math
import os
import random
import re
import sys
from collections import defaultdict

from af3tfm.config import PATHS
from af3tfm.dataset import DATA_DIR

PWM_DIR = DATA_DIR / "sequences/cisbp/motifs"
MANIFEST = DATA_DIR / "sequences/cisbp/motifs.txt"
PEAKS = PATHS["zebrahub_peaks_fasta"]
OUTDIR = DATA_DIR / "sequences/cisbp/nonmotif_negatives"
N_POS, NEG_PER_POS, SEED = 30, 5, 0  # 30 x 5 = 150 negatives per TF
PVAL = 1e-3
IC_CORE = 1.0  # bits; positions that count as "core"
MINSEED = 4  # forbid contiguous core matches >= this
PER_BIN = 500  # per needed GC bin (150 negatives/TF)
MAX_TRIES = 6_000_000  # safety cap; rarely hit for mid-GC bins
WINDOW = (0, 1, -1, 2, -2)  # GC offsets draw() falls back through

BASES = "ACGT"
COMP = str.maketrans("ACGTacgtN", "TGCATGCAN")


def revcomp(s):
    return s.translate(COMP)[::-1]


# PWM handling
def read_pwm(path):
    """Return PPM as list of dicts (one per position), pseudocount-smoothed."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(">") or line.startswith("#"):
                continue
            vals = re.split(r"[\s,]+", line)
            nums = []
            for v in vals:
                try:
                    nums.append(float(v))
                except ValueError:
                    pass
            if nums:
                rows.append(nums)

    if not rows:
        sys.exit(f"No numeric rows parsed from {path}")

    # CisBP style: leading position-index column (Pos A C G T) -> drop the index
    if all(len(r) == 5 for r in rows):
        rows = [r[1:] for r in rows]

    # JASPAR style: 4 rows (A,C,G,T) x width. HOMER style: width rows x 4 cols.
    if len(rows) == 4 and len(rows[0]) != 4:
        cols = list(zip(*rows))
    else:
        if any(len(r) != 4 for r in rows):
            sys.exit(f"Expected 4 values per position in {path}")
        cols = rows

    ppm = []
    for col in cols:
        tot = sum(col)
        if tot <= 0:
            sys.exit("PWM column sums to zero")
        # pseudocount so log-odds never diverges
        probs = [(c / tot) * 0.998 + 0.0005 for c in col]
        ppm.append(dict(zip(BASES, probs)))
    return ppm


def log_odds(seq, ppm, bg=0.25):
    """Max log-odds score over both strands; -inf if seq has non-ACGT."""

    def score_fwd(s):
        tot = 0.0
        for i, ch in enumerate(s):
            p = ppm[i].get(ch)
            if p is None:
                return float("-inf")
            tot += math.log2(p / bg)
        return tot

    return max(score_fwd(seq), score_fwd(revcomp(seq)))


def score_threshold(ppm, peak_seqs, rng, p=1e-3, n_sample=200000):
    """Empirical log-odds cutoff at the upper p-quantile of genomic background."""
    k = len(ppm)
    scores = []
    tries = 0
    while len(scores) < n_sample and tries < n_sample * 20:
        tries += 1
        s = rng.choice(peak_seqs)
        if len(s) <= k:
            continue
        i = rng.randrange(0, len(s) - k + 1)
        sub = s[i : i + k].upper()
        sc = log_odds(sub, ppm)
        if sc != float("-inf"):
            scores.append(sc)
    if not scores:
        sys.exit("Could not build background score distribution")
    scores.sort()
    idx = min(len(scores) - 1, int(round((1.0 - p) * len(scores))))
    return scores[idx]


# Filtering
def low_complexity(seq):
    """Reject homopolymer runs >=5 or any base above 60% of the k-mer."""
    if re.search(r"(A{5,}|C{5,}|G{5,}|T{5,})", seq):
        return True
    for b in BASES:
        if seq.count(b) / len(seq) > 0.60:
            return True
    return False


def gc_count(seq):
    return seq.count("G") + seq.count("C")


# Sampling
def sample_positive(ppm, rng):
    return "".join(
        rng.choices(BASES, weights=[ppm[i][b] for b in BASES], k=1)[0]
        for i in range(len(ppm))
    )


def read_fasta(path):
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur).upper())
                    cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append("".join(cur).upper())
    return [s for s in seqs if s]


# Core seeding
def ic(coldict):
    return 2.0 + sum(p * math.log2(p) for p in coldict.values())


def core_consensus(ppm, thr=IC_CORE):
    """Consensus string over the contiguous span of IC >= thr positions ('' if flat)."""
    keep = [i for i, c in enumerate(ppm) if ic(c) >= thr]
    if not keep:
        return ""
    lo, hi = min(keep), max(keep)
    return "".join(max(ppm[i], key=ppm[i].get) for i in range(lo, hi + 1))


def core_seeds(core, L=MINSEED):
    """Every length-L window of the core consensus plus reverse-complements."""
    S = set()
    for i in range(len(core) - L + 1):
        w = core[i : i + L]
        S.add(w)
        S.add(revcomp(w))
    return S


def index_by_gc_core(peak_seqs, k, ppm, cutoff, seeds, rng, needed):
    """Bin genomic k-mers by GC, tagging each seed-clean (True) or seed-carrying.

    Only the GC bins in `needed` (the ones draw() will consume) are filled; k-mers
    at any other GC are discarded before the expensive log-odds check. Stops once
    every needed bin holds PER_BIN entries -- or MAX_TRIES is hit, which only
    happens for motifs whose GC sits at the rare extremes of the genomic range.
    """
    bins = defaultdict(list)  # gc -> list of (seq, is_clean)
    seen = set()
    tries = 0

    while (
        tries < MAX_TRIES and min((len(bins[g]) for g in needed), default=0) < PER_BIN
    ):
        tries += 1
        s = rng.choice(peak_seqs)
        i = rng.randrange(0, len(s) - k + 1)
        sub = s[i : i + k]
        g = gc_count(sub)
        if g not in needed or len(bins[g]) >= PER_BIN:  # unwanted or full GC bin
            continue
        if sub in seen or any(c not in BASES for c in sub):  # skip non-ACGT
            continue
        if low_complexity(sub):  # skip homopolymers or high-mono content
            continue
        if log_odds(sub, ppm) > cutoff:  # full-width site: always excluded
            continue
        seen.add(sub)
        is_clean = not any(seed in sub for seed in seeds)  # seed on either strand
        bins[g].append((sub, is_clean))
    return bins


def draw(bins, g, n, used, prefer_clean=True):
    """Draw up to n unused seqs at GC=g (then +/-1,2), clean-first. Returns
    (list of (seq, is_clean))."""
    picked = []

    def take(want_clean):
        for d in WINDOW:
            for seq, clean in bins.get(g + d, []):
                if len(picked) >= n:
                    return
                if seq in used:
                    continue
                if clean != want_clean:
                    continue
                used.add(seq)
                picked.append((seq, clean))

    if prefer_clean:
        take(True)
        if len(picked) < n:
            take(False)
    else:
        take(True)
        take(False)
    return picked


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print("loading peaks ...", flush=True)
    peak_seqs = read_fasta(PEAKS)
    print(f"loaded {len(peak_seqs)} peak seqs\n", flush=True)

    # each TF's cognate PWM is pulled from motifs/<motif_id>.txt via the
    # "dbid,tf_name,motif_id" motifs.txt manifest (CisBP for direct TFs, JASPAR
    # for inferred ones); TFs with no usable motif are absent from the manifest
    entries = []
    with open(MANIFEST, newline="") as fh:
        for _dbid, tf_name, motif_id in csv.reader(fh):
            entries.append((tf_name, os.path.join(PWM_DIR, f"{motif_id}.txt")))

    hdr = (
        f"{'TF':14s} {'k':>2s} {'core':16s} {'seeds':>5s} "
        f"{'made':>4s} {'clean':>5s} {'leak':>4s} {'warn':>4s}"
    )
    print(hdr)
    print("-" * len(hdr))

    combined = []
    summary = []
    for name, path in entries:
        rng = random.Random(SEED)
        ppm = read_pwm(path)
        k = len(ppm)
        cutoff = score_threshold(ppm, peak_seqs, rng, p=PVAL)

        core = core_consensus(ppm)
        seeds = core_seeds(core) if len(core) >= MINSEED else set()
        core_str = core if core else "(none)"
        if not seeds:  # core shorter than a seed -> unfilterable
            core_str += " *"

        # Sample positives first so we know which GC bins draw() will consume, then
        # fill only those bins (plus the +/-WINDOW fallbacks) instead of all 0..k.
        positives = [sample_positive(ppm, rng) for _ in range(N_POS)]
        pos_gc = [gc_count(p) for p in positives]
        needed = {g + d for g in pos_gc for d in WINDOW if 0 <= g + d <= k}

        bins = index_by_gc_core(peak_seqs, k, ppm, cutoff, seeds, rng, needed)

        used = set()
        recs = []
        made = clean_n = leak_n = warn = 0
        for pi, g in enumerate(pos_gc):
            picks = draw(bins, g, NEG_PER_POS, used, prefer_clean=True)
            if len(picks) < NEG_PER_POS:
                warn += 1
            for ni, (neg, is_clean) in enumerate(picks):
                nid = f"{name}_neg{pi:03d}_{ni}"
                recs.append((nid, neg))
                made += 1
                clean_n += int(is_clean)
                leak_n += int(not is_clean)
                combined.append(
                    (
                        name,
                        nid,
                        neg,
                        gc_count(neg),
                        f"{log_odds(neg, ppm):.3f}",
                        "clean" if is_clean else "leak",
                    )
                )

        with open(os.path.join(OUTDIR, f"{name}_negatives.fa"), "w") as fh:
            for nid, seq in recs:
                fh.write(f">{nid}\n{seq}\n")

        print(
            f"{name:14s} {k:2d} {core_str:16s} {len(seeds):5d} "
            f"{made:4d} {clean_n:5d} {leak_n:4d} {warn:4d}",
            flush=True,
        )
        summary.append((name, made, clean_n, leak_n, bool(seeds)))

    with open(os.path.join(OUTDIR, "all_negatives.tsv"), "w") as fh:
        fh.write("tf\tnegative_id\tsequence\tgc\tlogodds\tcore_status\n")
        for row in combined:
            fh.write("\t".join(str(x) for x in row) + "\n")

    print(f"\n{len(entries)} TFs, {sum(s[1] for s in summary)} negatives total.")
    unfiltered = [s[0] for s in summary if not s[4]]
    leaky = [s for s in summary if s[4] and s[3] > 0]  # had seeds yet still leaked
    if unfiltered:
        print(
            f"Unfilterable (core < {MINSEED} bp, no seed to avoid): "
            f"{', '.join(unfiltered)}"
        )
    if leaky:
        print("Residual seed leak despite filtering (GC bin ran short of clean):")
        for n, made, c, lk, _ in sorted(leaky, key=lambda x: -x[3]):
            print(
                f"  {n:14s} {lk:4d}/{made} carry a core seed ({100 * lk / made:.0f}%)"
            )
    if not leaky:
        print("No residual seed leak on any filtered TF.")


if __name__ == "__main__":
    main()
