"""
Shared PWM consensus/clustering helpers used by the specificity notebooks
(notebooks/03_building-ppms/03_tune-consensus.py, notebooks/04_database-and-applications/04_cluster-consistency.py) and
data/database/export_data.py.

UPGMA clustering of predicted PPMs, medoid-seeded consensus averaging,
IC-weighted recovery scoring against a reference motif, aligned logo plotting,
and B-strand PPM extraction from specificity NPZ files.

Ben Iovino 8/6/26 CZ-Biohub
"""

from pathlib import Path

import logomaker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from af3tfm.utils import pwm_similarity, reverse_complement_pwm

BASE_COLORS = {"A": "#1a9850", "C": "#4393c3", "G": "#d6604d", "T": "#762a83"}


def _to_logo_df(pwm: np.ndarray) -> pd.DataFrame:
    """
    Convert (L, 4) probability matrix to logomaker weight (bits) DataFrame.

    Args:
        pwm: (L, 4) array of probabilities for A, C, G, T in that order

    Returns:
        DataFrame with columns A, C, G, T and index 1-based position, containing bits values.
    """

    df = pd.DataFrame(pwm, columns=list("ACGT"))
    df.index = range(1, len(df) + 1)

    return logomaker.transform_matrix(
        df, from_type="probability", to_type="weight"
    ).clip(lower=0)


def _pad(pwm: np.ndarray, left: int, total: int) -> np.ndarray:
    """
    Pad a PWM with uniform (0.25) rows to fit a total length coordinate.

    Args:
        pwm: (L, 4) array of probabilities for A, C, G, T in that order
        left: number of uniform rows to add to the left
        total: total length after padding

    Returns:
        Padded PWM as (total, 4) array
    """

    right = total - left - len(pwm)
    uniform = np.full((1, 4), 0.25)
    parts = []
    if left > 0:
        parts.append(np.tile(uniform, (left, 1)))
    parts.append(pwm)
    if right > 0:
        parts.append(np.tile(uniform, (right, 1)))
    return np.vstack(parts)


def cluster_upgma(
    scored_with_null: list,
    cor_threshold: float = 0.73,
) -> list:
    """
    UPGMA (average linkage) hierarchical clustering with a pure correlation threshold.

    Merges the pair of clusters with the highest average-linkage cor, provided
    cor >= cor_threshold. Similarity is computed with min_overlap=6, so short
    spurious overlaps are already excluded without an Ncor penalty.
    Stops when no valid merge remains.

    Inter-cluster similarities are updated with the Lance-Williams average-linkage
    formula so that no pairwise recomputation is needed after each merge.

    Args:
        scored_with_null : list of (ppm, dna, null_mean, null_std, null_sims)
        cor_threshold : float, minimum correlation for merging clusters

    Returns:
        list of (orig_rank, cluster_id, ppm, dna, align_offset, align_rc, is_medoid)
    """

    n = len(scored_with_null)
    ppms = [item[0] for item in scored_with_null]

    # Precompute all pairwise correlations
    pair_cor = {}
    for i in range(n):
        for j in range(i + 1, n):
            cor, _, _ = pwm_similarity(ppms[i], ppms[j])
            pair_cor[(i, j)] = cor

    def raw_cor(i, j):
        return pair_cor[(min(i, j), max(i, j))]

    # UPGMA state: parallel lists of (label, size, member_set)
    labels = list(range(n))
    sizes = [1] * n
    members = [{i} for i in range(n)]
    next_label = n

    # Inter-cluster similarities; keys are (lower, higher) cluster labels
    ic_cor = dict(pair_cor)

    while len(labels) > 1:
        best_val, best_pair = -np.inf, None
        for a in range(len(labels)):
            for b in range(a + 1, len(labels)):
                key = (min(labels[a], labels[b]), max(labels[a], labels[b]))
                c = ic_cor.get(key, -np.inf)
                if c >= cor_threshold and c > best_val:
                    best_val, best_pair = c, (a, b)

        if best_pair is None:
            break

        a, b = best_pair
        lbl_a, lbl_b = labels[a], labels[b]
        sz_a, sz_b = sizes[a], sizes[b]

        new_lbl = next_label
        next_label += 1
        new_sz = sz_a + sz_b
        new_members = members[a] | members[b]

        # Lance-Williams update for every remaining cluster c
        remaining = [i for i in range(len(labels)) if i not in (a, b)]
        for idx_c in remaining:
            lbl_c = labels[idx_c]
            key_ac = (min(lbl_a, lbl_c), max(lbl_a, lbl_c))
            key_bc = (min(lbl_b, lbl_c), max(lbl_b, lbl_c))
            key_new = (min(new_lbl, lbl_c), max(new_lbl, lbl_c))
            ic_cor[key_new] = (sz_a * ic_cor[key_ac] + sz_b * ic_cor[key_bc]) / new_sz

        labels = [lbl for i, lbl in enumerate(labels) if i not in (a, b)] + [new_lbl]
        sizes = [sz for i, sz in enumerate(sizes) if i not in (a, b)] + [new_sz]
        members = [m for i, m in enumerate(members) if i not in (a, b)] + [new_members]

    # Sort clusters by lowest member index for deterministic IDs
    final = sorted(members, key=min)
    cluster_id = {}
    for cid, mem_set in enumerate(final):
        for i in mem_set:
            cluster_id[i] = cid + 1

    # Medoid: member with highest total cor to all other members (consensus-closest)
    def recompute_medoid(idxs):
        if len(idxs) == 1:
            return idxs[0]
        totals = {i: sum(raw_cor(i, j) for j in idxs if j != i) for i in idxs}
        return max(totals, key=totals.get)

    medoid_of = {}
    for mem_set in final:
        idxs = list(mem_set)
        med = recompute_medoid(idxs)
        for i in idxs:
            medoid_of[i] = med

    # Alignment offsets relative to each cluster's medoid
    align = {}
    for mem_set in final:
        med = medoid_of[list(mem_set)[0]]
        for i in mem_set:
            if i == med:
                align[i] = (0, False)
            else:
                _, off, rc = pwm_similarity(ppms[med], ppms[i])
                align[i] = (off, rc)

    return [
        (
            k + 1,
            cluster_id[k],
            ppms[k],
            scored_with_null[k][1],
            align[k][0],
            align[k][1],
            medoid_of[k] == k,
        )
        for k in range(n)
    ]


def cluster_consensus_pwm(members: list) -> np.ndarray:
    """
    Average cluster member PPMs into a single consensus in the seed's coordinate frame.

    members[0] is the seed (offset=0, rc=False); others are aligned via pwm_similarity offsets.
    Positions where only the seed contributes keep the seed's probabilities unchanged.

    Args:
        members: list of (ppm, offset, rc) for each cluster member, where ppm is the original PPM.

    Returns:
        Consensus PPM as (L, 4) array, where L is the seed's length.
    """

    seed_ppm = members[0][0]
    L = len(seed_ppm)
    accum = seed_ppm.astype(float).copy()
    weight = np.ones(L)

    # For each non-seed member, add its probabilities to the consensus at the aligned positions
    for ppm_j, off, rc in members[1:]:
        ppm_use = reverse_complement_pwm(ppm_j) if rc else ppm_j
        L_j = len(ppm_use)
        s = max(0, off)
        e = min(L, off + L_j)
        if s >= e:
            continue
        accum[s:e] += ppm_use[s - off : e - off]
        weight[s:e] += 1

    return accum / weight[:, np.newaxis]


def pred_recovery_fraction(
    pred_ppm: np.ndarray,
    jaspar_ppm: np.ndarray,
    offset: int,
    is_rc: bool,
) -> float:
    """
    JASPAR-IC-weighted recovery fraction.

    Denominator is total JASPAR IC across all positions; numerator is
    JASPAR-IC-weighted per-position similarity in the overlap region
    (non-overlapping JASPAR positions contribute 0). Directly answers
    "what fraction of the reference motif's information content is recovered?"

    Per-position similarity: (sum_b min(pred_b, ref_b) - baseline) / (1 - baseline)
    where baseline = sum_b min(0.25, ref_b); 0 = uniform predictor, 1 = perfect.
    Final score clipped to [0, 1].

    Args:
        pred_ppm: (L, 4) array of probabilities for A, C, G, T in that order
        jaspar_ppm: (L_ref, 4) array of probabilities for A, C, G, T in that order
        offset: int, alignment offset of pred relative to jaspar (positive means pred shifted right)
        is_rc: bool, whether pred is reverse-complemented relative to jaspar

    Returns:
        Recovery fraction as a float in [0, 1]
    """

    ref = reverse_complement_pwm(jaspar_ppm) if is_rc else jaspar_ppm
    la, lb = len(pred_ppm), len(ref)

    ref_ic = (ref * np.log2(ref / 0.25 + 1e-9)).sum(axis=1)
    ref_ic = np.maximum(ref_ic, 0.0)
    total_ic = ref_ic.sum()
    if total_ic == 0:
        return 0.0

    s_pred = max(0, offset)
    e_pred = min(la, lb + offset)
    s_ref = max(0, -offset)
    e_ref = min(lb, la - offset)

    if s_pred >= e_pred:
        return 0.0

    overlap = np.minimum(pred_ppm[s_pred:e_pred], ref[s_ref:e_ref]).sum(axis=1)

    ref_slice = ref[s_ref:e_ref]
    baseline = np.minimum(ref_slice, 0.25).sum(axis=1)
    denominator = 1.0 - baseline
    safe_denom = np.where(denominator > 1e-9, denominator, 1.0)
    sim = np.maximum((overlap - baseline) / safe_denom, 0.0)

    numerator = (ref_ic[s_ref:e_ref] * sim).sum()

    return float(np.clip(numerator / total_ic, 0.0, 1.0))


def plot_aligned_logos(
    panels: list,
    tf_name: str,
    subtitle: str,
    ref_panel: tuple | None = None,
    out_path: str | None = None,
):
    """
    Show a stack of PWM logos in one shared coordinate frame, one panel per PWM.

    Args:
        panels: list of (ppm, label, metrics) drawn top-to-bottom beneath the reference
        tf_name: str, TF name for title
        subtitle: str, description of the panel stack for the figure title
        ref_panel: (ppm, label) | None, reference motif drawn at the top in its own frame
        out_path: str | None, path to save the plot
    """

    if ref_panel is not None:
        frame_ppm = ref_panel[0]
        placed = [(frame_ppm, 0, False, ref_panel[1], "")]
        rest = panels
    else:
        frame_ppm, first_label, first_metrics = panels[0]
        placed = [(frame_ppm, 0, False, first_label, first_metrics)]
        rest = panels[1:]

    # Align every PWM to the reference frame; offset is positive when shifted right
    for ppm, label, metrics in rest:
        _, off, rc = pwm_similarity(frame_ppm, ppm)
        ppm_use = reverse_complement_pwm(ppm) if rc else ppm
        placed.append((ppm_use, off, rc, label, metrics))

    left_margin = -min(0, min(off for _, off, *_ in placed))
    total = max(left_margin + off + len(ppm_use) for ppm_use, off, *_ in placed)

    n = len(placed)
    fig, axes = plt.subplots(n, 1, figsize=(max(4, 0.55 * total), 1.6 * n), sharex=True)
    axes = np.atleast_1d(axes)

    for ax, (ppm_use, off, rc, label, metrics) in zip(axes, placed):
        padded = _pad(ppm_use, left_margin + off, total)

        tags = []
        if rc:
            tags.append("RC")
        if off != 0:
            tags.append(f"off={off:+d}")
        parts = [label]
        if tags:
            parts.append(f"[{', '.join(tags)}]")
        if metrics:
            parts.append(metrics)

        logo = logomaker.Logo(
            _to_logo_df(padded),
            ax=ax,
            color_scheme=BASE_COLORS,
            font_name="DejaVu Sans",
        )
        logo.style_spines(visible=False)
        logo.style_spines(spines=["left", "bottom"], visible=True)
        ax.set_ylabel("bits")
        ax.set_title("  ".join(parts))

    axes[-1].set_xlabel("Position")
    plt.suptitle(f"{tf_name} — {subtitle}")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()


_DNA_INDICES = [21, 22, 23, 24]


def _load_b_strand_ppm(npz_path: Path) -> np.ndarray:
    """
    Return (L, 4) B-strand PPM from a specificity NPZ file.

    Args:
        npz_path: Path to the NPZ file containing "predicted_ppm" and "dna_mask"

    Returns:
        (L, 4) array of probabilities for A, C, G, T in that order, for the B strand only
    """

    data = np.load(str(npz_path), allow_pickle=True)
    ppm_full = data["predicted_ppm"]  # (N, 33)
    dna_mask = data["dna_mask"].astype(bool)
    dna_idx = np.where(dna_mask)[0]
    ppm_dna = ppm_full[dna_idx][:, _DNA_INDICES]  # (n_dna, 4)
    encoded = data["encoded_residues"][dna_idx]
    fwd_chain = encoded[0][
        0
    ]  # First DNA chain label ("B" for monomer, "C" for homodimer)
    b_mask = np.array([e[0] == fwd_chain for e in encoded])

    return ppm_dna[b_mask]
