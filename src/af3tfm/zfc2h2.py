"""Shared utilities for zf-C2H2 recognition code extraction and comparison.

Ben Iovino 4/28/26 Biohub
"""

import re

# Regex for a single C2H2 zinc finger.
# Anchored on the conserved spine [FYL]-X5-[LIVMFY]-X2-H, where the [LIVMFY] slot is the
# helix +4 hydrophobic and the first zinc His sits at helix +7. The five residues between
# the two anchors are helix -2,-1,+1,+2,+3, so the specificity residues are captured
# at fixed offsets from that spine:
#   group 1 (p-1, 1 aa)  – helix -1 (reads 3' base; most important contact)
#   group 2 (p2, 1 aa)   – helix +2 (cross-strand base)
#   group 3 (p3, 1 aa)   – helix +3 (central base)
#   group 4 (p5, 1 aa)   – helix +5 (bystander; between +4 anchor and first His)
#   group 5 (p6, 1 aa)   – helix +6 (reads 5' base)
#   group 6 (linker)     – H1–H2 linker (not stored in code)
# The +4 anchor and the pre-helix aromatic are hydrophobic-core positions (they pack the
# ββα fold, they do NOT read DNA), so their classes are broadened beyond the classic F/Y
# and L/I: natural C2H2 fingers carry F/M/V/Y at +4 and tolerate L at the aromatic slot.
# Broadening recovers ~13% more annotated DNA-binding fingers (80.5%->93.5% on tfdb4_rep)
# without altering any code extracted under the stricter [FY]/[LI] classes — the captured
# specificity positions are unchanged; only additional fingers are matched.
ZF_RE = re.compile(r"C.{2,4}C.{3}[FYL].(.).(.)(.)[LIVMFY](.)(.)H(.{3,5})H")

# Recognition-code position map (indices into the 6-char code string)
#   code[0] = p-1 -> exact match required
#   code[1] = p2  -> substitution allowed
#   code[2] = p3  -> exact match required
#   code[3] = H   -> always 'H'; included for readability
#   code[4] = p5  -> captured for reference but NOT scored (helix +5 bystander)
#   code[5] = p6  -> substitution allowed
_P_EXACT = (0, 2)
_P_FLEX = (1, 5)  # p2, p6 — DNA-contacting flexible positions (p5 excluded)
_P_SCORED = (0, 1, 2, 5)  # p-1, p2, p3, p6 — the four DNA-contacting positions
_INF = 10_000


def extract_zf_codes(sequence: str) -> list[str]:
    """
    Find all C2H2 zinc finger repeats in protein sequence and return recognition
    codes ordered N→C.

    Each code is a fixed 6-character string: p-1 | p2 | p3 | H | p5 | p6
    where positions are numbered relative to the recognition alpha-helix
    (the conserved [LI] hydrophobic is at helix +4, the first zinc His at +7).

    Args:
        sequence: full protein sequence

    Returns:
        List of recognition codes, ordered N→C. Each code is a 6-character string
    """

    codes = []
    for m in ZF_RE.finditer(sequence):
        p_neg1, p2, p3, p5, p6 = (
            m.group(1),  # helix -1
            m.group(2),  # helix +2
            m.group(3),  # helix +3
            m.group(4),  # helix +5
            m.group(5),  # helix +6
        )
        code = p_neg1 + p2 + p3 + "H" + p5 + p6  # always 6 characters
        codes.append((m.start(), code))

    return [c for _, c in sorted(codes)]


def extract_zf_codes_indexed(sequence: str) -> list[tuple[int, int, str]]:
    """
    As extract_zf_codes, but keeping each finger's span so it can be intersected with
    domain annotations.

    Args:
        sequence: full protein sequence

    Returns:
        List of (start, end, code) ordered N→C, with 1-based inclusive spans to match
        the coordinates interproscan reports.
    """

    hits = []
    for m in ZF_RE.finditer(sequence):
        code = m.group(1) + m.group(2) + m.group(3) + "H" + m.group(4) + m.group(5)
        hits.append((m.start() + 1, m.end(), code))

    return sorted(hits)


def dna_binding_finger_indices(
    sequence: str, ranges: list[tuple[int, int]], pad: int = 2
) -> list[int]:
    """
    Indices (into the full N→C finger array) of the fingers that engage DNA.

    Only a subset of a long array contacts DNA, and that subset is what sets specificity,
    so scoring is restricted to it. A finger counts as DNA-binding when its regex span
    overlaps an annotated DNA-binding range; pad absorbs the few-residue disagreement
    between where the regex starts (the first zinc Cys) and where interproscan calls the
    domain boundary.

    Args:
        sequence: full protein sequence
        ranges: (start, end) 1-based inclusive spans annotated as DNA-binding
        pad: residues of slack allowed at each end when testing overlap

    Returns:
        Sorted finger indices, empty if no finger overlaps any range.
    """

    fingers = extract_zf_codes_indexed(sequence)
    return [
        i
        for i, (s, e, _) in enumerate(fingers)
        if any(not (e < rs - pad or s > re_ + pad) for rs, re_ in ranges)
    ]


def build_fingerprint(codes: list[str]) -> tuple[int, tuple[str, ...]]:
    """
    Convert an ordered list of recognition codes into a hashable fingerprint:
    (n_zf, (code_1, code_2, ...)).

    Args:
        codes: list of recognition codes (6-char strings), ordered N→C

    Returns:
        Tuple of (number of ZFs, tuple of recognition codes)
    """

    return (len(codes), tuple(codes))


def zf_pair_distance(ca: str, cb: str) -> int:
    """
    Distance between two single-ZF recognition codes (each 6 chars).

    Returns _INF if either exact-match position (p-1 or p3) differs.
    Otherwise returns the number of mismatches at the flexible positions
    (p2, p6): 0, 1, or 2. (p5 is captured in the code but not scored.)

    Args:
        ca, cb: two recognition codes (6-character strings)

    Returns:
        0, 1, or 2 for flexible mismatches; _INF for incompatible ZF pairs
    """

    for i in _P_EXACT:
        if ca[i] != cb[i]:
            return _INF

    return sum(1 for i in _P_FLEX if ca[i] != cb[i])


def fingerprint_distance(fp_a: tuple, fp_b: tuple) -> int:
    """
    Distance between two protein fingerprints of the same ZF count.
    Returns the sum of per-ZF distances. A single incompatible ZF pair
    (p-1 or p3 mismatch) raises the total to _INF, preventing merging.

    Args:
        fp_a, fp_b: two fingerprints (n_zf, (code_1, code_2, ...))

    Returns:
        Total distance (sum of per-ZF distances) or _INF if any incompatible ZF pair
    """

    _, codes_a = fp_a
    _, codes_b = fp_b
    total = 0
    for ca, cb in zip(codes_a, codes_b):
        d = zf_pair_distance(ca, cb)
        if d >= _INF:
            return _INF
        total += d
    return total


def best_core_match(
    codes_a: list[str],
    codes_b: list[str],
    k_min: int = 3,
    max_count_diff: int = 2,
) -> dict | None:
    """
    Find the best contiguous, gapless recognition-code core shared between two finger
    arrays that may differ in length (evolutionary finger gain/loss).

    Rationale: only a contiguous run of ZFs engages DNA (Zif268/EGR1 modular model),
    and flanking fingers of a long array need not bind DNA. So two orthologs whose total
    finger counts differ by one or two may still share the same core specificity. We slide
    array B against array A at every relative offset (register is unknown when lengths
    differ), and within each overlap find the longest contiguous run where every finger
    passes the exact-position gate (p-1 and p3 match). Flexible-position (p2, p6)
    mismatches do not break the run but lower the score.

    Args:
        codes_a, codes_b: recognition code arrays (6-char strings), ordered N->C
        k_min: minimum contiguous core length (in fingers) to report a match
        max_count_diff: reject the pair outright if the total finger counts differ by
            more than this (default 2)

    Returns:
        None if the count difference exceeds max_count_diff or no run of length >= k_min
        exists. Otherwise a dict:
          core_len : length of the best contiguous compatible run
          score    : 1 - (flex mismatches in core)/(core_len*2), in [0, 1]
          a_start  : index of the core's first finger in codes_a
          b_start  : index of the core's first finger in codes_b
          a_len, b_len : finger counts of the two arrays
          ratio    : core_len / max(a_len, b_len) (1.0 = core spans the longer array)
          a_interior, b_interior : True if the core touches neither terminus of that array
    """

    la, lb = len(codes_a), len(codes_b)
    if abs(la - lb) > max_count_diff:
        return None

    best = None
    # relative offset g: align codes_b[j] with codes_a[j + g]
    for g in range(-(lb - 1), la):
        run_len, run_flex, run_a_start = 0, 0, 0
        # j indexes codes_b; i = j + g indexes codes_a; only where both valid
        j_lo = max(0, -g)
        j_hi = min(lb, la - g)
        for j in range(j_lo, j_hi):
            i = j + g
            d = zf_pair_distance(codes_a[i], codes_b[j])
            if d >= _INF:  # exact-position mismatch breaks the contiguous run
                run_len = 0
                run_flex = 0
                continue
            if run_len == 0:
                run_a_start = i
            run_len += 1
            run_flex += d
            if run_len >= k_min:
                score = 1.0 - run_flex / (run_len * len(_P_FLEX))
                a_start = run_a_start
                b_start = run_a_start - g
                cand = (run_len, score, a_start, b_start)
                if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                    best = cand

    if best is None:
        return None

    core_len, score, a_start, b_start = best
    a_end, b_end = a_start + core_len - 1, b_start + core_len - 1
    return {
        "core_len": core_len,
        "score": round(score, 4),
        "a_start": a_start,
        "b_start": b_start,
        "a_len": la,
        "b_len": lb,
        "ratio": round(core_len / max(la, lb), 4),
        "a_interior": a_start > 0 and a_end < la - 1,
        "b_interior": b_start > 0 and b_end < lb - 1,
    }


def _pair_residue_matches(ca: str, cb: str) -> int:
    """
    Matching DNA-contacting residues (of 4) between two fingers. A substitution at an
    exact-match position (p-1 or p3) voids the whole finger, so it scores 0 rather than
    partial credit — those two residues set specificity and a change at either reassigns
    the base being read.

    Args:
        ca, cb: two recognition codes (6-character strings)

    Returns:
        0-4 matching residues at p-1, p2, p3, p6
    """

    if any(ca[p] != cb[p] for p in _P_EXACT):
        return 0
    return sum(1 for p in _P_SCORED if ca[p] == cb[p])


def binding_finger_similarity(
    binding_codes: list[str],
    binding_offsets: list[int],
    human_codes: list[str],
) -> float | None:
    """
    Fraction of matching DNA-contacting residues between a zebrafish TF's DNA-binding
    fingers and the best-matching stretch of a human TF's finger array.

    Specificity comes from the fingers that actually engage DNA, not from the whole array —
    in a long zinc finger protein most fingers never touch the duplex — so only the
    DNA-binding fingers are scored. Their spacing within the zebrafish array is preserved
    and the block is slid along the human array at every register, best register winning.
    Each finger contributes its four DNA-contacting positions (p-1, p2, p3, p6), so 12 of
    16 matching residues over four binding fingers is 75%. A binding finger with no human
    partner at a register scores 0 of 4, which is what makes registers hanging off the end
    of the human array lose.

    Args:
        binding_codes: recognition codes of the DNA-binding fingers, ordered N->C
        binding_offsets: each binding finger's index within the full zebrafish array, so
            gaps between non-adjacent binding fingers are carried into the alignment
        human_codes: the human TF's full recognition code array, ordered N->C

    Returns:
        Matching-residue fraction in [0, 1], or None if either side has no fingers.
    """

    if not binding_codes or not human_codes:
        return None

    span = binding_offsets[-1] - binding_offsets[0]  # fingers spanned, gaps included
    best = 0
    # register g aligns the first binding finger with human_codes[g]; the block may start
    # before or run past the human array so short arrays still find their best fit
    for g in range(-span, len(human_codes)):
        matches = 0
        for code, off in zip(binding_codes, binding_offsets):
            j = g + (off - binding_offsets[0])
            if 0 <= j < len(human_codes):
                matches += _pair_residue_matches(code, human_codes[j])
        best = max(best, matches)

    return best / (len(binding_codes) * len(_P_SCORED))


def core_residue_similarity(
    codes_a: list[str],
    codes_b: list[str],
    k_min: int = 3,
    max_count_diff: int = 2,
) -> float | None:
    """
    Fraction of matching DNA-contacting residues between two whole finger arrays, aligned
    at the register set by their best contiguous gapless core (see best_core_match).

    Superseded by binding_finger_similarity for ortholog scoring, which restricts the
    comparison to the fingers annotated as DNA-binding rather than using the whole array.
    Retained for TFs that have no DNA-binding annotation to restrict to.

    Args:
        codes_a, codes_b: recognition code arrays (6-char strings), ordered N->C
        k_min: minimum contiguous core length (in fingers) for the register to be defined
        max_count_diff: reject the pair outright if finger counts differ by more than this

    Returns:
        Matching-residue fraction in [0, 1], or None if the arrays share no scoreable core
        (see best_core_match) and so have no alignment register.
    """

    info = best_core_match(codes_a, codes_b, k_min=k_min, max_count_diff=max_count_diff)
    if info is None:
        return None

    la, lb = len(codes_a), len(codes_b)
    offset = (
        info["a_start"] - info["b_start"]
    )  # codes_b[j] aligns with codes_a[j + offset]
    matches = 0
    for j in range(lb):
        i = j + offset
        if not 0 <= i < la:
            continue  # overhanging finger, no partner: 0 of its 4 positions match
        matches += _pair_residue_matches(codes_a[i], codes_b[j])

    return matches / (max(la, lb) * len(_P_SCORED))


def same_count_sim(codes_a: list[str], codes_b: list[str], max_bad: int = 0) -> float:
    """
    Normalized similarity [0, 1] between two recognition code arrays of equal length.
    By default (max_bad=0) any ZF that fails the exact-position gate (p-1 or p3
    mismatch) zeroes the whole score.

    Args:
        codes_a, codes_b: two lists of recognition codes (6-character strings) of equal length
        max_bad: number of exact-position-mismatch ZFs to tolerate (default 0)

    Returns:
        Normalized similarity score between 0 and 1, where 1 means identical codes and 0 means
        all ZFs have at least one mismatch
    """

    m = len(codes_a)
    total, n_bad = 0, 0
    for ca, cb in zip(codes_a, codes_b):
        d = zf_pair_distance(ca, cb)
        if d >= _INF:
            n_bad += 1
            if n_bad > max_bad:
                return 0.0
            total += len(
                _P_FLEX
            )  # tolerated exact-mismatch ZF: charge max per-ZF distance
        else:
            total += d

    return 1.0 - total / (m * len(_P_FLEX))  # flex positions per ZF
