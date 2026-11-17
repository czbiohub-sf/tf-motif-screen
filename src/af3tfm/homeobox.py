"""Shared utilities for homeodomain recognition code extraction and alignment.

Ben Iovino 4/28/26 Biohub
"""

import gzip
import io
from pathlib import Path

import pandas as pd
import pyhmmer

from af3tfm.dataset import DATA_DIR

HD_POSITIONS = [46, 49, 50, 51, 53]  # canonical: 47, 50, 51, 52, 54
CANONICAL_P51 = "N"
CANONICAL_P51_MS = 50
IPR_ACCESSION = "IPR001356"  # canonical Homeodomain
# Homeobox InterPro accessions whose sequences align to PF00046. TALE-class
# proteins (IRX/MEIS/PBX/PKNOX/TGIF/MKX/SIX) are annotated "KN homeodomain"
# (IPR008422), which is dropped if we filter on IPR001356 alone. Note IPR006899
# (HNF1 N-terminal) is deliberately excluded: it is the HNF1 dimerisation domain,
# not the DNA-binding homeodomain, and it does not align to PF00046.
HD_ACCESSIONS = {IPR_ACCESSION, "IPR008422"}
HMM_PATH = Path(f"{DATA_DIR}/tools/mmseqs/hd_clusters/PF00046.hmm")


def extract_hd_sequences(tfs_df: pd.DataFrame, doms_df: pd.DataFrame) -> dict[str, str]:
    """
    Extract homeodomain sequences from full-length TF sequences using
    InterProScan domain boundary annotations.

    Handles proteins with multiple homeodomain instances by appending
    an '__hd{n}' suffix to disambiguate.

    Args:
        tfs_df: DataFrame containing TF information with columns "DBID" and "Sequence"
        doms_df: DataFrame containing domain annotations with columns "DBID", "accession", "start", "end"

    Returns:
        dict mapping instance_id → homeodomain sequence
    """

    seq_map = dict(zip(tfs_df["DBID"], tfs_df["Sequence"]))
    hd_doms = doms_df[doms_df["accession"].isin(HD_ACCESSIONS)].copy()

    if hd_doms.empty:
        raise ValueError(
            f"No {sorted(HD_ACCESSIONS)} entries found in domain CSV. "
            "Check that InterProScan was run and that the accession "
            "column contains a homeobox accession (e.g. 'IPR001356')."
        )

    sequences: dict[str, str] = {}
    instance_counts: dict[str, int] = {}

    for _, row in hd_doms.iterrows():
        pid = row["DBID"]
        if pid not in seq_map:
            continue
        dom_seq = seq_map[pid][int(row["start"]) - 1 : int(row["end"])]
        n = instance_counts.get(pid, 0)
        key = pid if n == 0 else f"{pid}__hd{n + 1}"
        instance_counts[pid] = n + 1
        sequences[key] = dom_seq

    return sequences


def load_hmm(hmm_path: Path) -> pyhmmer.plan7.HMM:
    """
    Load an HMM file, transparently handling gzip compression.

    Args:
        hmm_path: Path to the HMM file (can be gzipped or plain text)

    Returns:
        pyhmmer.plan7.HMM object
    """

    with open(hmm_path, "rb") as fh:
        raw = fh.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    with pyhmmer.plan7.HMMFile(io.BytesIO(raw)) as f:
        return f.read()


def align_sequences(
    hmm: pyhmmer.plan7.HMM,
    sequences: dict[str, str],
) -> tuple[dict[str, str], list[int]]:
    """
    Align homeodomain sequences to the HMM using pyhmmer.

    Args:
        hmm: Loaded HMM object for the homeodomain (e.g. PF00046)
        sequences: dict mapping sequence name → raw sequence string

    Returns:
        aligned_seqs : dict mapping sequence name → aligned sequence string
        match_cols   : list of 0-based column indices for HMM match states
    """

    alphabet = pyhmmer.easel.Alphabet.amino()
    digital_seqs = [
        pyhmmer.easel.TextSequence(name=name.encode(), sequence=seq).digitize(alphabet)
        for name, seq in sequences.items()
    ]
    msa = pyhmmer.hmmer.hmmalign(hmm, digital_seqs, trim=False)

    buf = io.BytesIO()
    msa.write(buf, "stockholm")
    buf.seek(0)
    sto = buf.read().decode()

    aligned_seqs: dict[str, str] = {}
    rf_line = ""
    for line in sto.splitlines():
        if line.startswith("#=GC RF"):
            rf_line += line.split(None, 2)[2]
        elif line and not line.startswith("#") and line != "//":
            parts = line.split()
            if len(parts) == 2:
                name, seq_aln = parts
                aligned_seqs[name] = aligned_seqs.get(name, "") + seq_aln

    match_cols = [i for i, c in enumerate(rf_line) if c == "x"]

    return aligned_seqs, match_cols


def extract_recognition_codes(
    aligned_seqs: dict[str, str],
    match_cols: list[int],
) -> dict[str, dict[int, str]]:
    """
    For each sequence, extract amino acids at the canonical HD positions
    by indexing into the HMM match-state columns.

    Args:
        aligned_seqs: dict mapping sequence name → aligned sequence string
        match_cols: list of 0-based column indices for HMM match states

    Returns:
        dict mapping instance_id → {match_state_position: amino_acid}
    """

    n_match = len(match_cols)
    results: dict[str, dict[int, str]] = {}

    for name, aln_seq in aligned_seqs.items():
        rec: dict[int, str] = {}
        for pos in HD_POSITIONS:
            idx = pos - 1
            if idx >= n_match:
                rec[pos] = "-"
                continue
            col = match_cols[idx]
            aa = aln_seq[col] if col < len(aln_seq) else "-"
            rec[pos] = aa.upper()
        results[name] = rec

    return results


def recognition_code_str(rec: dict[int, str]) -> str:
    """
    Compact recognition code string: p47|p50|p51|p52|p54.

    Args:
        rec: dict mapping match_state_position → amino_acid

    Returns:
        String representation of the recognition code
    """

    return "".join(rec.get(p, "-") for p in HD_POSITIONS)
