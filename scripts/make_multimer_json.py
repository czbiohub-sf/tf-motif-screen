"""
Creates JSON files for AlphaFold3 multimer jobs.

Ben Iovino  05/15/26    Biohub
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from af3tfm.dataset import DATA_DIR

DATA_ROOT = DATA_DIR / "jobs/hero"
_DNA_INDICES = [21, 22, 23, 24]
_RUNS_QUERY = "SELECT dna, ipTM, dom_pLDDT FROM runs WHERE protein = ?"


def rev_comp(seq: str) -> str:
    """
    Returns the reverse compliment of a DNA sequence.

    Args:
        seq (str): The DNA sequence to reverse complement.

    Returns:
        str: The reverse complement of the input DNA sequence.
    """
    return (
        seq.replace("A", "t")
        .replace("T", "a")
        .replace("C", "g")
        .replace("G", "c")
        .upper()[::-1]
    )


def random_flank(n: int, rng: np.random.Generator, gc_content: float = 0.5) -> str:
    """
    Generate a random DNA sequence of length n with a specified GC content.

    Args:
        n (int): Length of the DNA sequence.
        rng (np.random.Generator): Numpy random number generator.
        gc_content (float): Desired GC content (default is 0.5).

    Returns:
        str: Random DNA sequence.
    """
    at = (1.0 - gc_content) / 2
    gc = gc_content / 2
    return "".join(rng.choice(list("ACGT"), n, p=[at, gc, gc, at]))


def load_output_data(ensembl_id: str) -> dict:
    path = DATA_ROOT / "output_data" / ensembl_id / f"{ensembl_id}_data.json"
    with open(path) as f:
        return json.load(f)


def build_protein_entry(ensembl_id: str, chain_id: str) -> dict:
    """Build a protein chain entry for an AF3 job JSON.

    Args:
        ensembl_id: Ensembl ID of the protein
        chain_id: single-letter chain ID (A, B, C, ...)

    Returns:
        Dictionary representing the protein chain entry for the AF3 job JSON
    """

    data = load_output_data(ensembl_id)
    protein = data["sequences"][0]["protein"]
    base = DATA_ROOT / "output_data" / ensembl_id / ensembl_id

    templates = []
    for i, t in enumerate(protein["templates"]):
        templates.append(
            {
                "mmcifPath": str(base) + f"_template_{i}.mmcif",
                "queryIndices": t["queryIndices"],
                "templateIndices": t["templateIndices"],
            }
        )

    return {
        "protein": {
            "sequence": protein["sequence"],
            "id": chain_id,
            "modifications": [],
            "unpairedMsaPath": str(base) + "_unpaired.a3m",
            "pairedMsaPath": str(base) + "_paired.a3m",
            "templates": templates,
        }
    }


def build_proteins(ensembl_ids: list[str]) -> list[dict]:
    """Build protein chain entries (A, B, C, ...) for a complex, one per Ensembl ID.
    Each chain points at its own paired MSA. AF3 re-pairs sequences across chains
    by organism (UniProt species mnemonic parsed from the headers), so no manual
    cross-chain MSA alignment is needed — passing each subunit's own paired MSA is
    sufficient for AF3 to pair orthologs across all chains.

    Args:
        ensembl_ids: list of Ensembl IDs for the protein chains

    Returns:
        List of protein chain entries (dicts) for the AF3 job JSON
    """
    chain_ids = [chr(ord("A") + i) for i in range(len(ensembl_ids))]
    return [build_protein_entry(eid, cid) for eid, cid in zip(ensembl_ids, chain_ids)]


def _top_runs(conn: sqlite3.Connection, eid: str, n: int) -> pd.DataFrame:
    """Top-n runs for one protein: score by ipTM x dom_pLDDT percentile-rank product,
    keep the highest-scoring row per DNA sequence (avoids duplicate job names).

    Args:
        conn: sqlite3 connection to the job database
        eid: Ensembl ID of the protein
        n: number of top runs to return

    Returns:
        DataFrame of top-n runs, sorted by combined score descending
    """
    df = pd.read_sql(_RUNS_QUERY, conn, params=(eid,))
    df["combined"] = df["ipTM"].rank(pct=True) * df["dom_pLDDT"].rank(pct=True)
    df = df.loc[df.groupby("dna")["combined"].idxmax()]
    return df.nlargest(n, "combined")


def get_top_dna_entries(ensembl_id: str, n: int = 25) -> list[str]:
    """Return the top-n unique DNA sequences for a TF, ranked by percentile rank product.

    Args:
        ensembl_id: Ensembl ID of the protein
        n: number of top DNA sequences to return

    Returns:
        List of the top-n unique DNA sequences, highest-scoring first
    """
    conn = sqlite3.connect(DATA_ROOT / "jobdb.sqlite")
    top = _top_runs(conn, ensembl_id, n)
    conn.close()
    return list(top["dna"])


def get_top_dna_entries_union(*ensembl_ids: str, n: int = 25) -> list[str]:
    """Return unique DNA sequences from the union of top-n runs of each Ensembl ID.

    Yields up to len(ensembl_ids) * n sequences (fewer when they overlap), keeping
    the highest-scoring run per unique DNA sequence and sorting by combined score.
    """
    conn = sqlite3.connect(DATA_ROOT / "jobdb.sqlite")
    dfs = [_top_runs(conn, eid, n) for eid in ensembl_ids]
    conn.close()

    combined = pd.concat(dfs)
    combined = combined.loc[combined.groupby("dna")["combined"].idxmax()]
    combined = combined.sort_values("combined", ascending=False)
    return list(combined["dna"])


def make_complex_job(name: str, proteins: list[dict], dna: str) -> dict:
    """Build an AF3 job dict: N protein chains plus DNA and its reverse complement.

    Args:
        name: job name (used in output JSON)
        proteins: list of protein chain entries (dicts) for the AF3 job JSON
        dna: DNA sequence for the complex

    Returns:
        Dictionary representing the AF3 job JSON
    """
    n = len(proteins)
    dna_id = chr(ord("A") + n)
    rc_id = chr(ord("A") + n + 1)
    return {
        "name": name,
        "modelSeeds": [1, 2, 3, 4, 5],
        "sequences": [
            *proteins,
            {"dna": {"sequence": dna, "id": dna_id}},
            {"dna": {"sequence": rev_comp(dna), "id": rc_id}},
        ],
        "dialect": "alphafold3",
        "version": 1,
    }


def write_job(job_data: dict, job_dir: Path, job_name: str):
    """Write an AF3 job JSON to the specified directory.

    Args:
        job_data: Dictionary representing the AF3 job JSON
        job_dir: Path to the output directory for job JSONs
        job_name: Name of the job (used for the output JSON file name)
    """

    out = job_dir / job_name
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"{job_name}_data.json"
    if out_file.exists():
        return
    with open(out_file, "w") as f:
        json.dump(job_data, f, indent=4)


def _make_homodimer_jobs(
    ensembl_id: str,
    job_dir: Path,
    n: int,
    tag: str,
    name_suffix: str,
    label: str,
    dna_transform=lambda dna, i: dna,
):
    """Shared scaffold for the homodimer job variants.

    Builds both chains (A, B) from a single TF, pulls its top-n DNA entries, and
    writes one job per entry. Each chain uses its own paired MSA; AF3 pairs the two
    chains by organism.

    Args:
        ensembl_id: Ensembl ID of the TF to fold
        job_dir: output directory for job JSONs
        n: number of top DNA entries to use
        tag: job-folder tag, used as `<ensembl_id>_<tag>_<i>`
        name_suffix: appended to the AF3 job name after "homodimer"
        label: human-readable descriptor used in the progress message
        dna_transform: maps (dna, index) -> the DNA sequence to actually fold
            (identity for the plain homodimer; flank/palindrome/core builders otherwise)
    """
    protein_a = build_protein_entry(ensembl_id, "A")
    protein_b = build_protein_entry(ensembl_id, "B")
    db_entries = get_top_dna_entries(ensembl_id, n)
    print(f"Creating {len(db_entries)} {label} jobs for {ensembl_id}")

    for i, dna in enumerate(db_entries):
        job_data = make_complex_job(
            f"{ensembl_id}_homodimer{name_suffix}",
            [protein_a, protein_b],
            dna_transform(dna, i),
        )
        write_job(job_data, job_dir, f"{ensembl_id}_{tag}_{i}")


def make_homodimer_jobs(ensembl_id: str, job_dir: Path, n: int = 25):
    """
    Create homodimer jobs using the top-n DNA sequences for a single TF.

    Args:
        ensembl_id: Ensembl ID of the TF to fold
        job_dir: output directory for job JSONs
        n: number of top DNA entries to use (default 25)
    """
    _make_homodimer_jobs(ensembl_id, job_dir, n, "top", "", "homodimer")


def _get_ic_core(
    npz_path: Path, ic_mass: float = 0.9, min_core: int = 3
) -> tuple[int, int]:
    """Return (start, end) 0-indexed slice of the informative core in the forward-strand PPM.

    Knowledge-agnostic core selection: find the *shortest contiguous window* that
    captures at least `ic_mass` of the profile's total information content.  If the
    resulting window is shorter than `min_core`, fall back to the highest-IC contiguous
    window of width `min_core` rather than returning a sub-min_core core.

    Args:
        npz_path: path to the NPZ file containing the predicted PPM and DNA mask
        ic_mass: fraction of total IC the core window must capture (default 0.9)
        min_core: minimum core length; sliding-window fallback uses this width if the IC-mass
            core is shorter (default 3)

    Returns:
        Tuple of (start, end) indices for the core window in the forward-strand PPM
    """

    data = np.load(str(npz_path), allow_pickle=True)
    ppm_full = data["predicted_ppm"]
    dna_mask = data["dna_mask"].astype(bool)
    dna_idx = np.where(dna_mask)[0]
    ppm_dna = ppm_full[dna_idx][:, _DNA_INDICES]
    encoded = data["encoded_residues"][dna_idx]
    fwd_chain = encoded[0][0]
    b_mask = np.array([e[0] == fwd_chain for e in encoded])
    ppm_fwd = ppm_dna[b_mask]

    ic = np.maximum(
        np.sum(ppm_fwd * np.log2(np.clip(ppm_fwd, 1e-9, 1.0) / 0.25), axis=1),
        0.0,
    )
    L = len(ic)
    total = ic.sum()
    if L <= min_core or total <= 0.0:
        return (0, L)

    # Shortest contiguous window capturing >= ic_mass of total IC; on equal width,
    # prefer the window with the larger captured IC (the (-icsum) tuple element).
    target = ic_mass * total
    best = None  # (width, -icsum, start, end)
    for i in range(L):
        s = 0.0
        for j in range(i, L):
            s += ic[j]
            if s >= target:
                cand = (j - i + 1, -s, i, j + 1)
                if best is None or cand < best:
                    best = cand
                break
    s_idx, e_idx = best[2], best[3]

    # Expand a too-short mass window to the highest-IC window of width min_core.
    if (e_idx - s_idx) < min_core:
        window_sums = np.convolve(ic, np.ones(min_core), mode="valid")
        s_idx = int(np.argmax(window_sums))
        e_idx = s_idx + min_core
    return (s_idx, e_idx)


def make_homodimer_palindrome_core_jobs(
    ensembl_id: str,
    job_dir: Path,
    n: int = 25,
    spacer_len: int = 3,
    ic_mass: float = 0.9,
    min_core: int = 3,
):
    """Create homodimer jobs using IC-trimmed half-site + random spacer + half-site.
    The core is the shortest contiguous window capturing `ic_mass` of the NA-MPNN
    profile's total information content. The two half-sites are then arranged as a
    convergent inverted repeat, the canonical homodimer-palindrome geometry.

    Args:
        ensembl_id: Ensembl ID of the TF to fold
        job_dir: output directory for job JSONs
        n: number of top DNA entries to use (default 25)
        spacer_len: number of random bases to insert between the two half-sites (default 3)
        ic_mass: fraction of total IC the core window must capture (default 0.9)
        min_core: minimum core length; sliding-window fallback uses this width if the IC-mass core is shorter (default 3)
    """
    spec_dir = DATA_ROOT / "specificities" / ensembl_id / "specificity"
    rng = np.random.default_rng(42)

    def build_palindrome_core(dna, i):
        npz_matches = list(spec_dir.glob(f"{ensembl_id}_*_{dna}_*.npz"))
        if npz_matches:
            s, e = _get_ic_core(npz_matches[0], ic_mass, min_core)
            core_dna = dna[s:e]
        else:
            print(f"  Warning: no NPZ found for {dna}, using full sequence")
            core_dna = dna

        spacer_seq = random_flank(spacer_len, rng, gc_content=0.4)
        palindrome_dna = rev_comp(core_dna) + spacer_seq + core_dna
        print(f"  {i:2d}  {dna} -> core={core_dna}  palindrome={palindrome_dna}")
        return palindrome_dna

    _make_homodimer_jobs(
        ensembl_id,
        job_dir,
        n,
        "palindrome_core",
        "_palindrome_core",
        "palindrome-core homodimer",
        build_palindrome_core,
    )


def _dimer_dna_scores(dimer_db: Path, eid: str) -> dict[str, float]:
    """Return {dna: ipTM x dom_pLDDT percentile-rank product} for one homodimer job.

    Each DNA is an already-assembled palindrome-core half-site; the score is only used
    to order the union deterministically, not to filter (selection is knowledge-agnostic).
    """
    conn = sqlite3.connect(str(dimer_db))
    df = pd.read_sql(
        "SELECT dna, ipTM, dom_pLDDT FROM runs WHERE protein = ?", conn, params=(eid,)
    )
    conn.close()
    df["combined"] = df["ipTM"].rank(pct=True) * df["dom_pLDDT"].rank(pct=True)
    return df.groupby("dna")["combined"].mean().to_dict()


def make_homodimer_tetramer_jobs(
    eid_a: str,
    eid_b: str,
    dimer_db_a: Path,
    dimer_db_b: Path,
    job_dir: Path,
    inter_spacer: int = 0,
):
    """Create hetero-tetramer (dimer-of-dimers) jobs: A,B = eid_a homodimer,
    C,D = eid_b homodimer, bound to a full p53-family response element.

    Knowledge-agnostic DNA selection (mirrors the top-N monomer approach, but here uses
    *every* validated homodimer half-site): take all palindrome-core DNAs from both
    homodimer job DBs and, for each, build the full element as a tandem duplicate of
    that half-site (the canonical p53 RE is a direct repeat of two decamers, here
    H + H with an optional inter_spacer):

        H + inter_spacer + H

    Every per-job half-site is kept (a DNA present in both jobs yields two jobs), each
    job sorted by its own score so the two TFs' lists stay self-consistent.  One job per
    half-site entry; the NA-MPNN PWM-construction step recovers the motif from the
    predictions rather than us hand-picking a specific site.  Chains: A,B (eid_a) and
    C,D (eid_b) are protein; DNA strand = E, rev-comp = F.

    Args:
        eid_a: Ensembl ID of the first TF (homodimer A,B)
        eid_b: Ensembl ID of the second TF (homodimer C,D)
        dimer_db_a: path to the homodimer job DB for eid_a
        dimer_db_b: path to the homodimer job DB for eid_b
        job_dir: output directory for job JSONs
        inter_spacer: number of random bases to insert between the two half-sites (default 0, canonical)
    """
    proteins = [
        build_protein_entry(eid_a, "A"),
        build_protein_entry(eid_a, "B"),
        build_protein_entry(eid_b, "C"),
        build_protein_entry(eid_b, "D"),
    ]

    scores_a = _dimer_dna_scores(dimer_db_a, eid_a)
    scores_b = _dimer_dna_scores(dimer_db_b, eid_b)
    half_sites = sorted(scores_a, key=lambda d: scores_a[d], reverse=True) + sorted(
        scores_b, key=lambda d: scores_b[d], reverse=True
    )

    name = f"{eid_a}_{eid_b}_tetramer"
    rng = np.random.default_rng(42)
    print(
        f"Creating {len(half_sites)} tetramer jobs for {name} "
        f"({len(scores_a)} {eid_a} + {len(scores_b)} {eid_b} half-sites)"
    )

    for i, half in enumerate(half_sites):
        inter = random_flank(inter_spacer, rng, gc_content=0.4) if inter_spacer else ""
        full_dna = half + inter + half
        print(f"  {i:2d}  half-site={half}  full ({len(full_dna)} bp)={full_dna}")
        job_data = make_complex_job(name, proteins, full_dna)
        write_job(job_data, job_dir, f"{name}_top_{i}")


def make_heterodimer_jobs(eid_a: str, eid_b: str, job_dir: Path, n: int = 25):
    """Create heterodimer jobs using the union of top-n sequences from each TF.

    Args:
        eid_a: Ensembl ID of the first TF (chain A)
        eid_b: Ensembl ID of the second TF (chain B)
        job_dir: output directory for job JSONs
        n: number of top DNA entries to use from each TF (default 25)
    """
    proteins = build_proteins([eid_a, eid_b])
    db_entries = get_top_dna_entries_union(eid_a, eid_b, n=n)
    name = f"{eid_a}_{eid_b}"
    print(f"Creating {len(db_entries)} heterodimer jobs for {name}")

    for i, dna in enumerate(db_entries):
        job_name = f"{name}_top_{i}"
        job_data = make_complex_job(name, proteins, dna)
        write_job(job_data, job_dir, job_name)


def make_trimer_jobs(ensembl_ids: list[str], job_dir: Path, n: int = 25):
    """Create trimer jobs (three protein chains + DNA) for a complex like NF-Y.

    DNA sequences are the union of top-n from all three subunits' folding runs;
    each chain uses its own paired MSA and AF3 pairs the three chains by organism.

    Args:
        ensembl_ids: list of three Ensembl IDs for the protein chains
        job_dir: output directory for job JSONs
        n: number of top DNA entries to use from each TF (default 25)
    """
    proteins = build_proteins(ensembl_ids)
    db_entries = get_top_dna_entries_union(*ensembl_ids, n=n)
    name = "_".join(ensembl_ids)
    print(f"Creating {len(db_entries)} trimer jobs for {name}")

    for i, dna in enumerate(db_entries):
        job_name = f"{name}_top_{i}"
        job_data = make_complex_job(name, proteins, dna)
        write_job(job_data, job_dir, job_name)


def make_homodimer_flanked_jobs(
    ensembl_id: str, job_dir: Path, n: int = 25, flank: int = 3
):
    """Create homodimer jobs with random flanking sequences added to each end of the DNA.

    Args:
        ensembl_id: Ensembl ID of the TF
        job_dir: output directory for job JSONs
        n: number of top DNA entries to use from the TF (default 25)
        flank: number of random flanking bases to add to each end of the DNA (default 3)
    """
    rng = np.random.default_rng(42)
    _make_homodimer_jobs(
        ensembl_id,
        job_dir,
        n,
        "flanked",
        "_flanked",
        f"flanked homodimer (flank={flank})",
        lambda dna, i: random_flank(flank, rng) + dna + random_flank(flank, rng),
    )


def make_homodimer_palindrome_jobs(ensembl_id: str, job_dir: Path, n: int = 25):
    """Create homodimer jobs with DNA extended to its palindrome (seq + rev_comp(seq)).

    Args:
        ensembl_id: Ensembl ID of the TF
        job_dir: output directory for job JSONs
        n: number of top DNA entries to use from the TF (default 25)
    """
    _make_homodimer_jobs(
        ensembl_id,
        job_dir,
        n,
        "palindrome",
        "_palindrome",
        "palindrome homodimer",
        lambda dna, i: dna + rev_comp(dna),
    )


def main():
    parser = argparse.ArgumentParser(description="Create AF3 dimer/monomer job JSONs.")
    parser.add_argument(
        "--mode",
        choices=[
            "homodimer",
            "homodimer_flanked",
            "homodimer_palindrome",
            "homodimer_palindrome_core",
            "heterodimer",
            "trimer",
            "homodimer_tetramer",
        ],
        help="Job type to create",
    )
    parser.add_argument(
        "--protein",
        nargs="+",
        metavar="ENSEMBL_ID",
        help="One Ensembl ID (homodimer modes), two (heterodimer: A then B), "
        "or three (trimer: A, B, C)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT.parent / "dimers" / "input_fold",
        help="Output directory for job folders (default: data/jobs/dimers/input_fold)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of top DNA entries to use (default: 50)",
    )
    parser.add_argument(
        "--flank",
        type=int,
        default=3,
        help="Flanking bases to add each side for homodimer_flanked mode (default: 3)",
    )
    parser.add_argument(
        "--spacer",
        type=int,
        default=3,
        help="Spacer bases between half-sites for homodimer_palindrome_core mode (default: 3)",
    )
    parser.add_argument(
        "--inter-spacer",
        type=int,
        default=0,
        help="Spacer bases between the two decamer half-sites for homodimer_tetramer mode (default: 0, canonical)",
    )
    parser.add_argument(
        "--dimer-jobs",
        nargs=2,
        metavar="JOB_NAME",
        help="homodimer_tetramer mode: the two homodimer job dir names under data/jobs "
        "(e.g. dimers-tp63 dimers-tp73), in the same A,B order as the Ensembl IDs; "
        "each TF's validated top half-site DNA is read from <job>/jobdb.sqlite",
    )
    parser.add_argument(
        "--ic-mass",
        type=float,
        default=0.9,
        help="Fraction of total information content the core window must capture (default: 0.9)",
    )
    parser.add_argument(
        "--min-core",
        type=int,
        default=3,
        help="Minimum core length; sliding-window fallback uses this width if the IC-mass core is shorter (default: 3)",
    )
    args = parser.parse_args()

    if args.mode in ("heterodimer", "homodimer_tetramer"):
        if len(args.protein) != 2:
            parser.error(f"{args.mode} requires exactly two Ensembl IDs")
        if args.mode == "homodimer_tetramer" and not args.dimer_jobs:
            parser.error("homodimer_tetramer requires --dimer-jobs JOB_A JOB_B")
    elif args.mode == "trimer":
        if len(args.protein) != 3:
            parser.error("trimer requires exactly three Ensembl IDs")
    else:
        if len(args.protein) != 1:
            parser.error(f"{args.mode} requires exactly one Ensembl ID")

    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "homodimer":
        make_homodimer_jobs(args.protein[0], args.output, args.n)
    elif args.mode == "homodimer_flanked":
        make_homodimer_flanked_jobs(args.protein[0], args.output, args.n, args.flank)
    elif args.mode == "homodimer_palindrome":
        make_homodimer_palindrome_jobs(args.protein[0], args.output, args.n)
    elif args.mode == "homodimer_palindrome_core":
        make_homodimer_palindrome_core_jobs(
            args.protein[0],
            args.output,
            args.n,
            spacer_len=args.spacer,
            ic_mass=args.ic_mass,
            min_core=args.min_core,
        )
    elif args.mode == "heterodimer":
        make_heterodimer_jobs(args.protein[0], args.protein[1], args.output, args.n)
    elif args.mode == "trimer":
        make_trimer_jobs(args.protein, args.output, args.n)
    elif args.mode == "homodimer_tetramer":
        jobs_root = DATA_ROOT.parent
        db_a = jobs_root / args.dimer_jobs[0] / "jobdb.sqlite"
        db_b = jobs_root / args.dimer_jobs[1] / "jobdb.sqlite"
        for db in (db_a, db_b):
            if not db.exists():
                parser.error(f"dimer job DB not found: {db}")
        make_homodimer_tetramer_jobs(
            args.protein[0],
            args.protein[1],
            db_a,
            db_b,
            args.output,
            inter_spacer=args.inter_spacer,
        )


if __name__ == "__main__":
    main()
