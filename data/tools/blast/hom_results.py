"""
Here we look through the BLAST alignments between Zebrafish TFs and human TFs to find the DNA
binding domain as annotated in the TF dataset, and compute the percent similarity of that region
for the orthologous human TF (labeled by ZFIN). Only the ZFIN ortholog is reported: its DBD
percent similarity is recorded whether or not it carries a JASPAR motif, and a TF with no ZFIN
ortholog among its BLAST hits is left unassigned.

Ben Iovino  4/15/26 Biohub
"""

# %%
import re
import time
from pathlib import Path

import pandas as pd
import requests

from af3tfm.dataset import DATA_DIR, DS
from af3tfm.utils import fetch_jaspar_motif

dataset = DS["tfdb4_rep"]
tf_df = pd.read_csv(dataset.seq_csv, sep=",", header=0)
doms_df = pd.read_csv(dataset.dom_csv, sep=",", header=0)

results_path = Path(f"{DATA_DIR}/tools/blast/results")
results_path.mkdir(parents=True, exist_ok=True)

# ZFIN ortholog mappings (built by get_orthos.py)
ZFIN_DIR = Path(f"{DATA_DIR}/tools/blast/zfin")
orthos_df = pd.read_csv(ZFIN_DIR / "ensdarg_human_orthos.csv")

# ensdarg (== DBID) -> list of human ortholog symbols
dbid_orthologs = {
    row.ensdarg: [s for s in str(row.human_symbols).split("|") if s]
    for row in orthos_df.itertuples()
}

# API endpoints. /stream now answers any multi-gene OR query with a 200 whose *body* is
# "Error encountered when streaming data", so it silently resolved nothing; /search handles
# the batched query but caps each response at a page and continues via a Link header.
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"
SYMBOL_BATCH = 100  # gene symbols per UniProt query
UNIPROT_PAGE = 500  # rows per /search page

# BLAST alignment regexes
query_regex = re.compile(r"^Query\s+(\d+)\s+([A-Z\-]+)\s+\d+")
align_regex = re.compile(r"^\s{12}([A-Za-z\+\| ]+)")  # Exactly 12 spaces at start
subject_regex = re.compile(r"^Sbjct\s+(\d+)\s+([A-Z\-]+)\s+\d+")

# Families whose homolog/motif assignments come from recognition-code scripts
RECOG_CODE_FAMILIES = {"zf-C2H2", "Homeobox", "ZBTB"}


# %%
def blast_alignment_info(path: str | Path) -> dict[str, list[tuple[int, str]]]:
    """
    Parse a BLASTP pair-wise report (plain-text format).

    Returns:
        {hit_description: [(query_position, relation), ...]}
        where relation is 'match', 'mismatch', 'subject_gap', or 'query_gap'.
    """
    out: dict[str, list[tuple[int, str]]] = {}
    hit = None

    with Path(path).expanduser().open() as fh:
        for line in fh:
            if line.startswith(">"):
                hit = line[1:].strip()
                out.setdefault(hit, [])
                continue

            query_match = query_regex.match(line)
            if not (hit and query_match):
                continue

            query_start, query_seq = int(query_match.group(1)), query_match.group(2)
            try:
                match_seq = align_regex.match(next(fh, "")).group(1)
            except AttributeError:  # 11 spaces
                continue
            subject_line = next(fh, "")
            subject_match = subject_regex.match(subject_line)
            if not subject_match:  # bad HSP
                continue
            subject_seq = subject_match.group(2)

            query_pos = query_start
            for query_char, match_char, subject_char in zip(
                query_seq, match_seq, subject_seq
            ):
                if query_char != "-":
                    if subject_char == "-":
                        relation = "subject_gap"
                    elif query_char == subject_char or match_char == "+":
                        relation = "match"
                    else:
                        relation = "mismatch"
                    out[hit].append((query_pos, relation))
                    query_pos += 1
                else:
                    # Subject insertion — no query position consumed
                    out[hit].append((query_pos, "query_gap"))

    return out


def fetch_refseq_to_symbol(symbols: set[str]) -> dict[str, str]:
    """
    Build a RefSeq protein accession -> HGNC gene symbol map for a set of human genes.

    Args:
        symbols: Human gene symbols (any case) to resolve.

    Returns:
        Dict mapping versionless RefSeq accession (e.g. 'NP_003241') to gene symbol.
    """
    refseq_to_symbol: dict[str, str] = {}
    ordered = sorted({s.upper() for s in symbols if s})

    for i in range(0, len(ordered), SYMBOL_BATCH):
        batch = ordered[i : i + SYMBOL_BATCH]
        query = f"(organism_id:9606) AND ({' OR '.join(f'gene:{g}' for g in batch)})"
        url = UNIPROT_SEARCH
        params = {
            "query": query,
            "fields": "gene_names,xref_refseq",
            "format": "tsv",
            "size": UNIPROT_PAGE,
        }

        # Follow the 'next' cursor until the batch is exhausted; a batch of 100 symbols
        # overflows one page, and stopping early would silently drop those accessions.
        while url:
            try:
                resp = requests.get(url, params=params, timeout=120)
                resp.raise_for_status()
                if "Error encountered" in resp.text:  # 200, but the body is an error
                    raise RuntimeError("UniProt returned a body-level error")
            except Exception as e:
                print(f"UniProt lookup error (batch {i // SYMBOL_BATCH + 1}): {e}")
                break

            for line in resp.text.strip().split("\n")[1:]:
                parts = line.split("\t")
                if len(parts) < 2 or not parts[0]:
                    continue
                gene = (
                    parts[0].split()[0].upper()
                )  # primary symbol; the rest are synonyms
                for ref in parts[1].split(";"):
                    ref = ref.split("[")[
                        0
                    ].strip()  # drop isoform tag: 'NP_003241.2 [P-1]'
                    if ref:
                        refseq_to_symbol.setdefault(ref.split(".")[0], gene)

            url = resp.links.get("next", {}).get("url")
            params = None  # the cursor url already carries query/fields/size

        if i + SYMBOL_BATCH < len(ordered):
            time.sleep(0.1)

    return refseq_to_symbol


def _score_hit_vs_dbd(
    mappings: list[tuple[int, str]], dbd_start: int, dbd_end: int, dbd_length: int
) -> tuple[float, int]:
    """
    Return (similarity, covered_positions) for one BLAST hit against the DBD region.

    Args:
        mappings: List of (query_position, relation) for the hit.
        dbd_start: Start position of the DBD in the query sequence.
        dbd_end: End position of the DBD in the query sequence.
        dbd_length: Length of the DBD region (dbd_end - dbd_start + 1).

    Returns:
        sim: Percent similarity (matches / dbd_length) for the DBD region.
        covered: Number of DBD positions covered by the hit (matches + mismatches).
    """

    matching, covered = 0, 0
    seen: set[int] = set()
    for pos, rel in mappings:
        if rel == "query_gap":
            continue  # not a real residue; shares pos with the next real entry
        if pos in seen:
            continue  # already counted from a higher-scoring HSP
        seen.add(pos)
        if dbd_start <= pos <= dbd_end:
            if rel == "match":
                matching += 1
                covered += 1
            elif rel == "mismatch":
                covered += 1
    sim = matching / dbd_length if dbd_length > 0 else 0.0
    return sim, covered


def select_homolog(candidates: list[tuple]) -> tuple | None:
    """
    Choose the ZFIN ortholog to report for one TF from its BLAST candidates.

    A motif is what makes the assignment usable downstream, so a motif-bearing ortholog wins
    over a motif-less one even at lower DBD similarity; ties within each group go to the
    highest similarity. Most TFs have a single ZFIN ortholog and never reach this.

    Args:
        candidates: List of (gene, sim, covered, jaspar) tuples, one per ZFIN ortholog
            found among the TF's BLAST hits.

    Returns:
        The chosen candidate tuple, or None if the TF has no ZFIN ortholog among its hits.
    """

    if not candidates:
        return None
    return max(candidates, key=lambda c: (bool(c[3]), c[1]))


def analyze_blast_results(
    tf_df: pd.DataFrame,
    doms_df: pd.DataFrame,
    results_path: Path,
) -> tuple[dict, dict]:
    """
    For each TF, DBD-score its human BLAST hits and report the ZFIN ortholog among them
    (see select_homolog), whether or not that ortholog carries a JASPAR motif.

    Args:
        tf_df: DataFrame with TF info, must include 'DBID', 'TF_Name' and 'Family'.
        doms_df: DataFrame with domain info, must include 'DBID' and 'DNA-Binding'.
        results_path: Path to directory containing BLAST result files.

    Returns:
        results: {family: [sim, ...]} for every TF with a reported ZFIN ortholog.
        top_hits: {dbid: (sim_pct, gene_symbol, jaspar_id)} for those TFs; jaspar_id is
            None when the ortholog has no JASPAR motif.
    """

    # Compute per-TF sorted hit lists, keyed by DBID (unique; TF_Name is not)
    per_tf_hits: dict[str, tuple] = {}
    for dbid, tf, tf_fam in zip(tf_df["DBID"], tf_df["TF_Name"], tf_df["Family"]):
        if tf_fam in RECOG_CODE_FAMILIES:
            continue  # handled by recognition-code scripts, not single-DBD BLAST
        result_file = results_path / f"{dbid}_results.txt"
        if not result_file.exists():
            continue

        dbd_rows = doms_df.loc[
            (doms_df["DBID"] == dbid) & (doms_df["DNA-Binding"])
        ].values
        if len(dbd_rows) == 0:
            print(f"No DBD info for {tf} ({dbid}), skipping...")
            continue
        dbd_start, dbd_end = dbd_rows[0][4], dbd_rows[0][5]
        dbd_length = dbd_end - dbd_start + 1

        hit_sims = []
        for hit, mappings in blast_alignment_info(result_file).items():
            sim, covered = _score_hit_vs_dbd(mappings, dbd_start, dbd_end, dbd_length)
            acc_match = re.match(r"^((?:NP|XP)_\d+(?:\.\d+)?)", hit)
            accession = acc_match.group(1) if acc_match else None
            hit_name = re.sub(r"^(?:NP|XP)_\d+(?:\.\d+)?\s*", "", hit).strip()
            hit_sims.append((sim, covered, hit_name, accession))

        hit_sims.sort(key=lambda x: x[0], reverse=True)
        per_tf_hits[dbid] = (tf, tf_fam, dbd_length, hit_sims)

    # Resolve the ZFIN ortholog symbols each TF could report, then fetch RefSeq accessions
    # for that gene set only (far cheaper than symbol-ing every hit)
    tf_symbols: dict[str, set] = {
        dbid: {o.upper() for o in dbid_orthologs.get(dbid, [])} for dbid in per_tf_hits
    }

    relevant_symbols = {s for syms in tf_symbols.values() for s in syms}
    print(f"Fetching RefSeq accessions for {len(relevant_symbols)} ortholog symbols...")
    refseq_to_symbol = fetch_refseq_to_symbol(relevant_symbols)
    resolved = len(set(refseq_to_symbol.values()) & relevant_symbols)
    print(
        f"  Symbols resolved: {resolved}/{len(relevant_symbols)} "
        f"({len(refseq_to_symbol)} RefSeq accessions)"
    )
    # Without accessions no hit can be tied to an ortholog, so every TF would be silently
    # left unassigned and the csv wiped. Fail here instead of writing that out.
    if relevant_symbols and not resolved:
        raise RuntimeError(
            "UniProt resolved no ortholog symbols - check the endpoint before re-running"
        )

    # Report the ZFIN ortholog per TF, motif or not
    results: dict[str, list] = {}
    top_hits: dict[str, tuple] = {}

    for dbid, (tf, tf_fam, dbd_length, hits) in per_tf_hits.items():
        orthologs = tf_symbols[dbid]

        # Build the ortholog candidates from the hits (hits sorted by sim desc, so the
        # first time a gene is seen is its best-scoring HSP; skip duplicate isoforms).
        candidates, seen = [], set()
        for sim, covered, hit_name, acc in hits:
            gene = refseq_to_symbol.get(acc.split(".")[0], "") if acc else ""
            if not gene or gene in seen or gene not in orthologs:
                continue
            seen.add(gene)
            candidates.append((gene, sim, covered, fetch_jaspar_motif(gene)))

        chosen = select_homolog(candidates)
        if chosen is None:
            continue  # no ZFIN ortholog among the BLAST hits -> leave unassigned
        gene, sim, covered, jaspar = chosen

        coverage_pct = covered / dbd_length if dbd_length > 0 else 0.0
        jaspar_str = f", JASPAR={jaspar}" if jaspar else ", no JASPAR motif"
        print(
            f"TF: {tf}, ortholog: {gene}, sim: {sim:.2%}, "
            f"coverage: {covered}/{dbd_length} DBD positions{jaspar_str}"
        )
        if coverage_pct < 0.8:
            print(f"  WARNING: low DBD coverage for {tf} ({coverage_pct:.1%})")

        top_hits[dbid] = (round(sim * 100, 1), gene, jaspar)
        results.setdefault(str(tf_fam), []).append(sim)

    return results, top_hits


def get_no_match_families(tf_df: pd.DataFrame, results: dict) -> list[str]:
    """
    Return families that have no ZFIN-ortholog similarity in results.

    Args:
        tf_df: DataFrame with TF info, must include 'Family'.
        results: Dictionary of per-family similarities.

    Returns:
        List of family names with no reported ortholog.
    """

    all_families = {str(f) for f in tf_df["Family"].dropna().unique()}
    return sorted(all_families - set(results.keys()) - RECOG_CODE_FAMILIES - {"nan"})


# %%
# Run analysis
results, top_hits = analyze_blast_results(tf_df, doms_df, results_path)
n_motif = sum(1 for h in top_hits.values() if h[2])
print(
    f"\nTFs with a reported ZFIN ortholog: {len(top_hits)} ({n_motif} carry a JASPAR motif)"
)
no_match = get_no_match_families(tf_df, results)
print(f"Families with no reported ortholog: {no_match}")


# %%
# Save top hit info back to tf_df
def _unpack_hit(dbid):
    h = top_hits.get(dbid)
    return (h[0], h[1], h[2]) if h else (None, None, None)


hom_cols = ["Hom_Sim", "Hom_Gene", "Hom_JASPAR"]
for col in hom_cols:
    if col not in tf_df.columns:
        tf_df[col] = None
tf_df = tf_df.drop(columns=["Hom_Source"], errors="ignore")  # paralog-era column

# Only update non-recognition-code families; leave recog-code rows untouched so their
# values (written by zfc2h2_hom.py / zbtb_hom.py) are never overwritten here.
mask = ~tf_df["Family"].isin(RECOG_CODE_FAMILIES)
tf_df.loc[mask, hom_cols] = pd.DataFrame(
    tf_df.loc[mask, "DBID"].map(_unpack_hit).tolist(), index=tf_df.loc[mask].index
).values
tf_df.to_csv(dataset.seq_csv, index=False)
