"""Sequence metadata and inclusion rules for the released ZebraMotif database."""

import argparse
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Explicit exclusions for supplemental genes outside the release catalog.
EXCLUDED_GENE_IDS: frozenset[str] = frozenset({"ENSDARG00000054778"})
DOMAIN_ID = re.compile(r"^ENSDARG\d+_\d+$")
DOMAIN_COLUMNS = [
    "DBID",
    "TF_Name",
    "accession",
    "description",
    "start",
    "end",
    "DNA-Binding",
]
ZFIN_REFERENCE = Path("tools/blast/zfin/zfin_ids_release.parquet")
MEME_HEADER = (
    "MEME version 4\n\n"
    "ALPHABET= ACGT\n\n"
    "strands: + -\n\n"
    "Background letter frequencies\n"
    "A 0.25 C 0.25 G 0.25 T 0.25\n"
)


def sequence_type(dbid: str) -> str:
    """Identify full-length gene entries and suffixed domain predictions."""
    return "domain" if DOMAIN_ID.fullmatch(dbid.upper()) else "full_length"


def filter_database_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply explicit exclusions; supplemental catalog entries are retained."""
    gene_ids = frame["DBID"].str.upper().str.replace(r"_\d+$", "", regex=True)
    return frame.loc[~gene_ids.isin(EXCLUDED_GENE_IDS)].copy()


def add_sequence_types(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply release inclusion rules and describe the sequence for each entry."""
    result = filter_database_rows(frame)
    result["sequence_type"] = result["DBID"].map(sequence_type)
    return result


def validate_catalog_coverage(lookup: pd.DataFrame, catalog: pd.DataFrame) -> None:
    """Require every original AnimalTFDB4 gene ID to resolve to a released motif."""
    catalog_ids = set(catalog["DBID"])
    missing = catalog_ids - set(lookup["DBID"])
    unresolved = set(
        lookup.loc[lookup["DBID"].isin(catalog_ids) & lookup["Rep_DBID"].isna(), "DBID"]
    )
    if missing or unresolved:
        raise ValueError(
            "Incomplete AnimalTFDB4 catalog coverage: "
            f"missing query IDs {sorted(missing)}; "
            f"unresolved query IDs {sorted(unresolved)}"
        )


def reconcile_representatives(
    lookup: pd.DataFrame, modes: pd.DataFrame
) -> pd.DataFrame:
    """Give a query's own exported primary mode precedence over older redirects.

    Preserve other query assignments: refreshing a released export must not
    silently replace its clustering with the current research input files.
    """
    primary = modes.loc[modes["tier"].eq("primary")]
    if primary["DBID"].duplicated().any() or modes["motif_id"].duplicated().any():
        raise ValueError("Primary DBIDs and motif IDs must be unique")
    if lookup["DBID"].duplicated().any():
        raise ValueError("Query DBIDs must be unique")
    representatives = set(primary["DBID"])
    if set(modes["DBID"]) - representatives:
        raise ValueError("Every exported representative must have a primary mode")
    missing = representatives - set(lookup["DBID"])
    if missing:
        raise ValueError(
            f"Modeled representatives missing from the lookup: {sorted(missing)}"
        )

    result = lookup.copy()
    own = result["DBID"].isin(representatives)
    result.loc[own, "Rep_DBID"] = result.loc[own, "DBID"]
    result["is_representative"] = own
    result.loc[own, "resolution"] = "self"
    dangling = set(result["Rep_DBID"].dropna()) - representatives
    if dangling:
        raise ValueError(
            f"Queries resolve to representatives without primary modes: {sorted(dangling)}"
        )
    return result


def meme_document(rows: pd.DataFrame, names: dict | None = None) -> str:
    """Serialize a collection or one motif with the shared MEME representation."""
    blocks = []
    for row in rows.itertuples():
        pwm = np.vstack(row.pwm)
        if pwm.ndim != 2 or pwm.shape[1] != 4 or not np.isfinite(pwm).all():
            raise ValueError(
                f"Expected a finite (L, 4) probability matrix for {row.motif_id}"
            )
        name = row.TF_Name if names is None else names.get(row.motif_id, row.TF_Name)
        lines = [
            f"MOTIF {row.motif_id} {name}",
            "",
            f"letter-probability matrix: alength= 4 w= {len(pwm)} nsites= {int(row.n_members)} E= 0",
        ]
        lines += [
            "  ".join(f"{probability:.6f}" for probability in position)
            for position in pwm
        ]
        blocks.append("\n".join(lines))
    return MEME_HEADER + "\n" + "\n\n".join(blocks) + "\n"


def meme_documents(modes: pd.DataFrame, lookup: pd.DataFrame) -> dict[str, str]:
    """Serialize both collections using symbols from the final query lookup.

    Match ZebraMotif downloads: A,C,G,T probabilities to six decimal places,
    cluster support as nsites, and sorted, deduplicated query symbols as names.
    Sorting by DBID, tier, and rank makes output independent of worker completion.
    """
    if modes["motif_id"].duplicated().any():
        raise ValueError("Motif IDs must be unique")
    if not modes["tier"].isin(["primary", "alternate"]).all():
        raise ValueError("Unknown binding-mode tier")
    symbols = (
        lookup.groupby("Rep_DBID")["TF_Name"]
        .apply(lambda names: ",".join(sorted(set(names.dropna().astype(str)))))
        .to_dict()
    )
    ordered = modes.sort_values(
        ["DBID", "tier", "mode_rank"],
        key=lambda column: column.map({"primary": 0, "alternate": 1})
        if column.name == "tier"
        else column,
    )
    names = {
        row.motif_id: symbols.get(row.DBID) or row.TF_Name
        for row in ordered.itertuples()
    }
    return {
        "zebramotif_primary_modes.meme": meme_document(
            ordered.loc[ordered["tier"].eq("primary")], names
        ),
        "zebramotif_all_modes.meme": meme_document(ordered, names),
    }


def _write_text(text: str, path: Path) -> None:
    """Replace a changed text export atomically, preserving file permissions."""
    if path.exists() and path.read_text() == text:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        temporary.write_text(text)
        permissions_source = (
            path if path.exists() else path.parent / "tf_to_rep.parquet"
        )
        if permissions_source.exists():
            shutil.copymode(permissions_source, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_meme_exports(
    modes: pd.DataFrame, lookup: pd.DataFrame, database_dir: Path
) -> None:
    """Create or refresh both MEME files from the exported tables."""
    documents = meme_documents(modes, lookup)
    for name, text in documents.items():
        _write_text(text, Path(database_dir) / name)


def load_zfin_ids(path: Path) -> pd.DataFrame:
    """Read a frozen release mapping or a cached ZFIN text cross-reference."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=["DBID", "zfin_id"])
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["zfin_id", "so_term", "symbol", "DBID"],
        dtype=str,
    )[["DBID", "zfin_id"]]


def add_lookup_annotations(
    lookup: pd.DataFrame, catalog: pd.DataFrame, zfin_ids: pd.DataFrame
) -> pd.DataFrame:
    """Attach query-specific ZFIN IDs and exact original-catalog membership.

    Domain entries are not rows in the original gene catalog, and do not inherit
    the parent gene's ZFIN ID. Missing mappings remain null.
    """
    mapping = zfin_ids[["DBID", "zfin_id"]].dropna().drop_duplicates()
    if mapping["DBID"].duplicated().any():
        raise ValueError("ZFIN cross-reference has conflicting IDs for the same DBID")
    result = add_sequence_types(lookup)
    result["zfin_id"] = (
        result["DBID"].map(mapping.set_index("DBID")["zfin_id"]).astype("string")
    )
    result["in_original_catalog"] = result["DBID"].isin(catalog["DBID"])
    return result


def select_domain_annotations(domains: pd.DataFrame, keep_dbids: set) -> pd.DataFrame:
    """Preserve all annotated domains and complete DNA-placement flags."""
    flags = domains["DNA-Binding"]
    if not flags.empty and (
        flags.isna().any() or not pd.api.types.is_bool_dtype(flags)
    ):
        raise ValueError(
            "Domain annotations require complete boolean DNA-Binding flags"
        )
    result = domains.loc[domains["DBID"].isin(keep_dbids), DOMAIN_COLUMNS].copy()
    result["DNA-Binding"] = result["DNA-Binding"].astype(bool)
    return result.reset_index(drop=True)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Replace one table atomically while preserving its file permissions."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".parquet", delete=False
    ) as f:
        temporary = Path(f.name)
    try:
        frame.to_parquet(temporary, index=False)
        permissions_source = (
            path if path.exists() else path.parent / "tf_to_rep.parquet"
        )
        if permissions_source.exists():
            shutil.copymode(permissions_source, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_domain_annotations(domains: pd.DataFrame, database_dir: Path) -> None:
    """Export one complete domain table and retire its two CSV predecessors."""
    _write_parquet(domains, database_dir / "dbd_domains.parquet")
    for name in ["domain_annotations.csv", "dbd_domains.csv"]:
        (database_dir / name).unlink(missing_ok=True)


def consolidate_database_annotations(
    database_dir: Path,
    catalog: pd.DataFrame,
    zfin_ids: pd.DataFrame,
    domains: pd.DataFrame,
) -> dict:
    """Consolidate website annotations without rebuilding motifs or changing joins."""
    database_dir = Path(database_dir)
    lookup = add_lookup_annotations(
        pd.read_parquet(database_dir / "tf_to_rep.parquet"), catalog, zfin_ids
    )
    keep_dbids = set(
        pd.read_parquet(database_dir / "pwm_modes.parquet", columns=["DBID"])["DBID"]
    )
    domain_table = select_domain_annotations(domains, keep_dbids)
    _write_parquet(lookup, database_dir / "tf_to_rep.parquet")
    write_domain_annotations(domain_table, database_dir)
    for name in ["zfin_ids.parquet", "animaltfdb4_tf_ids.txt"]:
        (database_dir / name).unlink(missing_ok=True)
    return {
        "original_catalog_entries": int(lookup["in_original_catalog"].sum()),
        "zfin_links": int(lookup["zfin_id"].notna().sum()),
        "domain_annotations": len(domain_table),
        "dna_binding_domains": int(domain_table["DNA-Binding"].sum()),
    }


def refresh_database_metadata(database_dir: Path) -> dict:
    """Update an existing export without recomputing predictions or motif matrices.

    Repair modeled queries' self-resolution, regenerate MEME files from the
    final lookup, remove excluded genes and structures, and add sequence types.
    Retain the release's other query assignments and all retained motif matrices.
    The original research inputs and job outputs are untouched.
    """
    database_dir = Path(database_dir)
    original_modes = pd.read_parquet(database_dir / "pwm_modes.parquet")
    modes = add_sequence_types(original_modes)
    original_lookup = pd.read_parquet(database_dir / "tf_to_rep.parquet")
    lookup = reconcile_representatives(add_sequence_types(original_lookup), modes)
    documents = meme_documents(modes, lookup)

    removed = original_modes.loc[~original_modes["DBID"].isin(set(modes["DBID"]))]
    structure_paths = set(removed["structure_path"].dropna()) - set(
        modes["structure_path"].dropna()
    )
    for relative in structure_paths:
        path = database_dir / relative
        if not path.resolve().is_relative_to(database_dir.resolve()):
            raise ValueError(f"Structure path falls outside the export: {relative}")

    tables = {}
    if (
        len(original_modes) != len(modes)
        or "sequence_type" not in original_modes
        or not original_modes["sequence_type"].equals(modes["sequence_type"])
    ):
        tables["pwm_modes.parquet"] = modes
    if not original_lookup.equals(lookup):
        tables["tf_to_rep.parquet"] = lookup
    for name in [
        "interface_residues.parquet",
        "confidence_metrics.parquet",
        "dbd_domains.parquet",
        "zfin_ids.parquet",
    ]:
        path = database_dir / name
        if path.exists():
            original = pd.read_parquet(path)
            filtered = filter_database_rows(original)
            if len(filtered) != len(original):
                tables[name] = filtered

    for name, frame in tables.items():
        _write_parquet(frame, database_dir / name)

    for name in ["dbd_domains.csv", "domain_annotations.csv"]:
        domains_path = database_dir / name
        if domains_path.exists():
            domains = filter_database_rows(pd.read_csv(domains_path))
            domains.to_csv(domains_path, index=False)

    for name, text in documents.items():
        _write_text(text, database_dir / name)

    for relative in structure_paths:
        (database_dir / relative).unlink(missing_ok=True)

    return {
        "full_length_entries": int(lookup["sequence_type"].eq("full_length").sum()),
        "domain_entries": int(lookup["sequence_type"].eq("domain").sum()),
        "queryable_entries": len(lookup),
        "primary_modes": int(modes["tier"].eq("primary").sum()),
        "alternate_modes": int(modes["tier"].eq("alternate").sum()),
        "removed_modes": len(removed),
    }


def main() -> None:
    from af3tfm.dataset import DATA_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-dir", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DATA_DIR / "sequences/animaltfdb/tfdb4_tfs.csv",
    )
    parser.add_argument(
        "--zfin-cross-reference",
        type=Path,
        default=DATA_DIR / ZFIN_REFERENCE,
    )
    parser.add_argument(
        "--domains",
        type=Path,
        default=DATA_DIR / "sequences/animaltfdb/tfdb4_rep_doms.csv",
    )
    args = parser.parse_args()
    catalog = pd.read_csv(args.catalog)
    zfin_ids = load_zfin_ids(args.zfin_cross_reference)
    domains = pd.read_csv(args.domains)
    summary = refresh_database_metadata(args.database_dir)
    summary.update(
        consolidate_database_annotations(args.database_dir, catalog, zfin_ids, domains)
    )
    print(summary)


if __name__ == "__main__":
    main()
