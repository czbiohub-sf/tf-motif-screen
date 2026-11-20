"""Prepare and package a complete release without requiring a website checkout."""

import hashlib
import html
import json
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import date
from pathlib import Path
from string import Template

import pandas as pd

from export_tools.database_export import (
    refresh_database_metadata,
    validate_catalog_coverage,
)
from export_tools.motif_downloads import MotifDownloads

AF3_TERMS = (
    "https://github.com/google-deepmind/alphafold3/blob/main/OUTPUT_TERMS_OF_USE.md"
)
AF3_PAPER = "10.1038/s41586-024-07487-w"
EXPORT_PATH = Path("data/database")
PIPELINE_REPOSITORY = "tf-motif-screen"
REPOSITORY_OWNER_URL = "https://github.com/czbiohub-sf"
PIPELINE_REPOSITORY_URL = f"{REPOSITORY_OWNER_URL}/{PIPELINE_REPOSITORY}"
PRODUCER = f"{PIPELINE_REPOSITORY}/{EXPORT_PATH.as_posix()}/export_data.py"
TABLES = (
    "pwm_modes.parquet",
    "tf_to_rep.parquet",
    "confidence_metrics.parquet",
    "interface_residues.parquet",
    "dbd_domains.parquet",
)
EXPORTS = (
    *TABLES,
    "zebramotif_primary_modes.meme",
    "zebramotif_all_modes.meme",
    "SCHEMA.md",
    "release_metadata.json",
    "zebramotif_primary_modes.metadata.tsv",
    "zebramotif_all_modes.metadata.tsv",
)
RIGHTS_DESCRIPTION = (
    "Applies to AlphaFold3 structures and derivatives, including predicted "
    "PPM/MEME motifs, confidence summaries, residue contacts, and AF3-derived "
    "domain-placement flags, following the source repositories' data "
    "policies. Preserve the full Output Terms, Legally Binding Terms of Use "
    "notice, modification descriptions, and AlphaFold3 citation. Source "
    "software is MIT-licensed. Reference annotations retain their providers' "
    "terms and attributions. See the deposited LICENSE.md for the scope by file."
)


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def file_record(path, relative_to):
    with path.open("rb") as handle:
        hasher = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def repository_state(root, url, generated_paths=()):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    scope = ["."]
    for path in [root / EXPORT_PATH / "export", *generated_paths]:
        if path.resolve().is_relative_to(root.resolve()):
            relative = path.resolve().relative_to(root.resolve()).as_posix()
            scope.append(f":(exclude){relative}/**")
    try:
        head = git("rev-parse", "HEAD")
        changed = bool(
            git("status", "--porcelain=v1", "--untracked-files=normal", "--", *scope)
        )
    except (OSError, subprocess.CalledProcessError):
        head, changed = None, None
    return {
        "url": url,
        "head_at_metadata_preparation": head,
        "has_local_changes": changed,
        "status_scope": (
            "Revision and status describe the local checkout used for packaging. "
            "Status excludes generated exports. Null means Git metadata is unavailable. "
            "The URL identifies the documented pipeline."
        ),
    }


def summarize(data):
    frames = {name: pd.read_parquet(data / name, engine="pyarrow") for name in TABLES}
    modes, queries = frames[TABLES[0]], frames[TABLES[1]]
    confidence, contacts, domains = (frames[name] for name in TABLES[2:])
    primary = modes.loc[modes["tier"].eq("primary")]
    reps = set(primary["DBID"])
    if reps - set(queries["DBID"]):
        raise ValueError("A modeled representative is missing from the query lookup")
    if primary["DBID"].duplicated().any():
        raise ValueError("A representative must have exactly one primary mode")
    if not primary["sequence_type"].isin(["full_length", "domain"]).all():
        raise ValueError("Every representative must have a known sequence type")
    if modes["motif_id"].duplicated().any() or queries["DBID"].duplicated().any():
        raise ValueError("Duplicate motif or query identifiers")
    if set(queries["Rep_DBID"].dropna()) - reps:
        raise ValueError("A query references a representative without a primary mode")
    own = queries.loc[queries["DBID"].isin(reps)]
    if not (
        own["DBID"].eq(own["Rep_DBID"])
        & own["is_representative"]
        & own["resolution"].eq("self")
    ).all():
        raise ValueError(
            "A modeled protein's query must resolve to its own predictions"
        )
    paths = set(primary["structure_path"].dropna())
    structures = {
        p.relative_to(data).as_posix() for p in (data / "structures").glob("*.cif.gz")
    }
    if paths != structures or len(paths) != len(primary):
        raise ValueError("Primary structure references and files do not agree")
    for rel in paths:
        if not (data / rel).resolve().is_relative_to(data.resolve()):
            raise ValueError(f"Structure path outside the dataset: {rel}")

    def missing(frame):
        return primary.loc[
            ~primary["DBID"].isin(frame["DBID"]), ["DBID", "TF_Name"]
        ].to_dict("records")

    counts = {
        "modes": len(modes),
        "primary_modes": len(primary),
        "full_length_representatives": int(
            primary["sequence_type"].eq("full_length").sum()
        ),
        "domain_representatives": int(primary["sequence_type"].eq("domain").sum()),
        "alternate_modes": int(modes["tier"].eq("alternate").sum()),
        "query_entries": len(queries),
        "catalog_genes": int(queries["in_original_catalog"].sum()),
        "full_length_entries": int(queries["sequence_type"].eq("full_length").sum()),
        "domain_entries": int(queries["sequence_type"].eq("domain").sum()),
        "resolved_queries": int(queries["Rep_DBID"].notna().sum()),
        "structures": len(structures),
        "contact_rows": len(contacts),
        "contact_proteins": contacts["DBID"].nunique(),
        "confidence_rows": len(confidence),
        "confidence_proteins": confidence["DBID"].nunique(),
        "domain_rows": len(domains),
        "annotated_proteins": domains["DBID"].nunique(),
        "missing_domain_proteins": len(missing(domains)),
    }
    coverage = {
        "domain_representatives": primary.loc[
            primary["sequence_type"].eq("domain"), ["DBID", "TF_Name"]
        ]
        .sort_values("TF_Name")
        .to_dict("records"),
        "unresolved_queries": queries.loc[
            queries["Rep_DBID"].isna(), ["DBID", "TF_Name"]
        ].to_dict("records"),
        "proteins_without_confidence": missing(confidence),
        "proteins_without_contacts": missing(contacts),
        "proteins_without_domain_annotations": missing(domains),
        "zero_domain_plddt_means": "Unavailable annotated-domain estimate, not zero-confidence folding.",
    }
    settings = {
        "num_specs": sorted(int(x) for x in modes["num_specs"].unique()),
        "cor_threshold": sorted(float(x) for x in modes["cor_threshold"].unique()),
        "maximum_alternates": 3,
        "primary_refined_similarity_cutoff": 0.72,
        "matrix_column_order": ["A", "C", "G", "T"],
    }
    if settings["num_specs"] != [47] or settings["cor_threshold"] != [0.73]:
        raise ValueError(
            "Export settings changed; update the methods documentation before releasing"
        )
    return counts, coverage, settings


def citation(config):
    def quote(value):
        return json.dumps(value, ensure_ascii=False)

    lines = [
        "cff-version: 1.2.0",
        "type: dataset",
        f"title: {quote(config['title'])}",
        f"version: {quote(config['version'])}",
        "message: >-",
        "  Cite the Zenodo record for the exact dataset version you used.",
        "  Also cite the AlphaFold3 paper listed in README.md.",
        "authors:",
    ]
    for creator in config["creators"]:
        lines += [
            f"  - family-names: {quote(creator['family_name'])}",
            f"    given-names: {quote(creator['given_name'])}",
        ]
        if creator.get("affiliation"):
            lines.append(f"    affiliation: {quote(creator['affiliation'])}")
        if creator.get("orcid"):
            lines.append(
                f"    orcid: {quote('https://orcid.org/' + creator['orcid'].removeprefix('https://orcid.org/'))}"
            )
    lines += [
        f"repository-code: {quote(config['repositories']['ZebraMotif'])}",
        f"license-url: {quote(AF3_TERMS)}",
    ]
    if config.get("doi"):
        lines.append(f"doi: {quote(config['doi'])}")
    if config.get("publication_date"):
        lines.append(f"date-released: {quote(config['publication_date'])}")
    return ("\n".join(lines) + "\n").encode()


def zenodo_metadata(config, counts):
    creators = []
    for creator in config["creators"]:
        person = {
            "type": "personal",
            "family_name": creator["family_name"],
            "given_name": creator["given_name"],
        }
        if creator.get("orcid"):
            person["identifiers"] = [
                {
                    "scheme": "orcid",
                    "identifier": creator["orcid"].removeprefix("https://orcid.org/"),
                }
            ]
        entry = {"person_or_org": person}
        if creator.get("affiliation"):
            entry["affiliations"] = [{"name": creator["affiliation"]}]
        creators.append(entry)
    description = (
        f"<p>{html.escape(config['description'])}</p>"
        f"<p>Version {html.escape(config['version'])} "
        f"({html.escape(config['release_label'])}): {counts['modes']:,} modes "
        f"({counts['primary_modes']:,} primary, {counts['alternate_modes']:,} alternate), "
        f"{counts['structures']:,} selected structures, and {counts['query_entries']:,} query entries. "
        f"The {counts['primary_modes']:,} modeled representatives comprise "
        f"{counts['full_length_representatives']:,} full-length proteins and "
        f"{counts['domain_representatives']:,} separately modeled domains. "
        "The included README, schema, and provenance describe methods, coverage, and file identities.</p>"
        f'<p>Predictions and derivatives are subject to the <a href="{AF3_TERMS}">AlphaFold3 Output Terms of Use</a>. '
        "See the deposited LICENSE.md for the scope by file, preserved software MIT notices, and reference attributions.</p>"
        f'<p>Also cite Abramson et al., Nature (2024), <a href="https://doi.org/{AF3_PAPER}">{AF3_PAPER}</a>.</p>'
    )
    metadata = {
        "resource_type": {"id": "dataset"},
        "title": config["title"],
        "version": config["version"],
        "creators": creators,
        "description": description,
        "subjects": [{"subject": keyword} for keyword in config["keywords"]],
        "rights": [
            {
                "title": {"en": "Other"},
                "description": {"en": RIGHTS_DESCRIPTION},
                "link": AF3_TERMS,
            }
        ],
        "related_identifiers": [
            {
                "identifier": url,
                "scheme": "url",
                "relation_type": {"id": "isdocumentedby"},
            }
            for url in config["repositories"].values()
        ]
        + [
            {
                "identifier": AF3_PAPER,
                "scheme": "doi",
                "relation_type": {"id": "references"},
            }
        ],
    }
    if config.get("publication_date"):
        metadata["publication_date"] = config["publication_date"]
    document = {"access": {"record": "public", "files": "public"}, "metadata": metadata}
    if config.get("doi"):
        document["pids"] = {
            "doi": {"identifier": config["doi"], "provider": "datacite"}
        }
    return json_bytes(document)


def migrate_pipeline_repository(config):
    """Use the current pipeline identity when loading older release metadata."""
    repositories = config.get("repositories")
    missing_url = f"Set repositories.{PIPELINE_REPOSITORY} to the pipeline URL in release metadata"
    if not isinstance(repositories, dict):
        raise ValueError(missing_url)
    if PIPELINE_REPOSITORY in repositories:
        url = repositories[PIPELINE_REPOSITORY]
        if not isinstance(url, str) or not url.strip():
            raise ValueError(missing_url)
        return

    candidates = [
        name
        for name, url in repositories.items()
        if name != "ZebraMotif"
        and isinstance(url, str)
        and url.rstrip("/").removesuffix(".git").casefold()
        == f"{REPOSITORY_OWNER_URL}/{name}".casefold()
    ]
    methods_basis = config["reported_versions"].get("basis")
    previous = methods_basis.partition("/")[0] if isinstance(methods_basis, str) else ""
    if previous not in candidates:
        if len(candidates) != 1 or set(repositories) - {"ZebraMotif"} != set(
            candidates
        ):
            raise ValueError(missing_url)
        (previous,) = candidates

    previous_url = repositories.pop(previous)
    repositories[PIPELINE_REPOSITORY] = PIPELINE_REPOSITORY_URL
    prefix = f"{previous}/"
    for field, replacement in (
        ("reported_versions", f"{PIPELINE_REPOSITORY}/"),
        ("observed_annotation_versions", ""),
    ):
        versions = config[field]
        basis = versions.get("basis")
        if isinstance(basis, str) and basis.startswith(prefix):
            versions["basis"] = replacement + basis.removeprefix(prefix)
    config["description"] = re.sub(
        rf"(?<![\w.-]){re.escape(previous)}(?![\w-])",
        PIPELINE_REPOSITORY,
        config["description"].replace(previous_url, PIPELINE_REPOSITORY_URL),
    )


def load_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config.get("version"), str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._+-]*", config["version"]
    ):
        raise ValueError("Dataset version must be a filename-safe text label")
    if not config["creators"]:
        raise ValueError("At least one dataset creator is required")
    for key in ["snapshot_date", "publication_date"]:
        if config.get(key):
            date.fromisoformat(config[key])
    if config.get("doi") and not re.fullmatch(r"10\.\d{4,9}/\S+", config["doi"]):
        raise ValueError("Use a DOI identifier, without a https://doi.org/ prefix")
    migrate_pipeline_repository(config)
    return config


def copy_database(database, raw):
    """Copy only the scientific release payload, checking it against its sources."""
    database, raw = Path(database), Path(raw)
    modes = pd.read_parquet(database / "pwm_modes.parquet", engine="pyarrow")
    primary = modes.loc[modes["tier"].eq("primary")]
    structures = set(primary["structure_path"].dropna())
    present = {
        path.relative_to(database).as_posix()
        for path in (database / "structures").glob("*.cif.gz")
    }
    if structures != present or len(structures) != len(primary):
        raise ValueError("Primary structure references and source files do not agree")
    raw.mkdir(parents=True)
    records = []
    for name in [*TABLES, *sorted(structures)]:
        source = database / name
        if source.is_symlink() or not source.resolve().is_relative_to(
            database.resolve()
        ):
            raise ValueError(f"Structure or table outside the source dataset: {name}")
        record = file_record(source, database)
        target = raw / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if file_record(target, raw)["sha256"] != record["sha256"]:
            raise ValueError(f"Source database changed while copying: {name}")
        records.append(record)
    return records


def collection_downloads(raw, metadata):
    modes = pd.read_parquet(raw / "pwm_modes.parquet", engine="pyarrow").sort_values(
        ["DBID", "tier", "mode_rank"],
        key=lambda column: column.map({"primary": 0, "alternate": 1})
        if column.name == "tier"
        else column,
    )
    queries = pd.read_parquet(raw / "tf_to_rep.parquet", engine="pyarrow")
    symbols = (
        queries.groupby("Rep_DBID")["TF_Name"]
        .apply(lambda names: ",".join(sorted(set(names.dropna().astype(str)))))
        .to_dict()
    )
    names = {
        row.motif_id: symbols.get(row.DBID) or row.TF_Name for row in modes.itertuples()
    }
    return MotifDownloads(
        {"primary": modes.loc[modes["tier"].eq("primary")], "all": modes},
        queries,
        names,
        raw,
        metadata,
    )


def prepare_documents(
    root, raw, database, config, source_records, config_path, generated_paths=()
):
    release = root / EXPORT_PATH / "release"
    payload_paths = [raw / name for name in EXPORTS] + sorted(
        (raw / "structures").glob("*.cif.gz")
    )
    payload = [file_record(path, raw) for path in payload_paths]
    counts, coverage, settings = summarize(raw)
    counts["payload_bytes"] = sum(row["size_bytes"] for row in payload)
    authors = "; ".join(
        f"{creator['family_name']}, {creator['given_name']}"
        for creator in config["creators"]
    )
    cite = f"{authors.rstrip('.')}. *{config['title']}*. Version {config['version']}."
    if config.get("doi"):
        cite += f" Zenodo. [https://doi.org/{config['doi']}](https://doi.org/{config['doi']})"
    else:
        cite += " Cite the version-specific Zenodo DOI once assigned; `CITATION.cff` records the dataset metadata."
    pipeline_url = config["repositories"][PIPELINE_REPOSITORY]
    pipeline_notice = f"notices/{PIPELINE_REPOSITORY}-MIT.txt"
    values = {key: f"{value:,}" for key, value in counts.items()}
    values.update(
        title=config["title"],
        version=config["version"],
        release_label=config["release_label"],
        citation=cite,
        pipeline_name=PIPELINE_REPOSITORY,
        pipeline_url=pipeline_url,
        pipeline_notice=pipeline_notice,
    )
    values["domain_representative_names"] = ", ".join(
        f"`{row['TF_Name']}`" for row in coverage["domain_representatives"]
    )
    unresolved = coverage["unresolved_queries"]
    missing_confidence = coverage["proteins_without_confidence"]
    values["unresolved_description"] = (
        "Unresolved query IDs: "
        + ", ".join(f"`{row['DBID']}`" for row in unresolved)
        + "."
        if unresolved
        else "All queries resolve to representative motifs."
    )
    values["confidence_description"] = (
        "Confidence summaries are absent for "
        + ", ".join(f"`{row['TF_Name']}`" for row in missing_confidence)
        + "."
        if missing_confidence
        else "Confidence summaries cover every modeled protein."
    )
    outputs = {
        "README.md": Template((release / "README.template.md").read_text())
        .substitute(values)
        .encode(),
        "LICENSE.md": Template((release / "LICENSE.md").read_text())
        .substitute(values)
        .encode(),
        "Legally Binding Terms of Use.txt": (
            release / "Legally Binding Terms of Use.txt"
        ).read_bytes(),
        "CITATION.cff": (
            citation(config) + (release / "citation-references.yml").read_bytes()
        ),
        "ZENODO_METADATA.json": zenodo_metadata(config, counts),
        "notices/ZebraMotif-MIT.txt": (
            release / "notices/ZebraMotif-MIT.txt"
        ).read_bytes(),
        pipeline_notice: (root / "LICENSE").read_bytes(),
    }
    notice_sources = json.loads((release / "notices/sources.json").read_text())
    for source in notice_sources["documents"]:
        name = source["filename"]
        if Path(name).name != name:
            raise ValueError(f"Notice filename must be a basename: {name}")
        content = (release / "notices" / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError(
                f"Upstream notice changed without updating its source record: {name}"
            )
        outputs[f"notices/{name}"] = content
    outputs["notices/sources.json"] = json_bytes(notice_sources)

    source_paths = [
        "README.md",
        "LICENSE",
        "data/database/README.md",
        "data/database/export_data.py",
        "scripts/get_domains.py",
        "scripts/get_contacts.py",
        "src/af3tfm/consensus.py",
        "data/database/export_tools/__init__.py",
        "data/database/export_tools/database_export.py",
        "data/database/export_tools/confidence_export.py",
        "data/database/export_tools/motif_downloads.py",
        "data/database/export_tools/release_export.py",
        "data/database/export_tools/metric_bands.py",
        "data/sequences/animaltfdb/tfdb4_tfs.csv",
        "data/sequences/animaltfdb/tfdb4_rep_tfs.csv",
        "data/sequences/animaltfdb/tfdb4_rep_doms.csv",
        "data/tools/homer/jaspar.homer.motif",
        "data/tools/blast/zfin/zfin_ids_release.parquet",
        "data/tools/blast/zfin/zfin_ids_release.provenance.json",
        "data/tools/ipscan/tfdb4_rep/tfdb4_rep_results.json",
    ]
    source_paths += [
        path.relative_to(root).as_posix()
        for path in sorted(release.rglob("*"))
        if path.is_file()
    ]
    source_paths += [
        path.relative_to(root).as_posix()
        for path in sorted(
            (root / "data/tools/mmseqs/animaltfdb4/results").glob("*_cluster.tsv")
        )
    ]
    observed = [
        {"repository": PIPELINE_REPOSITORY, **file_record(root / name, root)}
        for name in sorted(set(source_paths))
        if (root / name).is_file()
    ]
    source_database = (
        database.relative_to(root).as_posix()
        if database.is_relative_to(root)
        else str(database)
    )
    provenance = {
        "schema_version": 2,
        "dataset": {
            key: config[key]
            for key in [
                "title",
                "version",
                "release_label",
                "snapshot_date",
                "publication_date",
                "doi",
                "creators",
            ]
        },
        "counts": counts,
        "export_settings": settings,
        "coverage": coverage,
        "documented_snapshots": config["documented_snapshots"],
        "reported_versions": config["reported_versions"],
        "observed_annotation_versions": config["observed_annotation_versions"],
        "repositories": {
            PIPELINE_REPOSITORY: repository_state(root, pipeline_url, generated_paths)
        },
        "source_identity_scope": "Source-file hashes and repository revisions were recorded during packaging. Original model-generation inputs and dates are identified only where documented.",
        "observed_source_files": observed,
        "source_database": source_database,
        "release_configuration": {
            **file_record(config_path, config_path.parent),
            "path": config_path.relative_to(root).as_posix()
            if config_path.is_relative_to(root)
            else str(config_path),
        },
        "input_export_files": source_records,
        "payload_files": payload,
    }
    outputs["PROVENANCE.json"] = json_bytes(provenance)
    hashes = {
        path.relative_to(raw).as_posix(): file_record(path, raw)["sha256"]
        for path in sorted(raw.rglob("*"))
        if path.is_file() and path.relative_to(raw).as_posix() not in outputs
    }
    hashes.update(
        {name: hashlib.sha256(content).hexdigest() for name, content in outputs.items()}
    )
    for record in payload:
        if hashes[record["path"]] != record["sha256"]:
            raise ValueError(
                "Dataset changed during preparation; rerun with a stable snapshot"
            )
    outputs["SHA256SUMS"] = "".join(
        f"{digest}  {name}\n" for name, digest in sorted(hashes.items())
    ).encode()
    for name, content in outputs.items():
        target = raw / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return counts


def inventory(directory):
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink in release: {path.relative_to(directory)}")
        if path.is_file():
            record = file_record(path, directory)
            files[record["path"]] = record
    return files


def build_archives(directory, downloads):
    raw = directory / "raw"
    names = [f"zebramotif-data-{downloads.version}.zip"]
    with zipfile.ZipFile(directory / names[0], "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(raw.rglob("*")):
            if path.is_file():
                downloads._write_member(
                    archive, path.relative_to(raw).as_posix(), path.read_bytes()
                )
    for collection in ["primary", "all"]:
        name = downloads.filename(collection, "zip")
        with (directory / name).open("wb") as handle:
            downloads.write_zip(handle, collection)
        names.append(name)
    records = [file_record(directory / name, directory) for name in names]
    sums = "".join(f"{row['sha256']}  {row['path']}\n" for row in records)
    (directory / "SHA256SUMS").write_text(sums)
    for record in records:
        (directory / (record["path"] + ".sha256")).write_text(
            f"{record['sha256']}  {record['path']}\n"
        )


def validate_output_directory(output):
    """Replace only a release previously managed by this exporter."""
    if output.is_symlink():
        raise ValueError(f"Release output cannot be a symlink: {output}")
    if not output.exists():
        return
    if not output.is_dir():
        raise ValueError(f"Release output must be a directory: {output}")
    if not any(output.iterdir()):
        return
    manifest = output / "EXPORT_MANIFEST.json"
    if not manifest.is_file():
        raise ValueError(f"Nonempty output is not a managed release: {output}")
    saved = json.loads(manifest.read_text())
    producer = saved.get("producer")
    owned = {row["path"] for row in saved.get("files", ())}
    required_files = {
        "raw/release_metadata.json",
        "raw/PROVENANCE.json",
        f"zebramotif-data-{saved.get('version')}.zip",
    }
    # Releases from either exporter path can be rebuilt after a repository rename.
    if (
        not isinstance(producer, str)
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*/(?:data/database|scripts)/export_data\.py",
            producer,
        )
        or saved.get("schema_version") != 1
        or not required_files <= owned
    ):
        raise ValueError(f"Output belongs to a different producer: {output}")
    owned.add("EXPORT_MANIFEST.json")
    present = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in release: {path.relative_to(output)}")
        if path.is_file():
            present.add(path.relative_to(output).as_posix())
    extras = present - owned
    if extras:
        raise ValueError(f"Unmanaged files in release output: {sorted(extras)}")


def export_release(
    root, database, output=None, *, schema, metadata_path=None, check=False
):
    """Build all deliverables in staging, then publish or check the complete release."""
    root, database = Path(root).resolve(), Path(database).resolve()
    config_path = (
        Path(metadata_path).resolve()
        if metadata_path
        else root / EXPORT_PATH / "release/metadata.json"
    )
    config = load_config(config_path)
    output = (
        Path(output).absolute()
        if output
        else root / EXPORT_PATH / "export" / config["version"]
    )
    resolved_output = output.resolve()
    if (
        root.is_relative_to(resolved_output)
        or database.is_relative_to(resolved_output)
        or resolved_output.is_relative_to(database)
    ):
        raise ValueError(
            "Release output cannot contain the repository or overlap the source database"
        )
    validate_output_directory(output)
    if check and not output.is_dir():
        raise ValueError(f"No completed release to check: {output}")
    if not check:
        output.parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = None if check else output.parent
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.export-", dir=temporary_parent
    ) as temporary:
        staging = Path(temporary) / "release"
        staging.mkdir()
        raw = staging / "raw"
        source_records = copy_database(database, raw)
        refresh_database_metadata(raw)
        catalog_path = root / "data/sequences/animaltfdb/tfdb4_tfs.csv"
        if catalog_path.is_file():
            validate_catalog_coverage(
                pd.read_parquet(raw / "tf_to_rep.parquet"),
                pd.read_csv(catalog_path, usecols=["DBID"]),
            )
        (raw / "SCHEMA.md").write_text(schema)
        counts, _, settings = summarize(raw)
        metadata = {
            key: value
            for key, value in config.items()
            if key not in ["description", "keywords"]
        }
        metadata["counts"] = counts
        metadata["methods"] = {
            **metadata["methods"],
            **settings,
            "num_specs": settings["num_specs"][0],
            "cor_threshold": settings["cor_threshold"][0],
        }
        (raw / "release_metadata.json").write_bytes(json_bytes(metadata))
        downloads = collection_downloads(raw, metadata)
        for collection in ["primary", "all"]:
            (raw / downloads.filename(collection, "metadata.tsv")).write_text(
                downloads.metadata_tsv(collection)
            )
        counts = prepare_documents(
            root,
            raw,
            database,
            config,
            source_records,
            config_path,
            [database, output, Path(temporary)],
        )
        build_archives(staging, downloads)
        expected = inventory(staging)
        manifest = {
            "schema_version": 1,
            "producer": PRODUCER,
            "version": config["version"],
            "files": list(expected.values()),
        }
        (staging / "EXPORT_MANIFEST.json").write_bytes(json_bytes(manifest))
        expected["EXPORT_MANIFEST.json"] = file_record(
            staging / "EXPORT_MANIFEST.json", staging
        )
        actual = inventory(output) if output.is_dir() else {}
        changed = sorted(
            name
            for name in set(actual) | set(expected)
            if actual.get(name) != expected.get(name)
        )
        if check:
            if changed:
                preview = ", ".join(changed[:8])
                raise ValueError(f"Release is stale ({len(changed)} files): {preview}")
            print(f"Release documents, raw files, ZIPs, and checksums match: {output}")
            return output
        if not changed:
            print(f"Release already matches: {output}")
            return output
        previous = Path(temporary) / "previous"
        if output.exists():
            output.replace(previous)
        try:
            staging.replace(output)
        except OSError:
            if previous.exists():
                previous.replace(output)
            raise
        print(
            f"Exported {counts['modes']} modes, {counts['structures']} structures, raw files, and 3 ZIPs to {output}"
        )
        return output
