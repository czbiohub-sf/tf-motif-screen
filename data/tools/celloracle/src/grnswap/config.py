"""Central config + path resolution for the GRN motif-swap framework.

Every script reads grn_swap.yaml through here so paths/params stay consistent across the
CisBP, ZebraMotif, and hybrid arms. The only thing that ever differs between arms is the motif DB.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# grn/src/grnswap/config.py -> grn/
GRN_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = GRN_ROOT / "config" / "grn_swap.yaml"
ANNOTATIONS = ("global_annotation", "annotation_ML_coarse")

# Read af3tfm's config.yaml directly from the repository root.
PROJECT_ROOT = GRN_ROOT.parents[2]
AF3TFM_CONFIG = PROJECT_ROOT / "src" / "af3tfm" / "config.yaml"
AF3TFM_LOCAL_CONFIG = AF3TFM_CONFIG.with_name("config.local.yaml")  # personal overrides

# (section, key) in grn_swap.yaml given relative to PROJECT_ROOT
_REPO_PATHS = [
    ("motif_db", "zebramotif_meme"),
    ("motif_db", "zebramotif_meme_primary"),
    ("motif_db", "zebramotif_prefix"),
    ("motif_db", "cisbp_prefix"),
    ("motif_db", "hybrid_prefix"),
    ("outputs", "root"),
    ("outputs", "shared_root"),
]

# (section, key) in grn_swap.yaml -> key under `paths` in af3tfm's config.yaml
_EXTERNAL_PATHS = {
    ("motif_db", "cisbp_cache"): "cisbp_cache",
    ("inputs", "rna_h5ad"): "zebrahub_multiome_rna_h5ad",
    ("inputs", "zebrahub_processed"): "zebrahub_processed",
}


def add_annotation_argument(parser) -> None:
    """Add the per-run override for the config's annotation toggle."""
    parser.add_argument(
        "--annotation",
        choices=ANNOTATIONS,
        help="Override grn.annotation and select its output directory for this run",
    )


def load_config(
    path: str | Path | None = None,
    annotation: str | None = None,
    *,
    resolve_external: bool = True,
) -> dict:
    """Resolve shared settings and annotation-specific outputs from one config.

    Analysis scripts can skip external data-path resolution. When shared_root
    is configured, outputs.root is derived from the selected annotation.
    """
    cfg = yaml.safe_load(Path(path or CONFIG_PATH).read_text())
    if annotation is not None and annotation not in ANNOTATIONS:
        raise ValueError(
            f"Unknown annotation {annotation!r}; choose from {ANNOTATIONS}"
        )
    for sec, key in _REPO_PATHS:
        if key in cfg.get(sec, {}):
            cfg[sec][key] = str(PROJECT_ROOT / cfg[sec][key])
    if "shared_root" in cfg["outputs"]:
        selected = annotation or cfg["grn"]["annotation"]
        if selected not in ANNOTATIONS:
            raise ValueError(
                f"Unknown annotation {selected!r}; choose from {ANNOTATIONS}"
            )
        cfg["grn"]["annotation"] = selected
        shared = Path(cfg["outputs"]["shared_root"])
        cfg["outputs"]["root"] = str(shared / selected)
        cfg.setdefault("inputs", {}).setdefault(
            "base_grn_root", str(shared / "base_grn")
        )
    elif annotation is not None and annotation != cfg["grn"].get("annotation"):
        raise ValueError(
            "Switching annotation requires outputs.shared_root in the config"
        )
    if "base_grn_root" in cfg.get("inputs", {}):
        cfg["inputs"]["base_grn_root"] = str(
            PROJECT_ROOT / cfg["inputs"]["base_grn_root"]
        )
    if not resolve_external:
        return cfg
    ext = yaml.safe_load(AF3TFM_CONFIG.read_text())["paths"]
    if AF3TFM_LOCAL_CONFIG.exists():
        ext.update(
            (yaml.safe_load(AF3TFM_LOCAL_CONFIG.read_text()) or {}).get("paths", {})
        )
    for (sec, key), name in _EXTERNAL_PATHS.items():
        if key not in cfg.setdefault(sec, {}):
            cfg[sec][key] = ext[name]
    return cfg


def timepoint(cfg: dict, index: int) -> dict:
    """Return the timepoint entry at the given index in the config's timepoints list."""
    return cfg["timepoints"][index]


# Input paths (reused, read-only)
def peak_csv(cfg: dict, tp: dict) -> str:
    """Path to a timepoint's stage-03 processed peak CSV in the zebrahub cicero outputs."""
    root = cfg["inputs"]["zebrahub_processed"]
    sub = cfg["inputs"]["cicero_subdir"]
    i = tp["id"]
    return f"{root}/{sub}/{i}_cicero/03_{i}_processed_peak_file_danRer11.csv"


# Output paths (gitignored)
def _out(cfg: dict, *parts: str) -> Path:
    """Join parts under the outputs root, creating the parent directory, and return the path."""
    p = Path(cfg["outputs"]["root"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _base_grn_root(cfg: dict) -> Path:
    return Path(
        cfg.get("inputs", {}).get(
            "base_grn_root",
            Path(cfg["outputs"].get("shared_root", cfg["outputs"]["root"]))
            / "base_grn",
        )
    )


def base_grn_db(cfg: dict, tp: dict, db: str) -> str:
    """Shared stage-02 prior; its location does not depend on annotation."""
    path = _base_grn_root(cfg) / db / f"02_{tp['id']}_base_GRN_dataframe.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def base_grn_for_arm(cfg: dict, tp: dict, arm: str) -> str:
    """Read an arm's shared stage-02 prior, or the run's own prior by default."""
    return str(_base_grn_root(cfg) / arm / f"02_{tp['id']}_base_GRN_dataframe.parquet")


def motif_pfm_for_db(cfg: dict, db: str) -> str:
    """Path to the .pfm for a motif DB (db in {cisbp, zebramotif, hybrid})."""
    key = {
        "cisbp": "cisbp_prefix",
        "zebramotif": "zebramotif_prefix",
        "hybrid": "hybrid_prefix",
    }[db]
    return cfg["motif_db"][key] + ".pfm"


def tfinfo_out(cfg: dict, tp: dict, db: str = "zebramotif") -> str:
    """Path to the stage-02 TFinfo (scanned peaks x motifs) HDF5 for a motif DB."""
    path = (
        _base_grn_root(cfg) / db / f"02_{tp['id']}_peaks_motifs_TFs.celloracle.tfinfo"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def oracle_out(cfg: dict, tp: dict, arm: str) -> str:
    """Path to the stage-03 Oracle object for a timepoint/arm."""
    return str(_out(cfg, "oracle", arm, f"03_{tp['id']}.celloracle.oracle"))


def links_out(cfg: dict, tp: dict, arm: str) -> str:
    """Path to the stage-03 per-celltype Links object for a timepoint/arm."""
    return str(_out(cfg, "links", arm, f"03_{tp['id']}_celltype_GRNs.celloracle.links"))


def grn_csv_dir(cfg: dict, tp: dict, arm: str) -> str:
    """Directory for the per-celltype raw GRN CSVs of a timepoint/arm (created if absent)."""
    d = Path(cfg["outputs"]["root"]) / "grn_csv" / arm / f"03_{tp['id']}"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def comparison_dir(cfg: dict) -> str:
    """Directory for cross-arm comparison outputs (created if absent)."""
    d = Path(cfg["outputs"]["root"]) / "comparison"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
