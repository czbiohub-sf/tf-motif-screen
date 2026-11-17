"""
Summarise the per-protein distribution of AF3 confidence metrics for the Dash app, so it can
show how confident each TF fold and each protein-DNA interface is for a given protein.

Reads every prediction in data/jobs/hero/jobdb.sqlite (one row per protein x motif-DNA x seed)
and, for each protein, describes the distribution of each confidence metric. Two subsets are
emitted per protein:
    all    -- every prediction (the full DNA screen; its spread shows discrimination)
    top47  -- the NUM_SPECS structures actually used to build the PWM (ipTM x dom_pLDDT
              percentile-rank product, matching data/database/export_data.py)

Output: data/database/export/database/confidence_metrics.parquet, one row per (DBID, subset, metric):
    DBID, subset, metric, category, higher_is_better, n,
    mean, std, p0, p5, p10, p25, p50, p75, p90, p95, p100
The percentiles are enough to render a box/violin plot; category ('tf' | 'interface' |
'global') and higher_is_better tell the app how to group and colour each metric.

Ben Iovino 9/15/26 Biohub
"""

# %%
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from export_tools.database_export import filter_database_rows

NUM_SPECS = 47

# metric -> (category, higher_is_better)
METRICS = {
    "ipTM": ("interface", True),
    "iface_PAE": ("interface", False),
    "iface_PAE_contact": ("interface", False),
    "iface_contact_prob": ("interface", True),
    "n_contacted_bases": ("interface", True),
    "dna_pLDDT": ("interface", True),
    "dom_pLDDT": ("tf", True),
    "prot_pTM": ("tf", True),
    "fraction_disordered": ("tf", False),
    "ptm": ("global", True),
    "ranking_score": ("global", True),
    "has_clash": ("global", False),
}

PCTS = [0, 5, 10, 25, 50, 75, 90, 95, 100]


def _describe(df: pd.DataFrame, dbid: str, subset: str) -> list:
    """One summary row per metric for the given protein subset."""
    rows = []
    for metric, (cat, hib) in METRICS.items():
        vals = df[metric].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        qs = np.percentile(vals, PCTS)
        row = {
            "DBID": dbid,
            "subset": subset,
            "metric": metric,
            "category": cat,
            "higher_is_better": hib,
            "n": int(vals.size),
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
        }
        row.update({f"p{p}": round(float(q), 4) for p, q in zip(PCTS, qs)})
        rows.append(row)
    return rows


def export_confidence_metrics(
    jobdb_path: Path,
    out_dir: Path,
    keep_dbids: set | None = None,
    num_specs: int = NUM_SPECS,
) -> pd.DataFrame:
    """Export the existing confidence summaries without modifying the job database."""
    jobdb_path, out_dir = Path(jobdb_path), Path(out_dir)
    if not jobdb_path.is_file():
        raise FileNotFoundError(jobdb_path)
    cols = ["protein", "ipTM", "dom_pLDDT"] + [
        m for m in METRICS if m not in ("ipTM", "dom_pLDDT")
    ]
    with closing(
        sqlite3.connect(jobdb_path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as con:
        runs = pd.read_sql(f"SELECT {', '.join(cols)} FROM runs", con)
    runs["DBID"] = runs["protein"].str.upper()
    runs = filter_database_rows(runs)
    if keep_dbids is not None:
        runs = runs.loc[runs["DBID"].isin(keep_dbids)]
    print(f"loaded {len(runs)} runs across {runs['DBID'].nunique()} proteins")

    all_rows = []
    for dbid, g in runs.groupby("DBID"):
        all_rows.extend(_describe(g, dbid, "all"))
        # Top-47 by the same combined rank export_data uses for the PWM.
        combined = g["ipTM"].rank(pct=True) * g["dom_pLDDT"].rank(pct=True)
        top = g.loc[combined.nlargest(min(num_specs, len(g))).index]
        all_rows.extend(_describe(top, dbid, "top47"))

    conf = pd.DataFrame(
        all_rows,
        columns=[
            "DBID",
            "subset",
            "metric",
            "category",
            "higher_is_better",
            "n",
            "mean",
            "std",
            *[f"p{p}" for p in PCTS],
        ],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "confidence_metrics.parquet"
    with tempfile.NamedTemporaryFile(
        dir=out_dir, suffix=".parquet", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        conf.to_parquet(temporary, index=False)
        permissions_source = (
            out_path if out_path.exists() else out_dir / "pwm_modes.parquet"
        )
        if permissions_source.exists():
            shutil.copymode(permissions_source, temporary)
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"confidence_metrics: {len(conf)} rows | {conf['DBID'].nunique()} proteins | "
        f"{conf['metric'].nunique()} metrics x {conf['subset'].nunique()} subsets"
    )
    print(f"wrote {out_path}")
    return conf
