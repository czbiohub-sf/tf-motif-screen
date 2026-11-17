#!/usr/bin/env python
"""Estimate AF3 fold runtime by sampling SLURM logs and querying sacct.

Samples fold-job log files (named `<jobid>.log` by run_af3.py) across the
`input_fold_*/slurm_logs` dirs, resolves each job's Elapsed time and node via
`sacct`, maps the node to its GPU model via `sinfo`, and prints per-GPU
mean/median runtime + job share plus a pooled summary.

The `.log` files themselves are empty; timing lives in SLURM accounting.

Usage:
    python scripts/estimate_fold_runtime.py                 # default 100k sample
    python scripts/estimate_fold_runtime.py -n 10000
"""

import argparse
import random
import re
import subprocess
from collections import defaultdict
from pathlib import Path

# Fold-log dirs under data/jobs/hero, with rough sampling weights (~ dir sizes).
HERO = Path(__file__).resolve().parent.parent / "data/jobs/hero"
LOG_DIRS = {
    HERO / "input_fold_all/slurm_logs": 0.017,
    HERO / "input_fold_all2/slurm_logs": 0.665,
    HERO / "input_fold_top_50/slurm_logs": 0.318,
}
MIN_REAL_SEC = 20  # below this = instant "results exist; skipping" early-exits


def sample_ids(n: int) -> list[str]:
    """Sample n job IDs from the log dirs, weighted by LOG_DIRS."""
    ids: list[str] = []
    for d, w in LOG_DIRS.items():
        if not d.is_dir():
            print(f"  warning: {d} missing, skipping")
            continue
        files = [p.stem for p in d.glob("*.log")]
        take = min(len(files), round(n * w))
        ids.extend(random.sample(files, take))
        print(f"  {d.name} ({d.parent.name}): sampled {take} of {len(files)}")
    return ids


def node_gpu_map() -> dict[str, str]:
    """node name -> GPU model, parsed from `sinfo` gres field (gpu:MODEL:count)."""
    out = subprocess.run(
        ["sinfo", "-N", "-h", "-o", "%N|%G"],
        capture_output=True,
        text=True,
    ).stdout
    mapping = {}
    for line in out.splitlines():
        node, _, gres = line.partition("|")
        m = re.match(r"gpu:([^:]+):", gres)
        if m:
            mapping[node.strip()] = m.group(1)
    return mapping


def query_sacct(ids: list[str], batch: int = 1000) -> list[tuple[str, int, str]]:
    """Return (state, elapsed_sec, node) per job, batching sacct calls."""
    rows = []
    for i in range(0, len(ids), batch):
        chunk = ",".join(ids[i : i + batch])
        out = subprocess.run(
            [
                "sacct",
                "-j",
                chunk,
                "-X",
                "--noheader",
                "--format=JobID%22,State%14,ElapsedRaw,NodeList%16",
            ],
            capture_output=True,
            text=True,
        ).stdout
        for line in out.splitlines():
            f = line.split()
            if len(f) < 3:
                continue
            try:
                elapsed = int(f[2])
            except ValueError:
                continue
            rows.append((f[1], elapsed, f[3] if len(f) > 3 else ""))
    return rows


def fmt(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60}m{sec % 60:02d}s"


def pct(vals: list[int], p: float) -> float:
    vals = sorted(vals)
    k = (len(vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "-n",
        "--num",
        type=int,
        default=100_000,
        help="number of jobs to sample (default 100000)",
    )
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = ap.parse_args()
    random.seed(args.seed)

    print(f"Sampling ~{args.num} fold-job logs...")
    ids = sample_ids(args.num)
    print(f"Querying sacct for {len(ids)} jobs...")
    rows = query_sacct(ids)
    n2g = node_gpu_map()

    states = defaultdict(int)
    for s, _, _ in rows:
        states[s] += 1
    print(
        "\nStates: "
        + " | ".join(
            f"{s} {c} ({100*c/len(rows):.1f}%)"
            for s, c in sorted(states.items(), key=lambda x: -x[1])
        )
    )

    # completed real folds only for the timing distribution
    comp = [(e, node) for s, e, node in rows if s == "COMPLETED" and e >= MIN_REAL_SEC]
    v = [e for e, _ in comp]

    by_gpu = defaultdict(list)
    for e, node in comp:
        by_gpu[n2g.get(node, "?" + node)].append(e)

    print(f"\n{'GPU':<10}{'n':>9}{'share':>8}{'median':>9}{'mean':>9}{'p90':>9}")
    for g in sorted(by_gpu, key=lambda k: -len(by_gpu[k])):
        vv = by_gpu[g]
        print(
            f"{g:<10}{len(vv):>9}{100*len(vv)/len(v):>7.1f}%"
            f"{fmt(pct(vv, 50)):>9}{fmt(sum(vv)/len(vv)):>9}{fmt(pct(vv, 90)):>9}"
        )

    print(f"\nPooled (n={len(v)} completed real folds):")
    print(
        f"  median={fmt(pct(v, 50))}  mean={fmt(sum(v)/len(v))}  "
        f"p90={fmt(pct(v, 90))}  p99={fmt(pct(v, 99))}  max={fmt(max(v))}"
    )

    per = sum(e for _, e, _ in rows) / len(rows)
    print(f"  mean per submitted job (incl fails/skips): {per:.1f}s ({fmt(per)})")


if __name__ == "__main__":
    main()
