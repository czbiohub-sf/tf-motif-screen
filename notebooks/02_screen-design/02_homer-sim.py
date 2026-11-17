"""
Plotting within-cluster and all-motif similarity distributions for
HOMER results using global Pearson correlation over the aligned overlap.

Ben Iovino 4/23/26  Biohub
"""

# %%
import json
import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from af3tfm.dataset import DATA_DIR, PROJECT_ROOT
from af3tfm.motif import load_homer_motif_file, load_homer_multi_motif_file
from af3tfm.utils import pairwise_similarity_matrix, set_plotting_style

set_plotting_style()

FIG_DIR = PROJECT_ROOT / "notebooks" / "02_screen-design" / "figures"
FIG_DIR.mkdir(exist_ok=True)

PLOT_MINIMUM = -0.5

SIMILARITY_METHOD = {
    "name": "global_pearson_overlap_min6_v1",
    "formula": "sum((A - 0.25) * (B - 0.25)) / sqrt(sum((A - 0.25)**2) * sum((B - 0.25)**2))",
    "normalization": "overlapping positions only",
    "minimum_overlap_bp": 6,
    "alignment": "maximum across valid offsets and both DNA orientations",
    "negative_correlations": "retained",
    "implementation": "src/af3tfm/utils.py:pairwise_similarity_matrix",
}


def file_sha256(path: str | Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def intra_cluster_similarity(directory: str) -> dict[int, dict]:
    """
    Compute pairwise PWM similarity for every HOMER cluster in homerResults.

    A cluster is identified by motif<N>: includes motif<N>.motif (representative)
    and all motif<N>.similar*.motif files. RV (reverse-complement) files are excluded.

    Args:
        directory : str

    Returns:
        dict[int, dict]
            Returns a dict mapping cluster_id -> {
                'mean'    : float,
                'median'  : float,
                'min'     : float,
                'n_motifs': int,
                'scores'  : list[float],  # all pairwise scores
            }
    """

    clusters: dict[int, list[Path]] = defaultdict(list)
    for p in sorted(Path(directory).glob("motif*.motif")):
        m = re.match(r"motif(\d+)(\.similar\d+)?$", p.stem)
        if m:
            clusters[int(m.group(1))].append(p)

    results = {}
    for cid, paths in sorted(clusters.items()):
        if len(paths) < 2:
            continue
        pwms = [load_homer_motif_file(str(p))[1] for p in paths]
        mat = pairwise_similarity_matrix(pwms)
        scores = mat[np.triu_indices(len(pwms), k=1)].tolist()
        results[cid] = {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "min": float(np.min(scores)),
            "n_motifs": len(pwms),
            "scores": scores,
        }
    return results


def plot_intra_inter_similarity(
    inter_sim_path: str,
    homer_results_dir: str,
    out_path: str,
):
    """
    Violin plot comparing within-cluster and all-motif PWM similarities.
    The background is the upper triangle of the full motif similarity matrix;
    it includes within-cluster pairs and is not restricted to between clusters.
    Within-cluster scores are computed fresh using the same current similarity
    helper, pooling all pairwise scores across every multi-member cluster.
    Only the plotted values are capped below at PLOT_MINIMUM; saved matrices
    and the raw distribution summaries retain the original correlations.

    Args:
        inter_sim_path : str
            Path to .npy file containing precomputed pairwise similarity matrix for all motifs.
        homer_results_dir : str
            Path to HOMER homerResults directory containing clustered motif files.
        out_path : str
            Output stem; SVG, PNG, and a JSON score summary are saved.
    """

    inter_mat = np.load(inter_sim_path)
    inter_scores = inter_mat[np.triu_indices(inter_mat.shape[0], k=1)]

    intra_results = intra_cluster_similarity(homer_results_dir)
    intra_scores = np.asarray([s for v in intra_results.values() for s in v["scores"]])
    capped_pair_counts = {
        "within_clusters": int((intra_scores < PLOT_MINIMUM).sum()),
        "all_motif_pairs": int((inter_scores < PLOT_MINIMUM).sum()),
    }

    df = pd.DataFrame(
        {
            "type": np.repeat(
                ["intra", "inter"], [len(intra_scores), len(inter_scores)]
            ),
            "similarity": np.maximum(
                np.concatenate([intra_scores, inter_scores]), PLOT_MINIMUM
            ),
        }
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.violinplot(
        data=df,
        x="similarity",
        y="type",
        order=["intra", "inter"],
        density_norm="width",
        cut=0,
        inner="quart",
        inner_kws={"linewidth": 1, "color": "black"},
        ax=ax,
    )
    xlabel = "PWM similarity (global Pearson r)"
    if any(capped_pair_counts.values()):
        xlabel += f"\nValues below {PLOT_MINIMUM:g} capped for display"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.set_xlim(PLOT_MINIMUM, 1)
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_yticks([0, 1], ["Within clusters", "All motif pairs"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = Path(out_path)
    for suffix in (".svg", ".png"):
        export_path = out_path.with_suffix(suffix)
        fig.savefig(export_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {export_path}", flush=True)
    plt.close(fig)

    summary = {
        "similarity_method": SIMILARITY_METHOD,
        "implementation_sha256": file_sha256(PROJECT_ROOT / "src/af3tfm/utils.py"),
        "plot_source_sha256": file_sha256(__file__),
        "matrix": str(inter_sim_path),
        "matrix_sha256": file_sha256(inter_sim_path),
        "motif_count": int(inter_mat.shape[0]),
        "non_singleton_cluster_count": len(intra_results),
        "background_definition": "All unordered pairs from hits.all.motif, including within-cluster pairs",
        "axis_limits": [PLOT_MINIMUM, 1],
        "display": {
            "lower_cap": PLOT_MINIMUM,
            "scope": "plotted values only; raw scores unchanged",
            "capped_pair_counts": capped_pair_counts,
        },
        "distributions": {},
        "cluster_inputs": [
            {"path": str(p.relative_to(PROJECT_ROOT)), "sha256": file_sha256(p)}
            for p in sorted(Path(homer_results_dir).glob("motif*.motif"))
            if re.fullmatch(r"motif\d+(\.similar\d+)?", p.stem)
        ],
    }
    for name, scores in (
        ("within_clusters", intra_scores),
        ("all_motif_pairs", inter_scores),
    ):
        if not np.isfinite(scores).all() or (np.abs(scores) > 1 + 1e-12).any():
            raise ValueError(f"Invalid global Pearson scores in {name}")
        summary["distributions"][name] = {
            "pair_count": int(scores.size),
            "median": float(np.median(scores)),
            "minimum": float(scores.min()),
            "maximum": float(scores.max()),
            "negative_pair_count": int((scores < 0).sum()),
        }
    summary_path = out_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["distributions"], indent=2), flush=True)


# %%
if __name__ == "__main__":
    # Recompute both groups with the current helper. Keep the historical
    # data/tools/homer/inter_sim.npy cache separate from these versioned scores.
    path = f"{DATA_DIR}/tools/homer/hits.all.motif"
    names, pwms = load_homer_multi_motif_file(path)
    print(
        f"Loaded {len(pwms)} motifs (avg length {sum(len(p) for p in pwms)/len(pwms):.1f})"
    )
    mat = pairwise_similarity_matrix(pwms)
    sim_path = FIG_DIR / "intra_inter_sim_global_pearson.npy"
    np.save(sim_path, mat)
    metadata = {
        "similarity_method": SIMILARITY_METHOD,
        "implementation_sha256": file_sha256(PROJECT_ROOT / "src/af3tfm/utils.py"),
        "motif_input": str(Path(path).relative_to(PROJECT_ROOT)),
        "motif_input_sha256": file_sha256(path),
        "motif_ids_in_matrix_order": names,
        "matrix_sha256": file_sha256(sim_path),
    }
    sim_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    # Plot intra-cluster vs inter-cluster similarity distributions
    if sim_path.exists():
        plot_intra_inter_similarity(
            str(sim_path),
            f"{DATA_DIR}/tools/homer/jaspar_clustered/homerResults",
            f"{FIG_DIR}/intra_inter_sim.svg",
        )

# %%
