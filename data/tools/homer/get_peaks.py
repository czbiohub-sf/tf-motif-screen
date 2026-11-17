"""
Extract sequences from peak UMAP for each coarse leiden cluster for HOMER motif analysis.

Ben Iovino  8/14/25 Biohub
"""

from pathlib import Path

import pandas as pd
import scanpy as sc
from af3tfm.config import PATHS
from af3tfm.dataset import DATA_DIR


def main():
    """ """

    multiome_data_path = PATHS["zebrahub_objects_v2"]
    path = f"{multiome_data_path}/peaks_by_pb_leiden_0.4_merged_annotated_filtered.h5ad"
    df = sc.read_h5ad(path)

    metadata_df = pd.read_csv(
        f"{multiome_data_path}/all_peaks_leiden_clusters.csv", index_col=0
    )
    metadata_df = metadata_df["leiden_coarse_renumbered"]
    df = df.obs.join(metadata_df, how="left")

    # Write each leiden cluster to a separate df
    save_path = Path(f"{DATA_DIR}/tools/homer/jaspar_test/clusters")
    save_path.mkdir(parents=True, exist_ok=True)
    df["leiden_coarse_renumbered"] = df["leiden_coarse_renumbered"].astype("category")
    for cluster in df["leiden_coarse_renumbered"].cat.categories:
        cluster_df = df.loc[
            df["leiden_coarse_renumbered"] == cluster, ["chrom", "start", "end"]
        ]
        cluster_df["peak_name"] = [f"peak_{i}" for i in range(len(cluster_df))]
        cluster_df["chrom"] = ["chr" + str(i) for i in cluster_df["chrom"]]
        cluster_df.to_csv(save_path / f"{cluster}.csv", index=False, sep="\t")


if __name__ == "__main__":
    main()
