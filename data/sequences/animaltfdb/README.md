# AnimalTFDB4

### Downloading data

Download the zebrafish or human **TF protein FASTA** from the
[AnimalTFDB 4.0 download page](https://guolab.wchscu.cn/AnimalTFDB4/#/)
in your browser, following the site's terms and any required browser interaction.
Keep the downloaded file separate from the output below. Then run from the
project root:

```bash
python data/sequences/animaltfdb/get_seqs.py \
  --input /path/to/Danio_rerio_TF_protein.fasta

python data/sequences/animaltfdb/get_seqs.py \
  --input /path/to/Homo_sapiens_TF_protein.fasta \
  --filename data/sequences/animaltfdb/human_tfs.fasta
```

The script reads the AnimalTFDB protein identifiers and family labels, fetches
current protein sequences from Ensembl, and writes `tfdb4_tfs.csv` (zebrafish)
or `human_tfs.csv` (human) under `data/sequences/animaltfdb/`. Both contain
`DBID`, `TF_Name`, `Family`, and `Sequence` columns, alongside corresponding
FASTA outputs. Ensembl access still requires an internet connection.

For a directly accessible file, `--url URL` remains available; omitting both
`--input` and `--url` tries the zebrafish URL once. The downloader does not solve
browser challenges, set challenge cookies, or retry blocked requests. If a
request fails or returns HTML, use the manual download workflow above.

### Data sources
The zebrafish and human TF protein sequences are from [Ensembl](https://www.ensembl.org) (Harrison et al., *Nucleic Acids Res.* 2024). AnimalTFDB 4.0 provides the TF lists, protein identifiers, and family labels (Shen et al., *Nucleic Acids Res.* 2023). `get_seqs.py` retrieves sequences from the Ensembl REST API using those protein identifiers.

The pipeline reads `tfdb4_tfs.csv` and `human_tfs.csv`. The script also writes the corresponding `.fasta` files as local, ignored outputs; they are not required by downstream steps.

### Sequence clustering
Our end goal is to determine the DNA sequence binding specificity of all the TFs from AnimalTFDB4, but we first want to perform a comprehensive search on a smaller subset of TFs. To accomplish this, we:
1. Run `scripts/get_domains.py --dataset tfdb4 --fold-dirs <af3_output_dir>` to find the predicted domains of each TF using interproscan.
2. Cluster the sequences in each family using mmseqs2 by the similarity of the DNA binding domain to get representatives by running `python data/tools/mmseqs/cluster_tfdb4.py`. zf-C2H2 TFs are clustered separately with `python data/tools/mmseqs/cluster_zfc2h2.py`, zbtb TFs with `python data/tools/mmseqs/cluster_zbtb.py`, and homeodomain TFs with `python data/tools/mmseqs/cluster_hd.py`.
3. You can compute the intra and inter cluster similarities with `python data/tools/mmseqs/tfdb_clusters.py`, then plot them alongside the number of representatives per family with `notebooks/02_screen-design/02_mmseqs-tfdb-clusters.py`.

The resulting representatives are saved to `data/sequences/animaltfdb/tfdb_rep_tfs.csv`.

### Clustering validation
To make sure our clustering works as intended, we run all of the members of a few protein clusters against all of the members of a few motif clusters (from homer) and check their results against the representatives of each. We extract these proteins by running `python get_val_seqs.py` to get a set of proteins for this task in `data/sequences/animaltfdb/tfdb_val_tfs.csv`.
