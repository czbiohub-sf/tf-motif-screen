# MMseqs

### Installation
You can download install MMseqs2 via the instructions on their github: https://github.com/soedinglab/MMseqs2

### Usage
We use MMseqs to cluster the all of the protein sequences in each family (labeled by AnimalTFDB4) by their domains, as predicted by InterProScan, which captures the similarity of their domain architecture. You can run `python cluster_tfdb4.py` to cluster the sequences in this way.

### C2H2 Zinc Fingers
Clustering zf-C2H2 TFs with mmseqs gives us clusters with very low intra-cluster similarity, likely due to mmseqs clustering by similarity that all zinc fingers share in the invariant residues of their fingers, while the variant (and DNA recognizing) residues cause high dissimilarity between cluster members. To remedy this, we cluster C2H2 zinc fingers by their 6-residue recognition codes. Zinc fingeres with the same number of fingers and same fingerprints (set of recognition codes, one for each finger) are then put into the same cluster, with some variability allowed in the less constrained positions of the recognition codes. You can run `python cluster_zfc2h2.py` to see the results of this clustering and obtain the corresponding representatives for our dataset.

### ZBTB Domain TFs
ZBTB proteins carry C2H2 zinc fingers for DNA binding in addition to their BTB dimerization domain. Like zf-C2H2 TFs, MMseqs clusters them by invariant scaffold residues rather than the specificity-determining recognition code positions, producing very large heterogeneous clusters. We apply the same recognition code clustering as for zf-C2H2: proteins are grouped by ZF count, fingerprint distances are computed per-finger, and hierarchical average-linkage clustering is applied with a threshold of 1 mismatch per ZF. You can run `python cluster_zbtb.py` to cluster ZBTB TFs and update the representative set.

### Homeodomains
Like zf-C2H2 TFs, homeodomains are known to bind to DNA with a similar recognition code system. We align these TFs to the PFAM homeodomain HMM to determine their match state positions, and then group homeodomains with the exact matching residues at the positions that should be conserved and place them into a cluster. Sequences are also clustered by other domains they may contain - POU, PAX, etc. as these may affect DNA sequence binding specificity. You can run `python cluster_hd.py` to see the results of this cluster and obtain the corresponding representatives for our dataset.

### Cluster analysis
You can run `python tfdb_clusters.py` to compute the inter/intra cluster similarities for each family (written to `notebooks/reports/intra_inter_similarities.csv`). The plots of these similarities and of how many representative sequences each family has are in `notebooks/02_screen-design/02_mmseqs-tfdb-clusters.py`.