# HOMER

### Installation
You can install HOMER by following their guide http://homer.ucsd.edu/homer/introduction/install.html.

### Motif Database
We use JASPAR's CORE vertebrates collection (2024) as a database to search our DNA sequences from scATAC-seq. To download this database, run `python data/tools/homer/get_jaspar.py`.

### Peak Sequences
We use Zebrahub Multiome's (2025) scATAC-seq dataset, which contains over 640K peaks, to discover motifs from and search JASPAR with. We also separate these sequences into their leiden clusters as annotated in their project. To get these sequences per their cluster, run `python data/tools/homer/get_peaks.py`.

### Running HOMER
We want to discover motifs in each collection of peaks sequences. For this, run `sbatch homer_array.sbatch` to submit an array job. You can also run `sbatch homer_single.sbatch`. Each cluster will have it's own HOMER output, which you can run `python get_hits.py` (with HOMER on your `PATH`, e.g. `conda activate homer`) to condense and cluster these results into a single set of motifs that we use to predict DNA sequence binding specificity for our transcription factors.

We use a stringent threshold (p < 1e-12) for HOMER motifs matched to JASPAR so that we confidently assign known PWMs to TF binding preferences in our scATAC-seq data. For de novo motifs, we kept all HOMER motifs without significance filtering in order to maximize sequence diversity across our predictions, from high-confidence binding motifs to low-information sequences.

### Analysis
Our HOMER run results in 2566 motifs from all 36 clusters, mostly from the JASPAR database. You can visualize the difference between JASPAR and de novo hits with `notebooks/02_screen-design/02_homer-clusters.py`. We call `compareMotifs.pl` on all of the 2566 motifs to reduce redundancy, leaving us with 312 in the end. You can visualize the motif similarity before and after clustering with `notebooks/02_screen-design/02_homer-sim.py`. 