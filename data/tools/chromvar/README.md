# ChromVAR

### Installation
Install chromVAR from the Greenleaf lab github - https://github.com/GreenleafLab/chromVAR

### Goal
We want to test whether our predicted ZebraMotif PWMs behave like real binding motifs by asking if a motif's chromatin accessibility track the RNA expression of the TF it belongs to using the Zebrahub-Multiome dataset. We run the same test on CisBP's zebrafish motifs to use as a baseline for ZebraMotif's performance.

### Pipeline
`collect_seacells.R` collates Zebrahub-Multiome's SEACells metacell partitions (7 datasets, ~75 cells each) into one per-cell assignment table keyed to the integrated object's barcodes. Produces `results/seacells_assignment.csv`, the metacell partition used by stage 02.

`01_scan_peaks.R` scans peak x motif using motifmatchr's `matchMotifs(pwm_list, peaks, genome, p.cutoff = 5e-5, out = "matches")`, calling a hit at each motif's analytic per-motif p-value threshold. Uses the multiome peaks and both the ZebraMotif (primary binding modes only) and CisBP databases. Produces `results/scan_cache.rds`.

`02_deviations.R` aggregates the ATAC signal onto the SEACells metacells (~1200 metacells after a >=20-cell floor) to build denser profiles, then runs chromVAR against the stage-01 cache. Produces an accessibility deviation score per motif, per metacell.

`03_correlate_stats.R` correlates each motif's accessibility deviation with its own TF's expression, using a null of that motif correlated against all other TFs. Every testable TF (whitelist + IC floor) is evaluated and its own-TF correlation is ranked against the full TF pool (`own_pctile`). Produces `correlations.tsv` / `percentile_stats.tsv`.

### Analysis
`rank_coupled_tfs.R` ranks every tested TF by its own-TF coupling and annotates each with the cell types driving that coupling (per-metacell decomposition of the Spearman rho) and its CellOracle GRN out-edge count, flagging which coupled TFs are drawable as subGRNs. `rank_coupled_tfs_timepoint.R` breaks this down further by timepoint, a variable at which SEACells are computed for each.

The [Python figure notebook](../../../notebooks/04_database-and-applications/04_chromvar-own-tf-coupling.py) plots TF expression against own-motif deviation and illustrates the own-TF percentile with a TF-swap correlation distribution. It reads the saved stage-02/03 results, defaults to otx2b's primary ZebraMotif motif, and saves SVG/PNG figures beside the notebook.

The [Python family notebook](../../../notebooks/04_database-and-applications/04_chromvar-family.py) draws the family distributions directly from `correlations.tsv`, with median crossbars, tested-TF counts, and highlighted ZebraMotif TFs. It ranks classes using the canonical family crosswalk and summed coverage from the current PWM export, JASPAR vertebrates, and CisBP zebrafish. It saves SVG/PNG figures and plotted-data, summary, and coverage TSVs beside the notebook, using shared plotting defaults and the narrower panel width. No R or pre-rendered SVG is needed to generate this plot.

`family_stratified_stats.R` and the Python family notebook compare CisBP vs ZebraMotif by DBD family (the notebook writes `percentile_family.svg` and `.png`, Fig 4D). The pooled comparison is confounded by composition (CisBP is small and curated, ~20% zf-C2H2; ZebraMotif is comprehensive, ~69% zf-C2H2), so these restrict to families present in both resources to compare motif quality rather than family composition.

`chromvar.slurm` runs the full pipeline (01 -> 02 -> 03 -> ranking -> family stats) end to end; `collect_seacells.R` is run once up front and its output reused.

### Workflow configuration
Aggregation uses the precomputed SEACells partition (~75 cells per metacell, ~1200 metacells after a >=20-cell floor). SEACells give high cell-type purity and also resolve time-point specificity across 10-24 hours post-fertilization. Shared config and paths live in `utils.R`.
