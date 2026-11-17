#!/usr/bin/env Rscript
#
# module load r/4.4 && Rscript 02_deviations.R
#
# Aggregates the multiome onto Yang-Joon Kim's precomputed SEACells partition
# (collated per-cell by collect_seacells.R) and runs chromVAR on those dense
# aggregated profiles using the cached peak x motif scan (stage 01). Aggregating
# before chromVAR means deviations are computed on ~1200 dense SEACell profiles
# rather than ~10^5 sparse cells -- cheaper, and it removes the dropout that would
# otherwise dominate the RNA side of every correlation.
#
# Reads:  scan_cache.rds  (from stage 01), the SEACells assignment CSV
# Writes into cfg$outdir (base_outdir/seacells, see utils.R):
#   metacells.rds            list(atac_mc, rna_mc, mc_celltype)
#   chromvar_deviations.rds  the chromVAR RangedSummarizedExperiment

suppressPackageStartupMessages({
  library(Matrix)
  library(GenomicRanges)
  library(SummarizedExperiment)
  library(chromVAR)
  library(matrixStats)
  library(BiocParallel)
})

# computeDeviations bplapply's over the ~1953 motifs, getBackgroundPeaks over
# peak-groups, and deviations on the ~1200 metacells, so we parallelize over
# motifs (the largest dimension) and let each motif's background sampling and
# deviation computation run serially
n_bp <- suppressWarnings(as.integer(Sys.getenv("R_BP_WORKERS")))
if (!is.na(n_bp) && n_bp > 1) {
  register(MulticoreParam(workers = n_bp))
  message("R_BP_WORKERS set -- MulticoreParam(", n_bp, ")")
} else {
  register(SerialParam())
}

source("utils.R")
set.seed(cfg$seed)
dir.create(cfg$outdir, recursive = TRUE, showWarnings = FALSE)

if (!file.exists(cfg$scan_cache))
  stop("scan cache not found: ", cfg$scan_cache, "\n  run 01_scan_peaks.R first")

genome <- load_genome()
dat    <- load_multiome(genome)

# SEACells partition: a precomputed per-cell assignment (Yang-Joon Kim's SEACells,
# collated by collect_seacells.R). `mc` is the metacell label per cell, in the
# column order of dat$atac; cells with no SEACell assignment are dropped.
msg("Using SEACells partition: ", cfg$external_metacells_csv)
asn <- read.csv(cfg$external_metacells_csv, stringsAsFactors = FALSE)
stopifnot(all(c("cell", "metacell") %in% colnames(asn)))
i  <- match(colnames(dat$atac), asn$cell)
mc <- asn$metacell[i]
n_unassigned <- sum(is.na(mc))
if (n_unassigned) {
  msg("  ", n_unassigned, " of ", length(mc),
      " cells have no SEACell; dropping them")
  keep <- !is.na(mc)
  dat$atac <- dat$atac[, keep]; dat$rna <- dat$rna[, keep]
  dat$celltype <- dat$celltype[keep]
  mc <- mc[keep]
}
msg("  ", length(unique(mc)), " SEACells over ", length(mc), " cells")

keep_mc <- names(which(table(mc) >= cfg$min_cells_per_mc))
keep_cell <- mc %in% keep_mc
mc <- mc[keep_cell]
msg("  kept ", length(keep_mc), " SEACells covering ", sum(keep_cell), " cells")

atac_mc <- aggregate_to_metacells(dat$atac[, keep_cell], mc, fun = "sum")
rna_mc  <- aggregate_to_metacells(dat$rna[,  keep_cell], mc, fun = "mean")

# dominant cell type per metacell, for the inset scatters only
mc_celltype <- tapply(dat$celltype[keep_cell], mc, function(x) names(sort(table(x), TRUE))[1])
mc_celltype <- mc_celltype[colnames(atac_mc)]

saveRDS(list(atac_mc = atac_mc, rna_mc = rna_mc, mc_celltype = mc_celltype),
        file.path(cfg$outdir, "metacells.rds"))

# chromVAR deviations on metacells
msg("Reusing cached peak x motif scan: ", cfg$scan_cache)
sc <- readRDS(cfg$scan_cache)
motif_ix_use <- sc$motif_ix_use
stopifnot(nrow(motif_ix_use) == length(dat$peaks))   # cache rows must be in this peak order
msg("  ", ncol(motif_ix_use), " motifs retained for deviation computation (cached)")

msg("Computing chromVAR deviations")
se <- SummarizedExperiment(
  assays  = list(counts = atac_mc),
  rowRanges = dat$peaks,
  colData = DataFrame(celltype = mc_celltype, row.names = colnames(atac_mc))
)
# dat$peaks carries no names, so tag each row with its integer position BEFORE
# filtering. filterPeaks() drops rows but preserves rownames, so the surviving
# names recover the original peak indices needed to realign motif_ix_use (whose
# rows are in dat$peaks order).
stopifnot(nrow(se) == nrow(motif_ix_use))
rownames(se) <- as.character(seq_len(nrow(se)))
se <- addGCBias(se, genome = genome)
se <- filterPeaks(se, non_overlapping = TRUE)

motif_ix_use <- motif_ix_use[as.integer(rownames(se)), , drop = FALSE]
# Re-seed so the stochastic background sampling (and hence the deviations) is
# reproducible regardless of whether the scan above was cached or freshly run
set.seed(cfg$seed)
bg  <- getBackgroundPeaks(se, niterations = 50)
dev <- computeDeviations(object = se, annotations = motif_ix_use, background_peaks = bg)
saveRDS(dev, file.path(cfg$outdir, "chromvar_deviations.rds"))
