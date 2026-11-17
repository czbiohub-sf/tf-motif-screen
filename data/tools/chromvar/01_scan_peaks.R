#!/usr/bin/env Rscript
#
# Run - module load r/4.4 && Rscript 01_scan_peaks.R
#
# Scans every peak against every PWM with motifmatchr's analytic per-motif
# p-value threshold (matchMotifs default p.cutoff = 5e-5).
#
# Writes to base_outdir (see utils.R):
#   scan_cache.rds          list(motif_ix_use, n_matched) -- consumed by stage 02/03
#   motif_ic.tsv            per-motif total/mean IC (reported; the IC floor lives in stage 03)
#   matched_peak_counts.tsv per-motif genome-wide matched-peak counts
#
# Delete scan_cache.rds to force a rescan (e.g. after a DB change). If the cache
# already exists this script will not run.

suppressPackageStartupMessages({
  library(Matrix)
  library(GenomicRanges)
  library(universalmotif)
  library(TFBSTools)
  library(motifmatchr)
})

source("utils.R")
set.seed(cfg$seed)
dir.create(cfg$base_outdir, recursive = TRUE, showWarnings = FALSE)

if (file.exists(cfg$scan_cache)) {
  msg("Scan cache already present, delete to re-run: ", cfg$scan_cache)
  quit(save = "no")
}

genome <- load_genome()

# Peaks (only the peak set is needed here; atac/rna/emb ignored)
dat   <- load_multiome(genome)
peaks <- dat$peaks
rm(dat); gc()

# Motifs
zm <- read_motifs(cfg$zebramotif_path)
msg("  ", length(zm), " ZebraMotif PWMs read")

# Keep only the PRIMARY binding mode per ZebraMotif id (lowest numeric suffix,
# e.g. ZM...​.1 over .2/.3/.4)
if (isTRUE(cfg$zm_primary_only)) {
  zm_ids  <- vapply(zm, function(x) x["name"], character(1))
  zm_base <- sub("\\.[0-9]+$", "", zm_ids)
  zm_mode <- suppressWarnings(as.integer(sub(".*\\.", "", zm_ids)))
  zm_mode[is.na(zm_mode)] <- 1L
  ord     <- order(zm_base, zm_mode)
  keep    <- ord[!duplicated(zm_base[ord])]
  zm      <- zm[sort(keep)]
  msg("  ", length(zm), " primary-mode ZebraMotif PWMs kept (cfg$zm_primary_only)")
}

cis   <- read_cisbp_dir(cfg$cisbp_pwm_dir, cfg$cisbp_tf_info)
cisbp <- cis$motifs

all_um  <- c(zm, cisbp)
names(all_um) <- vapply(all_um, function(x) x["name"], character(1))
all_um  <- all_um[!duplicated(names(all_um))]
pwm_list <- convert_motifs(all_um, "TFBSTools-PWMatrix")
pwm_list <- do.call(PWMatrixList, pwm_list)
msg("  ", length(pwm_list), " PWMs (", length(zm), " ZebraMotif, ",
    length(cisbp), " CisBP)")

# record IC per motif (reported in motif_ic.tsv; the IC floor is applied in stage 03)
ic_tbl <- data.frame(
  motif_id = names(all_um),
  total_ic = vapply(all_um, function(x) sum(x["icscore"]), numeric(1)),
  mean_ic  = vapply(all_um, function(x) mean(colSums(x["motif"] * log2((x["motif"] + 1e-9) / 0.25))), numeric(1)),
  stringsAsFactors = FALSE
)
write.table(ic_tbl, file.path(cfg$base_outdir, "motif_ic.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)

# motifmatchr's analytic per-motif p-value threshold.
msg("Scanning via motifmatchr matchMotifs (p.cutoff = ", cfg$scan_alpha, ")")
mm <- matchMotifs(pwm_list, peaks, genome = genome,
                  p.cutoff = cfg$scan_alpha, out = "matches")
motif_ix <- as(motifMatches(mm), "CsparseMatrix") * 1   # peaks x motifs, 0/1
colnames(motif_ix) <- names(pwm_list)
rm(mm); gc()
msg("  median matched peaks per motif: ", median(Matrix::colSums(motif_ix)))

# Retain every motif that matched at least one peak. Differing match counts
# across motifs are handled downstream by chromVAR's background-peak correction
n_matched <- Matrix::colSums(motif_ix)
motif_ix_use <- motif_ix[, n_matched > 0, drop = FALSE]
msg("  ", ncol(motif_ix_use), " motifs retained for deviation computation")

# scan_cache.rds keeps the historical {motif_ix_use, n_matched} format so existing
# caches remain valid. Its rows are in peaks order; stage 02 re-derives the SAME
# peaks via the shared loader and asserts the row count matches before use.
saveRDS(list(motif_ix_use = motif_ix_use, n_matched = n_matched), cfg$scan_cache)
write.table(data.frame(motif_id = names(n_matched), n_matched = as.integer(n_matched)),
            file.path(cfg$base_outdir, "matched_peak_counts.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
