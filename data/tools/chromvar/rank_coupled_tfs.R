#!/usr/bin/env Rscript
#
# module load r/4.4 && Rscript rank_coupled_tfs.R
#
# Ranks every tested TF by its own-TF coupling (own_pctile, the TF-swap percentile
# from stage 03) and annotates each with:
#   1) coupling: own_pctile, rho, top5, n_matched, total_ic (from stage 03).
#      One row per (tf, source), so a TF appears once per DB that has a motif for it.
#   2) metacell decomposition: the TF's Spearman rho is the sum over metacells of
#      rank(motif z-deviation) * rank(TF expression); this decomposes that sum by
#      the metacell's dominant cell type, so `frac_of_rho` is the share of the
#      coupling coming from each cell type
#   3) GRN connectivity: grn_regulator_edges, the number of edges where the TF
#      is the source, summed over every cell-type x timepoint hybrid CellOracle
#      GRN. Flags which coupled TFs are drawable as subGRNs
#
# Reads stage 01/02/03 outputs under cfg$base_outdir (see utils.R) and,
# if present, the hybrid GRN under ../celloracle/outputs/global_annotation/grn_csv/hybrid.
# Writes to cfg$outdir (results/seacells/):
#   tf_coupling_ranked.tsv         one row per (tf, source), ranked; top-3 cell types inline
#   tf_coupling_celltype_decomp.tsv  long: (tf, source) x cell type, full decomposition

suppressPackageStartupMessages({ library(SummarizedExperiment) })
source("utils.R")

# Inputs
cor_tab <- read.delim(file.path(cfg$outdir, "correlations.tsv"), stringsAsFactors = FALSE)
dev     <- readRDS(file.path(cfg$outdir, "chromvar_deviations.rds"))
zmat    <- assays(dev)$z
mcs     <- readRDS(file.path(cfg$outdir, "metacells.rds"))
rna     <- mcs$rna_mc

# metacell -> dominant cell type (majority vote from the per-cell SEACells table)
asg <- read.csv(file.path(cfg$base_outdir, "seacells_assignment.csv"), stringsAsFactors = FALSE)
dom <- tapply(asg$celltype, asg$metacell, function(x) names(sort(table(x), decreasing = TRUE))[1])

cols <- colnames(zmat)
ct   <- dom[cols]
if (any(is.na(ct)))
  msg("WARNING: ", sum(is.na(ct)), " metacells without a cell-type label (dropped from decomposition)")
keep <- !is.na(ct); cols <- cols[keep]; ct <- ct[keep]
msg(length(cols), " metacells with cell-type labels; ", length(unique(ct)), " cell types")

# Family
fam_csv  <- read.csv(cfg$tf_family_csv, stringsAsFactors = FALSE)
fam_map  <- setNames(fam_csv$Family, tolower(fam_csv$TF_Name))

# Coupling + decomposition - keep tested motifs with a defined coupling and present z/rna rows
cor_tab <- cor_tab[!is.na(cor_tab$own_pctile), ]
cor_tab <- cor_tab[cor_tab$motif_id %in% rownames(zmat) & cor_tab$tf %in% rownames(rna), ]
msg(nrow(cor_tab), " tested motifs with defined coupling")

decomp_rows <- vector("list", nrow(cor_tab))
top_ct <- matrix(NA_character_, nrow(cor_tab), 3)
top_fr <- matrix(NA_real_,      nrow(cor_tab), 3)

for (i in seq_len(nrow(cor_tab))) {
  mid <- cor_tab$motif_id[i]; tf <- cor_tab$tf[i]; src <- cor_tab$source[i]
  a <- rank(zmat[mid, cols]); a <- a - mean(a); a <- a / sqrt(sum(a^2))
  b <- rank(rna[tf, cols]);   b <- b - mean(b); b <- b / sqrt(sum(b^2))
  contrib <- a * b                      # sums to Spearman rho over these metacells
  rho <- sum(contrib)
  agg <- tapply(contrib, ct, sum); n <- table(ct)[names(agg)]
  d <- data.frame(tf = tf, source = src, celltype = names(agg),
                  n_metacells = as.integer(n),
                  sum_contrib = as.numeric(agg),
                  frac_of_rho = as.numeric(agg) / rho,
                  mean_per_mc = as.numeric(agg) / as.numeric(n),
                  stringsAsFactors = FALSE)
  d <- d[order(-d$sum_contrib), ]
  decomp_rows[[i]] <- d
  k <- min(3, nrow(d))
  top_ct[i, seq_len(k)] <- d$celltype[seq_len(k)]
  top_fr[i, seq_len(k)] <- round(d$frac_of_rho[seq_len(k)], 3)
}

# GRN regulator connectivity - count edges where the TF is the source (regulator),
# summed across every cell-type x timepoint GRN in the hybrid CellOracle output.
# Flags which coupled TFs are drawable as subGRNs. NA everywhere if the GRN tree is absent.
grn_dir   <- "../celloracle/outputs/global_annotation/grn_csv/hybrid"
grn_cache <- file.path(cfg$outdir, "grn_regulator_edge_counts.tsv")
grn_files <- list.files(grn_dir, pattern = "^raw_GRN_for_.*\\.csv$",
                        recursive = TRUE, full.names = TRUE)
grn_counts <- NULL
if (file.exists(grn_cache)) {
  msg("reading cached GRN edge counts: ", grn_cache)
  gtb <- read.delim(grn_cache, stringsAsFactors = FALSE)
  grn_counts <- setNames(as.integer(gtb$edges), tolower(gtb$tf))
} else if (length(grn_files)) {
  msg("counting GRN regulator edges across ", length(grn_files), " hybrid GRN files (awk; cached after)")
  awk <- sprintf("awk -F, 'FNR>1{c[$2]++} END{for(k in c) print k\"\\t\"c[k]}' %s",
                 paste(shQuote(grn_files), collapse = " "))
  out <- system(awk, intern = TRUE)
  sp  <- strsplit(out, "\t", fixed = TRUE)
  grn_counts <- setNames(as.integer(vapply(sp, `[`, character(1), 2)),
                         tolower(vapply(sp, `[`, character(1), 1)))
  write.table(data.frame(tf = names(grn_counts), edges = as.integer(grn_counts)),
              grn_cache, sep = "\t", quote = FALSE, row.names = FALSE)
  msg("  ", length(grn_counts), " distinct regulator TFs; cached to ", grn_cache)
} else {
  msg("WARNING: no hybrid GRN files under ", grn_dir, " -- grn_regulator_edges = NA")
}
if (!is.null(grn_counts)) {
  grn_edges <- as.integer(grn_counts[tolower(cor_tab$tf)]); grn_edges[is.na(grn_edges)] <- 0L
} else {
  grn_edges <- rep(NA_integer_, nrow(cor_tab))
}

# Ranked table
ranked <- data.frame(
  tf         = cor_tab$tf,
  family     = fam_map[tolower(cor_tab$tf)],
  source     = cor_tab$source,
  own_pctile = round(cor_tab$own_pctile, 4),
  rho        = round(cor_tab$rho, 4),
  top5       = cor_tab$top5,
  n_matched  = cor_tab$n_matched,
  total_ic   = round(cor_tab$total_ic, 2),
  grn_regulator_edges = grn_edges,
  top1_celltype = top_ct[, 1], top1_frac = top_fr[, 1],
  top2_celltype = top_ct[, 2], top2_frac = top_fr[, 2],
  top3_celltype = top_ct[, 3], top3_frac = top_fr[, 3],
  stringsAsFactors = FALSE
)
ranked <- ranked[order(-ranked$own_pctile), ]
ranked <- cbind(rank = seq_len(nrow(ranked)), ranked)

out1 <- file.path(cfg$outdir, "tf_coupling_ranked.tsv")
out2 <- file.path(cfg$outdir, "tf_coupling_celltype_decomp.tsv")
write.table(ranked, out1, sep = "\t", quote = FALSE, row.names = FALSE)
decomp <- do.call(rbind, decomp_rows)
decomp$sum_contrib <- round(decomp$sum_contrib, 5)
decomp$frac_of_rho <- round(decomp$frac_of_rho, 4)
decomp$mean_per_mc <- round(decomp$mean_per_mc, 6)
write.table(decomp, out2, sep = "\t", quote = FALSE, row.names = FALSE)

cat("\n=== tested motifs by source ===\n")
print(table(source = ranked$source))
cat("\n=== top 15 by coupling ===\n")
print(head(ranked[, c("rank","tf","family","source","own_pctile","rho","grn_regulator_edges","top1_celltype","top1_frac")], 15), row.names = FALSE)
