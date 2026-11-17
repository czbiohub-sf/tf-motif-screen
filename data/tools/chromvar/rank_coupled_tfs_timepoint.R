#!/usr/bin/env Rscript
#
# module load r/4.4 && Rscript rank_coupled_tfs_timepoint.R
#
# Same metacell decomposition of each TF's Spearman rho
#(rank(motif z-deviation) * rank(TF expression), summed over metacells),
# but grouped by timepoint and by celltype x timepoint. Because SEACells
# are built per dataset, every metacell has exactly one dev_stage, so the
# timepoint grouping is exact (no majority vote, unlike the cell-type label).
#
# Reads the same stage 02/03 outputs as rank_coupled_tfs.R plus
# seacells_assignment.csv. Writes to cfg$outdir (results/seacells/):
#   tf_coupling_timepoint_decomp.tsv           long: (tf, source) x timepoint
#   tf_coupling_celltype_timepoint_decomp.tsv  long: (tf, source) x celltype x timepoint
#   tf_coupling_timepoint_ranked.tsv           one row per (tf, source); top-3 timepoints
#                                              and top-3 (celltype x timepoint) cells inline

suppressPackageStartupMessages({ library(SummarizedExperiment) })
source("utils.R")

# Inputs (identical to rank_coupled_tfs.R)
cor_tab <- read.delim(file.path(cfg$outdir, "correlations.tsv"), stringsAsFactors = FALSE)
dev     <- readRDS(file.path(cfg$outdir, "chromvar_deviations.rds"))
zmat    <- assays(dev)$z
mcs     <- readRDS(file.path(cfg$outdir, "metacells.rds"))
rna     <- mcs$rna_mc

# metacell -> dominant cell type (majority vote) and -> timepoint (exact, one per mc)
asg <- read.csv(file.path(cfg$base_outdir, "seacells_assignment.csv"), stringsAsFactors = FALSE)
dom <- tapply(asg$celltype,  asg$metacell, function(x) names(sort(table(x), decreasing = TRUE))[1])
tp  <- tapply(asg$dev_stage, asg$metacell, function(x) names(sort(table(x), decreasing = TRUE))[1])

cols <- colnames(zmat)
ct   <- dom[cols]
st   <- tp[cols]
keep <- !is.na(ct) & !is.na(st); cols <- cols[keep]; ct <- ct[keep]; st <- st[keep]
msg(length(cols), " metacells with labels; ", length(unique(ct)), " cell types; ",
    length(unique(st)), " timepoints")

# Order timepoints developmentally when they follow the "Nsomites" convention
tp_levels <- unique(unname(st))
tp_num    <- suppressWarnings(as.numeric(sub("somites?$", "", tp_levels)))
tp_order  <- if (all(!is.na(tp_num))) tp_levels[order(tp_num)] else sort(tp_levels)

# Family
fam_csv  <- read.csv(cfg$tf_family_csv, stringsAsFactors = FALSE)
fam_map  <- setNames(fam_csv$Family, tolower(fam_csv$TF_Name))

# Coupling + decomposition - keep tested motifs with a defined coupling and present z/rna rows
cor_tab <- cor_tab[!is.na(cor_tab$own_pctile), ]
cor_tab <- cor_tab[cor_tab$motif_id %in% rownames(zmat) & cor_tab$tf %in% rownames(rna), ]
msg(nrow(cor_tab), " tested motifs with defined coupling")

# celltype x timepoint grouping key
ctp <- paste(ct, st, sep = "@")

tp_rows  <- vector("list", nrow(cor_tab))   # per-timepoint decomposition
ctp_rows <- vector("list", nrow(cor_tab))   # per-(celltype x timepoint) decomposition
top_tp   <- matrix(NA_character_, nrow(cor_tab), 3); top_tp_fr  <- matrix(NA_real_, nrow(cor_tab), 3)
top_ctp  <- matrix(NA_character_, nrow(cor_tab), 3); top_ctp_fr <- matrix(NA_real_, nrow(cor_tab), 3)

for (i in seq_len(nrow(cor_tab))) {
  mid <- cor_tab$motif_id[i]; tf <- cor_tab$tf[i]; src <- cor_tab$source[i]
  a <- rank(zmat[mid, cols]); a <- a - mean(a); a <- a / sqrt(sum(a^2))
  b <- rank(rna[tf, cols]);   b <- b - mean(b); b <- b / sqrt(sum(b^2))
  contrib <- a * b                      # sums to Spearman rho over these metacells
  rho <- sum(contrib)

  # by timepoint
  agg <- tapply(contrib, st, sum); n <- table(st)[names(agg)]
  d <- data.frame(tf = tf, source = src, timepoint = names(agg),
                  n_metacells = as.integer(n),
                  sum_contrib = as.numeric(agg),
                  frac_of_rho = as.numeric(agg) / rho,
                  mean_per_mc = as.numeric(agg) / as.numeric(n),
                  stringsAsFactors = FALSE)
  d <- d[order(-d$sum_contrib), ]
  tp_rows[[i]] <- d
  k <- min(3, nrow(d))
  top_tp[i, seq_len(k)]    <- d$timepoint[seq_len(k)]
  top_tp_fr[i, seq_len(k)] <- round(d$frac_of_rho[seq_len(k)], 3)

  # by celltype x timepoint
  agg2 <- tapply(contrib, ctp, sum); n2 <- table(ctp)[names(agg2)]
  parts <- strsplit(names(agg2), "@", fixed = TRUE)
  d2 <- data.frame(tf = tf, source = src,
                   celltype  = vapply(parts, `[`, character(1), 1),
                   timepoint = vapply(parts, `[`, character(1), 2),
                   n_metacells = as.integer(n2),
                   sum_contrib = as.numeric(agg2),
                   frac_of_rho = as.numeric(agg2) / rho,
                   mean_per_mc = as.numeric(agg2) / as.numeric(n2),
                   stringsAsFactors = FALSE)
  d2 <- d2[order(-d2$sum_contrib), ]
  ctp_rows[[i]] <- d2
  k2 <- min(3, nrow(d2))
  top_ctp[i, seq_len(k2)]    <- paste0(d2$celltype[seq_len(k2)], "@", d2$timepoint[seq_len(k2)])
  top_ctp_fr[i, seq_len(k2)] <- round(d2$frac_of_rho[seq_len(k2)], 3)
}

# Ranked table (mirrors rank_coupled_tfs.R, timepoint axis inline)
ranked <- data.frame(
  tf         = cor_tab$tf,
  family     = fam_map[tolower(cor_tab$tf)],
  source     = cor_tab$source,
  own_pctile = round(cor_tab$own_pctile, 4),
  rho        = round(cor_tab$rho, 4),
  n_matched  = cor_tab$n_matched,
  total_ic   = round(cor_tab$total_ic, 2),
  top1_timepoint = top_tp[, 1], top1_tp_frac = top_tp_fr[, 1],
  top2_timepoint = top_tp[, 2], top2_tp_frac = top_tp_fr[, 2],
  top3_timepoint = top_tp[, 3], top3_tp_frac = top_tp_fr[, 3],
  top1_celltype_tp = top_ctp[, 1], top1_ctp_frac = top_ctp_fr[, 1],
  top2_celltype_tp = top_ctp[, 2], top2_ctp_frac = top_ctp_fr[, 2],
  top3_celltype_tp = top_ctp[, 3], top3_ctp_frac = top_ctp_fr[, 3],
  stringsAsFactors = FALSE
)
ranked <- ranked[order(-ranked$own_pctile), ]
ranked <- cbind(rank = seq_len(nrow(ranked)), ranked)

out_rank <- file.path(cfg$outdir, "tf_coupling_timepoint_ranked.tsv")
out_tp   <- file.path(cfg$outdir, "tf_coupling_timepoint_decomp.tsv")
out_ctp  <- file.path(cfg$outdir, "tf_coupling_celltype_timepoint_decomp.tsv")
write.table(ranked, out_rank, sep = "\t", quote = FALSE, row.names = FALSE)

tp_decomp <- do.call(rbind, tp_rows)
tp_decomp$sum_contrib <- round(tp_decomp$sum_contrib, 5)
tp_decomp$frac_of_rho <- round(tp_decomp$frac_of_rho, 4)
tp_decomp$mean_per_mc <- round(tp_decomp$mean_per_mc, 6)
write.table(tp_decomp, out_tp, sep = "\t", quote = FALSE, row.names = FALSE)

ctp_decomp <- do.call(rbind, ctp_rows)
ctp_decomp$sum_contrib <- round(ctp_decomp$sum_contrib, 5)
ctp_decomp$frac_of_rho <- round(ctp_decomp$frac_of_rho, 4)
ctp_decomp$mean_per_mc <- round(ctp_decomp$mean_per_mc, 6)
write.table(ctp_decomp, out_ctp, sep = "\t", quote = FALSE, row.names = FALSE)

cat("\n=== timepoints (developmental order) ===\n"); print(tp_order)
cat("\n=== top 15 by coupling: leading timepoint and celltype x timepoint ===\n")
print(head(ranked[, c("rank","tf","family","source","own_pctile","rho",
                      "top1_timepoint","top1_tp_frac","top1_celltype_tp","top1_ctp_frac")], 15),
      row.names = FALSE)
