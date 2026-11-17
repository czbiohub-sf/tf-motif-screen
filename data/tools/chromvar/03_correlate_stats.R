#!/usr/bin/env Rscript
#
# module load r/4.4 && Rscript 03_correlate_stats.R
#
# From metacells + chromVAR deviations, restrict to testable TFs (whitelist
#+ IC floor), correlate each motif's accessibility deviation with its own TF's
# expression, score the TF-swap own-TF percentile null.
#
# Reads (from outdir unless noted):
#   metacells.rds            (stage 02)
#   chromvar_deviations.rds  (stage 02)
#   motif_ic.tsv             (stage 01, in base_outdir)
#   matched_peak_counts.tsv  (stage 01, in base_outdir)

suppressPackageStartupMessages({
  library(Matrix)
  library(SummarizedExperiment)
  library(chromVAR)
  library(universalmotif)      # for build_m2tf_from_cfg()
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

source("utils.R")

# Load saved metacells + deviations from stage 02
mcs    <- readRDS(file.path(cfg$outdir, "metacells.rds"))
rna_mc <- mcs$rna_mc; mc_celltype <- mcs$mc_celltype
dev    <- readRDS(file.path(cfg$outdir, "chromvar_deviations.rds"))
zmat   <- assays(dev)$z            # motifs x metacells
zmat   <- zmat[complete.cases(zmat), , drop = FALSE]
ic_tbl <- read.delim(file.path(cfg$base_outdir, "motif_ic.tsv"))
nmt    <- read.delim(file.path(cfg$base_outdir, "matched_peak_counts.tsv"))
n_matched <- setNames(nmt$n_matched, nmt$motif_id)

m2tf <- build_m2tf_from_cfg()
msg("  motif->TF map: ", sum(m2tf$source == "zebramotif"), " ZebraMotif, ",
    sum(m2tf$source == "cisbp"), " CisBP entries")

# Correlations and nulls
msg("Correlating deviations with TF expression")
# One row per (motif_id, TF) pair: a CisBP PWM spans a paralog group, so a motif
# is testable against several TFs. Filter the pairs to the tested population.
pairs <- m2tf[m2tf$motif_id %in% rownames(zmat), ]
pairs <- pairs[pairs$tf_gene_symbol %in% rownames(rna_mc), ]

# a TF must actually be expressed somewhere or the correlation is meaningless
expressed <- rownames(rna_mc)[rowMeans(rna_mc > 0) >= 0.05]
pairs <- pairs[pairs$tf_gene_symbol %in% expressed, ]
msg("  ", nrow(pairs), " motif-TF pairs expressed")

# Optional curated whitelist first (drops non-sequence-specific genes such as
# dnmt1/mbd3b/dnajc1 that carry a motif but are not TFs).
if (!is.null(cfg$tf_list_file)) {
  tf_ok  <- trimws(readLines(cfg$tf_list_file))
  pairs <- pairs[pairs$tf_gene_symbol %in% tf_ok, ]
  msg("  ", nrow(pairs), " pairs after TF whitelist")
}
# Information-content floor: drop near-zero-bit PWMs that match tens of thousands
# of peaks and self-rank spuriously.
if (!is.null(cfg$min_total_ic)) {
  ic_ok  <- ic_tbl$motif_id[ic_tbl$total_ic >= cfg$min_total_ic]
  pairs <- pairs[pairs$motif_id %in% ic_ok, ]
  msg("  ", nrow(pairs), " pairs after total_ic >= ", cfg$min_total_ic, " bit floor")
}
res <- data.frame(
  motif_id = pairs$motif_id,
  tf       = pairs$tf_gene_symbol,
  source   = pairs$source,
  n_matched = as.integer(n_matched[pairs$motif_id]),
  total_ic  = ic_tbl$total_ic[match(pairs$motif_id, ic_tbl$motif_id)],
  rho       = mapply(function(mid, tf) spearman(zmat[mid, ], rna_mc[tf, colnames(zmat)]),
                     pairs$motif_id, pairs$tf_gene_symbol, USE.NAMES = FALSE),
  stringsAsFactors = FALSE
)

# TF-swap specificity null (own-TF percentile; see utils.R)
# Each own-TF's correlation is ranked against all OTHER pool TFs (same-family
# paralogs excluded); the percentile reflects factor-among-factors specificity.
own_percentile <- make_own_percentile(rna_mc, expressed, colnames(zmat))
res$own_pctile <- vapply(seq_len(nrow(res)),
  function(k) own_percentile(zmat, res$motif_id[k], res$tf[k]), numeric(1))
res$top5 <- res$own_pctile >= 0.95            # own TF in top 5% of other pool TFs

# Use ZebraMotif primary mode, CisBP already carries one motif per TF. The
# full per-motif table is written separately as a supplementary artifact.
# Collapse WITHIN each resource (source, tf): a gene present in BOTH CisBP and
# ZebraMotif is a separate evaluation in each and must be kept in both -- a global
# dedup on tf alone would drop the CisBP row for any shared gene.
write.table(res, file.path(cfg$outdir, "correlations_bypair.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
mode_idx <- suppressWarnings(as.integer(sub(".*\\.", "", res$motif_id)))
mode_idx[is.na(mode_idx)] <- 99L
res <- res[order(res$source, res$tf, mode_idx), ]
res <- res[!duplicated(res[, c("source", "tf")]), ]
msg("  collapsed motif modes -> ", nrow(res), " unique per-resource TFs (primary mode each; ",
    sum(res$source == "zebramotif"), " ZebraMotif, ", sum(res$source == "cisbp"), " CisBP)")

write.table(res, file.path(cfg$outdir, "correlations.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)


# Statistics and figure

# Three resource configurations: CisBP only, ZebraMotif only, combined (pooled).
zm_only  <- res$source == "zebramotif"
cis_only <- res$source == "cisbp"

# TF-swap percentile of the own-TF correlation. Null median is 0.5;
# a working resource shifts it toward 1 and enriches own_pctile >= 0.95
# ("own TF in the top 5% of other pool TFs").
# n_tested counts only pairs with a defined percentile (own_pctile / top5 non-NA);
# an undefined percentile (e.g. constant deviation row) is not a tested pair. All
# three summaries share this denominator so frac_top5, the printed n_top5/n_tested
# label, and the binomial trial count agree.
rate_row <- function(mask) {
  t5 <- res$top5[mask]
  ok <- !is.na(t5)
  n  <- sum(ok); ntop <- sum(t5[ok])
  data.frame(
    n_tested      = n,
    median_pctile = round(median(res$own_pctile[mask], na.rm = TRUE), 4),
    n_top5        = ntop,
    frac_top5     = if (n > 0) round(ntop / n, 4) else NA_real_,
    # HEADLINE test: is the top-5% rate enriched over the 5% null? The signal is a
    # tail effect (only cell-type-specific activators self-rank high), so a tail
    # binomial is the right test -- a whole-distribution Wilcoxon on the percentiles
    # is diluted by the null-like bulk and understates it.
    top5_binom_p  = if (n > 0) signif(binom.test(ntop, n, p = 0.05,
                                     alternative = "greater")$p.value, 3) else NA_real_
  )
}
stat_tbl <- rbind(
  cbind(group = "CisBP only",                    rate_row(cis_only)),
  cbind(group = "ZebraMotif only",               rate_row(zm_only)),
  cbind(group = "CisBP + ZebraMotif (combined)", rate_row(cis_only | zm_only))
)
# ZebraMotif vs CisBP percentile distributions (the head-to-head): Mann-Whitney.
mwu <- wilcox.test(res$own_pctile[zm_only], res$own_pctile[cis_only])
stat_tbl$zm_vs_cisbp_p <- c(NA, round(mwu$p.value, 4), NA)

write.table(stat_tbl, file.path(cfg$outdir, "percentile_stats.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
print(stat_tbl)
msg("  ZebraMotif vs CisBP percentile distribution: Mann-Whitney p = ",
    signif(mwu$p.value, 3))

# Paired head-to-head on TFs testable in BOTH resources (same gene, each resource's
# own primary motif). Removes the family-composition confound of the unpaired MWU
# (ZebraMotif's ~700 extra zf-C2H2); diff = CisBP - ZebraMotif, so > 0 favors CisBP.
fam <- load_tf_family()
shared <- merge(res[zm_only,  c("tf", "own_pctile")],
                res[cis_only, c("tf", "own_pctile")],
                by = "tf", suffixes = c(".zm", ".cis"))
shared <- shared[!is.na(shared$own_pctile.zm) & !is.na(shared$own_pctile.cis), ]
shared$family <- unname(fam[tolower(shared$tf)])
shared$family[is.na(shared$family) | !nzchar(shared$family)] <- "unknown"
shared$diff <- shared$own_pctile.cis - shared$own_pctile.zm

paired_row <- function(d) {
  data.frame(
    n_shared      = nrow(d),
    median_cisbp  = round(median(d$own_pctile.cis), 4),
    median_zm     = round(median(d$own_pctile.zm), 4),
    median_diff   = round(median(d$diff), 4),
    frac_cis_wins = round(mean(d$diff > 0), 4),
    wilcox_p      = if (nrow(d) > 1) signif(wilcox.test(d$own_pctile.cis, d$own_pctile.zm,
                                           paired = TRUE)$p.value, 3) else NA_real_
  )
}
paired_tbl <- rbind(
  cbind(set = "all",          paired_row(shared)),
  cbind(set = "no_zf-C2H2",   paired_row(shared[shared$family != "zf-c2h2", ])),
  cbind(set = "zf-C2H2_only", paired_row(shared[shared$family == "zf-c2h2", ]))
)
write.table(paired_tbl, file.path(cfg$outdir, "paired_stats.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(shared, file.path(cfg$outdir, "paired_shared_tfs.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)
print(paired_tbl)
msg("  Paired CisBP - ZebraMotif on ", nrow(shared), " shared TFs: median diff = ",
    paired_tbl$median_diff[1], ", Wilcoxon signed-rank p = ", paired_tbl$wilcox_p[1])

# distribution of the TF-swap percentile per resource (null median 0.5)
plot_df <- bind_rows(
  data.frame(pct = res$own_pctile[cis_only],           grp = "CisBP only"),
  data.frame(pct = res$own_pctile[zm_only],            grp = "ZebraMotif only"),
  data.frame(pct = res$own_pctile[cis_only | zm_only], grp = "CisBP + ZebraMotif (combined)")
) %>% filter(!is.na(pct))

plot_df$grp <- factor(plot_df$grp, levels = c(
  "CisBP only", "ZebraMotif only", "CisBP + ZebraMotif (combined)"))

# CisBP amber, ZebraMotif blue, combined green
pal <- c("CisBP only"                    = "#E2A32D",
         "ZebraMotif only"               = "#2166ac",
         "CisBP + ZebraMotif (combined)" = "#5aae61")

p_main <- ggplot(plot_df, aes(pct, fill = grp, colour = grp)) +
  geom_density(alpha = 0.45, adjust = 1.1) +
  geom_vline(xintercept = 0.5, linetype = 3, colour = "grey40") +
  scale_fill_manual(values = pal) + scale_colour_manual(values = pal) +
  labs(x = "percentile of own-TF correlation among other pool TFs",
       y = "density", fill = NULL, colour = NULL) +
  theme_classic(base_size = 11) +
  theme(legend.position = c(0.02, 0.98), legend.justification = c(0, 1),
        legend.background = element_blank())

ggsave(file.path(cfg$outdir, "percentile_distribution.svg"), p_main, width = 5.6, height = 3.6)

# headline panel: fraction of TFs with own TF in the top 5% vs other TFs
rate_df <- stat_tbl
rate_df$group <- factor(rate_df$group, levels = rate_df$group)
p_rate <- ggplot(rate_df, aes(group, frac_top5, fill = group)) +
  geom_col(width = 0.65) +
  geom_hline(yintercept = 0.05, linetype = 2, colour = "grey40") +
  annotate("text", x = 0.6, y = 0.07, label = "null ~5%",
           hjust = 0, size = 3, colour = "grey40") +
  geom_text(aes(label = paste0(n_top5, "/", n_tested)), vjust = -0.4, size = 3) +
  scale_fill_manual(values = c("#E2A32D", "#2166ac", "#5aae61")) +
  labs(x = NULL, y = "fraction of TFs with own TF in\ntop 5% vs other TFs") +
  theme_classic(base_size = 11) +
  theme(legend.position = "none",
        axis.text.x = element_text(angle = 20, hjust = 1))
ggsave(file.path(cfg$outdir, "fraction_top5.svg"), p_rate, width = 4.6, height = 3.8)

# Strongest ZebraMotif-only TFs
top <- res[zm_only & !is.na(res$rho), ]
top <- head(top[order(-top$rho), ], 3)

inset_df <- bind_rows(lapply(seq_len(nrow(top)), function(k) data.frame(
  tf       = top$tf[k],
  dev      = zmat[top$motif_id[k], ],
  expr     = as.numeric(rna_mc[top$tf[k], colnames(zmat)]),
  celltype = mc_celltype
)))

p_inset <- ggplot(inset_df, aes(dev, expr, colour = celltype)) +
  geom_point(size = 0.9, alpha = 0.85) +
  facet_wrap(~ tf, scales = "free", nrow = 1) +
  labs(x = "chromVAR deviation z-score", y = "TF expression (log-norm)", colour = NULL) +
  theme_classic(base_size = 10) + theme(legend.position = "bottom")

ggsave(file.path(cfg$outdir, "zm_only_insets.svg"), p_inset, width = 8.5, height = 3.2)
