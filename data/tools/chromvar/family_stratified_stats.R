#!/usr/bin/env Rscript
#
# module load r/4.4 && Rscript family_stratified_stats.R
#
# Reads the collapsed per-TF coupling table (correlations.tsv, from stage 03),
# joins the animalTFDB4 family (utils.R::load_tf_family), and reports
# the own-TF percentile summary per (resource, family).
#
# Reads:  <outdir>/correlations.tsv, tf_family_csv (via cfg)
# Writes: <outdir>/percentile_stats_by_family.tsv
#         <outdir>/percentile_stats_zfC2H2_excluded.tsv

suppressPackageStartupMessages({ library(dplyr); library(ggplot2); library(tidyr) })
source("utils.R")

res <- read.delim(file.path(cfg$outdir, "correlations.tsv"), stringsAsFactors = FALSE)
fam <- load_tf_family()                       # tolower(TF) -> tolower(Family)
res$family <- unname(fam[tolower(res$tf)])
res$family[is.na(res$family) | !nzchar(res$family)] <- "unknown"

# Same summary as stage 03's rate_row, over an arbitrary row subset.
rate_row <- function(d) {
  t5 <- d$top5; ok <- !is.na(t5)
  n <- sum(ok); ntop <- sum(t5[ok])
  data.frame(
    n_tested      = n,
    median_pctile = round(median(d$own_pctile, na.rm = TRUE), 4),
    n_top5        = ntop,
    frac_top5     = if (n > 0) round(ntop / n, 4) else NA_real_,
    top5_binom_p  = if (n > 0) signif(binom.test(ntop, n, p = 0.05,
                                    alternative = "greater")$p.value, 3) else NA_real_
  )
}

# Long-form label used for both resources and the pooled "combined" view.
groups <- list(
  cisbp      = res[res$source == "cisbp", ],
  zebramotif = res[res$source == "zebramotif", ],
  combined   = res
)

# Per (resource, family)
by_family <- do.call(rbind, lapply(names(groups), function(g) {
  d <- groups[[g]]
  do.call(rbind, lapply(split(d, d$family), function(df) {
    cbind(resource = g, family = df$family[1], rate_row(df))
  }))
}))
# sort within resource by n_tested desc
by_family <- by_family[order(match(by_family$resource, names(groups)),
                             -by_family$n_tested), ]
rownames(by_family) <- NULL

write.table(by_family, file.path(cfg$outdir, "percentile_stats_by_family.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# E resource with zf-C2H2 removed
excl <- do.call(rbind, lapply(names(groups), function(g) {
  d <- groups[[g]]
  rbind(
    cbind(resource = g, set = "all",           rate_row(d)),
    cbind(resource = g, set = "no_zf-C2H2",    rate_row(d[d$family != "zf-c2h2", ])),
    cbind(resource = g, set = "zf-C2H2_only",  rate_row(d[d$family == "zf-c2h2", ]))
  )
}))
rownames(excl) <- NULL
write.table(excl, file.path(cfg$outdir, "percentile_stats_zfC2H2_excluded.tsv"),
            sep = "\t", quote = FALSE, row.names = FALSE)

# ZebraMotif vs CisBP head-to-head, with and without zf-C2H2.
zm <- groups$zebramotif; cis <- groups$cisbp
mwu_all <- wilcox.test(zm$own_pctile, cis$own_pctile)$p.value
mwu_nz  <- wilcox.test(zm$own_pctile[zm$family != "zf-c2h2"],
                       cis$own_pctile[cis$family != "zf-c2h2"])$p.value

cat("\n=== Per-resource, zf-C2H2 partition ===\n")
print(excl, row.names = FALSE)
cat("\nZebraMotif vs CisBP percentile MWU:  all p =", signif(mwu_all, 3),
    " | zf-C2H2 excluded p =", signif(mwu_nz, 3), "\n")

cat("\n=== Top families by n_tested (ZebraMotif) ===\n")
print(head(by_family[by_family$resource == "zebramotif", ], 15), row.names = FALSE)

msg("Wrote percentile_stats_by_family.tsv and percentile_stats_zfC2H2_excluded.tsv to ", cfg$outdir)

# Per-family CisBP vs ZebraMotif comparison figures
pf0 <- by_family[by_family$resource %in% c("cisbp", "zebramotif"), ]
cov <- pf0 %>% group_by(family) %>%
  summarise(in_cis = any(resource == "cisbp" & n_tested > 0),
            in_zm  = any(resource == "zebramotif" & n_tested > 0), .groups = "drop") %>%
  mutate(coverage = ifelse(in_cis & in_zm, "both",
                    ifelse(in_zm, "ZebraMotif only", "CisBP only")))
pf0 <- left_join(pf0, cov[, c("family", "coverage")], by = "family")
pf0$resource <- factor(pf0$resource, levels = c("cisbp", "zebramotif"),
                       labels = c("CisBP", "ZebraMotif"))
pal <- c("CisBP" = "#E2A32D", "ZebraMotif" = "#2166ac")

mk <- function(df, yvar, xlab, vline, title) {
  # order families by best (max) median across resources; C2H2 sinks to bottom.
  ord <- df %>% group_by(family) %>% summarise(m = max(median_pctile), .groups = "drop") %>%
    arrange(m) %>% pull(family)
  df$family <- factor(df$family, levels = ord)
  # append a coverage tag to exclusive families' labels
  tag <- setNames(df$coverage, df$family)[levels(df$family)]
  labs_fam <- ifelse(tag == "both", levels(df$family),
                     paste0(levels(df$family), " *"))
  ggplot(df, aes(family, .data[[yvar]], fill = resource)) +
    geom_col(position = position_dodge(width = 0.7), width = 0.65) +
    geom_text(aes(label = n_tested), position = position_dodge(width = 0.7),
              hjust = -0.15, size = 2.2, colour = "grey30") +
    geom_hline(yintercept = vline, linetype = 3, colour = "grey40") +
    scale_fill_manual(values = pal) +
    scale_x_discrete(labels = labs_fam) +
    coord_flip() + expand_limits(y = max(df[[yvar]], na.rm = TRUE) * 1.12) +
    labs(x = NULL, y = xlab, fill = NULL,
         caption = "* = family present in only one resource; bar label = n TFs tested") +
    ggtitle(title) +
    theme_classic(base_size = 11) +
    theme(legend.position = "top")
}

# powered subset (n>=5 in either resource)
powered_fam <- unique(pf0$family[pf0$n_tested >= 5])
pf5 <- pf0[pf0$family %in% powered_fam, ]

save2 <- function(p, stem, h) {
  ggsave(file.path(cfg$outdir, paste0(stem, ".svg")), p, width = 6.4, height = h)
}
save2(mk(pf5, "median_pctile", "median own-TF percentile (null = 0.5)", 0.5,
         "Own-TF percentile by family (n>=5)"), "percentile_by_family", 5.2)
save2(mk(pf5, "frac_top5", "fraction with own TF in top 5% (null = 0.05)", 0.05,
         "Top-5% rate by family (n>=5)"), "top5_by_family", 5.2)
# full panel: every family in either resource (exclusives included)
save2(mk(pf0, "median_pctile", "median own-TF percentile (null = 0.5)", 0.5,
         "Own-TF percentile by family (all)"), "percentile_by_family_all", 9.0)
save2(mk(pf0, "frac_top5", "fraction with own TF in top 5% (null = 0.05)", 0.05,
         "Top-5% rate by family (all)"), "top5_by_family_all", 9.0)
