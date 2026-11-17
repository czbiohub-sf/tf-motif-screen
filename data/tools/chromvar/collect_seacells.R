#!/usr/bin/env Rscript
#
# module load r/4.4 && Rscript collect_seacells.R
#
# Collect precomputed SEACells into a single, per-cell assignment table keyed to
# the integrated object's cell barcodes. SEACells were computed per dataset, so
# their IDs restart at "SEACell-0" in every set, each is prefixed with its dataset
# ("TDR126_SEACell-36") to be globally  unique. The integrated object appends a
# merge suffix ("-1_1".."-1_7") to the bare 10x barcode, and bare barcodes collide
# across datasets, so cells are joined on (dataset, bare barcode)
#
# Reads:  integrated_RNA_ATAC.rds (barcodes + `dataset` metadata),
#         objects_75cells_per_metacell/<data>_seacells_obs_annotation_ML_coarse.csv
# Writes: results/seacells_assignment.csv  (cell, metacell, dataset, dev_stage, celltype)

suppressPackageStartupMessages({ library(Seurat); library(Signac) })
source("utils.R")

seacell_dir <- PATHS$zebrahub_seacells_dir

# obs csv basename -> dataset id as it appears in integrated obj's `dataset` column
# (the integrated object drops the "reseq" suffix).
files <- c(
  TDR126      = "TDR126_seacells_obs_annotation_ML_coarse.csv",
  TDR127      = "TDR127_seacells_obs_annotation_ML_coarse.csv",
  TDR128      = "TDR128_seacells_obs_annotation_ML_coarse.csv",
  TDR118reseq = "TDR118reseq_seacells_obs_annotation_ML_coarse.csv",
  TDR119reseq = "TDR119reseq_seacells_obs_annotation_ML_coarse.csv",
  TDR124reseq = "TDR124reseq_seacells_obs_annotation_ML_coarse.csv",
  TDR125reseq = "TDR125reseq_seacells_obs_annotation_ML_coarse.csv"
)

# Gather the per-dataset SEACells assignments
msg("Reading ", length(files), " SEACells obs tables")
sea <- do.call(rbind, lapply(names(files), function(fkey) {
  dataset <- sub("reseq$", "", fkey)                 # TDR118reseq -> TDR118
  df <- read.csv(file.path(seacell_dir, files[[fkey]]),
                 stringsAsFactors = FALSE, check.names = FALSE)
  stopifnot(all(c("index", "SEACell") %in% colnames(df)))
  df <- df[nzchar(df$SEACell) & !is.na(df$SEACell), ]
  data.frame(
    dataset  = dataset,
    bare     = df$index,                             # 10x barcode, e.g. AAACAGCCAAACGGGC-1
    metacell = paste0(dataset, "_", df$SEACell),     # globally unique
    dev_stage = df[["dev_stage"]],
    celltype  = df[["annotation_ML_coarse"]],
    stringsAsFactors = FALSE
  )
}))
sea$key <- paste(sea$dataset, sea$bare, sep = "|")
stopifnot(!any(duplicated(sea$key)))                 # barcodes unique within a dataset
msg("  ", nrow(sea), " cells, ", length(unique(sea$metacell)),
    " metacells across ", length(unique(sea$dataset)), " datasets")

# Key to the integrated object's barcodes
msg("Loading integrated object metadata")
obj <- readRDS(cfg$signac_rds)
md  <- obj[[]]
md$cell <- rownames(md)
stopifnot("dataset" %in% colnames(md))
md$bare <- sub("_[0-9]+$", "", md$cell)              # AAACAGCC...-1_3 -> AAACAGCC...-1
md$key  <- paste(md$dataset, md$bare, sep = "|")

i <- match(md$key, sea$key)
out <- data.frame(
  cell      = md$cell,
  metacell  = sea$metacell[i],
  dataset   = md$dataset,
  dev_stage = sea$dev_stage[i],
  celltype  = sea$celltype[i],
  stringsAsFactors = FALSE
)
matched <- !is.na(out$metacell)
msg("  matched ", sum(matched), " / ", nrow(out), " integrated cells to a SEACell (",
    sprintf("%.1f%%", 100 * mean(matched)), ")")
unmatched_sea <- setdiff(sea$key, md$key)
if (length(unmatched_sea))
  msg("  ", length(unmatched_sea), " SEACells cells had no integrated barcode (dropped)")

out <- out[matched, ]
# per-dataset sanity so a silent join failure can't pass as success
print(table(out$dataset))

outfile <- file.path(cfg$base_outdir, "seacells_assignment.csv")
write.csv(out, outfile, row.names = FALSE)
msg("Wrote ", outfile, "  (", nrow(out), " cells, ",
    length(unique(out$metacell)), " metacells)")
