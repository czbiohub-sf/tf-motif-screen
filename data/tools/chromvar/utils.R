# utils.R -- shared config + helpers for the chromVAR pipeline

suppressPackageStartupMessages({
  library(Matrix)
  library(matrixStats)
})

# Project root (scripts are run from data/tools/chromvar) and external data paths
# from src/af3tfm/config.yaml, overridden by config.local.yaml if present
PROJECT_ROOT <- normalizePath("../../..")
.af3tfm_cfg <- yaml::read_yaml(file.path(PROJECT_ROOT, "src/af3tfm/config.yaml"))
.af3tfm_local <- file.path(PROJECT_ROOT, "src/af3tfm/config.local.yaml")
if (file.exists(.af3tfm_local)) .af3tfm_cfg <- modifyList(.af3tfm_cfg, yaml::read_yaml(.af3tfm_local))
PATHS <- .af3tfm_cfg$paths

# File and hyperparameter configuration for pipeline

cfg <- list(
  # Input objects (stage 01)
  signac_rds      = PATHS$zebrahub_signac_rds,        # path to Seurat object w/ ATAC + RNA assays

  signac_atac_assay = "peaks_integrated",  # ChromatinAssay, 640,834 peaks, integer counts
  signac_rna_assay  = "RNA",               # log-normalized data slot

  # Motif filename
  zebramotif_path = file.path(PROJECT_ROOT, "data/database/zebramotif_all_modes.meme"),
  zm_primary_only = TRUE,  # Multiple binding modes per motif, keep only primary mode

  # ZebraMotif contains one PWM per representative TF, with related TFs mapped to a rep whose PWM they borrow
  zm_tf_to_rep    = file.path(PROJECT_ROOT, "data/database/tf_to_rep.parquet"),

  # CisBP zebrafish, used as the incumbent-resource comparison
  cisbp_pwm_dir   = file.path(PROJECT_ROOT, "data/sequences/cisbp/pwms"),  # one .txt per motif
  cisbp_tf_info   = file.path(PROJECT_ROOT, "data/sequences/cisbp/TF_Information.txt"),

  # BSgenome object
  genome_pkg      = "BSgenome.Drerio.UCSC.danRer11",

  # Metacell partition (stage 02). Aggregation is onto Zebrahub-Multiome's precomputed
  # SEACells (~75 cells/metacell), collated per-cell by collect_seacells.R into
  # columns cell,metacell,dataset,dev_stage,celltype. min_cells_per_mc drops the
  # few undersized SEACells left after the barcode join.
  external_metacells_csv = "results/seacells_assignment.csv",
  min_cells_per_mc = 20,

  # Per-motif scan cutoffs (stage 01). motifmatchr's
  # analytic per-motif p-value at its default p.cutoff = 5e-5
  scan_alpha      = 5e-5,          # matchmotifs p.cutoff

  # Tested-population restriction (stage 03)
  tf_list_file    = file.path(PROJECT_ROOT, "data/sequences/animaltfdb/tfdb4_tf_whitelist.txt"),  # optional whitelist of TFs to test
  min_total_ic    = 1, # Minimum total IC for a PWM to be testable

  # TF-family source (animalTFDB4; columns TF_Name, Family). Same-family paralogs
  # are excluded from each own-TF's null pool (they share the motif and would rank
  # spuriously high).
  tf_family_csv   = file.path(PROJECT_ROOT, "data/sequences/animaltfdb/tfdb4_tfs.csv"),

  # base_outdir holds the metacell-independent files
  base_outdir     = "results",
  seed            = 1
)

# Env-var overrides -- isolate a whole scan+deviations+stats world in its own dir:
# R_BASE_OUTDIR=<dir> : root for scan_cache + IC/matched TSVs + outputs (default "results").
# R_SCAN_ALPHA=<float> : matchmotifs p.cutoff (stage 01).
if (nzchar(Sys.getenv("R_BASE_OUTDIR"))) cfg$base_outdir <- Sys.getenv("R_BASE_OUTDIR")
if (nzchar(Sys.getenv("R_SCAN_ALPHA"))) cfg$scan_alpha  <- as.numeric(Sys.getenv("R_SCAN_ALPHA"))


# The scan cache lives in base_outdir and is shared across scan worlds.
# Delete it to force a rescan (e.g. after a DB change).
cfg$scan_cache <- file.path(cfg$base_outdir, "scan_cache.rds")

# Deviations/stats for the SEACells partition go under base_outdir/seacells/, one
# level below the scan cache (which is partition-independent and shared).
cfg$outdir <- file.path(cfg$base_outdir, "seacells")


# Helpers

msg <- function(...) cat(format(Sys.time(), "[%H:%M:%S] "), ..., "\n", sep = "")

# Load the configured BSgenome object and return it
load_genome <- function() {
  suppressPackageStartupMessages(library(cfg$genome_pkg, character.only = TRUE))
  get(cfg$genome_pkg)
}

# Aggregate a features x cells matrix into features x metacells
aggregate_to_metacells <- function(mat, mc_assign, fun = c("sum", "mean")) {
  fun <- match.arg(fun)
  mc <- factor(mc_assign)
  # indicator matrix cells x metacells
  ind <- sparse.model.matrix(~ 0 + mc)
  colnames(ind) <- levels(mc)
  agg <- mat %*% ind
  if (fun == "mean") agg <- sweep(agg, 2, table(mc)[colnames(agg)], "/")
  as.matrix(agg)
}

# Data loader: returns peaks (GRanges), atac (peaks x cells, sparse), rna (genes x
# cells, sparse, log-normalised), and celltype (character vector, for plotting only).
load_signac <- function(genome) {
  suppressPackageStartupMessages({ library(Seurat); library(Signac) })
  obj <- readRDS(cfg$signac_rds)
  DefaultAssay(obj) <- cfg$signac_atac_assay

  peaks <- granges(obj[[cfg$signac_atac_assay]])
  atac  <- GetAssayData(obj, assay = cfg$signac_atac_assay, slot = "counts")

  # This object stores peaks in Ensembl seqlevel style ("1","2",...,"MT") while
  # the BSgenome uses UCSC ("chr1",...,"chrM"); getSeq()/addGCBias() would
  # otherwise find zero matching seqlevels. Harmonize names, restrict to genome
  # contigs, attach the genome's chromosome lengths, and drop peaks whose
  # coordinates fall outside those bounds (peak calling can extend a few peaks
  # past a chromosome end, which makes getSeq error). Every drop is mirrored on
  # `atac` so the peak x cell matrix stays row-aligned with `peaks`.
  stopifnot(length(peaks) == nrow(atac))
  drop_rows <- function(mask, why) {
    if (any(!mask)) message("  dropping ", sum(!mask), " peaks (", why, ")")
    peaks <<- peaks[mask]; atac <<- atac[mask, , drop = FALSE]
  }
  seqlevelsStyle(peaks) <- "UCSC"
  drop_rows(as.character(seqnames(peaks)) %in% seqnames(genome),
            "contig absent from genome")
  seqlevels(peaks) <- seqlevelsInUse(peaks)
  seqinfo(peaks)   <- seqinfo(genome)[seqlevels(peaks)]
  slen <- seqlengths(peaks)[as.character(seqnames(peaks))]
  drop_rows(start(peaks) >= 1 & end(peaks) <= slen, "out of chromosome bounds")

  list(
    peaks    = peaks,
    atac     = atac,
    rna      = GetAssayData(obj, assay = cfg$signac_rna_assay,  slot = "data"),
    celltype = as.character(Idents(obj))
  )
}

# Load the multiome, intersect to cells present in all modalities, and align.
load_multiome <- function(genome) {
  msg("Loading multiome data")
  dat <- if (!is.null(cfg$signac_rds)) load_signac(genome)
  else stop("No multiome source configured (cfg$signac_rds is NULL)")

  # keep cells present in both modalities
  common <- intersect(colnames(dat$atac), colnames(dat$rna))
  stopifnot(length(common) > 0)
  dat$atac <- dat$atac[, common]; dat$rna <- dat$rna[, common]
  dat$celltype <- dat$celltype[match(common, colnames(dat$atac))]
  msg("  ", length(common), " cells, ", length(dat$peaks), " peaks, ", nrow(dat$rna), " genes")
  dat
}

# Motif reading and motif -> TF mapping

# Read motifs from a single multi-motif file OR a directory of motif files.
read_motifs <- function(path) {
  fs <- if (dir.exists(path))
    list.files(path, pattern = "\\.(meme|jaspar|txt|pfm)$", full.names = TRUE)
  else path
  do.call(c, lapply(fs, function(f) {
    m <- tryCatch(read_meme(f), error = function(e)
         tryCatch(read_jaspar(f), error = function(e2) read_homer(f)))
    if (!is.list(m)) m <- list(m)
    m
  }))
}

# CisBP ships one .txt per motif in its pwms dir, and a large fraction of them
# are header-only stubs for TFs with no matrix. Those must be dropped or
# read_cisbp() errors out and the control set silently shrinks. Returns both the
# motif list and the TF_Information table (used to build the motif->TF map).
read_cisbp_dir <- function(dir, tf_info) {
  info <- read.delim(tf_info, stringsAsFactors = FALSE)
  info <- info[nzchar(info$Motif_ID) & info$Motif_ID != ".", ]
  fs <- file.path(dir, paste0(unique(info$Motif_ID), ".txt"))
  fs <- fs[file.exists(fs)]
  nonempty <- fs[vapply(fs, function(f) length(readLines(f, warn = FALSE)) > 1, logical(1))]
  msg("  CisBP: ", length(nonempty), " non-empty matrices of ", length(fs), " referenced")
  ms <- lapply(nonempty, function(f) {
    m <- tryCatch(read_cisbp(f), error = function(e) NULL)
    if (is.null(m)) return(NULL)
    if (is.list(m)) m <- m[[1]]
    m["name"] <- sub("\\.txt$", "", basename(f))
    m
  })
  ms <- Filter(Negate(is.null), ms)
  list(motifs = ms, info = info)
}

# Resolve a stable Ensembl gene id (ENSDARG...) to the current animaltfdb4 gene symbol
load_ens2sym <- function() {
  f <- read.csv(cfg$tf_family_csv, stringsAsFactors = FALSE)
  f <- f[!duplicated(f$DBID), ]
  setNames(f$TF_Name, f$DBID)
}

# ZebraMotif ids embed the gene's Ensembl id: "ZM00000011235.1" ->
# ENSDARG00000011235. Returns the ENSDARG, or NA if the id does not conform.
zm_id_to_ensdarg <- function(motif_id) {
  m <- regmatches(motif_id, regexec("^ZM0*([0-9]+)\\.", motif_id))[[1]]
  if (length(m) < 2) return(NA_character_)
  sprintf("ENSDARG%011d", as.integer(m[2]))
}

# ZebraMotif cluster membership for the transferred-PWM expansion. Returns a
# data.frame(rep_dbid, member_sym) of the resolution == "cluster" members (the
# rep's PWM is transferred to them)
load_zm_clusters <- function() {
  empty <- data.frame(rep_dbid = character(0), member_sym = character(0))
  if (is.null(cfg$zm_tf_to_rep)) return(empty)
  t2r <- as.data.frame(arrow::read_parquet(cfg$zm_tf_to_rep))
  t2r <- t2r[!t2r$is_representative & t2r$resolution == "cluster" &
             !is.na(t2r$Rep_DBID), , drop = FALSE]
  if (!nrow(t2r)) return(empty)
  ens2sym <- load_ens2sym()
  sym <- unname(ens2sym[t2r$DBID]); sym[is.na(sym)] <- t2r$TF_Name[is.na(sym)]
  out <- data.frame(rep_dbid = t2r$Rep_DBID, member_sym = sym,
                    stringsAsFactors = FALSE)
  out[!is.na(out$member_sym) & nzchar(out$member_sym), , drop = FALSE]
}

# Build the motif -> TF map, keyed on the stable ENSDARG rather than free-text
# gene symbols
build_m2tf <- function(zm, cis_info) {
  first_char <- function(x) if (length(x) == 0) NA_character_ else as.character(x)[1]
  ens2sym <- load_ens2sym()

  # ZebraMotif: one PWM per gene, keyed by its OWN ENSDARG (embedded in the id).
  # Map id -> ENSDARG -> current symbol; fall back to the first altname only when
  # the ENSDARG is absent from animalTFDB4. (The altname is an ortholog-group list,
  # so first_char(altname) previously mismapped ~130 PWMs to a paralog, e.g.
  # ZM00000006120 for tbx2b was scored against tbx2a.)
  zm_ids <- vapply(zm, function(x) first_char(x["name"]),    character(1))
  zm_alt <- vapply(zm, function(x) first_char(x["altname"]), character(1))
  zm_sym <- unname(ens2sym[vapply(zm_ids, zm_id_to_ensdarg, character(1))])
  zm_sym[is.na(zm_sym)] <- zm_alt[is.na(zm_sym)]
  zm_map <- data.frame(motif_id = zm_ids, tf_gene_symbol = zm_sym,
                       source = "zebramotif", stringsAsFactors = FALSE)
  zm_missing <- is.na(zm_map$tf_gene_symbol) | !nzchar(zm_map$tf_gene_symbol)
  if (any(zm_missing))
    msg("  WARNING: ", sum(zm_missing), " ZebraMotif PWMs have no resolvable TF symbol; dropped")
  zm_map <- zm_map[!zm_missing, ]

  # Expand each rep's PWM to its transferred cluster members: for every motif id
  # of a rep, add one (motif_id, member) row per member sharing that rep. This is
  # the ZebraMotif analogue of the CisBP paralog expansion below -- a shared PWM
  # tested against each related TF's expression, using the rep's deviation row.
  clusters <- load_zm_clusters()
  if (nrow(clusters)) {
    zm_map$rep_dbid <- vapply(zm_map$motif_id, zm_id_to_ensdarg, character(1))
    memb <- merge(zm_map[, c("motif_id", "rep_dbid")], clusters, by = "rep_dbid")
    if (nrow(memb)) {
      zm_map <- rbind(
        zm_map[, c("motif_id", "tf_gene_symbol", "source")],
        data.frame(motif_id = memb$motif_id, tf_gene_symbol = memb$member_sym,
                   source = "zebramotif", stringsAsFactors = FALSE))
    } else {
      zm_map$rep_dbid <- NULL
    }
    msg("  ZebraMotif: expanded to ", sum(!duplicated(clusters$member_sym)),
        " transferred cluster members")
  }

  # CisBP: TF_Information lists every paralog for a Motif_ID; expand to one
  # (motif, symbol) pair per paralog (symbol resolved via ENSDARG, falling
  # back to the CisBP TF_Name) so every expressed paralog is testable
  cis_sym <- unname(ens2sym[cis_info$DBID])
  cis_sym[is.na(cis_sym)] <- cis_info$TF_Name[is.na(cis_sym)]
  cis_map <- data.frame(motif_id = cis_info$Motif_ID, tf_gene_symbol = cis_sym,
                        source = "cisbp", stringsAsFactors = FALSE)
  cis_map <- cis_map[!is.na(cis_map$tf_gene_symbol) & nzchar(cis_map$tf_gene_symbol), ]

  # Keyed on the (motif_id, tf, source) triple; motif_id alone is not unique.
  m2tf <- rbind(zm_map, cis_map)
  m2tf <- m2tf[!duplicated(m2tf[, c("motif_id", "tf_gene_symbol", "source")]), ]
  rownames(m2tf) <- NULL
  m2tf
}

# Rebuild the motif -> TF map straight from the configured ZebraMotif file and
# CisBP TF_Information, without re-reading the CisBP matrix directory.
build_m2tf_from_cfg <- function() {
  zm <- read_motifs(cfg$zebramotif_path)
  info <- read.delim(cfg$cisbp_tf_info, stringsAsFactors = FALSE)
  info <- info[nzchar(info$Motif_ID) & info$Motif_ID != ".", ]
  build_m2tf(zm, info)
}

# For each motif, correlate its accessibility deviation with every candidate TF's
# expression across metacells (Spearman), then rank the OWN-TF correlation among
# all OTHER TFs (not arbitrary genes), excluding same-family paralogs

# TF-family map for the swap null: tolower(TF_Name) -> tolower(Family).
# Empty vector if no family CSV is configured (then no paralog exclusion).
load_tf_family <- function() {
  if (is.null(cfg$tf_family_csv)) return(setNames(character(0), character(0)))
  f <- read.csv(cfg$tf_family_csv, stringsAsFactors = FALSE)
  setNames(tolower(f$Family), tolower(f$TF_Name))
}

# Own-TF percentile function (TF-swap null).
# The `expressed` argument is intersected with the TF universe (whitelist if set,
# else every TF in the family map) so the null pool is TFs, not all genes.
make_own_percentile <- function(rna_mc, expressed, cols) {
  fam      <- load_tf_family()
  universe <- if (!is.null(cfg$tf_list_file))
                tolower(trimws(readLines(cfg$tf_list_file))) else names(fam)
  row_lc   <- setNames(rownames(rna_mc), tolower(rownames(rna_mc)))
  pool_lc  <- intersect(intersect(universe, tolower(expressed)), names(row_lc))
  pool     <- unname(row_lc[pool_lc])           # back to rna_mc row casing
  Etf      <- as.matrix(rna_mc[pool, cols, drop = FALSE])
  Er <- t(apply(Etf, 1, rank))
  Er <- Er - rowMeans(Er)
  Er <- Er / sqrt(pmax(rowSums(Er^2), .Machine$double.eps))
  function(zmat, mid, tf) {
    if (is.na(match(tf, rownames(Er)))) return(NA_real_)   # own TF not in the pool
    dr <- rank(zmat[mid, cols])
    dr <- dr - mean(dr)
    ss <- sum(dr^2)
    if (ss <= 0) return(NA_real_)               # constant deviation row -> correlation undefined
    dr <- dr / sqrt(ss)
    rr <- as.numeric(Er %*% dr)                 # spearman with every pool TF
    names(rr) <- rownames(Er)
    # Drop same-family paralogs of the own TF (keep self).
    own_fam <- fam[tolower(tf)]
    sibs <- if (!is.na(own_fam)) setdiff(names(fam)[fam == own_fam], tolower(tf)) else character(0)
    rr   <- rr[!(tolower(names(rr)) %in% sibs)]
    i    <- match(tf, names(rr))
    mean(rr <= rr[i])                           # percentile among all other pool TFs
  }
}

spearman <- function(a, b) suppressWarnings(cor(a, b, method = "spearman"))

# Coupling table for a given metacell resolution and tested-TF set

# Build the collapsed per-TF own-TF coupling table from saved metacells +
# deviations, applying the SAME testability filter as stage 03 (whitelist ->
# IC floor) and the full-pool TF-swap null.
build_coupling_table <- function(rna_mc, zmat, m2tf, ic_tbl, n_matched) {
  # One row per (motif_id, TF) pair (a CisBP PWM spans a paralog group).
  pairs <- m2tf[m2tf$motif_id %in% rownames(zmat), ]
  pairs <- pairs[pairs$tf_gene_symbol %in% rownames(rna_mc), ]
  expressed <- rownames(rna_mc)[rowMeans(rna_mc > 0) >= 0.05]
  pairs <- pairs[pairs$tf_gene_symbol %in% expressed, ]
  if (!is.null(cfg$tf_list_file)) {
    tf_ok  <- trimws(readLines(cfg$tf_list_file))
    pairs <- pairs[pairs$tf_gene_symbol %in% tf_ok, ]
  }
  if (!is.null(cfg$min_total_ic)) {
    ic_ok  <- ic_tbl$motif_id[ic_tbl$total_ic >= cfg$min_total_ic]
    pairs <- pairs[pairs$motif_id %in% ic_ok, ]
  }

  res <- data.frame(
    motif_id  = pairs$motif_id,
    tf        = pairs$tf_gene_symbol,
    source    = pairs$source,
    n_matched = as.integer(n_matched[pairs$motif_id]),
    total_ic  = ic_tbl$total_ic[match(pairs$motif_id, ic_tbl$motif_id)],
    rho       = mapply(function(mid, tf) spearman(zmat[mid, ], rna_mc[tf, colnames(zmat)]),
                       pairs$motif_id, pairs$tf_gene_symbol, USE.NAMES = FALSE),
    stringsAsFactors = FALSE
  )
  own_percentile <- make_own_percentile(rna_mc, expressed, colnames(zmat))
  res$own_pctile <- vapply(seq_len(nrow(res)),
    function(k) own_percentile(zmat, res$motif_id[k], res$tf[k]), numeric(1))
  res$top5 <- res$own_pctile >= 0.95

  # collapse motif modes -> primary (lowest numeric suffix) per (source, tf)
  mode_idx <- suppressWarnings(as.integer(sub(".*\\.", "", res$motif_id)))
  mode_idx[is.na(mode_idx)] <- 99L
  res <- res[order(res$source, res$tf, mode_idx), ]
  res[!duplicated(res[, c("source", "tf")]), ]
}


# Single home for all results and figures
cfg$resultsdir <- cfg$base_outdir
dir.create(cfg$resultsdir, showWarnings = FALSE, recursive = TRUE)
