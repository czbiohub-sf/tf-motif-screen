# CellOracle

CellOracle analysis for comparing gene regulatory networks inferred with CisBP, ZebraMotif, and their union (`hybrid`). Motif scans define candidate TF–target links, and RNA expression is used to infer signed, weighted networks for each cell type and developmental stage.

### Setup

Requires [CellOracle](https://github.com/morris-lab/CellOracle) and the repository's `grn-plots` extra for plotting dependencies.

External data and environment paths are configured in `src/af3tfm/config.local.yaml`. Motif databases, annotations, stages, and inference parameters are configured in `config/grn_swap.yaml`.

### Pipeline

- `src/grnswap/` contains helpers to convert motif formats and build the CisBP, ZebraMotif, and hybrid databases.
- `scripts/01_validate_motif_db.py` checks motif database formats, gene identifiers, and scanning.
- `scripts/02_compute_base_grn.py` scans accessible peaks to construct motif-based regulatory priors.
- `scripts/03_compute_celltype_grns.py` preprocesses RNA counts, imputes expression, and fits cell-type GRNs using those priors.
- Optional stages 04–05 compute developmental pseudotime and run in silico TF-knockout simulations.

### Analysis

- [04_chromvar-egoregulons.py](../../../notebooks/04_database-and-applications/04_chromvar-egoregulons.py) displays a TF's strongest significant targets as ego networks and ranked coefficient bars, allowing comparison across stages.
- [04_grn-networkx.py](../../../notebooks/04_database-and-applications/04_grn-networkx.py) compares CisBP and hybrid subnetworks around selected TFs. Each arm independently selects its strongest targets; a shared layout highlights additions to the selected hybrid network. Nodes distinguish TFs from other genes.

Fitted networks remain under `outputs/<annotation>/`. Plotting notebooks save figures and their companion CSVs under `notebooks/04_database-and-applications/figures/celloracle/<annotation>/`, in `chromvar_egoregulons/` and `grn_networkx/` respectively. Each notebook supports interactive cells and command-line options through `--help`; `--output-dir` overrides its figure directory. Paths beginning with `notebooks/` are relative to the repository root.

The Slurm runner `scripts/slurm/run_pyscript.sh` accepts repository-relative notebook paths as well as pipeline script names.
