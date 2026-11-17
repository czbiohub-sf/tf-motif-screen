#!/bin/bash
# Submit the NA-MPNN parameter sweep (number_of_batches x temperature) for the
# high-homology TFs with JASPAR orthologs (data/jobs/hero/highsim_jaspar_tfs.txt).
# Each grid cell writes to its own spec-dir so the NPZs (whose filenames do not
# encode these params) don't collide. Downstream: notebooks/03_building-ppms/03_sweep-na-mpnn.py
# gains num_inferences x temperature as two more sweep axes on top of
# num_specs x cor_threshold.
set -euo pipefail

# Activate the pipeline environment, or set AF3TFM_PYTHON to its interpreter.
PY="${AF3TFM_PYTHON:-python}"
JOBDB=data/jobs/hero/jobdb.sqlite
TFS=data/jobs/hero/highsim_jaspar_tfs.txt
NSTRUCTS=30

for NI in 1 5 10; do
  for T in 0.1 0.2 0.3 0.4 0.5; do
    SPEC="spec_ni${NI}_t${T}"
    echo "=== $SPEC (n_batches=$NI, temperature=$T) ==="
    "$PY" scripts/run_spec.py \
      --dataset tfdb4_rep \
      --jobdb "$JOBDB" \
      --partition preempted \
      --gpus 0 \
      --n-structs "$NSTRUCTS" \
      --proteins-file "$TFS" \
      --spec-dir "$SPEC" \
      --n-batches "$NI" \
      --temperature "$T"
  done
done
