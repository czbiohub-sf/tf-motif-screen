#!/bin/bash
#SBATCH --job-name=homer_motif
#SBATCH --time=8:00:00
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=homer_%A_%a.ou
#SBATCH --error=homer_%A_%a.err
#SBATCH --array=1-36


module load anaconda/latest
conda activate homer

# submit from project root
ROOT="$SLURM_SUBMIT_DIR"
CLUSTER_PATH="$ROOT/data/homer/jaspar/clusters"
OUT_PATH="$ROOT/data/homer/jaspar/output"
GENOME_PATH=$(bash "$ROOT/src/af3tfm/config_get.sh" genome_fa)
DATABASE_FILE_PATH="$ROOT/data/homer/jaspar.homer.motif"

# Pick the N-th *.csv file for this array element
csv_file=$(ls -1 "${CLUSTER_PATH}"/*.csv | sed -n "${SLURM_ARRAY_TASK_ID}p")

# Exit cleanly if the ID is out of range
if [[ -z "$csv_file" ]]; then
    echo "No CSV file found for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

base_name=$(basename "$csv_file" .csv)
out_dir="${OUT_PATH}/${base_name}"
mkdir -p "$out_dir"

findMotifsGenome.pl "$csv_file" "$GENOME_PATH" "$out_dir" -size 200 -mknown "$DATABASE_FILE_PATH" -mcheck "$DATABASE_FILE_PATH"