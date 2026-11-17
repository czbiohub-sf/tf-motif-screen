#!/bin/bash
#SBATCH --job-name=homer_motif
#SBATCH --time=8:00:00
#SBATCH --partition=cpu
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=homer_%A_%a.ou
#SBATCH --error=homer_%A_%a.err

module load anaconda/latest
conda activate homer

# submit from project root
ROOT="$SLURM_SUBMIT_DIR"
CLUSTER_FILE_PATH="$ROOT/data/homer/jaspar/clusters/0.csv"
GENOME_PATH=$(bash "$ROOT/src/af3tfm/config_get.sh" genome_fa)
OUTPUT_FILE_PATH="$ROOT/data/homer/jaspar/output/0"
DATABASE_FILE_PATH="$ROOT/data/homer/jaspar.homer.motif"

findMotifsGenome.pl "$CLUSTER_FILE_PATH" "$GENOME_PATH" "$OUTPUT_FILE_PATH" -size 200 -mknown "$DATABASE_FILE_PATH" -mcheck "$DATABASE_FILE_PATH"