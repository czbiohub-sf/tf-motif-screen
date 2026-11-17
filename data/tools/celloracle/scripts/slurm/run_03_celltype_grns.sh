#!/bin/bash
#SBATCH --job-name=grnswap_03_celltypeGRN
#SBATCH --partition=cpu
#SBATCH --output=data/tools/celloracle/logs/03_celltypeGRN_%A_%a.out
#SBATCH --error=data/tools/celloracle/logs/03_celltypeGRN_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
# Usage (from project root): sbatch --array=0-5 data/tools/celloracle/scripts/slurm/run_03_celltype_grns.sh <arm> [--annotation global_annotation|annotation_ML_coarse] [--config config.yaml]
# arm = cisbp | zebramotif | hybrid. The config supplies the default annotation.

set -euo pipefail
ARM="${1:?provide arm: cisbp|zebramotif|hybrid}"
shift
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"   # submit from project root
GRN="$ROOT/data/tools/celloracle"

# Mail notifications go to mail_user from src/af3tfm/config(.local).yaml; #SBATCH lines
# can't read the config, so the running job adds the recipient to itself.
MAIL_USER=$(bash "$ROOT/src/af3tfm/config_get.sh" mail_user 2>/dev/null || true)
if [[ -n "$MAIL_USER" && -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol update JobId="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" MailUser="$MAIL_USER" || true
fi

ENV=$(bash "$ROOT/src/af3tfm/config_get.sh" celloracle_env)

export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"   # needed for igraph (get_network_score)

# Concurrent CellOracle imports must not share genomepy's sqlite cache.
JOB_CACHE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/celloracle_${SLURM_JOB_ID:-local}_XXXXXX")
export XDG_CACHE_HOME="$JOB_CACHE"
export MPLCONFIGDIR="$JOB_CACHE/matplotlib"
export NUMBA_CACHE_DIR="$JOB_CACHE/numba"
trap 'rm -rf "$JOB_CACHE"' EXIT

CONFIG_ARGS=()
if [[ $# -gt 0 && "$1" != --* ]]; then
    # Accept a positional config path used by saved run commands.
    CONFIG_ARGS=(--config "$1")
    shift
fi

"$ENV/bin/python" -u "$GRN/scripts/03_compute_celltype_grns.py" --index "${SLURM_ARRAY_TASK_ID:-0}" --arm "$ARM" "${CONFIG_ARGS[@]}" "$@"
