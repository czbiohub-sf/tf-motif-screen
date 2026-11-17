#!/bin/bash
#SBATCH --job-name=grnswap_02_baseGRN
#SBATCH --partition=cpu
#SBATCH --output=data/tools/celloracle/logs/02_baseGRN_%A_%a.out
#SBATCH --error=data/tools/celloracle/logs/02_baseGRN_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
# Array over timepoint indices (6 timepoints), from project root: sbatch --array=0-5 data/tools/celloracle/scripts/slurm/run_02_base_grn.sh [cisbp|zebramotif|hybrid] [args...]

set -euo pipefail
DB="${1:-zebramotif}"   # which motif DB: cisbp | zebramotif | hybrid
if [[ $# -gt 0 ]]; then shift; fi
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"   # submit from project root
GRN="$ROOT/data/tools/celloracle"

# Mail notifications go to mail_user from src/af3tfm/config(.local).yaml; #SBATCH lines
# can't read the config, so the running job adds the recipient to itself.
MAIL_USER=$(bash "$ROOT/src/af3tfm/config_get.sh" mail_user 2>/dev/null || true)
if [[ -n "$MAIL_USER" && -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol update JobId="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}" MailUser="$MAIL_USER" || true
fi

ENV=$(bash "$ROOT/src/af3tfm/config_get.sh" celloracle_env)

# gimmemotifs cache: job-local dir so parallel array tasks don't corrupt a shared cache
NEW_CACHE=$(bash "$ROOT/src/af3tfm/config_get.sh" job_cache)/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$NEW_CACHE"
export XDG_CACHE_HOME="$NEW_CACHE"

# environment (direct interpreter; no conda activate needed)
export PATH="$ENV/bin:$PATH"
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"   # conda libstdc++ ahead of R libs (igraph GLIBCXX)

"$ENV/bin/python" -u "$GRN/scripts/02_compute_base_grn.py" --index "${SLURM_ARRAY_TASK_ID:-0}" --db "$DB" "$@"

# cleanup job-local cache
rm -rf "$NEW_CACHE"
