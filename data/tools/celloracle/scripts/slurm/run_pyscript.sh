#!/bin/bash
#SBATCH --job-name=grnswap_py
#SBATCH --partition=cpu
#SBATCH --output=data/tools/celloracle/logs/py_%x_%j.out
#SBATCH --error=data/tools/celloracle/logs/py_%x_%j.err
#SBATCH --mail-type=FAIL
# Generic runner (from project root): sbatch --job-name=<tag> [--mem=..] [--time=..] data/tools/celloracle/scripts/slurm/run_pyscript.sh <script.py> [args...]
# Runs a pipeline script or repository-relative notebook in the CellOracle env (paths.celloracle_env).

set -euo pipefail
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
export LD_LIBRARY_PATH="$ENV/lib:${LD_LIBRARY_PATH:-}"

# job-local cache so concurrent jobs don't race on genomepy's shared sqlite diskcache
# (importing celloracle touches it; a shared dir => "sqlite3 disk I/O error"). Matches run_02.
JOB_CACHE=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/celloracle_${SLURM_JOB_ID:-local}_XXXXXX")
export XDG_CACHE_HOME="$JOB_CACHE"
export MPLCONFIGDIR="$JOB_CACHE/matplotlib"
export NUMBA_CACHE_DIR="$JOB_CACHE/numba"
trap 'rm -rf "$JOB_CACHE"' EXIT

SCRIPT="$1"; shift
if [[ "$SCRIPT" != /* ]]; then
    if [[ -f "$ROOT/$SCRIPT" ]]; then
        SCRIPT="$ROOT/$SCRIPT"
    else
        SCRIPT="$GRN/scripts/$SCRIPT"
    fi
fi
"$ENV/bin/python" -u "$SCRIPT" "$@"
