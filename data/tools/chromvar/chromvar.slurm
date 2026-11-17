#!/bin/bash
#SBATCH --job-name=chromvar
#SBATCH --time=10:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=FAIL,END
#SBATCH --output=data/tools/chromvar/slurm_logs/chromvar_%j.log
#SBATCH --error=data/tools/chromvar/slurm_logs/chromvar_%j.err

set -euo pipefail

# Mail notifications go to mail_user from src/af3tfm/config(.local).yaml; #SBATCH lines
# can't read the config, so the running job adds the recipient to itself.
MAIL_USER=$(bash "$SLURM_SUBMIT_DIR/src/af3tfm/config_get.sh" mail_user 2>/dev/null || true)
if [[ -n "$MAIL_USER" ]]; then
    scontrol update JobId="$SLURM_JOB_ID" MailUser="$MAIL_USER" || true
fi

module load r/4.4

cd "$SLURM_SUBMIT_DIR/data/tools/chromvar"   # submit from project root

export R_BP_WORKERS=12

echo "[$(date)] stage 01 scan"; Rscript 01_scan_peaks.R
echo "[$(date)] stage 02 deviations"; Rscript 02_deviations.R
echo "[$(date)] stage 03 correlations"; Rscript 03_correlate_stats.R
echo "[$(date)] rank coupled TFs"; Rscript rank_coupled_tfs.R
echo "[$(date)] family stratified stats"; Rscript family_stratified_stats.R
echo "[$(date)] plot matched families"; Rscript plot_matched_families.R
echo "[$(date)] done"
