#!/bin/bash
#SBATCH --job-name=dml_acs
#SBATCH --output=dml_acs_%j.log
#SBATCH --error=dml_acs_%j.err
#SBATCH --partition=hpg-default
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# ──────────────────────────────────────────────────────────────
# Theory-Driven Scalar DML — Job B: Full vs No-ACS Controls
# 22 features × 4 specs (RF/GB × Full/NoACS) × B=500
# Resume-safe: resubmit same script to continue from checkpoint.
# ──────────────────────────────────────────────────────────────

echo "============================================="
echo "  Theory DML — Job B: ACS Controls"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $(hostname)"
echo "  CPUs:   $SLURM_CPUS_PER_TASK"
echo "  Start:  $(date)"
echo "============================================="

module purge
module load conda
conda activate "${CONDA_ENV:-kight_environment}"  # override via: export CONDA_ENV=your_env

export PYTHONUNBUFFERED=1
python -u theory_dml_acs.py

EXIT_CODE=$?

echo ""
echo "============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ Job B complete"
else
    echo "  ✗ Exited with code $EXIT_CODE (resubmit to resume)"
fi
echo "  Wall time: $SECONDS seconds"
echo "  End: $(date)"
echo "============================================="
