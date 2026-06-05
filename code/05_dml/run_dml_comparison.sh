#!/bin/bash
#SBATCH --job-name=dml_multi_backend
#SBATCH --output=dml_multi_backend_%j.log
#SBATCH --error=dml_multi_backend_%j.err
#SBATCH --partition=hpg-default
#SBATCH --mem=32gb
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#
# ──────────────────────────────────────────────────────────────
# Multi-Backend DML Comparison — HiPerGator SLURM
# ──────────────────────────────────────────────────────────────
#
# Runs Estimand B (vector DML) across all available embedding
# backends and produces comparison tables + figures.
#
# CPU-only job. RF nuisance models are the bottleneck (~15 min
# per backend with 500 trees on 8 cores).
#
# PREREQUISITES:
#   Run these first to generate embedding files:
#     1. embedding_pipeline_v3.py (BACKEND='compare')
#     2. llm_zoning_features.py (--backend local)
#   Plus the standard data files:
#     tampatrips_1.csv, tpazoning_clean.csv,
#     zone_features_combined.csv, acs_blockgroup_tampa.csv
#
# USAGE:
#   sbatch run_dml_comparison.sh
#   sbatch run_dml_comparison.sh "minilm,llm_scored,hybrid"
# ──────────────────────────────────────────────────────────────

BACKENDS="${1:-}"

echo "============================================="
echo "  Multi-Backend DML Comparison"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
if [ -n "$BACKENDS" ]; then
    echo "  Backends: $BACKENDS"
else
    echo "  Backends: all available"
fi
echo "============================================="

# ── Load Python environment
module purge
module load conda
conda activate <your_env_name>   # <-- EDIT: your conda env

# ── List available embedding files
echo ""
echo "Available embedding files:"
ls -la zone_embeddings_*.csv 2>/dev/null || echo "  (none found)"
echo ""

# ── Check for required data files
for f in tampatrips_1.csv zone_features_combined.csv acs_blockgroup_tampa.csv; do
    if [ ! -f "$f" ]; then
        # Try the processed_data path
        echo "WARNING: $f not found in $(pwd)"
    fi
done

# ── Run
BACKEND_FLAG=""
if [ -n "$BACKENDS" ]; then
    BACKEND_FLAG="--backends $BACKENDS"
fi

python dml_multi_backend.py \
    --n-folds 5 \
    --n-bootstrap 500 \
    --build-hybrid \
    $BACKEND_FLAG

EXIT_CODE=$?

echo ""
echo "============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✓ DML comparison complete"
    for f in dml_backend_summary.csv dml_backend_comparison.csv dml_estimand_c_comparison.csv fig_dml_backend_comparison.png; do
        [ -f "$f" ] && echo "  Output: $f"
    done
else
    echo "  ✗ Script exited with code $EXIT_CODE"
fi
echo "  Wall time: $SECONDS seconds"
echo "============================================="

exit $EXIT_CODE