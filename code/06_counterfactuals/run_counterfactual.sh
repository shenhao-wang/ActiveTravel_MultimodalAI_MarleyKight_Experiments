#!/bin/bash
#SBATCH --job-name=counterfactual_zoning
#SBATCH --output=counterfactual_zoning_%j.log
#SBATCH --error=counterfactual_zoning_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --mem=48gb
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=<your_qos>        # <-- EDIT: your SLURM QOS
#
# Generates counterfactual zoning texts using a local LLM,
# embeds them, and scores via DML theta projection.
#
# USAGE:
#   sbatch run_counterfactual.sh              # Mistral-7B (default)
#   sbatch run_counterfactual.sh llama8b      # Llama-3-8B
#
# NOTE: Also requires sentence-transformers for embedding.
#   The nlp/1.3 module includes both transformers and sentence-transformers.

MODEL_PRESET="${1:-mistral}"

case "$MODEL_PRESET" in
    mistral|default)
        MODULE="nlp/1.3"
        MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
        ;;
    llama8b|llama)
        MODULE="llama/3"
        MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
        ;;
    *)
        MODULE="nlp/1.3"
        MODEL_ID="$MODEL_PRESET"
        ;;
esac

echo "============================================="
echo "  Counterfactual Zoning Text Generation"
echo "  Model: $MODEL_ID"
echo "  Job ID: $SLURM_JOB_ID"
echo "============================================="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
module purge
module load $MODULE

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

python code/06_counterfactuals/counterfactual_zoning.py \
    --backend local \
    --model "$MODEL_ID" \
    --quantize 4bit \
    --dml-backend minilm \
    --n-variants 2

echo "Wall time: $SECONDS seconds"