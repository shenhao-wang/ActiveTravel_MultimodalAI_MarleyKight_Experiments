#!/bin/bash
#SBATCH --job-name=llm_zoning_features
#SBATCH --output=llm_zoning_features_%j.log
#SBATCH --error=llm_zoning_features_%j.err
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --mem=48gb
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=<your_qos>        # <-- EDIT: your SLURM QOS
#
# ──────────────────────────────────────────────────────────────
# LLM-Scored Theory-Driven Zoning Features — HiPerGator SLURM
# ──────────────────────────────────────────────────────────────
#
# Partition notes (UF HiPerGator):
#   hpg-turin  — L4 GPUs (24 GB VRAM each, 3 per node)
#   hpg-b200   — B200 GPUs (180 GB VRAM each, high demand)
#
# The L4 (24 GB) can run 7-9B models in 4-bit quantization (~8-10 GB).
# For full fp16, use hpg-b200 or request 2× L4.
#
# PREREQUISITES:
#   1. Hugging Face token for gated models (Llama, Gemma):
#      huggingface-cli login --token <your_token>
#      OR uncomment the HF_TOKEN line below
#   2. bitsandbytes installed: pip install bitsandbytes --break-system-packages
#   3. zone_text_corpora.csv in working directory
#
# USAGE:
#   sbatch run_llm_features.sh                  # Mistral-7B 4-bit (default)
#   sbatch run_llm_features.sh llama8b          # Llama-3-8B 4-bit
#   sbatch run_llm_features.sh gemma            # Gemma-2-9B 4-bit
#   sbatch run_llm_features.sh llama8b-fp16     # Llama-3-8B fp16 (needs B200)
# ──────────────────────────────────────────────────────────────

# ── Hugging Face token (uncomment and fill in)
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

MODEL_PRESET="${1:-mistral}"

case "$MODEL_PRESET" in
    mistral|default)
        MODULE="nlp/1.3"
        MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3"
        CORPUS_CHARS=6000
        PASSES=3
        QUANT="4bit"
        ;;
    llama8b|llama)
        MODULE="llama/3"
        MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
        CORPUS_CHARS=6000
        PASSES=3
        QUANT="4bit"
        ;;
    gemma)
        MODULE="gemma_llm"
        MODEL_ID="google/gemma-2-9b-it"
        CORPUS_CHARS=6000
        PASSES=3
        QUANT="4bit"
        ;;
    llama8b-fp16)
        MODULE="llama/3"
        MODEL_ID="meta-llama/Meta-Llama-3-8B-Instruct"
        CORPUS_CHARS=8000
        PASSES=3
        QUANT="none"
        ;;
    *)
        MODULE="nlp/1.3"
        MODEL_ID="$MODEL_PRESET"
        CORPUS_CHARS=6000
        PASSES=3
        QUANT="4bit"
        ;;
esac

echo "============================================="
echo "  LLM Zoning Feature Scoring"
echo "  Model: $MODEL_ID"
echo "  Module: $MODULE"
echo "  Quantization: $QUANT"
echo "  Passes: $PASSES"
echo "  Job ID: $SLURM_JOB_ID"
echo "============================================="

module purge
module load $MODULE

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "WARNING: nvidia-smi failed"

if [ ! -f "zone_text_corpora.csv" ]; then
    echo "ERROR: zone_text_corpora.csv not found in $(pwd)"
    exit 1
fi

RESUME_FLAG=""
if [ -f "zone_llm_features_raw_partial.csv" ]; then
    N_DONE=$(tail -n +2 zone_llm_features_raw_partial.csv | wc -l)
    echo "Found partial results ($N_DONE zones done). Resuming..."
    RESUME_FLAG="--resume zone_llm_features_raw_partial.csv"
fi

QUANT_FLAG=""
if [ "$QUANT" = "4bit" ]; then
    QUANT_FLAG="--quantize 4bit"
fi

python llm_zoning_features.py \
    --backend local \
    --model "$MODEL_ID" \
    --passes $PASSES \
    --corpus-chars $CORPUS_CHARS \
    --temperature 0.3 \
    --max-new-tokens 2048 \
    $QUANT_FLAG \
    $RESUME_FLAG

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Scoring complete"
    [ -f "zone_embeddings_llm_scored.csv" ] && echo "Output: zone_embeddings_llm_scored.csv"
else
    echo "✗ Script exited with code $EXIT_CODE"
fi
echo "Wall time: $SECONDS seconds"
exit $EXIT_CODE