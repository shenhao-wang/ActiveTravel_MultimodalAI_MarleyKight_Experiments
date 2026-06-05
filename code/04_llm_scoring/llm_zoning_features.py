"""
Tampa Zoning — LLM-Scored Theory-Driven Feature Extraction
============================================================
Uses an LLM to read the raw zoning corpus for each of Tampa's 56
zone classes and score it on 10 planning-theoretic dimensions derived
from the built environment and active travel literature.

Unlike the sentence-transformer pipeline, which produces data-driven
PCA dimensions requiring post-hoc interpretation, this script
produces features whose meaning is defined *a priori* by theory.
Every dimension has a planning-interpretable name and a scoring
rubric grounded in the literature.

Theoretical grounding:
  - Use separation / mixing: Cervero & Kockelman 1997 ("diversity" D)
  - Density permissions: Ewing & Cervero 2010 (density elasticities)
  - Pedestrian-street interface: Talen 2013 (form-based vs Euclidean)
  - Parking intensity: Shoup 2005; Salazar-Miranda et al. 2025
  - Design regulation (form-based): Parolek et al. 2008; Talen 2013
  - Use flexibility: Talen et al. 2016 (zoning-use mismatch)
  - Transit orientation: Serrano et al. 2023; Thrun & Connors 2016
  - Open space / green infrastructure: Ewing & Cervero 2010
  - Setback & building placement: Talen 2013; this paper's T3
  - Transparency & activation: NMU standards in Tampa Ch. 27

Output format matches zone_embeddings.csv:
  zone_class, context, feat_000 ... feat_009

Additionally outputs:
  zone_llm_features_raw.csv     — includes dimension names + rationales
  zone_llm_features_prompt.txt  — the exact prompt used (reproducibility)

Backends:
  'local'     — Local HuggingFace model on GPU (default, for HiPerGator)
  'anthropic' — Claude via API (uses ANTHROPIC_API_KEY)
  'openai'    — GPT-4o via API (uses OPENAI_API_KEY)
  'mock'      — Deterministic heuristic scorer for testing without GPU/API

Usage on HiPerGator:
  # Submit via SLURM (see run_llm_features.sh)
  sbatch run_llm_features.sh

  # Or interactively on a GPU dev node:
  module load llama/3       # or nlp/1.3
  srun --partition=gpu --gpus=1 --mem=32gb --time=02:00:00 --pty bash
  python llm_zoning_features.py --backend local \\
      --model meta-llama/Meta-Llama-3-8B-Instruct --passes 3

  # Test without GPU:
  python llm_zoning_features.py --backend mock

Supported local models (tested):
  meta-llama/Meta-Llama-3-8B-Instruct     (~16 GB VRAM, module load llama/3)
  meta-llama/Meta-Llama-3-70B-Instruct    (~140 GB, needs multi-GPU)
  mistralai/Mistral-7B-Instruct-v0.3      (~14 GB, module load nlp/1.3)
  google/gemma-2-9b-it                    (~18 GB, module load gemma_llm)

For gated models you need: huggingface-cli login --token <your_token>
or set HF_TOKEN in your SLURM environment.
"""

import argparse
import json
import os
import sys
import time
import re
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CORPUS_PATH = 'processed_data/zone_text_corpora_deduped.csv'
ZON_CSV_PATH = 'processed_data/tpazoning_clean.csv'
FEAT_PATH = 'processed_data/zone_features_combined.csv'

RANDOM_STATE = 42

# 10 theory-driven dimensions from urban planning literature (Cervero & Kockelman
# 1997, Ewing & Cervero 2010, Shoup 2005, Talen 2013). Chosen a priori so each
# feature has a planning-interpretable name, unlike PCA dims that need post-hoc labeling.
# Each tuple: (short_name, display_name, rubric_description)
DIMENSIONS = [
    ('use_mixing',
     'Use Mixing & Diversity',
     'To what extent does the zoning code permit a mix of land uses '
     '(residential, commercial, office, retail) within the same zone '
     'or on the same parcel? Score 0 if strictly single-use (e.g., '
     'single-family residential only). Score 10 if the code explicitly '
     'permits or requires mixed-use development with ground-floor retail.'),

    ('density_permissions',
     'Density Permissions',
     'What residential or employment density does the code permit? '
     'Consider maximum dwelling units per acre, FAR, building coverage, '
     'and minimum lot area. Score 0 for very low density (e.g., RS-150 '
     'with 15,000 sq ft lots). Score 10 for highest density (e.g., CBD '
     'with no density cap and no maximum height).'),

    ('ped_street_interface',
     'Pedestrian-Street Interface',
     'How does the code regulate the relationship between buildings and '
     'the street? Consider front setback requirements, build-to lines, '
     'and whether buildings must address the street. Score 0 if large '
     'setbacks create auto-oriented development. Score 10 if zero-setback '
     'or build-to-line requirements create a continuous street wall.'),

    ('parking_intensity',
     'Parking Requirements',
     'How much off-street parking does the code require? Consider '
     'minimum parking ratios, surface lot allowances, and whether '
     'the code has parking maximums or reductions for transit proximity. '
     'Score 0 if no minimum parking (most pedestrian-friendly). Score 10 '
     'if the code requires high parking ratios with surface lots '
     '(most auto-oriented). NOTE: this dimension is reverse-scored — '
     'higher values mean MORE parking, which is LESS pedestrian-friendly.'),

    ('form_based_design',
     'Form-Based Design Standards',
     'Does the code regulate building form, massing, and design rather '
     'than (or in addition to) land use? Consider transparency/fenestration '
     'requirements, articulation standards, façade treatment, street-level '
     'activation, and building height stepbacks. Score 0 if purely Euclidean '
     '(use-based only). Score 10 if the code is fully form-based with '
     'detailed design standards for the public realm.'),

    ('use_flexibility',
     'Use Flexibility & Conditional Uses',
     'How flexible is the code regarding permitted uses? Consider the '
     'breadth of by-right uses, availability of conditional/special uses, '
     'accessory dwelling units, live-work units, and home occupations. '
     'Score 0 if the zone permits only one or two use categories with no '
     'conditional uses. Score 10 if the zone permits a wide range of uses '
     'by right with generous conditional use provisions.'),

    ('transit_orientation',
     'Transit Orientation',
     'Does the code reference or incentivize transit-oriented development? '
     'Consider parking reductions near transit, density bonuses for transit '
     'proximity, TOD overlay provisions, and multimodal connectivity '
     'requirements. Score 0 if there are no transit-related provisions. '
     'Score 10 if the code has comprehensive TOD standards.'),

    ('open_space_green',
     'Open Space & Green Infrastructure',
     'What open space, landscaping, and green infrastructure does the code '
     'require? Consider minimum open space ratios, tree canopy requirements, '
     'park dedications, and landscaping buffers. Score 0 if no open space '
     'provisions. Score 10 if the code has detailed green infrastructure '
     'requirements with specific area and canopy minimums.'),

    ('setback_building_placement',
     'Setback & Building Placement',
     'What is the overall setback regime? Consider front, side, and rear '
     'setbacks together with any build-to-zone or build-to-line provisions. '
     'Score 0 if setbacks are large on all sides (auto-oriented suburban). '
     'Score 10 if setbacks are minimal or zero with build-to requirements '
     '(urban pedestrian-oriented).'),

    ('transparency_activation',
     'Ground-Floor Transparency & Activation',
     'Does the code require ground-floor transparency (window percentage), '
     'active ground-floor uses, or other provisions for pedestrian-level '
     'engagement? Score 0 if no transparency or activation requirements. '
     'Score 10 if the code mandates high transparency percentages (>50%) '
     'and requires active ground-floor uses.'),
]

N_DIMS = len(DIMENSIONS)
DIM_NAMES = [d[0] for d in DIMENSIONS]


# ─────────────────────────────────────────────────────────────
# ZONE CONTEXT (same as all other scripts)
# ─────────────────────────────────────────────────────────────
ALL_ZONES = [
    'RS-150','RS-100','RS-75','RS-60','RS-50',
    'RM-12','RM-16','RM-18','RM-24','RM-35','RM-50','RM-75',
    'RO-1','RO','OP-1','OP','CN','CG','CI','IG','IH',
    'PD-A','PD',
    'CBD-1','CBD-2',
    'SH-RS-A','SH-RS','SH-RM','SH-RO','SH-CN','SH-CG','SH-CI','SH-PD',
    'NMU-35','NMU-24','NMU-16',
    'M-AP-4','M-AP-3','M-AP-2','M-AP-1',
    'YC-9','YC-8','YC-7','YC-6','YC-5','YC-4','YC-3','YC-2','YC-1',
    'CD-3','CD-2','CD-1',
    'CU','UC','AS-1','RM-24/18',
]

ZONE_CONTEXT = {}
_RES = [
    'RS-150','RS-100','RS-75','RS-60','RS-50',
    'RM-12','RM-16','RM-18','RM-24','RM-35','RM-50','RM-75',
    'RO','RO-1','SH-RS','SH-RS-A','SH-RM','SH-RO','SH-PD',
    'YC-2','YC-4','YC-8','YC-9','RM-24/18',
]
_NONRES = [
    'CG','CI','CN','OP','OP-1','IG','IH','CBD-1','CBD-2',
    'CD-1','CD-2','CD-3','NMU-35','NMU-24','NMU-16',
    'SH-CG','SH-CI','SH-CN','YC-1','YC-3','YC-5','YC-6','YC-7',
]
for z in ALL_ZONES:
    if z in _RES:
        ZONE_CONTEXT[z] = 'Residential'
    elif z in _NONRES:
        ZONE_CONTEXT[z] = 'Non-Residential'
    else:
        ZONE_CONTEXT[z] = 'Other/PD'


# ─────────────────────────────────────────────────────────────
# PROMPT CONSTRUCTION
# ─────────────────────────────────────────────────────────────

def build_system_prompt():
    """System prompt for the LLM scorer."""
    return (
        "You are an expert urban planner and zoning code analyst. You will "
        "be given the full text corpus for a single zoning district from "
        "the City of Tampa, Florida's Land Development Code (Chapter 27). "
        "Your task is to score this zone on exactly 10 planning-theoretic "
        "dimensions.\n\n"
        # 1-5 scale (mapped to 0-10 integers): coarse enough to be interpretable,
        # fine enough to discriminate across Tampa's 56 zone classes.
        "For each dimension, you must provide:\n"
        "  1. A numeric score from 0 to 10 (integers only)\n"
        # Requiring rationales forces grounded scoring -- prevents arbitrary numbers
        # and lets us audit whether the LLM is citing real provisions.
        "  2. A brief rationale (1-2 sentences) citing specific provisions "
        "from the text\n\n"
        "If the corpus text does not contain enough information to score a "
        "dimension confidently, score it based on what you can infer from "
        "the zone classification and any dimensional standards present, "
        "and note the uncertainty in your rationale.\n\n"
        # JSON output format ensures structured, machine-parseable responses.
        # Free-text output would require fragile regex extraction of scores.
        "Respond ONLY with a valid JSON object, no markdown fences, "
        "no preamble. The JSON must have this structure:\n"
        '{\n'
        '  "zone_class": "<zone code>",\n'
        '  "scores": [\n'
        '    {"dimension": "<dim_name>", "score": <int 0-10>, '
        '"rationale": "<1-2 sentences>"},\n'
        '    ...\n'
        '  ]\n'
        '}\n'
    )


def build_user_prompt(zone_class, corpus_text, max_chars=12000):
    """User message with zone corpus and dimension rubrics."""
    dim_block = ""
    for i, (short, display, rubric) in enumerate(DIMENSIONS):
        dim_block += f"\n  {i+1}. {display} (dimension name: \"{short}\")\n"
        dim_block += f"     {rubric}\n"

    # Truncate corpus to avoid token limits
    truncated = corpus_text[:max_chars]
    if len(corpus_text) > max_chars:
        truncated += "\n\n[... corpus truncated for length ...]"

    return (
        f"Zone class: {zone_class}\n"
        f"Zone context: {ZONE_CONTEXT.get(zone_class, 'Unknown')}\n\n"
        f"=== SCORING DIMENSIONS ===\n{dim_block}\n"
        f"=== ZONING CODE CORPUS ===\n{truncated}\n\n"
        f"Score this zone on all 10 dimensions. Respond with JSON only."
    )


# ─────────────────────────────────────────────────────────────
# API BACKENDS
# ─────────────────────────────────────────────────────────────

def score_anthropic(zone_class, corpus_text, model='claude-sonnet-4-20250514'):
    """Score via Anthropic API."""
    import anthropic
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        system=build_system_prompt(),
        messages=[
            {"role": "user", "content": build_user_prompt(zone_class, corpus_text)}
        ],
    )
    text = response.content[0].text.strip()
    return parse_llm_response(text, zone_class)


def score_openai(zone_class, corpus_text, model='gpt-4o'):
    """Score via OpenAI API. GPT-4o replaced local Mistral for the final paper
    version -- higher scoring consistency and better calibration across zones."""
    import openai
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=model,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": build_user_prompt(zone_class, corpus_text)},
        ],
    )
    text = response.choices[0].message.content.strip()
    return parse_llm_response(text, zone_class)


def score_mock(zone_class, corpus_text):
    """
    Deterministic heuristic scorer for testing without API access.
    Uses keyword counting as a rough proxy. Not for production.
    """
    text_lower = corpus_text.lower()

    def kw_score(keywords, max_hits=10):
        count = sum(text_lower.count(kw) for kw in keywords)
        return min(10, int(count / max_hits * 10))

    scores = {}
    scores['use_mixing'] = kw_score(
        ['mixed use', 'mixed-use', 'commercial and residential',
         'retail', 'office', 'live-work', 'live/work'], max_hits=15)
    scores['density_permissions'] = kw_score(
        ['units per acre', 'dwelling units', 'far ', 'floor area ratio',
         'no maximum', 'unlimited'], max_hits=8)
    scores['ped_street_interface'] = kw_score(
        ['build-to', 'build to line', 'zero setback', 'street wall',
         'frontage', 'pedestrian'], max_hits=10)
    scores['parking_intensity'] = kw_score(
        ['parking space', 'parking ratio', 'minimum parking',
         'off-street parking', 'surface lot'], max_hits=12)
    scores['form_based_design'] = kw_score(
        ['transparency', 'fenestration', 'articulation', 'facade',
         'building form', 'stepback', 'design standard'], max_hits=8)
    scores['use_flexibility'] = kw_score(
        ['conditional use', 'special use', 'accessory', 'home occupation',
         'by right', 'permitted use'], max_hits=12)
    scores['transit_orientation'] = kw_score(
        ['transit', 'bus', 'streetcar', 'brt', 'tod',
         'multimodal', 'transit-oriented'], max_hits=6)
    scores['open_space_green'] = kw_score(
        ['open space', 'landscape', 'tree', 'canopy', 'park',
         'green', 'buffer'], max_hits=10)
    scores['setback_building_placement'] = 10 - kw_score(
        ['setback', 'minimum setback', 'front setback',
         'rear setback', 'side setback'], max_hits=15)
    scores['transparency_activation'] = kw_score(
        ['transparency', 'ground floor', 'active use', 'window',
         'percent minimum', 'storefront'], max_hits=8)

    # Clamp
    for k in scores:
        scores[k] = max(0, min(10, scores[k]))

    result = []
    for dim_name, _, _ in DIMENSIONS:
        result.append({
            'dimension': dim_name,
            'score': scores[dim_name],
            'rationale': f'Mock heuristic score based on keyword frequency.'
        })
    return result


# ─────────────────────────────────────────────────────────────
# LOCAL HUGGINGFACE BACKEND (HiPerGator GPU)
# ─────────────────────────────────────────────────────────────

# Global model/tokenizer — loaded once, reused for all zones
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None
_LOCAL_CONFIG = {}

def load_local_model(model_name, max_new_tokens=2048, temperature=0.3,
                     quantize=None):
    """
    Load a HuggingFace causal LM for local inference.

    Args:
        quantize: None for fp16, '4bit' for 4-bit quantization via
                  bitsandbytes. 4-bit cuts VRAM ~4× (8B model → ~8 GB),
                  making it fit on an L4 (24 GB) with room to spare.

    Tested models (HiPerGator modules):
      - module load llama/3  → meta-llama/Meta-Llama-3-8B-Instruct
      - module load nlp/1.3  → mistralai/Mistral-7B-Instruct-v0.3
      - module load gemma_llm → google/gemma-2-9b-it

    For gated models (Llama, Gemma), you need:
      huggingface-cli login --token <your_token>
    or set HF_TOKEN in your environment.
    """
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_CONFIG
    if _LOCAL_MODEL is not None:
        return  # already loaded

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f'\n  Loading local model: {model_name}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU count: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'    GPU {i}: {torch.cuda.get_device_name(i)} '
                  f'({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')

    device_map = 'auto' if torch.cuda.is_available() else 'cpu'

    # Build model kwargs based on quantization setting
    model_kwargs = dict(
        device_map=device_map,
        trust_remote_code=True,
    )

    # 4-bit NF4 quantization: enables single-GPU inference on an L4 (24 GB).
    # 8B model goes from ~16 GB to ~4 GB VRAM.
    if quantize == '4bit':
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs['quantization_config'] = bnb_config
            print(f'  Quantization: 4-bit NF4 (bitsandbytes)')
        except ImportError:
            print('  WARNING: bitsandbytes not available, falling back to fp16')
            print('  Install with: pip install bitsandbytes --break-system-packages')
            model_kwargs['torch_dtype'] = torch.float16
    else:
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs['torch_dtype'] = dtype
        print(f'  Quantization: none (dtype={model_kwargs["torch_dtype"]})')

    print(f'  device_map: {device_map}')
    print(f'  Loading tokenizer...')
    _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)

    # Set pad token if missing (common for Llama)
    if _LOCAL_TOKENIZER.pad_token is None:
        _LOCAL_TOKENIZER.pad_token = _LOCAL_TOKENIZER.eos_token

    print(f'  Loading model weights (this may take a few minutes)...')
    _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name, **model_kwargs)
    _LOCAL_MODEL.eval()

    _LOCAL_CONFIG = {
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'model_name': model_name,
        'quantize': quantize,
    }

    n_params = sum(p.numel() for p in _LOCAL_MODEL.parameters()) / 1e9
    print(f'  Model loaded: {n_params:.1f}B parameters')
    print(f'  Generation config: max_new_tokens={max_new_tokens}, '
          f'temperature={temperature}')


def score_local(zone_class, corpus_text, corpus_chars=6000):
    """
    Score via local HuggingFace model on GPU.

    Uses chat template if available (Llama-3-Instruct, Mistral-Instruct,
    Gemma-it all support this). Falls back to raw prompt concatenation.
    """
    import torch
    global _LOCAL_MODEL, _LOCAL_TOKENIZER, _LOCAL_CONFIG

    system_msg = build_system_prompt()
    # Tighter truncation for local models with smaller context windows
    user_msg = build_user_prompt(zone_class, corpus_text[:corpus_chars])

    # Build prompt using chat template if available
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        # Most instruct models support apply_chat_template
        prompt_text = _LOCAL_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: simple concatenation
        prompt_text = (
            f"### System:\n{system_msg}\n\n"
            f"### User:\n{user_msg}\n\n"
            f"### Assistant:\n"
        )

    inputs = _LOCAL_TOKENIZER(
        prompt_text, return_tensors='pt', truncation=True,
        max_length=8192  # leave room for generation
    ).to("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        output_ids = _LOCAL_MODEL.generate(
            **inputs,
            max_new_tokens=_LOCAL_CONFIG['max_new_tokens'],
            temperature=_LOCAL_CONFIG['temperature'],
            do_sample=_LOCAL_CONFIG['temperature'] > 0,
            top_p=0.9 if _LOCAL_CONFIG['temperature'] > 0 else 1.0,
            pad_token_id=_LOCAL_TOKENIZER.pad_token_id,
        )

    # Decode only the new tokens (skip the prompt)
    new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
    text = _LOCAL_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()

    return parse_llm_response(text, zone_class)


def parse_llm_response(text, zone_class):
    """Parse JSON response from LLM. Returns list of score dicts."""
    # Strip markdown fences if present
    text = re.sub(r'^```json\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        print(f'  WARNING: JSON parse failed for {zone_class}: {e}')
        print(f'  Raw response (first 500 chars): {text[:500]}')
        # Return NaN scores
        return [{'dimension': d[0], 'score': np.nan,
                 'rationale': 'PARSE_FAILED'} for d in DIMENSIONS]

    scores = data.get('scores', [])

    # Validate we got all dimensions
    got_dims = {s['dimension'] for s in scores}
    for dim_name, _, _ in DIMENSIONS:
        if dim_name not in got_dims:
            scores.append({
                'dimension': dim_name,
                'score': np.nan,
                'rationale': 'MISSING_FROM_RESPONSE'
            })

    return scores


# ─────────────────────────────────────────────────────────────
# MULTI-PASS SCORING (robustness)
# ─────────────────────────────────────────────────────────────

def score_zone_multi_pass(zone_class, corpus_text, backend_fn, n_passes=3):
    """
    Score a zone n_passes times and take the median.
    # WHY multi-pass: LLM scores are stochastic; median across passes reduces
    # noise without sacrificing the ordinal signal. n=3 is enough for stable
    # medians on a 0-10 scale (odd count avoids ties).
    For the mock backend, one pass suffices (deterministic).
    """
    all_scores = {d[0]: [] for d in DIMENSIONS}

    for p in range(n_passes):
        result = backend_fn(zone_class, corpus_text)
        for item in result:
            dim = item['dimension']
            if dim in all_scores and not np.isnan(item.get('score', np.nan)):
                all_scores[dim].append(item['score'])

    # Median aggregation
    final = {}
    for dim_name in DIM_NAMES:
        vals = all_scores[dim_name]
        if vals:
            final[dim_name] = int(np.median(vals))
        else:
            final[dim_name] = np.nan

    return final


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='LLM-scored theory-driven zoning features')
    parser.add_argument('--backend', choices=['local', 'anthropic', 'openai', 'mock'],
                        default='local',
                        help='LLM backend (default: local)')
    parser.add_argument('--passes', type=int, default=3,
                        help='Number of scoring passes per zone (default: 3)')
    parser.add_argument('--model', type=str, default=None,
                        help='Override model name. For local: HuggingFace model ID '
                             '(default: meta-llama/Meta-Llama-3-8B-Instruct). '
                             'For anthropic: claude-sonnet-4-20250514. For openai: gpt-4o.')
    parser.add_argument('--corpus', type=str, default=CORPUS_PATH,
                        help='Path to zone_text_corpora.csv')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to partial results CSV to resume from')
    parser.add_argument('--max-new-tokens', type=int, default=2048,
                        help='Max new tokens for local generation (default: 2048)')
    parser.add_argument('--temperature', type=float, default=0.3,
                        help='Sampling temperature for local model (default: 0.3)')
    parser.add_argument('--corpus-chars', type=int, default=6000,
                        help='Max corpus chars per zone sent to LLM (default: 6000). '
                             'Reduce for smaller-context models.')
    parser.add_argument('--quantize', choices=['4bit', 'none'], default=None,
                        help='Quantization for local backend. 4bit cuts VRAM ~4× '
                             '(fits 8B model on 24GB L4). Default: none.')
    args = parser.parse_args()

    # ── Load corpus
    print('=' * 65)
    print('  Tampa Zoning — LLM-Scored Theory-Driven Features')
    print(f'  Backend: {args.backend}  |  Passes: {args.passes}')
    print('=' * 65)

    corpus_df = pd.read_csv(args.corpus)
    print(f'Loaded {len(corpus_df)} zone corpora from {args.corpus}')
    corpus_map = dict(zip(corpus_df['zone_class'], corpus_df['corpus']))

    # ── Resume support
    completed = set()
    partial_rows = []
    if args.resume and os.path.exists(args.resume):
        partial = pd.read_csv(args.resume)
        completed = set(partial['zone_class'].tolist())
        partial_rows = partial.to_dict('records')
        print(f'Resuming: {len(completed)} zones already scored')

    # ── Select backend function
    if args.backend == 'local':
        model = args.model or 'meta-llama/Meta-Llama-3-8B-Instruct'
        quantize = args.quantize if args.quantize != 'none' else None
        load_local_model(model, max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, quantize=quantize)
        corpus_chars = args.corpus_chars
        backend_fn = lambda zc, ct: score_local(zc, ct, corpus_chars=corpus_chars)
    elif args.backend == 'anthropic':
        model = args.model or 'claude-sonnet-4-20250514'
        backend_fn = lambda zc, ct: score_anthropic(zc, ct, model=model)
    elif args.backend == 'openai':
        model = args.model or 'gpt-4o'
        backend_fn = lambda zc, ct: score_openai(zc, ct, model=model)
    else:
        backend_fn = score_mock
        args.passes = 1  # deterministic, one pass suffices

    # ── Save the exact prompt for reproducibility
    sample_zone = ALL_ZONES[0]
    sample_corpus = corpus_map.get(sample_zone, '[no corpus]')
    prompt_log = (
        "=== SYSTEM PROMPT ===\n" + build_system_prompt() + "\n\n"
        "=== USER PROMPT (example: " + sample_zone + ") ===\n"
        + build_user_prompt(sample_zone, sample_corpus[:2000])
    )
    with open('zone_llm_features_prompt.txt', 'w') as f:
        f.write(prompt_log)
    print('Saved: zone_llm_features_prompt.txt')

    # ── Score all zones
    print(f'\nScoring {len(ALL_ZONES)} zones...')
    rows = list(partial_rows)  # start with any resumed rows

    for i, zone in enumerate(ALL_ZONES):
        if zone in completed:
            continue

        corpus_text = corpus_map.get(zone, '')
        if not corpus_text:
            print(f'  [{i+1:>2}/{len(ALL_ZONES)}] {zone:<10}  SKIPPED (no corpus)')
            row = {'zone_class': zone, 'context': ZONE_CONTEXT[zone]}
            for d in DIM_NAMES:
                row[d] = np.nan
                row[f'{d}_rationale'] = 'NO_CORPUS'
            rows.append(row)
            continue

        print(f'  [{i+1:>2}/{len(ALL_ZONES)}] {zone:<10}  ', end='', flush=True)
        t0 = time.time()

        try:
            # Single-pass for raw output (keep rationales from first pass)
            first_pass = backend_fn(zone, corpus_text)
            rationales = {item['dimension']: item.get('rationale', '')
                         for item in first_pass}

            # Multi-pass for robust scores
            if args.passes > 1:
                scores = score_zone_multi_pass(
                    zone, corpus_text, backend_fn, n_passes=args.passes)
            else:
                scores = {}
                for item in first_pass:
                    scores[item['dimension']] = item.get('score', np.nan)

        except Exception as e:
            print(f'ERROR: {e}')
            scores = {d: np.nan for d in DIM_NAMES}
            rationales = {d: f'ERROR: {e}' for d in DIM_NAMES}

        elapsed = time.time() - t0
        score_str = ' '.join(f'{scores.get(d, 0):>2}' for d in DIM_NAMES)
        print(f'{score_str}  ({elapsed:.1f}s)')

        row = {'zone_class': zone, 'context': ZONE_CONTEXT[zone]}
        for d in DIM_NAMES:
            row[d] = scores.get(d, np.nan)
            row[f'{d}_rationale'] = rationales.get(d, '')
        rows.append(row)

        # Save intermediate results after each zone (crash recovery)
        pd.DataFrame(rows).to_csv('zone_llm_features_raw_partial.csv', index=False)

        # Rate limiting
        if args.backend != 'mock' and i < len(ALL_ZONES) - 1:
            time.sleep(1.0)

    # ── Build output DataFrames
    raw_df = pd.DataFrame(rows)
    # Ensure zone ordering matches ALL_ZONES
    zone_order = pd.DataFrame({'zone_class': ALL_ZONES, '_order': range(len(ALL_ZONES))})
    raw_df = (raw_df.merge(zone_order, on='zone_class', how='left')
              .sort_values('_order')
              .drop(columns='_order')
              .reset_index(drop=True))

    # Save raw (with rationales)
    raw_df.to_csv('zone_llm_features_raw.csv', index=False)
    print(f'\nSaved: zone_llm_features_raw.csv  ({len(raw_df)} zones × {N_DIMS} dims)')

    # ── Build embeddings-format output for DML pipeline
    # Column naming: feat_000 ... feat_009 (matches emb_000 convention)
    emb_cols = [f'feat_{i:03d}' for i in range(N_DIMS)]
    emb_df = pd.DataFrame()
    emb_df['zone_class'] = raw_df['zone_class']
    emb_df['context'] = raw_df['context']
    for i, dim_name in enumerate(DIM_NAMES):
        emb_df[emb_cols[i]] = raw_df[dim_name].astype(float)

    # Standardise to zero-mean unit-variance (matching sentence-transformer scale)
    for col in emb_cols:
        vals = emb_df[col].values
        mu, sd = np.nanmean(vals), np.nanstd(vals)
        if sd > 0:
            emb_df[col] = (vals - mu) / sd
        else:
            emb_df[col] = 0.0

    emb_df.to_csv('zone_embeddings_llm_scored.csv', index=False)
    print(f'Saved: zone_embeddings_llm_scored.csv  ({len(emb_df)} zones × {N_DIMS} dims)')

    # ── Also save a dimension name mapping
    dim_map = pd.DataFrame({
        'col': emb_cols,
        'dimension': DIM_NAMES,
        'display_name': [d[1] for d in DIMENSIONS],
        'rubric': [d[2] for d in DIMENSIONS],
    })
    dim_map.to_csv('zone_llm_dimension_map.csv', index=False)
    print(f'Saved: zone_llm_dimension_map.csv')

    # Clean up partial file
    if os.path.exists('zone_llm_features_raw_partial.csv'):
        os.remove('zone_llm_features_raw_partial.csv')

    # ── Summary statistics
    print('\n── Dimension Summary ──')
    print(f'  {"Dimension":<30} {"Mean":>6} {"Std":>6} {"Min":>4} {"Max":>4}')
    print('  ' + '─' * 54)
    for dim_name in DIM_NAMES:
        vals = raw_df[dim_name].dropna()
        if len(vals) > 0:
            print(f'  {dim_name:<30} {vals.mean():>6.1f} {vals.std():>6.1f} '
                  f'{vals.min():>4.0f} {vals.max():>4.0f}')

    print(f'\n✓ LLM feature extraction complete ({args.backend}).')


if __name__ == '__main__':
    main()