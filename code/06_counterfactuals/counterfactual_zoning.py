"""
Tampa Zoning — Generative Counterfactual Zoning Texts
======================================================
Uses a generative LLM to rewrite existing zoning code text to be
more active-travel-promoting, then evaluates the counterfactual
through the trained DML causal model.

Pipeline:
  1. Select target zones (car-promoting from Estimand C)
  2. Feed LLM: original corpus + DML findings + rewrite instruction
  3. LLM generates a counterfactual zoning text
  4. Embed counterfactual with SAME sentence-transformer + PCA
     (PCA fitted on 56 original zones; counterfactual projected in)
  5. Compute Estimand C causal score: s = e · θ
  6. Report Δs = s_counterfactual - s_original

Backends for text generation:
  'local'     — HuggingFace model on GPU (HiPerGator)
  'anthropic' — Claude API
  'openai'    — GPT-4o API
  'mock'      — Template-based rewrite (testing, no GPU/API)

Usage:
  python counterfactual_zoning.py --backend mock
  python counterfactual_zoning.py --backend local --model mistralai/Mistral-7B-Instruct-v0.3
  python counterfactual_zoning.py --backend anthropic

Requires in working directory:
  zone_text_corpora.csv
  zone_embeddings_minilm.csv
  dml_backend_comparison_merged.csv
"""

import argparse
import json
import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
CORPUS_PATH       = 'zone_text_corpora.csv'
EMB_PATH          = 'zone_embeddings_minilm.csv'
DML_RESULTS_PATH  = 'dml_backend_comparison_merged.csv'
# Must use MiniLM -- counterfactual deltas are only valid against the same
# embedding space used to estimate theta in the DML stage.
ST_MODEL_NAME     = 'all-MiniLM-L6-v2'
EMBEDDING_DIM     = 32
RANDOM_STATE      = 42

# Target zones for counterfactual rewriting.
TARGET_ZONES = [
    'RS-60',     # single-family, large setbacks, no mixed use
    'CG',        # general commercial, auto-oriented strip
    'NMU-24',    # mixed-use overlay — counterintuitively car-promoting in DML
    'IG',        # general industrial
    'RS-150', 'RS-100', 'RS-75', 'RS-50',
    'RM-12', 'RM-16', 'RM-18', 'RM-24', 'RM-35', 'RM-50', 'RM-75',
    'RO-1', 'RO', 'OP-1', 'OP', 'CN', 'CI', 'IH',
    'PD-A', 'PD', 'CBD-1', 'CBD-2',
    'SH-RS-A', 'SH-RS', 'SH-RM', 'SH-RO', 'SH-CN', 'SH-CG', 'SH-CI', 'SH-PD',
    'NMU-35', 'NMU-16',
    'M-AP-4', 'M-AP-3', 'M-AP-2', 'M-AP-1',
    'YC-9', 'YC-8', 'YC-7', 'YC-6', 'YC-5', 'YC-4', 'YC-3', 'YC-2', 'YC-1',
    'CD-3', 'CD-2', 'CD-1',
    'CU', 'UC', 'AS-1', 'RM-24/18',
]

# Two variants bracket the reform range: standard = moderate (TOD, -25-50% parking,
# ped connectivity); aggressive = full form-based conversion (zero parking mins,
# zero setbacks, 60% transparency). Lets us measure dose-response.
N_VARIANTS = 2   # 'standard' and 'aggressive'


# ─────────────────────────────────────────────────────────────
# REWRITE PROMPTS
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert urban planner specializing in form-based codes \
and active transportation policy. You will be given the full text of a zoning \
district from Tampa, Florida's Land Development Code (Chapter 27), along with \
findings from a causal analysis of how zoning text affects active travel rates.

Your task is to rewrite the zoning text to make it MORE supportive of active \
travel (walking and cycling), while keeping the rewrite realistic and legally \
implementable. The rewrite should:
  1. Maintain the same general zone classification and purpose
  2. Modify specific provisions identified as causally important
  3. Add provisions that are absent but would promote active travel
  4. Use language consistent with Tampa's existing code style
  5. Be a complete replacement text, not tracked changes

Respond with ONLY the rewritten zoning text. No preamble, no explanation, \
no markdown fences. Just the code text as it would appear in the LDC."""


DML_FINDINGS = """CAUSAL ANALYSIS FINDINGS (from Double Machine Learning, 50 cross-fits):

The following regulatory text features have the strongest causal effects on
active travel rates, after controlling for demographic self-selection:

1. TRANSIT ORIENTATION (strongest positive effect, θ = +0.13):
   - References to transit stops, BRT, bus access, multimodal connectivity
   - Parking reductions near transit, density bonuses for transit proximity
   - Zones with this language causally promote walking/cycling

2. GROUND-FLOOR TRANSPARENCY & ACTIVATION (significant, θ = -0.09):
   - Transparency minimums, active ground-floor uses, storefront requirements
   - Counterintuitively NEGATIVE in our model — zones with these requirements
     attract walkability-seekers, so the text itself shows negative residual
   - Still include these provisions; they indicate walkable intent

3. PARKING REQUIREMENTS (large absolute effect):
   - Minimum parking ratios are the strongest auto-orientation signal
   - Reducing or eliminating minimums has the largest predicted AT improvement
   - Adding parking maximums or shared parking provisions helps

4. SETBACK & BUILDING PLACEMENT:
   - Large setbacks create auto-oriented development
   - Build-to lines, zero setbacks, street wall requirements promote AT
   - Zones with 25+ ft front setbacks score as car-promoting

5. USE MIXING:
   - Permitting residential + commercial on same parcel promotes AT
   - Ground-floor retail requirements, live-work permissions
   - Exclusively single-use zones are strongly car-promoting"""


def build_user_prompt(zone_class, corpus_text, variant='standard'):
    """Build the rewrite prompt."""
    intensity = {
        'standard': (
            "Make MODERATE improvements to promote active travel. "
            "Add transit-oriented provisions, reduce parking minimums by 25-50%, "
            "add pedestrian connectivity requirements, and permit mixed uses where "
            "currently prohibited. Keep changes realistic for a near-term code update."
        ),
        'aggressive': (
            "Make AGGRESSIVE improvements to maximize active travel promotion. "
            "Eliminate minimum parking requirements entirely (or set maximums), "
            "require build-to lines with zero front setbacks, mandate ground-floor "
            "retail/active uses on all street-facing facades, require transit impact "
            "assessments, add bicycle parking minimums, require sidewalk construction, "
            "and permit the full range of mixed uses by right. This represents a "
            "complete conversion to a form-based, transit-oriented code."
        ),
    }

    # Truncate corpus for local models
    truncated = corpus_text[:8000]
    if len(corpus_text) > 8000:
        truncated += "\n[... truncated ...]"

    return (
        f"Zone class: {zone_class}\n\n"
        f"{DML_FINDINGS}\n\n"
        f"REWRITE INSTRUCTION:\n{intensity[variant]}\n\n"
        f"=== ORIGINAL ZONING CODE TEXT ===\n{truncated}\n\n"
        f"Rewrite this zone's code to be more active-travel-promoting. "
        f"Output ONLY the revised code text."
    )


# ─────────────────────────────────────────────────────────────
# GENERATION BACKENDS
# ─────────────────────────────────────────────────────────────

def generate_anthropic(zone_class, corpus_text, variant, model='claude-sonnet-4-20250514'):
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(zone_class, corpus_text, variant)}],
    )
    return response.content[0].text.strip()


def generate_openai(zone_class, corpus_text, variant, model='gpt-4o'):
    import openai
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=4000,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(zone_class, corpus_text, variant)},
        ],
    )
    return response.choices[0].message.content.strip()


# Global model holder for local backend
_LOCAL_MODEL = None
_LOCAL_TOKENIZER = None

# Mistral-7B for generation: runs locally on HiPerGator with no API cost for
# 112 generations (56 zones x 2 variants). GPT-4o is used only for scoring.
def generate_local(zone_class, corpus_text, variant, model_name='mistralai/Mistral-7B-Instruct-v0.3',
                   quantize='4bit', max_new_tokens=4000):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    global _LOCAL_MODEL, _LOCAL_TOKENIZER

    if _LOCAL_MODEL is None:
        print(f'  Loading model: {model_name}...')
        model_kwargs = dict(device_map='auto', trust_remote_code=True)
        # 4-bit NF4 quantization enables single-GPU inference on A100 hardware
        if quantize == '4bit':
            try:
                from transformers import BitsAndBytesConfig
                model_kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            except ImportError:
                model_kwargs['torch_dtype'] = torch.float16
        else:
            model_kwargs['torch_dtype'] = torch.float16

        _LOCAL_TOKENIZER = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if _LOCAL_TOKENIZER.pad_token is None:
            _LOCAL_TOKENIZER.pad_token = _LOCAL_TOKENIZER.eos_token
        _LOCAL_MODEL = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        _LOCAL_MODEL.eval()
        print(f'  Model loaded.')

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(zone_class, corpus_text, variant)},
    ]

    try:
        prompt_text = _LOCAL_TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt_text = f"### System:\n{SYSTEM_PROMPT}\n\n### User:\n{build_user_prompt(zone_class, corpus_text, variant)}\n\n### Assistant:\n"

    inputs = _LOCAL_TOKENIZER(prompt_text, return_tensors='pt', truncation=True, max_length=8192)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = _LOCAL_MODEL.generate(
            **inputs, max_new_tokens=max_new_tokens,
            # Temperature 0.7 balances regulatory plausibility with meaningful
            # variation. Lower = too conservative, higher = incoherent.
            temperature=0.7, do_sample=True, top_p=0.9,
            pad_token_id=_LOCAL_TOKENIZER.pad_token_id)

    new_tokens = output_ids[0, inputs['input_ids'].shape[1]:]
    return _LOCAL_TOKENIZER.decode(new_tokens, skip_special_tokens=True).strip()


def generate_mock(zone_class, corpus_text, variant):
    """Template-based mock rewrite for testing pipeline without GPU/API."""
    additions = {
        'standard': [
            "All new development within 0.25 miles of a transit stop shall receive a 25% reduction in minimum parking requirements.",
            "Sidewalks of minimum 5-foot width shall be required along all public street frontages.",
            "Mixed-use development combining residential and neighborhood commercial uses is permitted by right.",
            "Bicycle parking shall be provided at a minimum ratio of 1 space per 2,000 square feet of commercial floor area.",
            "Maximum front setback of 15 feet from the property line shall apply to encourage pedestrian-oriented development.",
        ],
        'aggressive': [
            "No minimum off-street parking is required. Maximum parking shall not exceed 1.5 spaces per dwelling unit.",
            "A build-to line of 0-5 feet from the front property line is required for all new construction.",
            "Ground-floor transparency minimum of 60% along all street-facing facades. Active uses (retail, restaurant, lobby) required on ground floor.",
            "Transit impact assessment required for all developments exceeding 10,000 square feet.",
            "All uses permitted in the NMU-24 district are permitted by right, including residential above commercial.",
            "Bicycle parking: 1 space per 500 sq ft commercial, 1 space per dwelling unit. Shower facilities required for commercial buildings over 20,000 sq ft.",
            "Street trees required at 30-foot intervals. Pedestrian-scale lighting required along all frontages.",
            "No drive-through facilities permitted. No auto-oriented uses (car wash, gas station) permitted.",
        ],
    }

    add_text = "\n".join(f"Section 27-XX.{i+1}. {provision}"
                         for i, provision in enumerate(additions[variant]))

    lines = corpus_text.split('\n')
    insert_point = min(5, len(lines))
    modified = '\n'.join(lines[:insert_point]) + '\n\n' + \
        f"--- ACTIVE TRAVEL PROVISIONS ({variant.upper()}) ---\n" + \
        add_text + '\n\n' + '\n'.join(lines[insert_point:])

    return modified


# ─────────────────────────────────────────────────────────────
# EMBEDDING & SCORING
# ─────────────────────────────────────────────────────────────

def build_embedding_pipeline(corpus_texts, model_name=ST_MODEL_NAME, n_dims=EMBEDDING_DIM):
    """
    Fit sentence-transformer + PCA on original 56 zone corpora.
    Returns the fitted PCA, the ST model, and the original embeddings.

    PCA is fit on original zones and counterfactuals are projected into that
    same space -- fit on originals, transform counterfactual. This ensures
    deltas are measured in the same coordinate system as the DML theta.
    """
    from sentence_transformers import SentenceTransformer
    print(f'  Loading sentence-transformer: {model_name}...')
    st_model = SentenceTransformer(model_name)

    print(f'  Encoding {len(corpus_texts)} original zone corpora...')
    raw_vecs = st_model.encode(corpus_texts, show_progress_bar=False,
                                batch_size=8, normalize_embeddings=True)

    pca = PCA(n_components=n_dims, random_state=RANDOM_STATE)
    original_embs = pca.fit_transform(raw_vecs)
    var = pca.explained_variance_ratio_.sum()
    print(f'  PCA fitted: {raw_vecs.shape[1]}d → {n_dims}d (var={var:.1%})')

    return st_model, pca, original_embs


def embed_counterfactual(text, st_model, pca):
    """Embed a single counterfactual text into the fitted PCA space."""
    raw_vec = st_model.encode([text], normalize_embeddings=True)
    pca_vec = pca.transform(raw_vec)
    return pca_vec[0]


def load_theta(backend='minilm', dml_path=DML_RESULTS_PATH):
    """Load θ vector from DML results."""
    if os.path.exists(dml_path):
        df = pd.read_csv(dml_path)
        sub = df[df.backend == backend].sort_values('dimension')
        if len(sub) > 0:
            theta = sub['theta'].values
            print(f'  Loaded θ from {dml_path} (backend={backend}, {len(theta)} dims)')
            return theta

    # Fallback: uniform theta for testing
    print(f'  WARNING: No DML results found. Using uniform θ for testing.')
    emb_df = pd.read_csv(EMB_PATH)
    emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
    return np.ones(len(emb_cols)) / len(emb_cols)


def causal_score(embedding, theta):
    """Estimand C: s = e · θ
    Closed-form projection; only valid because DML theta is a linear
    coefficient on the embedding. Delta_s = (e_cf . theta) - (e_orig . theta).
    """
    d = min(len(embedding), len(theta))
    return float(embedding[:d] @ theta[:d])


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generative counterfactual zoning texts')
    parser.add_argument('--backend', choices=['local', 'anthropic', 'openai', 'mock'],
                        default='mock')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name for local/API backend')
    parser.add_argument('--zones', type=str, default=None,
                        help='Comma-separated zone classes to rewrite (default: RS-60,CG,NMU-24,IG)')
    parser.add_argument('--dml-backend', type=str, default='minilm',
                        help='Which DML backend theta to use for scoring (default: minilm)')
    parser.add_argument('--quantize', choices=['4bit', 'none'], default='4bit')
    parser.add_argument('--n-variants', type=int, default=2,
                        help='1 for standard only, 2 for standard+aggressive')
    args = parser.parse_args()

    print('=' * 65)
    print('  Tampa Zoning — Generative Counterfactual Zoning Texts')
    print(f'  Backend: {args.backend}')
    print('=' * 65)

    # ── Load corpus
    corpus_df = pd.read_csv(CORPUS_PATH)
    corpus_map = dict(zip(corpus_df['zone_class'], corpus_df['corpus']))
    print(f'Loaded {len(corpus_df)} zone corpora')

    # ── Select target zones
    target_zones = args.zones.split(',') if args.zones else TARGET_ZONES
    print(f'Target zones: {target_zones}')

    # ── Select generation backend
    if args.backend == 'local':
        model_name = args.model or 'mistralai/Mistral-7B-Instruct-v0.3'
        gen_fn = lambda zc, ct, v: generate_local(zc, ct, v, model_name=model_name,
                                                   quantize=args.quantize)
    elif args.backend == 'anthropic':
        model_name = args.model or 'claude-sonnet-4-20250514'
        gen_fn = lambda zc, ct, v: generate_anthropic(zc, ct, v, model=model_name)
    elif args.backend == 'openai':
        model_name = args.model or 'gpt-4o'
        gen_fn = lambda zc, ct, v: generate_openai(zc, ct, v, model=model_name)
    else:
        gen_fn = generate_mock

    # ── Build embedding pipeline (fit PCA on all 56 original zones)
    print('\nBuilding embedding pipeline...')
    all_zone_classes = corpus_df['zone_class'].tolist()
    all_corpus_texts = corpus_df['corpus'].tolist()
    st_model, pca, original_embs = build_embedding_pipeline(all_corpus_texts)

    # Map zone_class → original PCA embedding
    original_emb_map = {z: original_embs[i] for i, z in enumerate(all_zone_classes)}

    # ── Load θ vector
    print('\nLoading DML θ vector...')
    theta = load_theta(backend=args.dml_backend)

    # ── Generate counterfactuals
    variants = ['standard', 'aggressive'] if args.n_variants >= 2 else ['standard']
    os.makedirs('counterfactual_texts', exist_ok=True)

    results = []
    print(f'\nGenerating counterfactuals ({len(target_zones)} zones × {len(variants)} variants)...')

    for zone in target_zones:
        corpus_text = corpus_map.get(zone, '')
        if not corpus_text:
            print(f'  {zone}: SKIPPED (no corpus)')
            continue

        # Original score
        orig_emb = original_emb_map.get(zone)
        if orig_emb is None:
            print(f'  {zone}: SKIPPED (no original embedding)')
            continue
        orig_score = causal_score(orig_emb, theta)

        for variant in variants:
            print(f'  {zone} ({variant})...', end=' ', flush=True)
            t0 = time.time()

            try:
                counterfactual_text = gen_fn(zone, corpus_text, variant)
            except Exception as e:
                print(f'ERROR: {e}')
                continue

            elapsed = time.time() - t0

            # Save counterfactual text
            fname = f'counterfactual_texts/{zone.replace("/", "-")}_{variant}.txt'
            with open(fname, 'w') as f:
                f.write(counterfactual_text)

            # Embed counterfactual
            cf_emb = embed_counterfactual(counterfactual_text, st_model, pca)
            cf_score = causal_score(cf_emb, theta)
            delta = cf_score - orig_score

            # Cosine similarity between original and counterfactual
            cos_sim = float(np.dot(orig_emb, cf_emb) /
                           (np.linalg.norm(orig_emb) * np.linalg.norm(cf_emb) + 1e-10))

            print(f'Δs = {delta:+.4f} (orig={orig_score:.4f} → cf={cf_score:.4f})  '
                  f'cos_sim={cos_sim:.3f}  ({elapsed:.1f}s)')

            results.append({
                'zone_class': zone,
                'variant': variant,
                'original_score': orig_score,
                'counterfactual_score': cf_score,
                'delta_score': delta,
                'pct_change': delta / abs(orig_score) * 100 if orig_score != 0 else 0,
                'cosine_similarity': cos_sim,
                'original_chars': len(corpus_text),
                'counterfactual_chars': len(counterfactual_text),
                'generation_time': elapsed,
                'text_file': fname,
            })

    if not results:
        print('\nNo counterfactuals generated.')
        return

    # ── Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('counterfactual_results.csv', index=False)
    print(f'\nSaved: counterfactual_results.csv ({len(results_df)} counterfactuals)')

    # ── Summary
    print('\n' + '=' * 75)
    print('  COUNTERFACTUAL RESULTS')
    print('=' * 75)
    print(f'  {"Zone":<10} {"Variant":<12} {"Orig":>8} {"CF":>8} {"Δs":>8} {"% Chg":>8} {"CosSim":>7}')
    print('  ' + '─' * 63)
    for _, r in results_df.iterrows():
        print(f'  {r["zone_class"]:<10} {r["variant"]:<12} {r["original_score"]:>8.4f} '
              f'{r["counterfactual_score"]:>8.4f} {r["delta_score"]:>+8.4f} '
              f'{r["pct_change"]:>7.1f}% {r["cosine_similarity"]:>7.3f}')

    # ── Figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Generative Counterfactual Zoning: Predicted Causal Improvement',
                 fontweight='bold', fontsize=13)

    # Panel 1: Bar chart of Δs
    ax = axes[0]
    zones_unique = results_df['zone_class'].unique()
    x = np.arange(len(zones_unique))
    width = 0.35
    colors = {'standard': '#1C7293', 'aggressive': '#065A82'}

    for i, variant in enumerate(variants):
        sub = results_df[results_df.variant == variant]
        vals = []
        for z in zones_unique:
            row = sub[sub.zone_class == z]
            vals.append(float(row.delta_score.values[0]) if len(row) > 0 else 0)
        offset = (i - (len(variants) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=variant.capitalize(),
                     color=colors.get(variant, '#888'), alpha=0.85)
        for j, v in enumerate(vals):
            ax.text(x[j] + offset, v + 0.001 * np.sign(v),
                   f'{v:+.3f}', ha='center', va='bottom' if v >= 0 else 'top',
                   fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(zones_unique, fontsize=10)
    ax.set_ylabel('Δ Causal Score (s_cf - s_orig)')
    ax.set_title('Predicted AT Improvement by Zone')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    # Panel 2: Cosine similarity
    ax = axes[1]
    for i, variant in enumerate(variants):
        sub = results_df[results_df.variant == variant]
        vals = []
        for z in zones_unique:
            row = sub[sub.zone_class == z]
            vals.append(float(row.cosine_similarity.values[0]) if len(row) > 0 else 0)
        offset = (i - (len(variants) - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=variant.capitalize(),
              color=colors.get(variant, '#888'), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(zones_unique, fontsize=10)
    ax.set_ylabel('Cosine Similarity (original ↔ counterfactual)')
    ax.set_title('Embedding Distance: How Much Did Text Change?')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig_counterfactual_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: fig_counterfactual_comparison.png')

    # ── Key interpretation
    avg_delta_std = results_df[results_df.variant == 'standard']['delta_score'].mean()
    if len(variants) > 1:
        avg_delta_agg = results_df[results_df.variant == 'aggressive']['delta_score'].mean()
        print(f'\nAverage Δs: standard={avg_delta_std:+.4f}, aggressive={avg_delta_agg:+.4f}')
    else:
        print(f'\nAverage Δs: standard={avg_delta_std:+.4f}')

    print(f'\n✓ Counterfactual generation complete.')
    print(f'  Texts saved in counterfactual_texts/')
    print(f'  Results: counterfactual_results.csv')
    print(f'  Figure: fig_counterfactual_comparison.png')


if __name__ == '__main__':
    main()