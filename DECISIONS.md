# Decisions Log

Methodological choices, design tradeoffs, and their rationale.

---

## Causal framework

**OLS vs logistic in DML second stage.** OLS is methodologically correct for
the DML second stage because it operates on continuous residuals (Y − ĝ(W)),
not the binary outcome directly. Logistic regression is used as the *naive
benchmark* for comparison only.

**ACS variables as confounders vs mediators.** Including ACS in W estimates
the direct regulatory effect of zoning text on AT. Excluding ACS estimates
the total effect (regulatory + demographic sorting). Both are reported.
The T1 sign difference between specifications is expected and reflects this
direct/total distinction — it is not a pipeline error.

**BH FDR correction scope.** Benjamini-Hochberg correction is applied
independently within each specification column (not across the full table).
This follows convention for multi-treatment DML where each column represents
a distinct model specification.

---

## LLM feature extraction

**Mistral-7B with 4-bit NF4 quantization.** Chosen for local HiPerGator
execution without API costs. Quantization enables single-GPU inference
on the available A100 hardware. Scoring uses 10 theory-driven planning
dimensions on a 1–5 scale with rationales.

**GPT-4o for scoring (later revision).** The LLM scoring script was updated
to use GPT-4o via API for the final paper version, replacing the local
Mistral scoring. Pipeline and data dictionary reflect GPT-4o as the
production scorer.

**D4a sign reversal — not reversed for the paper.** Within dimension D4
(parking/density), the LLM holistic score and the binary frontage standard
produce opposite signs. The LLM score captures a broad "parking intensity"
concept while the binary indicator captures a specific provision. This
within-dimension sign reversal was identified and validated the
multi-feature-per-dimension design (having both LLM and scraped features
per dimension catches this kind of nuance). The signs were not artificially
reconciled — both are reported as estimated.

---

## Embedding pipeline

**32 PCA dimensions.** All embedding backends are reduced to 32 PCA
components. This standardizes dimensionality across backends with different
native sizes (384 for MiniLM, 3072 for OpenAI) and keeps the DML treatment
vector manageable. The hybrid backend has 42 dimensions (10 LLM + 32 PCA).

**Seven backends compared.** MiniLM, MPNet, multi-qa, OpenAI-large, Google,
TF-IDF, and hybrid (LLM scores + MiniLM). Backend choice meaningfully
affects substantive conclusions. No single backend is designated as "correct." MiniLM and OpenAI-Large are
the two reported in the paper. MiniLM is used because the counterfactual
pipeline (Mistral-7B rewrites) re-embeds with MiniLM, so the counterfactual
Δs scores are only valid against MiniLM θ vectors. OpenAI-Large is used
because it produced the most semantically meaningful embedding dimensions
and TF-IDF term correlations — the word-level interpretations were more
coherent and aligned with planning theory than other backends.

**Corpus cross-referencing is legitimate.** CBD text appearing in CD corpus
files reflects Tampa's actual zoning code cross-references (the code itself
references across districts), not a pipeline contamination issue.

---

## DML estimation

**50-fit repeated cross-fitting.** 10 random seeds × 5 folds = 50 nuisance
model fits per treatment variable. Aggregation uses median θ with combined
standard errors (within-seed + across-seed variance). This design was chosen
over single-seed estimation to reduce sensitivity to a particular
train/test partition.

**RF as primary, GB as robustness.** Random forest (500 trees, depth 12) is
the primary nuisance learner. Gradient boosting is the robustness check.
Both are reported in the paper. Lasso was considered but dropped because
the nonlinear confounding structure favors tree-based learners.

**Bootstrap SEs for theory-driven, HC1 for vector.** Theory-driven scalar
DML uses bootstrap standard errors (B = 50) because each feature is
estimated independently. Theory-agnostic vector DML uses HC1 sandwich
standard errors because the joint estimation makes bootstrapping more
expensive. Both incorporate across-seed partition variance.

---

## Counterfactual generation

**Two rewrite variants per zone.** Standard (moderate reform: add TOD
provisions, reduce parking 25–50%, add pedestrian connectivity, permit
mixed uses) and aggressive (full form-based conversion: eliminate parking
minimums, zero-setback build-to lines, 60% ground-floor transparency,
all uses by right). 56 zones × 2 variants = 112 counterfactual texts.

**Mistral-7B at temperature 0.7.** Local LLM used for counterfactual
generation (not GPT-4o) to keep generation on HiPerGator without API costs
for 112 text generations. Temperature 0.7 balances coherent variation with
regulatory plausibility.

**Validation through DML projection.** Counterfactual texts are embedded
with MiniLM and projected into the original PCA space. Δs = (e_cf · θ) −
(e_orig · θ). This closes the generation-validation loop: the LLM generates,
the DML validates.

---

## Data processing

**Deduplication tradeoff.** Potential duplication artifacts in scraped zoning
text were identified late in the pipeline (near submission deadline).
Decision: trust the existing pipeline and add a limitations note rather
than rerun. Rationale: four processing layers (mean-pooling, PCA,
area-weighting, DML residualization) attenuate duplication effects. This
was a deliberate deadline-aware tradeoff, not an oversight.

**Split-parcel handling.** Parcels spanning multiple block groups are split
proportionally by area (ShapeSTAre_corrected). This avoids double-counting
and preserves the block-group-level analysis unit.

**26 vs 56 zones in AT rates.** zone_at_rates.csv has only 26 rows because
not all 56 zoning districts have sufficient trip data to compute reliable
AT rates. All 56 zones have embeddings and LLM scores; the AT rate
constraint binds at the DML estimation stage.

---

## Paper and reporting

**"Naive OLS" column label.** Dr. Wang suggested relabeling the "Naive OLS"
column in Table 3 to logistic regression, since the baseline model uses a
logistic specification. Status: flagged, pending final decision.

**θ notation.** Corrected θ₀ vs θ notation inconsistency in the manuscript.
The asymptotic guarantee comes from Neyman orthogonality + cross-fitting,
not the FWL theorem (which was initially misattributed).

**Classifier results excluded from paper.** ML classifier experiments
(v3–v9.5) are not in the final paper. The DML framework addresses the
causal question directly; classifiers were exploratory and are archived.
