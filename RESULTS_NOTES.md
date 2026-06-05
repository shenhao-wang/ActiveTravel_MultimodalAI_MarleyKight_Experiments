# Results Notes

Key findings, robustness checks, and specifications that did not work.

---

## Key findings

### Baseline associations (pre-DML)

Logistic regression and random forest models confirm expected statistical
associations between 3D/5D built environment features and active travel.
Trip distance is the dominant predictor: each standardized unit increase
reduces the log-odds of AT by 5.95 (logistic) / probability by 30.1 pp (RF).
Gender and income effects are significant only in the logistic specification.

### Theory-driven DML (21 features × 4 specifications)

Six regulatory dimensions promote active travel across most specifications:

- Transit-oriented planning (D7): θ = +0.246, largest effect, 2.5× the
  next largest. Remains dominant in the joint vector specification.
- Use flexibility (D2): θ = +0.096
- Parking intensity (D4): θ = −0.091 (more parking suppresses AT)
- Pedestrian street interface (D3): θ = +0.090
- Setback/building placement (D5): θ = +0.089 (smaller setbacks → more AT)
- Rear setback (D5, scraped): θ = −0.087 (larger rear setbacks → less AT)

Human-scale design (D6) produces a stable negative estimate (θ = −0.087)
across all four specifications. Interpretation: zones mandating human-scale
design attract walkability-seeking residents, compressing the residual causal
contribution of the text itself. This is a self-selection signature, not
evidence that human-scale design harms walkability.

### Theory-agnostic DML (vector embeddings)

MiniLM and OpenAI-Large produce complementary zone rankings. MiniLM
prioritizes lexical distinctiveness (Ybor City historic districts, CBD zones).
OpenAI captures finer semantic gradients (mixed-use and commercial zones rank
highest, single-family lowest), closer to conventional planning expectations.

Counterintuitive zone ranking: zones designed for walkability (CD-1/2/3,
NMU-24/35/16) score as car-promoting after DML removes self-selection.
Medium-density residential zones (RM-24/18, SH-RS, RS-60) score as most
AT-promoting. Interpretation: mixed-use zones' observed AT rates are a
product of who lives there, not what the code says. Residential zones serve
populations whose trip patterns make walking viable, and the code supports
rather than constrains that behavior.

Theory-agnostic embeddings also identify nontraditional regulatory dimensions
(rights-of-way language, land-conformance terminology, governance provision
density) that wouldn't appear in a theory-driven framework.

### Counterfactual analysis (112 LLM rewrites)

Three-category typology emerges:

- **Improvable zones** (Panel A): Industrial (IG, IH), airport (M-AP-4),
  residential office (RO-1) show largest positive Δs. M-AP-4 produces
  the maximum improvement (Δs = +0.052 aggressive, +0.064 standard).
- **Structurally constrained** (Panel B): Near-zero Δs. AS-1, CU, SH-PD,
  SH-RS lack strong auto- or pedestrian-oriented character.
- **AT-regressive** (Panel C): Ybor City (YC-1 through YC-9) and CBD zones
  worsen under rewriting (Δs = −0.036 for YC-5). Their text already
  occupies the AT-promoting pole; aggressive rewriting pushes embeddings
  toward the semantic center, diluting distinctive regulatory character.

Average Δs is +0.004 (standard) and +0.005 (aggressive). Distribution is
strongly heterogeneous with a "regression to the mean" pattern: AT-promoting
zones drop, car-promoting zones improve.

This independently confirms the self-selection finding from the main DML:
adding walkability language to already-walkable zones worsens predicted
causal scores.

---

## Robustness checks

### Scalar DML stability (Table 10 in paper)

Tested across four dimensions:

1. **Nuisance model**: RF vs GB. Point estimates change ≤30% in magnitude,
   no sign flips, no qualitative significance changes for T1 or T3. T2
   (ped score) remains marginal under both.
2. **Fold count**: K = 3, 5, 10. Estimates stable within 1.5 SEs.
3. **Confounder subsets**: Dropping ACS variables inflates T1 from −0.072
   to −0.104, confirming ACS captures confounding (not collinearity).
   ~3 pp of the residential-zone effect is demographically mediated.
   Dropping trip distance/duration and vehicle ownership produce small
   changes with no sign flips.
4. **Bootstrap replications**: B = 100, 500, 1000. SEs within 5% of each
   other across all three.

### Repeated cross-fitting design

10 seeds × 5 folds = 50 nuisance fits per treatment. Estimates aggregated
via median θ with combined SEs (within-seed sampling + across-seed partition
variance). BH FDR correction at α = 0.05 applied within each specification
column.

### ACS inclusion/exclusion interpretation

Including ACS as confounders (W) estimates the *direct* regulatory effect.
Excluding ACS estimates the *total* effect (including demographic mediation).
The sign flip on T1 between these specifications is expected and intentional,
not an error.

### Group-CV vs random-split classifiers

Group-CV (block groups held out as units) tested whether random-split
accuracy was inflated by BG-level memorization. Gap was modest, suggesting
random-split numbers approximately hold.

### Calibration / residualization diagnostics

RF-residualized features show improved log loss but near-identical Brier
scores. Interpretation: improvement reflects a small number of
confidently-wrong predictions corrected, not broad miscalibration.
Framed as suggestive, not conclusive.

---

## Specifications that did not work

### T2 (pedestrian score) marginal significance

The composite pedestrian score (T2) never reaches conventional significance
(p < 0.05) under any specification, hovering at p < 0.10. The score
aggregates 10+ binary indicators into a single sum, which may wash out
heterogeneous effects that the individual binary features capture.

### ML classifiers (v3–v9.5)

Extensive classifier experiments (logistic regression, RF, GB, SVM, KNN,
MLP, naive Bayes, etc.) were run across multiple feature sets and embedding
backends. These explored whether embeddings improve classification of
AT vs. non-AT trips. Results showed modest accuracy gains from embeddings
but the classifier approach was ultimately not included in the paper —
the DML framework superseded it for the causal question. Classifier
notebooks archived in `archive/old_classifiers/`.

### Joint vector DML specification

All 21 theory-driven features entered simultaneously as a joint treatment
vector. Results are reported as a robustness check in the appendix rather
than the main results because collinearity across correlated planning
dimensions makes individual coefficient interpretation less reliable than
the scalar-per-feature approach.

### Corpus deduplication uncertainty

Potential deduplication issues in the scraped zoning text were identified
late in the pipeline. Decision was made to trust the existing pipeline and
add a limitations note, reasoning that four processing layers (mean-pooling,
PCA, area-weighting, DML residualization) would absorb duplication artifacts.
Not rerun before submission.
