# Case Tricura — Nursing Facility Claims Prediction

Insurance claims in skilled nursing facilities are costly and often preventable. This project builds **predictive models for 7 types of claims** — not just to forecast *who* is at risk, but to explain *why*, so that facility staff can take targeted action before an incident occurs.

The core idea: if we can tell a nurse "this resident has a 72% chance of falling next month, primarily because their ADL scores declined and they have 4 active musculoskeletal diagnoses," that's actionable. A probability alone is not.

That's why every model here is paired with **SHAP explainability** at the individual patient level.

## Model Performance

| Claim Type | CV AUC | Validation AUC | Val F1 | Train Positives |
|---|---|---|---|---|
| **Fall** | 0.902 | 0.884 | 0.768 | 1,121 |
| **RTH (Return-to-Hospital)** | 0.907 | 0.901 | 0.472 | 883 |
| **Wound** | 0.869 | 0.867 | 0.713 | 341 |
| **Altercation** | 0.832 | 0.828 | 0.560 | 115 |
| **Medication Error** | 0.722 | 0.628 | — | 28 |
| **Elopement** | — | — | — | 6 |
| **Choking** | — | 0.172 | — | 6 |

> Models trained on data before Nov 2024, validated on Dec 2024 + Jan 2025, predictions generated for Feb 2025.

The models for Fall, RTH, Wound, and Altercation achieved strong AUCs (0.83–0.90), indicating reliable risk stratification. Medication Error, Elopement, and Choking had too few positive cases (6–28 in training) for the model to learn robust patterns — a known limitation discussed further in the Results section.

### Example: SHAP Feature Importance — Fall Claims

![SHAP Beeswarm - Falls](images/shap_beeswarm_fall.png)

*Each dot represents a resident. Position on the X-axis shows the SHAP value (impact on prediction). Color indicates the feature value (blue = low, red = high). Features sorted by global importance.*

### Example: Cross-Claim Feature Comparison

![Cross-Claim Heatmap](images/cross_claim_heatmap.png)

*Normalized SHAP values allow direct comparison of feature importance across different claim types.*

---

## The Problem

Nursing facility claims follow a rough distribution:

| Incident Type | % of Claims | Avg Cost per Incident |
|---|---|---|
| Falls | ~13% | ~$3,500 |
| Medication Errors | ~10% | ~$5,000 |
| Wounds / Pressure Injuries | ~7% | ~$4,000 |
| Return-to-Hospital (RTH) | ~7% | ~$20,000 |
| Elopement / Wandering | ~5% | ~$2,500 |
| Altercations | ~2% | ~$2,500 |

RTH events are by far the most expensive per incident. Falls are the most frequent. The goal is to predict which residents are most likely to generate each type of claim in the upcoming month, using only information available *before* that month begins.

---

## Approach & Design Decisions

### Why One Model Per Claim Type?

The risk profile for a fall is completely different from the risk profile for an elopement. A resident with declining mobility scores is a fall risk; a long-stay resident with a history of behavioral incidents is an elopement risk. Training a single multi-label model would force shared feature representations that blur these distinctions. Separate models let each claim type find its own signal, and separate SHAP analysis makes the drivers immediately interpretable per claim type.

### Why a Monthly Panel (resident × month)?

Clinical risk changes over time. A resident who was stable 3 months ago may have had a hospitalization last month and is now at elevated risk. By structuring the data as a panel with one row per resident per active month, the model can capture these temporal dynamics: "this resident had 2 falls in prior months and their ADL scores declined last month" is far more informative than a static snapshot.

### Why Strictly Historical Features (No Same-Month Data)?

This is the most critical design decision. In production, when we predict February's claims, we only have data through January. If we accidentally include February vitals or February incident counts as features, the model learns from information that wouldn't exist at prediction time — classic data leakage that inflates metrics but produces useless predictions.

Every feature in this pipeline uses data **strictly before** the target month:
- `_hist_` features aggregate all months before the target (cumulative history)
- `_prev_` features use only the immediately preceding month (recent signal)
- "Active" snapshots (diagnoses, orders, therapies) are taken as of the end of the previous month

This means the model's performance metrics reflect what it would actually achieve in deployment.

### Why Random Forest?

Three reasons: interpretability, robustness, and SHAP compatibility.

Random Forests handle mixed feature types (counts, averages, flags) without normalization, are resistant to overfitting with proper depth limits, and work natively with `TreeExplainer` in SHAP — which computes exact Shapley values in polynomial time rather than the exponential-time approximations needed for black-box models. For a healthcare use case where every prediction needs to be explainable, this matters more than squeezing an extra 1-2% AUC from gradient boosting.

### Why SMOTE Instead of Class Weights?

Most claim types are rare events (1-4% positive rate). `class_weight='balanced'` adjusts the loss function but doesn't change the feature space the model sees — it still learns decision boundaries from a handful of positive examples. SMOTE generates synthetic minority samples by interpolating between existing positive cases, giving the model a denser region of the feature space to learn from.

The tradeoff is that SMOTE can create unrealistic samples if the minority class is too small. That's why the pipeline has a tiered approach: SMOTE for claims with ≥3 positives, fallback to `class_weight='balanced'` for extremely rare claims (elopement, choking with only 6 cases).

SMOTE is applied inside each cross-validation fold separately, not before the split — otherwise synthetic samples based on validation data would leak into training, again inflating metrics.

### Why This Time Split?

The train/validate/predict split isn't random — it's **chronological**, which is essential for time-series prediction:

| Set | Period | Why |
|---|---|---|
| **Train** | Months < 2024-11 | The model learns patterns from the historical record |
| **Validate** | 2024-12 + 2025-01 | Two months of held-out data to check if the model generalizes to unseen time periods (not just unseen residents) |
| **Predict** | 2025-02 | The actual target month — generate predictions and SHAP explanations for operational use |

A random split would let the model see January data during training and predict December — effectively predicting the past. The chronological split ensures we're always predicting forward.

---

## Project Structure

```
├── README.md
├── requirements.txt
├── data/                          # Raw parquet files (not committed)
│   ├── residents.parquet
│   ├── incidents.parquet
│   ├── diagnoses.parquet
│   └── ... (17 tables)
│
├── notebooks/
│   ├── 01_data_treatment.ipynb    # Initial data exploration
│   ├── 02_build_panels.ipynb      # Build monthly panel datasets
│   ├── 03_model_training.ipynb    # Train RF + SHAP for all claims
│   ├── 04_analysis_fall.ipynb     # Deep dive: Fall claims
│   └── 05_analysis_all_claims.ipynb  # Full analysis: all 7 claim types
│
├── src/
│   ├── build_monthly_panels.py    # Panel dataset construction
│   └── model_claims.py            # Model training pipeline
│
├── output_data/                   # Generated outputs (not committed)
│   ├── claims_fall_monthly.parquet
│   ├── predictions_fall.parquet
│   ├── shap_normalized_fall.parquet
│   ├── model_rf_fall.joblib
│   └── ...
│
└── images/                        # Screenshots for README
    ├── shap_beeswarm_fall.png
    ├── cross_claim_heatmap.png
    └── ...
```

---

## Data Pipeline

### Source Data (17 tables)

Clinical and operational data from skilled nursing facilities, linked by `resident_id`:

| Table | Description | Key Fields |
|---|---|---|
| `residents` | Demographics | admission/discharge dates, DOB |
| `incidents` | Falls, wounds, elopements... | `incident_type`, `occurred_at` |
| `injuries` | Linked to incidents | injury type, location |
| `diagnoses` | ICD-10 codes | onset, resolved dates |
| `medications` | 1.4M+ records | description, status, schedule |
| `vitals` | Vital signs | type, value, measured_at |
| `lab_reports` | Lab results | severity status |
| `hospital_transfers` | RTH events | reason, emergency flag |
| `hospital_admissions` | Hospital stays | duration, emergency |
| `care_plans` | Active care plans | initiation, closure |
| `needs` | Patient needs | type, category |
| `physician_orders` | Active orders | category, frequency |
| `therapy_tracks` | PT/OT/SLP | discipline, duration |
| `adl_responses` | Daily living assessments | activity, response score |
| `gg_responses` | Functional assessments | task group, response code |
| `document_tags` | Clinical documents | doc type, confidence |
| `factors` | Incident contributing factors | factor type |

### Panel Construction

The raw data is event-level (one row per vital reading, one row per medication administration, etc.). For modeling, we need one row per resident per month with aggregated features. The panel construction handles this by computing three types of features:

**Cumulative history (`_hist_`)** answers "what is this resident's full track record?" — total incidents ever, total diagnoses ever. This captures long-term risk factors.

**Previous month (`_prev_`)** answers "what happened recently?" — vitals last month, ADL scores last month, lab results last month. This captures acute changes that might signal imminent risk.

**Active snapshots (`*_prev`)** answer "what is the resident's current clinical state?" — active diagnoses as of last month end, active orders, active therapies. A resident with 8 active diagnoses and 3 concurrent therapies has a different risk profile than one with 2 diagnoses and no therapy.

The combination of these three temporal perspectives gives the model both the long-term history and the recent trajectory for each resident.

---

## SHAP Explainability

### Why SHAP?

Standard feature importance (Gini importance from Random Forest) tells you which features matter *on average across all predictions*. SHAP tells you which features matter *for this specific resident*. Two residents can both have a 70% fall probability for completely different reasons — one because of declining ADL scores, another because of a history of 5 prior falls. SHAP captures this individual-level granularity, which is what makes the predictions operationally useful.

### What are SHAP Values?

**SHAP (SHapley Additive exPlanations)** values decompose each prediction into individual feature contributions. For every resident and every feature, SHAP tells you:

- **Sign**: Does this feature push the prediction toward a claim (+) or away from it (−)?
- **Magnitude**: How much does it contribute?

This is based on Shapley values from cooperative game theory — each feature's contribution is calculated as the average marginal contribution across all possible feature combinations. For tree-based models, `TreeExplainer` computes these exactly in polynomial time.

### Normalization for Cross-Claim Comparison

Raw SHAP values are model-specific — a SHAP value of 0.05 means something different for falls (common) vs. elopement (rare). To compare feature importance across claim types, we normalize:

For each resident, the normalized values sum to **100%**, representing the percentage contribution of each feature to that prediction. This lets us make statements like "age contributes 12% to fall predictions vs. 3% to elopement predictions" on equal footing.

```
normalized_SHAP_i = |SHAP_i| / Σ|SHAP_all| × 100%
```

### How to Read the Charts

- **Beeswarm plot**: Each dot = one resident. X = SHAP value. Color = feature value (blue=low, red=high). If red dots cluster on the right, high feature values increase risk.
- **Risk Tier Heatmap**: Shows which features matter most at each risk level. A feature that matters for high-risk residents but not low-risk ones is a targeted intervention opportunity.
- **SHAP vs Probability scatter**: How a single feature's contribution changes across the entire risk spectrum, colored by the actual feature value.

---

## Results & Insights

### Fall

![SHAP Magnitude per Risk Tier - Fall](images/shap_beeswarm_fall.png)

Residents with a high number of active physician orders — particularly dietary orders — show significantly elevated fall risk in the following month. This makes clinical sense: dietary orders often indicate swallowing difficulties or nutritional decline, which correlate with frailty and impaired balance. The model also picks up on ADL decline and prior fall history as strong predictors, consistent with established fall risk literature.

### Wound

![SHAP Magnitude per Risk Tier - Wound](images/shap_beeswarm_wound.png)

Wound recurrence dominates: residents with previous wound incidents are substantially more likely to develop new wounds. This aligns with clinical reality — once skin integrity is compromised (through pressure injuries, skin tears, or chronic wounds), the underlying risk factors (immobility, poor nutrition, incontinence) tend to persist. The model effectively identifies a "wound cycle" pattern where historical wound count is the strongest predictor.

### Return-to-Hospital

![SHAP Magnitude per Risk Tier - RTH](images/shap_beeswarm_rth.png)

Length of stay is the dominant driver: residents who have been in the facility longer accumulate more RTH events. This likely reflects the acuity profile — long-stay residents tend to have more complex medical conditions requiring hospital-level interventions. Prior hospital admissions compound this effect, suggesting that once a resident enters the hospitalization cycle, the pattern tends to repeat. Given that RTH events cost ~$20,000 each, identifying these residents early has the highest dollar-value impact of any claim type.

### Altercation

![SHAP Magnitude per Risk Tier - Altercation](images/shap_beeswarm_altercation.png)

The strongest signal comes from behavioral history: residents with prior altercations are far more likely to have future ones. Newly admitted residents show lower risk, which makes sense — behavioral patterns take time to manifest and escalate. An interesting secondary signal is medication frequency. Residents requesting medications more often may be experiencing pain, agitation, or psychiatric symptoms that also drive interpersonal conflict.

### Medication Error

![SHAP Magnitude per Risk Tier - Medication Error](images/shap_beeswarm_medication_error.png)

Prior medication errors are the strongest predictor of future ones. This points to systemic rather than random issues — when a medication error occurs for a resident, the root cause (complex medication regimen, communication gaps, specific staff assignments) likely persists. The model with only 28 training positives still achieved 0.72 CV AUC, suggesting the pattern is strong even with limited data, but the 0.63 validation AUC and zero F1 indicate it doesn't generalize well to new time periods. More data would likely improve this substantially.

### Elopement

![SHAP Magnitude per Risk Tier - Elopement](images/shap_beeswarm_elopement.png)

With only 6 training cases, the model cannot learn reliable patterns — the results should be interpreted with extreme caution. That said, the directional signal is interesting: longer-stay residents with accumulated incident histories show higher risk. This loosely aligns with the clinical profile of elopement-prone residents (cognitive decline, increasing restlessness over extended stays). A meaningful model would require either more data or a different approach entirely (e.g., rule-based flags from dementia diagnoses and wandering assessments).

### Choking

It seems that people with genitourinary diseases and people with more pharmacy orders happen to have higher choking risk. However, with only 6 positive training cases, the sample is far too small to draw conclusions. Any apparent patterns could be coincidental. This claim type would benefit most from additional data collection or integration with external risk factors (swallowing assessments, diet texture modifications).

---

## Limitations & Next Steps

**Data volume for rare claims**: Elopement (6 cases), Choking (6 cases), and Medication Error (28 cases) don't have enough positive examples for reliable modeling. Options include expanding the training window, pooling data across facilities, or using anomaly detection instead of classification.

**Feature granularity**: The current features are aggregated at the monthly level. Daily or weekly granularity could capture rapid deterioration patterns (e.g., a sudden drop in vitals over 3 days before a fall) that monthly averages smooth over.

**Model selection**: Random Forest was chosen for interpretability and SHAP compatibility. Gradient Boosting (XGBoost/LightGBM) would likely improve AUC by 2-5% and could be worth exploring for the higher-volume claim types while keeping RF for explainability.

**External validation**: The models are validated on 2 months of the same facilities. Validating on held-out facilities (rather than held-out time periods) would test whether the patterns generalize across different operational environments.

---

## Setup

### Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn shap joblib matplotlib seaborn
```

### Running the Pipeline

```bash
# 1. Build monthly panel datasets
python src/build_monthly_panels.py

# 2. Train models + generate predictions/SHAP
python src/model_claims.py

# 3. Open analysis notebooks
jupyter notebook notebooks/05_analysis_all_claims.ipynb
```

---

## License

This project was developed as a case study for claims prediction in skilled nursing facilities.
