"""
Random Forest Modeling + SHAP Analysis for Each Claim Type
==========================================================

For each of the 7 claim types:
  1. Load the monthly panel parquet
  2. Train a Random Forest on all months except the last
  3. Predict the last (next) month
  4. Compute SHAP values (TreeExplainer)
  5. Normalize SHAP values (% contribution) for cross-claim comparison
  6. Save predictions + SHAP values to parquet

Output files saved to: output_data directory
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, f1_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import shap

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r'C:\Users\whita\Documents\case-tricura\output_data'
OUTPUT_DIR = r'C:\Users\whita\Documents\case-tricura\output_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLAIM_TYPES = {
    'fall':             'target_fall',
    'wound':            'target_wound',
    'medication_error': 'target_medication_error',
    'elopement':        'target_elopement',
    'altercation':      'target_altercation',
    'choking':          'target_choking',
    'rth':              'target_rth',
}

# Columns to EXCLUDE from features
EXCLUDE_COLS = [
    'resident_id', 'facility_id', 'year_month',
    # All target columns (avoid leakage between claim types)
    'target_fall', 'target_wound', 'target_medication_error',
    'target_elopement', 'target_altercation', 'target_choking', 'target_rth',
]

# Random Forest hyperparameters
RF_PARAMS = {
    'n_estimators': 500,
    'max_depth': 15,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
    'random_state': 42,
    'n_jobs': -1,
}

RANDOM_STATE = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def prepare_features(df, target_col):
    """
    Prepare feature matrix X and target y.
    Drops excluded columns + the year_month column.
    """
    drop_cols = [c for c in EXCLUDE_COLS if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Fill any remaining NaN
    X = X.fillna(0)
    
    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Replace infinities
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, y, X.columns.tolist()


def compute_normalized_shap(shap_values, feature_names):
    """
    Normalize SHAP values so they sum to 100% per observation.
    This allows comparing feature importance ACROSS different claim models.
    
    Returns:
        shap_abs_norm: DataFrame with normalized absolute SHAP values (% contribution)
        shap_raw: DataFrame with raw SHAP values (signed)
    """
    # Raw SHAP values
    shap_raw = pd.DataFrame(shap_values, columns=feature_names)
    
    # Absolute SHAP values
    shap_abs = np.abs(shap_values)
    
    # Normalize per row: each feature's |SHAP| / sum(|SHAP|) * 100
    row_sums = shap_abs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid division by zero
    shap_pct = (shap_abs / row_sums) * 100
    
    shap_abs_norm = pd.DataFrame(shap_pct, columns=feature_names)
    
    return shap_abs_norm, shap_raw


def print_model_report(y_true, y_pred, y_proba, claim_name):
    """Print classification metrics."""
    print(f"\n  {'─'*50}")
    print(f"  Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0, 
                                labels=[0, 1],
                                target_names=['No Claim', 'Claim']))
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print(f"  Confusion Matrix:")
    print(f"    TN={cm[0,0]:,}  FP={cm[0,1]:,}")
    print(f"    FN={cm[1,0]:,}  TP={cm[1,1]:,}")
    
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        print(f"\n  ROC AUC: {auc:.4f}")
        print(f"  Average Precision (PR AUC): {ap:.4f}")
    else:
        print(f"\n  (Only one class in test set — AUC not computed)")


# ============================================================
# TIME SPLITS
# ============================================================
TRAIN_CUTOFF = pd.Timestamp('2024-11-01')      # train: all months < 2024-11
VAL_MONTHS = [pd.Timestamp('2024-12-01'), pd.Timestamp('2025-01-01')]
PREDICT_MONTH = pd.Timestamp('2025-02-01')

print(f"  Train:    months before {TRAIN_CUTOFF.strftime('%Y-%m')}")
print(f"  Validate: {', '.join(m.strftime('%Y-%m') for m in VAL_MONTHS)}")
print(f"  Predict:  {PREDICT_MONTH.strftime('%Y-%m')}")

all_model_summaries = []

for claim_key, target_col in CLAIM_TYPES.items():
    
    print(f"\n{'='*70}")
    print(f"  CLAIM TYPE: {claim_key.upper()}")
    print(f"{'='*70}")
    
    # ---- 1. Load data ----
    filepath = os.path.join(DATA_DIR, f'claims_{claim_key}_monthly.parquet')
    df = pd.read_parquet(filepath)
    print(f"\n  Loaded: {filepath}")
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    
    # ---- 2. Split: train / validate / predict ----
    df_train = df[df['year_month'] < TRAIN_CUTOFF]
    df_val = df[df['year_month'].isin(VAL_MONTHS)]
    df_predict = df[df['year_month'] == PREDICT_MONTH]
    
    print(f"  Train:    {len(df_train):,} rows (months < {TRAIN_CUTOFF.strftime('%Y-%m')})")
    print(f"  Validate: {len(df_val):,} rows ({', '.join(m.strftime('%Y-%m') for m in VAL_MONTHS)})")
    print(f"  Predict:  {len(df_predict):,} rows ({PREDICT_MONTH.strftime('%Y-%m')})")
    print(f"  Target (train): {df_train[target_col].value_counts().to_dict()}")
    print(f"  Target (val):   {df_val[target_col].value_counts().to_dict()}")
    print(f"  Target (pred):  {df_predict[target_col].value_counts().to_dict()}")
    
    # ---- 3. Prepare features ----
    X_train, y_train, feature_names = prepare_features(df_train, target_col)
    X_val, y_val, _ = prepare_features(df_val, target_col)
    X_predict, y_predict, _ = prepare_features(df_predict, target_col)
    
    X_val = X_val.reindex(columns=feature_names, fill_value=0)
    X_predict = X_predict.reindex(columns=feature_names, fill_value=0)
    
    print(f"  Features: {len(feature_names)}")
    
    # ---- 4. Cross-validation + SMOTE ----
    pos_count = int(y_train.sum())
    N_FOLDS = 5
    # In CV, each fold's train split has ~(N_FOLDS-1)/N_FOLDS of the positives
    min_pos_in_fold = int(np.floor(pos_count * (N_FOLDS - 1) / N_FOLDS))
    smote_k_cv = min(5, min_pos_in_fold - 1) if min_pos_in_fold > 1 else 1
    smote_k_full = min(5, pos_count - 1) if pos_count > 1 else 1
    use_smote = pos_count >= 3  # need at least 3 for SMOTE (k>=2)
    
    if pos_count >= 10 and smote_k_cv >= 1 and use_smote:
        print(f"\n  Running {N_FOLDS}-fold CV with SMOTE (k_cv={smote_k_cv})...")
        smote_cv = SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k_cv)
        pipeline_cv = ImbPipeline([
            ('smote', smote_cv),
            ('rf', RandomForestClassifier(**RF_PARAMS)),
        ])
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipeline_cv, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"  CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        cv_f1 = cross_val_score(pipeline_cv, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
        print(f"  CV F1:      {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")
    elif pos_count >= 10:
        print(f"\n  Running {N_FOLDS}-fold CV (class_weight=balanced, too few for SMOTE in folds)...")
        model_cv = RandomForestClassifier(**RF_PARAMS, class_weight='balanced')
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model_cv, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"  CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    else:
        cv_scores = None
        print(f"\n  (Skipping CV — only {int(pos_count)} positive samples in train)")
    
    # ---- 5. Train final model ----
    if use_smote:
        print(f"\n  Applying SMOTE (k={smote_k_full}) and training Random Forest...")
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=smote_k_full)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
        print(f"  After  SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")
        model = RandomForestClassifier(**RF_PARAMS)
        model.fit(X_train_res, y_train_res)
    else:
        print(f"\n  Too few positives ({pos_count}) for SMOTE — using class_weight='balanced'")
        model = RandomForestClassifier(**RF_PARAMS, class_weight='balanced')
        model.fit(X_train, y_train)
    
    # ---- 6. Validate on 2024-12 + 2025-01 ----
    print(f"\n  --- VALIDATION ({', '.join(m.strftime('%Y-%m') for m in VAL_MONTHS)}) ---")
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    print_model_report(y_val, y_val_pred, y_val_proba, claim_key)
    
    # Save validation predictions
    val_predictions = df_val[['resident_id', 'facility_id', 'year_month']].copy().reset_index(drop=True)
    val_predictions['claim_type'] = claim_key
    val_predictions['y_true'] = y_val.values
    val_predictions['y_pred'] = y_val_pred
    val_predictions['y_proba'] = y_val_proba
    val_predictions.to_parquet(
        os.path.join(OUTPUT_DIR, f'validation_{claim_key}.parquet'), index=False)
    
    # ---- 7. Predict on 2025-02 ----
    print(f"\n  --- PREDICTION ({PREDICT_MONTH.strftime('%Y-%m')}) ---")
    y_pred = model.predict(X_predict)
    y_proba = model.predict_proba(X_predict)[:, 1]
    
    if len(df_predict) > 0:
        print(f"  Predictions: {len(y_pred):,} residents")
        print(f"  Predicted positive: {y_pred.sum():,} ({y_pred.mean()*100:.2f}%)")
        print(f"  Mean probability: {y_proba.mean():.4f}")
        if len(np.unique(y_predict)) > 1:
            print_model_report(y_predict, y_pred, y_proba, claim_key)
    else:
        print(f"  No data for {PREDICT_MONTH.strftime('%Y-%m')}")
    
    # ---- 8. Feature importance (from RF) ----
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n  Top 15 features (RF importance):")
    for _, row in feat_imp.head(15).iterrows():
        print(f"    {row['importance']:.4f}  {row['feature']}")
    
    # ---- 9. SHAP values on prediction month ----
    print(f"\n  Computing SHAP values for {len(X_predict):,} predictions ({PREDICT_MONTH.strftime('%Y-%m')})...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_predict)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]
    else:
        shap_vals = shap_values
    
    # ---- 10. Normalize SHAP values ----
    shap_norm, shap_raw = compute_normalized_shap(shap_vals, feature_names)
    
    # ---- 11. Global SHAP summary ----
    global_shap = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_vals).mean(axis=0),
        'mean_norm_shap_pct': shap_norm.mean(axis=0).values,
    }).sort_values('mean_abs_shap', ascending=False)
    
    print(f"\n  Top 15 features (normalized SHAP %):")
    for _, row in global_shap.head(15).iterrows():
        print(f"    {row['mean_norm_shap_pct']:6.2f}%  {row['feature']}")
    
    # ---- 12. Build prediction output ----
    predictions = df_predict[['resident_id', 'facility_id', 'year_month']].copy().reset_index(drop=True)
    predictions['claim_type'] = claim_key
    predictions['y_true'] = y_predict.values
    predictions['y_pred'] = y_pred
    predictions['y_proba'] = y_proba
    
    # ---- 13. Build SHAP output ----
    shap_norm_out = df_predict[['resident_id', 'facility_id', 'year_month']].copy().reset_index(drop=True)
    shap_norm_out['claim_type'] = claim_key
    shap_norm_out = pd.concat([shap_norm_out, shap_norm.reset_index(drop=True)], axis=1)
    
    shap_raw_out = df_predict[['resident_id', 'facility_id', 'year_month']].copy().reset_index(drop=True)
    shap_raw_out['claim_type'] = claim_key
    shap_raw_out = pd.concat([shap_raw_out, shap_raw.reset_index(drop=True)], axis=1)
    
    # ---- 14. Save outputs ----
    pred_path = os.path.join(OUTPUT_DIR, f'predictions_{claim_key}.parquet')
    predictions.to_parquet(pred_path, index=False)
    
    shap_norm_path = os.path.join(OUTPUT_DIR, f'shap_normalized_{claim_key}.parquet')
    shap_norm_out.to_parquet(shap_norm_path, index=False)
    
    shap_raw_path = os.path.join(OUTPUT_DIR, f'shap_raw_{claim_key}.parquet')
    shap_raw_out.to_parquet(shap_raw_path, index=False)
    
    global_shap_path = os.path.join(OUTPUT_DIR, f'shap_global_{claim_key}.parquet')
    global_shap['claim_type'] = claim_key
    global_shap.to_parquet(global_shap_path, index=False)
    
    feat_imp_path = os.path.join(OUTPUT_DIR, f'feature_importance_{claim_key}.parquet')
    feat_imp['claim_type'] = claim_key
    feat_imp.to_parquet(feat_imp_path, index=False)
    
    # Save trained model
    model_path = os.path.join(OUTPUT_DIR, f'model_rf_{claim_key}.joblib')
    joblib.dump({
        'model': model,
        'feature_names': feature_names,
        'claim_type': claim_key,
        'target_col': target_col,
        'rf_params': RF_PARAMS,
        'train_cutoff': TRAIN_CUTOFF.strftime('%Y-%m'),
        'validation_months': [m.strftime('%Y-%m') for m in VAL_MONTHS],
        'prediction_month': PREDICT_MONTH.strftime('%Y-%m'),
    }, model_path)
    
    print(f"\n  Saved:")
    print(f"    {pred_path}")
    print(f"    {os.path.join(OUTPUT_DIR, f'validation_{claim_key}.parquet')}")
    print(f"    {shap_norm_path}")
    print(f"    {shap_raw_path}")
    print(f"    {global_shap_path}")
    print(f"    {feat_imp_path}")
    print(f"    {model_path}")
    
    # ---- 15. Collect summary ----
    summary = {
        'claim_type': claim_key,
        'train_rows': len(df_train),
        'val_rows': len(df_val),
        'predict_rows': len(df_predict),
        'train_positive': int(y_train.sum()),
        'val_positive': int(y_val.sum()),
        'predict_positive': int(y_predict.sum()),
        'n_features': len(feature_names),
        'prediction_month': PREDICT_MONTH.strftime('%Y-%m'),
    }
    
    # Validation metrics
    if len(np.unique(y_val)) > 1:
        summary['val_auc'] = roc_auc_score(y_val, y_val_proba)
        summary['val_avg_precision'] = average_precision_score(y_val, y_val_proba)
    else:
        summary['val_auc'] = None
        summary['val_avg_precision'] = None
    summary['val_f1'] = f1_score(y_val, y_val_pred, zero_division=0)
    
    if cv_scores is not None:
        summary['cv_auc_mean'] = cv_scores.mean()
        summary['cv_auc_std'] = cv_scores.std()
    else:
        summary['cv_auc_mean'] = None
        summary['cv_auc_std'] = None
    
    all_model_summaries.append(summary)

# ============================================================
# COMBINED OUTPUTS (for cross-claim comparison)
# ============================================================
print(f"\n\n{'='*70}")
print("  COMBINING OUTPUTS FOR CROSS-CLAIM COMPARISON")
print(f"{'='*70}")

all_predictions = []
all_validation = []
all_global_shap = []
all_feat_imp = []

for claim_key in CLAIM_TYPES:
    all_predictions.append(pd.read_parquet(os.path.join(OUTPUT_DIR, f'predictions_{claim_key}.parquet')))
    all_validation.append(pd.read_parquet(os.path.join(OUTPUT_DIR, f'validation_{claim_key}.parquet')))
    all_global_shap.append(pd.read_parquet(os.path.join(OUTPUT_DIR, f'shap_global_{claim_key}.parquet')))
    all_feat_imp.append(pd.read_parquet(os.path.join(OUTPUT_DIR, f'feature_importance_{claim_key}.parquet')))

combined_pred = pd.concat(all_predictions, ignore_index=True)
combined_pred.to_parquet(os.path.join(OUTPUT_DIR, 'predictions_all_claims.parquet'), index=False)

combined_val = pd.concat(all_validation, ignore_index=True)
combined_val.to_parquet(os.path.join(OUTPUT_DIR, 'validation_all_claims.parquet'), index=False)

combined_global_shap = pd.concat(all_global_shap, ignore_index=True)
combined_global_shap.to_parquet(os.path.join(OUTPUT_DIR, 'shap_global_all_claims.parquet'), index=False)

combined_feat_imp = pd.concat(all_feat_imp, ignore_index=True)
combined_feat_imp.to_parquet(os.path.join(OUTPUT_DIR, 'feature_importance_all_claims.parquet'), index=False)

summary_df = pd.DataFrame(all_model_summaries)
summary_df.to_parquet(os.path.join(OUTPUT_DIR, 'model_summary.parquet'), index=False)

print(f"\nCombined files saved:")
print(f"  predictions_all_claims.parquet  (2025-02 predictions)")
print(f"  validation_all_claims.parquet   (2024-12 + 2025-01 validation)")
print(f"  shap_global_all_claims.parquet")
print(f"  feature_importance_all_claims.parquet")
print(f"  model_summary.parquet")

# ============================================================
# FINAL SUMMARY TABLE
# ============================================================
print(f"\n\n{'='*70}")
print("  MODEL SUMMARY")
print(f"{'='*70}\n")

print(summary_df[['claim_type', 'train_rows', 'val_rows', 'predict_rows',
                   'train_positive', 'val_positive', 'cv_auc_mean',
                   'val_auc', 'val_f1', 'prediction_month']].to_string(index=False))

print(f"\n\nTrain:    months < {TRAIN_CUTOFF.strftime('%Y-%m')}")
print(f"Validate: {', '.join(m.strftime('%Y-%m') for m in VAL_MONTHS)}")
print(f"Predict:  {PREDICT_MONTH.strftime('%Y-%m')}")
print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("Done!")
