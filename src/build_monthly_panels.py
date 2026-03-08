"""
Build Monthly Panel DataFrames for Claim Type Prediction
=========================================================
POINT-IN-TIME STRICT: All features use data STRICTLY BEFORE the target month.
No same-month features are included.

Feature types:
  - _hist_:  cumulative from all months strictly before target month
  - _prev_:  value from the immediately preceding month (lag 1)
  - _active_prev_: snapshot of active records as of end of previous month

Creates 7 separate dataframes (one per claim type), each with:
  - Grain: resident_id x year_month
  - Filter: only months where the resident was active (insured)
  - Target: binary flag - did the resident have this claim type that month?
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. LOAD ALL DATA
# ============================================================
DATA_DIR = r'C:\Users\whita\Documents\case-tricura\data'
OUTPUT_DIR = r'C:\Users\whita\Documents\case-tricura\output_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.chdir(DATA_DIR)

print("Loading parquet files...")
tables = {}
for f in os.listdir('.'):
    if f.endswith('.parquet'):
        name = f.replace('.parquet', '')
        tables[name] = pd.read_parquet(f)
        print(f"  {name}: {tables[name].shape}")

# ============================================================
# 1. BUILD MONTHLY PANEL OF ACTIVE RESIDENTS
# ============================================================
print("\n[1/5] Building monthly panel of active residents...")

res = tables['residents'].copy()

all_dates = []
for name, df in tables.items():
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            valid = df[col].dropna()
            if len(valid) > 0:
                all_dates.extend([valid.min(), valid.max()])

data_min = pd.Timestamp(min(all_dates)).normalize().replace(day=1)
data_max = pd.Timestamp(max(all_dates)).normalize().replace(day=1)
months = pd.date_range(start=data_min, end=data_max, freq='MS')
print(f"  Date range: {data_min.strftime('%Y-%m')} to {data_max.strftime('%Y-%m')} ({len(months)} months)")

rows = []
for month_start in months:
    month_end = month_start + pd.offsets.MonthEnd(0)
    active = res[
        (res['admission_date'] <= month_end) &
        (res['discharge_date'].isna() | (res['discharge_date'] >= month_start))
    ].copy()
    active['year_month'] = month_start
    rows.append(active[['resident_id', 'facility_id', 'year_month',
                         'date_of_birth', 'admission_date', 'discharge_date',
                         'deceased_date', 'outpatient']])

panel = pd.concat(rows, ignore_index=True)
print(f"  Panel size: {len(panel):,} rows ({panel['resident_id'].nunique():,} residents)")

# ============================================================
# 2. RESIDENT-LEVEL STATIC + TIME-VARYING FEATURES
# ============================================================
print("\n[2/5] Computing resident features per month...")

panel['age_years'] = (panel['year_month'] - panel['date_of_birth']).dt.days / 365.25
panel['los_days'] = (panel['year_month'] - panel['admission_date']).dt.days
panel['is_outpatient'] = panel['outpatient'].astype(int)

panel.drop(columns=['date_of_birth', 'admission_date', 'discharge_date',
                     'deceased_date', 'outpatient'], inplace=True)

# ============================================================
# 3. HELPER FUNCTIONS - STRICTLY HISTORICAL
# ============================================================
print("\n[3/5] Aggregating features from all tables (strictly before target month)...")

def merge_historical_features(panel_df, source_df, date_col, resident_col,
                               agg_dict, prefix):
    """
    _hist_: cumulative from ALL months strictly before target month
    _prev_: value from the immediately preceding month only
    NO same-month features.
    """
    df = source_df.copy()
    df['_date'] = pd.to_datetime(df[date_col])
    df['_ym'] = df['_date'].dt.to_period('M').dt.to_timestamp()

    monthly = df.groupby([resident_col, '_ym']).agg(**agg_dict).reset_index()
    monthly.rename(columns={'_ym': 'year_month'}, inplace=True)
    monthly = monthly.sort_values([resident_col, 'year_month'])

    for col in agg_dict.keys():
        monthly[f'{prefix}_hist_{col}'] = (
            monthly.groupby(resident_col)[col]
            .transform(lambda x: x.cumsum().shift(1, fill_value=0))
        )
        monthly[f'{prefix}_prev_{col}'] = (
            monthly.groupby(resident_col)[col]
            .shift(1, fill_value=0)
        )

    keep_cols = [resident_col, 'year_month'] + \
        [f'{prefix}_hist_{c}' for c in agg_dict.keys()] + \
        [f'{prefix}_prev_{c}' for c in agg_dict.keys()]
    monthly = monthly[keep_cols]

    panel_df = panel_df.merge(monthly, on=[resident_col, 'year_month'], how='left')
    return panel_df


def merge_lagged_monthly_stats(panel_df, stats_df, resident_col='resident_id'):
    """
    Merge monthly statistics with a 1-month lag.
    For target month M, we use stats from month M-1.
    """
    lagged = stats_df.copy()
    lagged['year_month'] = lagged['year_month'] + pd.DateOffset(months=1)
    feat_cols = [c for c in lagged.columns if c not in [resident_col, 'year_month']]
    lagged.rename(columns={c: f'{c}_prev' for c in feat_cols}, inplace=True)
    panel_df = panel_df.merge(lagged, on=[resident_col, 'year_month'], how='left')
    return panel_df


def compute_active_prev_month(source_df, start_col, end_col, resident_col,
                               agg_dict, months_range):
    """
    Snapshot of active records as of end of PREVIOUS month.
    Returns with year_month = target month.
    """
    rows = []
    for month_start in months_range:
        prev_month_start = month_start - pd.DateOffset(months=1)
        prev_month_end = prev_month_start + pd.offsets.MonthEnd(0)
        active = source_df[
            (source_df[start_col] <= prev_month_end) &
            (source_df[end_col].isna() | (source_df[end_col] >= prev_month_start))
        ]
        if len(active) > 0:
            agg_result = active.groupby(resident_col).agg(**agg_dict).reset_index()
            agg_result['year_month'] = month_start
            rows.append(agg_result)
    if rows:
        return pd.concat(rows, ignore_index=True)
    else:
        cols = [resident_col, 'year_month'] + list(agg_dict.keys())
        return pd.DataFrame(columns=cols)


# ---- 3a. INCIDENTS ----
inc = tables['incidents'][tables['incidents']['strikeout'] == False].copy()

panel = merge_historical_features(
    panel, inc, 'occurred_at', 'resident_id',
    agg_dict={'count': ('incident_id', 'nunique')},
    prefix='incidents'
)

for itype in ['Fall', 'Wound', 'Elopement', 'Altercation', 'Medication Error', 'Choking']:
    inc_sub = inc[inc['incident_type'] == itype].copy()
    safe_name = itype.lower().replace(' ', '_')
    panel = merge_historical_features(
        panel, inc_sub, 'occurred_at', 'resident_id',
        agg_dict={'count': ('incident_id', 'nunique')},
        prefix=f'inc_{safe_name}'
    )

print(f"  After incidents: {panel.shape}")

# ---- 3b. INJURIES ----
inj = tables['injuries'].merge(
    inc[['incident_id', 'resident_id', 'occurred_at']].drop_duplicates(),
    on='incident_id', how='left'
)
panel = merge_historical_features(
    panel, inj, 'occurred_at', 'resident_id',
    agg_dict={
        'count': ('injury_id', 'nunique'),
        'post_incident': ('is_post_incident', 'sum'),
    },
    prefix='injuries'
)

# ---- 3c. FACTORS ----
fac = tables['factors'].merge(
    inc[['incident_id', 'resident_id', 'occurred_at']].drop_duplicates(),
    on='incident_id', how='left'
)
panel = merge_historical_features(
    panel, fac.dropna(subset=['resident_id']), 'occurred_at', 'resident_id',
    agg_dict={
        'count': ('factor_id', 'nunique'),
        'types': ('factor_type', 'nunique'),
    },
    prefix='factors'
)
print(f"  After injuries & factors: {panel.shape}")

# ---- 3d. DIAGNOSES ----
diag = tables['diagnoses'][tables['diagnoses']['strikeout'] == False].copy()

panel = merge_historical_features(
    panel, diag, 'onset_at', 'resident_id',
    agg_dict={
        'count': ('diagnosis_id', 'nunique'),
        'distinct_icd': ('icd_10_code', 'nunique'),
    },
    prefix='diag'
)

diag_active = compute_active_prev_month(
    diag, 'onset_at', 'resolved_at', 'resident_id',
    agg_dict={
        'diag_active_prev_count': ('diagnosis_id', 'nunique'),
        'diag_active_prev_distinct_icd': ('icd_10_code', 'nunique'),
    },
    months_range=months
)
panel = panel.merge(diag_active, on=['resident_id', 'year_month'], how='left')

diag['icd_chapter'] = diag['icd_10_code'].str[0].fillna('X')
icd_rows = []
for month_start in months:
    prev_start = month_start - pd.DateOffset(months=1)
    prev_end = prev_start + pd.offsets.MonthEnd(0)
    active_d = diag[
        (diag['onset_at'] <= prev_end) &
        (diag['resolved_at'].isna() | (diag['resolved_at'] >= prev_start))
    ]
    if len(active_d) > 0:
        pivot = active_d.groupby(['resident_id', 'icd_chapter'])['diagnosis_id'].nunique().unstack(fill_value=0)
        pivot.columns = [f'icd_ch_{c}_prev' for c in pivot.columns]
        pivot = pivot.reset_index()
        pivot['year_month'] = month_start
        icd_rows.append(pivot)

if icd_rows:
    icd_prev = pd.concat(icd_rows, ignore_index=True)
    panel = panel.merge(icd_prev, on=['resident_id', 'year_month'], how='left')
print(f"  After diagnoses: {panel.shape}")

# ---- 3e. CARE PLANS ----
cp = tables['care_plans'].copy()

cp_active = compute_active_prev_month(
    cp[cp['strikeout'] == False], 'initiated_at', 'closed_at', 'resident_id',
    agg_dict={'care_plan_active_prev_count': ('care_plan_id', 'nunique')},
    months_range=months
)
panel = panel.merge(cp_active, on=['resident_id', 'year_month'], how='left')

panel = merge_historical_features(
    panel, cp, 'initiated_at', 'resident_id',
    agg_dict={
        'count': ('care_plan_id', 'nunique'),
        'strikeout': ('strikeout', 'sum'),
    },
    prefix='careplan'
)
print(f"  After care plans: {panel.shape}")

# ---- 3f. NEEDS ----
needs = tables['needs'].copy()
needs_clean = needs[needs['strikeout'] == False]

needs_active = compute_active_prev_month(
    needs_clean, 'initiated_at', 'resolved_at', 'resident_id',
    agg_dict={
        'needs_active_prev_count': ('need_id', 'nunique'),
        'needs_active_prev_types': ('need_type', 'nunique'),
        'needs_active_prev_categories': ('need_category', 'nunique'),
    },
    months_range=months
)
panel = panel.merge(needs_active, on=['resident_id', 'year_month'], how='left')

need_cat_rows = []
for month_start in months:
    prev_start = month_start - pd.DateOffset(months=1)
    prev_end = prev_start + pd.offsets.MonthEnd(0)
    active_n = needs_clean[
        (needs_clean['initiated_at'] <= prev_end) &
        (needs_clean['resolved_at'].isna() | (needs_clean['resolved_at'] >= prev_start))
    ]
    if len(active_n) > 0:
        pivot = active_n.groupby(['resident_id', 'need_category'])['need_id'].nunique().unstack(fill_value=0)
        pivot.columns = [f'need_cat_{c.lower().replace(" ","_").replace("-","_").replace("/","_")}_prev' for c in pivot.columns]
        pivot = pivot.reset_index()
        pivot['year_month'] = month_start
        need_cat_rows.append(pivot)

if need_cat_rows:
    need_cat_prev = pd.concat(need_cat_rows, ignore_index=True)
    panel = panel.merge(need_cat_prev, on=['resident_id', 'year_month'], how='left')
print(f"  After needs: {panel.shape}")

# ---- 3g. VITALS (lagged by 1 month) ----
vit = tables['vitals'][tables['vitals']['strikeout'] == False].copy()
vit['_ym'] = vit['measured_at'].dt.to_period('M').dt.to_timestamp()

vit_types = vit['vital_type'].dropna().unique()
for vtype in vit_types:
    vt = vit[vit['vital_type'] == vtype].copy()
    safe_vt = vtype.lower().replace(' ', '_').replace('-', '_')
    monthly_stats = vt.groupby(['resident_id', '_ym']).agg(
        mean=('value', 'mean'),
        std=('value', 'std'),
        min_val=('value', 'min'),
        max_val=('value', 'max'),
        count=('value', 'count'),
    ).reset_index()
    monthly_stats.rename(columns={'_ym': 'year_month'}, inplace=True)
    monthly_stats.columns = ['resident_id', 'year_month'] + \
        [f'vital_{safe_vt}_{c}' for c in ['mean', 'std', 'min', 'max', 'count']]
    panel = merge_lagged_monthly_stats(panel, monthly_stats)

bp = vit[vit['dystolic_value'].notna()].copy()
if len(bp) > 0:
    bp_monthly = bp.groupby(['resident_id', '_ym'])['dystolic_value'].agg(
        ['mean', 'std', 'min', 'max']
    ).reset_index()
    bp_monthly.rename(columns={'_ym': 'year_month'}, inplace=True)
    bp_monthly.columns = ['resident_id', 'year_month',
                           'bp_diastolic_mean', 'bp_diastolic_std',
                           'bp_diastolic_min', 'bp_diastolic_max']
    panel = merge_lagged_monthly_stats(panel, bp_monthly)
print(f"  After vitals: {panel.shape}")

# ---- 3h. LAB REPORTS (lagged + historical) ----
labs = tables['lab_reports'].copy()
labs['_ym'] = labs['reported_at'].dt.to_period('M').dt.to_timestamp()

lab_monthly = labs.groupby(['resident_id', '_ym']).agg(
    lab_count=('lab_report_id', 'nunique'),
    lab_distinct_names=('lab_name', 'nunique'),
    lab_abnormal=('severity_status', lambda x: x.str.lower().str.contains('abnormal|critical|high|low', na=False).sum()),
).reset_index().rename(columns={'_ym': 'year_month'})
panel = merge_lagged_monthly_stats(panel, lab_monthly)

lab_sev = labs.groupby(['resident_id', '_ym', 'severity_status'])['lab_report_id'].nunique().reset_index()
lab_sev.rename(columns={'_ym': 'year_month', 'lab_report_id': 'count'}, inplace=True)
if len(lab_sev) > 0:
    lab_sev_pivot = lab_sev.pivot_table(
        index=['resident_id', 'year_month'], columns='severity_status',
        values='count', fill_value=0
    )
    lab_sev_pivot.columns = [f'lab_sev_{str(c).lower().replace(" ","_")}' for c in lab_sev_pivot.columns]
    lab_sev_pivot = lab_sev_pivot.reset_index()
    panel = merge_lagged_monthly_stats(panel, lab_sev_pivot)

panel = merge_historical_features(
    panel, labs, 'reported_at', 'resident_id',
    agg_dict={'count': ('lab_report_id', 'nunique')},
    prefix='labs'
)
print(f"  After labs: {panel.shape}")

# ---- 3i. MEDICATIONS (lagged + historical) ----
med = tables['medications'].copy()
med['_ym'] = med['scheduled_at'].dt.to_period('M').dt.to_timestamp()

med_monthly = med.groupby(['resident_id', '_ym']).agg(
    med_total=('medication_id', 'nunique'),
    med_distinct=('description', 'nunique'),
).reset_index().rename(columns={'_ym': 'year_month'})
panel = merge_lagged_monthly_stats(panel, med_monthly)

med_status = med.groupby(['resident_id', '_ym', 'status'])['medication_id'].nunique().reset_index()
med_status.rename(columns={'_ym': 'year_month', 'medication_id': 'count'}, inplace=True)
if len(med_status) > 0:
    med_status_pivot = med_status.pivot_table(
        index=['resident_id', 'year_month'], columns='status',
        values='count', fill_value=0
    )
    med_status_pivot.columns = [f'med_status_{str(c).lower().replace(" ","_").replace("-","_")}' for c in med_status_pivot.columns]
    med_status_pivot = med_status_pivot.reset_index()
    panel = merge_lagged_monthly_stats(panel, med_status_pivot)

panel = merge_historical_features(
    panel, med, 'scheduled_at', 'resident_id',
    agg_dict={'count': ('medication_id', 'nunique')},
    prefix='meds'
)
print(f"  After medications: {panel.shape}")

# ---- 3j. PHYSICIAN ORDERS (active as of prev month) ----
po = tables['physician_orders'].copy()

po_active = compute_active_prev_month(
    po, 'start_at', 'end_at', 'resident_id',
    agg_dict={
        'orders_active_prev_count': ('order_id', 'nunique'),
        'orders_active_prev_categories': ('category', 'nunique'),
    },
    months_range=months
)
panel = panel.merge(po_active, on=['resident_id', 'year_month'], how='left')

po_cat_rows = []
for month_start in months:
    prev_start = month_start - pd.DateOffset(months=1)
    prev_end = prev_start + pd.offsets.MonthEnd(0)
    active_po = po[
        (po['start_at'] <= prev_end) &
        (po['end_at'].isna() | (po['end_at'] >= prev_start))
    ]
    if len(active_po) > 0:
        pivot = active_po.groupby(['resident_id', 'category'])['order_id'].nunique().unstack(fill_value=0)
        pivot.columns = [f'order_cat_{c.lower().replace(" ","_").replace("-","_").replace("/","_")}_prev' for c in pivot.columns]
        pivot = pivot.reset_index()
        pivot['year_month'] = month_start
        po_cat_rows.append(pivot)

if po_cat_rows:
    po_cat_prev = pd.concat(po_cat_rows, ignore_index=True)
    panel = panel.merge(po_cat_prev, on=['resident_id', 'year_month'], how='left')
print(f"  After physician orders: {panel.shape}")

# ---- 3k. THERAPY TRACKS (active as of prev month) ----
tt = tables['therapy_tracks'].copy()

tt_active = compute_active_prev_month(
    tt, 'start_at', 'end_at', 'resident_id',
    agg_dict={
        'therapy_active_prev_count': ('therapy_id', 'nunique'),
        'therapy_active_prev_disciplines': ('discipline', 'nunique'),
    },
    months_range=months
)
panel = panel.merge(tt_active, on=['resident_id', 'year_month'], how='left')

tt_disc_rows = []
for month_start in months:
    prev_start = month_start - pd.DateOffset(months=1)
    prev_end = prev_start + pd.offsets.MonthEnd(0)
    active_tt = tt[
        (tt['start_at'] <= prev_end) &
        (tt['end_at'].isna() | (tt['end_at'] >= prev_start))
    ]
    if len(active_tt) > 0:
        pivot = active_tt.groupby(['resident_id', 'discipline'])['therapy_id'].nunique().unstack(fill_value=0)
        pivot.columns = [f'therapy_{c.lower().replace(" ","_").replace("-","_")}_prev' for c in pivot.columns]
        pivot = pivot.reset_index()
        pivot['year_month'] = month_start
        tt_disc_rows.append(pivot)

if tt_disc_rows:
    tt_disc_prev = pd.concat(tt_disc_rows, ignore_index=True)
    panel = panel.merge(tt_disc_prev, on=['resident_id', 'year_month'], how='left')
print(f"  After therapy tracks: {panel.shape}")

# ---- 3l. HOSPITAL TRANSFERS (historical only) ----
ht = tables['hospital_transfers'].copy()
panel = merge_historical_features(
    panel, ht, 'effective_date', 'resident_id',
    agg_dict={
        'count': ('transfer_id', 'nunique'),
        'emergency': ('emergency_flag', lambda x: (x == 1).sum()),
    },
    prefix='transfers'
)

# ---- 3m. HOSPITAL ADMISSIONS (historical only) ----
ha = tables['hospital_admissions'].copy()
panel = merge_historical_features(
    panel, ha, 'effective_date', 'resident_id',
    agg_dict={'count': ('admission_id', 'nunique')},
    prefix='hosp_adm'
)
print(f"  After hospital data: {panel.shape}")

# ---- 3n. ADL RESPONSES (lagged by 1 month) ----
adl = tables['adl_responses'].copy()
adl['response_num'] = pd.to_numeric(adl['response'], errors='coerce')
adl['_ym'] = adl['assessment_date'].dt.to_period('M').dt.to_timestamp()

adl_monthly = adl.groupby(['resident_id', '_ym']).agg(
    adl_count=('adl_response_id', 'nunique'),
    adl_mean_response=('response_num', 'mean'),
    adl_mean_change=('adl_change', 'mean'),
    adl_neg_changes=('adl_change', lambda x: (x < 0).sum()),
    adl_pos_changes=('adl_change', lambda x: (x > 0).sum()),
    adl_distinct_activities=('activity', 'nunique'),
).reset_index().rename(columns={'_ym': 'year_month'})
panel = merge_lagged_monthly_stats(panel, adl_monthly)

adl_cat = adl.groupby(['resident_id', '_ym', 'category'])['response_num'].mean().reset_index()
adl_cat.rename(columns={'_ym': 'year_month'}, inplace=True)
if len(adl_cat) > 0:
    adl_cat_pivot = adl_cat.pivot_table(
        index=['resident_id', 'year_month'], columns='category',
        values='response_num', fill_value=0
    )
    adl_cat_pivot.columns = [f'adl_avg_{c.lower().replace(" ","_").replace("-","_")}' for c in adl_cat_pivot.columns]
    adl_cat_pivot = adl_cat_pivot.reset_index()
    panel = merge_lagged_monthly_stats(panel, adl_cat_pivot)
print(f"  After ADL: {panel.shape}")

# ---- 3o. GG RESPONSES (lagged by 1 month) ----
gg = tables['gg_responses'].copy()
gg['_ym'] = gg['created_at'].dt.to_period('M').dt.to_timestamp()

gg_monthly = gg.groupby(['resident_id', '_ym']).agg(
    gg_count=('gg_response_id', 'nunique'),
    gg_mean_response_code=('response_code', 'mean'),
    gg_mean_change=('change', 'mean'),
    gg_neg_changes=('change', lambda x: (x < 0).sum()),
    gg_pos_changes=('change', lambda x: (x > 0).sum()),
    gg_distinct_tasks=('task_name', 'nunique'),
).reset_index().rename(columns={'_ym': 'year_month'})
panel = merge_lagged_monthly_stats(panel, gg_monthly)

gg_grp = gg.groupby(['resident_id', '_ym', 'task_group'])['response_code'].mean().reset_index()
gg_grp.rename(columns={'_ym': 'year_month'}, inplace=True)
if len(gg_grp) > 0:
    gg_grp_pivot = gg_grp.pivot_table(
        index=['resident_id', 'year_month'], columns='task_group',
        values='response_code', fill_value=0
    )
    gg_grp_pivot.columns = [f'gg_avg_{c.lower().replace(" ","_").replace("-","_").replace("/","_")}' for c in gg_grp_pivot.columns]
    gg_grp_pivot = gg_grp_pivot.reset_index()
    panel = merge_lagged_monthly_stats(panel, gg_grp_pivot)
print(f"  After GG: {panel.shape}")

# ---- 3p. DOCUMENT TAGS (historical only) ----
dt = tables['document_tags'].copy()
panel = merge_historical_features(
    panel, dt, 'created_at', 'resident_id',
    agg_dict={
        'count': ('document_tag_id', 'nunique'),
        'types': ('doc_type', 'nunique'),
        'avg_confidence': ('match_confidence', 'mean'),
    },
    prefix='doctags'
)
print(f"  After document tags: {panel.shape}")

# ============================================================
# 4. BUILD TARGET VARIABLES & SPLIT INTO 7 DATAFRAMES
# ============================================================
print("\n[4/5] Building target variables and splitting by claim type...")

inc['_ym'] = inc['occurred_at'].dt.to_period('M').dt.to_timestamp()

CLAIM_TYPES = {
    'fall':             'Fall',
    'wound':            'Wound',
    'medication_error': 'Medication Error',
    'elopement':        'Elopement',
    'altercation':      'Altercation',
    'choking':          'Choking',
}

ht_target = tables['hospital_transfers'].copy()
ht_target['_ym'] = ht_target['effective_date'].dt.to_period('M').dt.to_timestamp()
rth_monthly = ht_target.groupby(['resident_id', '_ym']).agg(
    rth_count=('transfer_id', 'nunique'),
).reset_index().rename(columns={'_ym': 'year_month'})
rth_monthly['target_rth'] = 1

CLAIM_TYPES['rth'] = '__rth__'

numeric_cols = panel.select_dtypes(include=[np.number]).columns
panel[numeric_cols] = panel[numeric_cols].fillna(0)

# ============================================================
# 5. EXPORT ONE DATAFRAME PER CLAIM TYPE
# ============================================================
print("\n[5/5] Exporting dataframes...\n")

output_dfs = {}

for claim_key, claim_value in CLAIM_TYPES.items():
    df = panel.copy()

    if claim_key == 'rth':
        df = df.merge(rth_monthly[['resident_id', 'year_month', 'target_rth']],
                       on=['resident_id', 'year_month'], how='left')
        df['target_rth'] = df['target_rth'].fillna(0).astype(int)
        target_col = 'target_rth'
    else:
        inc_type = inc[inc['incident_type'] == claim_value].copy()
        target_monthly = inc_type.groupby(['resident_id', '_ym']).agg(
            target_count=('incident_id', 'nunique'),
        ).reset_index().rename(columns={'_ym': 'year_month'})
        target_monthly[f'target_{claim_key}'] = 1

        df = df.merge(
            target_monthly[['resident_id', 'year_month', f'target_{claim_key}']],
            on=['resident_id', 'year_month'], how='left'
        )
        df[f'target_{claim_key}'] = df[f'target_{claim_key}'].fillna(0).astype(int)
        target_col = f'target_{claim_key}'

    df = df.sort_values(['resident_id', 'year_month']).reset_index(drop=True)

    total = len(df)
    positive = df[target_col].sum()
    residents = df['resident_id'].nunique()
    n_features = len([c for c in df.columns if c not in ['resident_id', 'facility_id', 'year_month', target_col]])

    print(f"  {claim_key.upper()}")
    print(f"    Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"    Residents: {residents:,}")
    print(f"    Target '{target_col}': {int(positive):,} / {total:,} ({positive/total*100:.2f}%)")
    print(f"    Features: {n_features}")

    filename = os.path.join(OUTPUT_DIR, f'claims_{claim_key}_monthly.parquet')
    df.to_parquet(filename, index=False)
    print(f"    Saved: {filename}\n")

    output_dfs[claim_key] = df

# ============================================================
# FINAL SUMMARY
# ============================================================
print("=" * 60)
print("ALL DONE!")
print("=" * 60)
print(f"\nGenerated 7 monthly panel dataframes in: {OUTPUT_DIR}")
for claim_key in CLAIM_TYPES:
    print(f"  - claims_{claim_key}_monthly.parquet")

print(f"\nFeature design (NO data leakage):")
print(f"  _hist_  = cumulative from all months BEFORE target month")
print(f"  _prev_  = value from the month immediately before target")
print(f"  _prev   = active snapshot as of end of previous month")
print(f"  Vitals/Labs/Meds/ADL/GG stats = lagged by 1 month")
print(f"\n  NO same-month (_curr_) features are included.")
