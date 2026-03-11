#!/usr/bin/env python
"""
GeoXCP NYC Income Data Generation Script
==========================================
Generates pre-computed results for the NYC Focus Circle visualization.

Dataset  : datasets/nyc_income.csv  (NYC census tract median incomes)
Target   : medianinco  (median household income, USD)
Method   : XGBoost + TreeSHAP + GeoXCP  (same pipeline as Seattle)
Output   : data/results_nyc.js  (loaded via window.GeoXCPDataNYC)

Key difference from Seattle:
  - All 2110 census tract points get GeoXCP explanations (not just test set),
    so the Focus Circle has dense spatial coverage for aggregation.

Usage:
    /opt/anaconda3/envs/UncertaintyGeoXAI/bin/python generate_data_nyc.py
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from GeoConformalizedExplainer import GeoConformalizedExplainer

t_start = time.time()

# ─────────────────────────────────────────────
# 1. Load and feature-engineer data
# ─────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading NYC Income data...")
print("=" * 60)

data_path = os.path.join(PROJECT_ROOT, 'datasets', 'nyc_income.csv')
data = pd.read_csv(data_path)
data = data.dropna(subset=['medianinco'])
data['medianinco'] = pd.to_numeric(data['medianinco'], errors='coerce')
data = data.dropna(subset=['medianinco'])

# Derived features (mirrors NYCIncome.ipynb)
data['male_p']         = data['male']       / (data['male'] + data['female'])
data['popover18_p']    = data['popover18']  / data['poptot']
data['european_p']     = data['european']   / data['poptot']
data['mixed_p']        = data['mixed']      / data['poptot']
data['asian_p']        = data['asian']      / data['poptot']
data['hispanic_p']     = data['hispanic']   / data['poptot']
data['african_p']      = data['african']    / data['poptot']
data['highschool_p']   = data['highschool'] / data['popover18']
data['bachelor_p']     = data['bachelor']   / data['popover18']
data['master_p']       = data['master']     / data['popover18']
data['doctorate_p']    = data['doctorate']  / data['popover18']
data['profes_p']       = data['profession'] / data['popinlabou']
data['labor_part_p']   = data['popinlabou'] / data['popover18']
data['pub_assist_p']   = data['withpubass'] / data['households']
data['ssi_assist_p']   = data['withssi']    / data['households']
data['long_commute_p'] = (data['com_90plus'] + data['comm_60_89']) / data['popinlabou']
data['dropout_p']      = (data['maledrop']   + data['femaledrop']) / data['poptot']
data['medianage']      = data['medianage'].fillna(data['medianage'].median())

data = data.replace([np.inf, -np.inf], 0).fillna(0)

# Feature lists
FEATURES = [
    'UNEMP_RATE', 'gini', 'profes_p', 'master_p', 'labor_part_p',
    'popover18_p', 'male_p', 'european_p', 'asian_p', 'african_p', 'hispanic_p',
    'highschool_p', 'bachelor_p', 'doctorate_p',
    'pub_assist_p', 'long_commute_p', 'dropout_p', 'medianage',
    'popdty', 'lon', 'lat'
]

FEATURE_LABELS = {
    'UNEMP_RATE':      'Unemployment Rate',
    'gini':            'Income Inequality (Gini)',
    'profes_p':        'Professional Occupation %',
    'master_p':        "Master's Degree %",
    'labor_part_p':    'Labor Participation Rate',
    'popover18_p':     'Pop. Over 18 %',
    'male_p':          'Male Pop. %',
    'european_p':      'White Pop. %',
    'asian_p':         'Asian Pop. %',
    'african_p':       'African Pop. %',
    'hispanic_p':      'Hispanic Pop. %',
    'highschool_p':    'High School Edu. %',
    'bachelor_p':      "Bachelor's Edu. %",
    'doctorate_p':     'Doctoral Edu. %',
    'pub_assist_p':    'Public Assistance Rate',
    'long_commute_p':  'Extreme Commute Rate',
    'dropout_p':       'Dropout Rate',
    'medianage':       'Median Age',
    'popdty':          'Population Density',
    'lon':             'Longitude',
    'lat':             'Latitude'
}

X   = data[FEATURES]
y   = data['medianinco']
loc = data[['lon', 'lat']]

print(f"  Loaded {len(data)} census tracts | {len(FEATURES)} features")
print(f"  Income range: ${y.min():,.0f} – ${y.max():,.0f}")

# ─────────────────────────────────────────────
# 2. Train / Calib / Test split  (80 / 10 / 10)
# ─────────────────────────────────────────────
print("\nStep 2: Splitting data (80/10/10)...")

X_train, X_temp, y_train, y_temp, loc_train, loc_temp = train_test_split(
    X, y, loc, train_size=0.8, random_state=42
)
X_calib, X_test, y_calib, y_test, loc_calib, loc_test = train_test_split(
    X_temp, y_temp, loc_temp, train_size=0.5, random_state=42
)
print(f"  Train: {len(X_train)}  |  Calib: {len(X_calib)}  |  Test: {len(X_test)}")

# ─────────────────────────────────────────────
# 3. Train XGBoost
# ─────────────────────────────────────────────
print("\nStep 3: Training XGBoost (n_estimators=500, max_depth=3)...")
t0 = time.time()

model = xgb.XGBRegressor(
    n_estimators=500, max_depth=3,
    min_child_weight=1.0, colsample_bytree=1.0,
    random_state=42
)
model.fit(X_train.values, y_train.values)

from sklearn.metrics import r2_score, root_mean_squared_error
train_r2   = r2_score(y_train, model.predict(X_train.values))
test_r2    = r2_score(y_test,  model.predict(X_test.values))
test_rmse  = root_mean_squared_error(y_test, model.predict(X_test.values))
print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Train R²={train_r2:.4f} | Test R²={test_r2:.4f} | RMSE=${test_rmse:,.0f}")

# ─────────────────────────────────────────────
# 4. TreeSHAP
# ─────────────────────────────────────────────
print("\nStep 4: Setting up TreeSHAP...")
tree_explainer = shap.TreeExplainer(model)

def tree_shap_f(x):
    return tree_explainer.shap_values(x)

print(f"  Validation shape: {tree_shap_f(X_test.values[:3]).shape}")

# ─────────────────────────────────────────────
# 5. GeoXCP pipeline
# ─────────────────────────────────────────────
print("\nStep 5: Running GeoXCP pipeline (MLP SHAP predictor)...")
print("  (MLP training ~2–5 min)")

explainer_geo = GeoConformalizedExplainer(
    prediction_f    = model.predict,
    x_train         = X_train,
    x_calib         = X_calib,
    coord_calib     = loc_calib.values,
    shap_value_f    = tree_shap_f,
    miscoverage_level = 0.1,   # 90% coverage
    band_width      = 0.15,    # degrees (~17 km)
    feature_names   = FEATURES,
    is_single_model = True     # shared MLP
)

t0 = time.time()

# Generate explanations for ALL 2110 points so the Focus Circle has
# dense spatial coverage regardless of where the user places it.
results = explainer_geo.uncertainty_aware_explain(
    x_test     = X,
    coord_test = loc.values
)
print(f"  GeoXCP done in {time.time()-t0:.1f}s")

# ─────────────────────────────────────────────
# 6. Accuracy summary (on held-out test set)
# ─────────────────────────────────────────────
print("\nAccuracy summary (test set):")
# Re-run on test set only to get proper out-of-sample accuracy metrics
results_test = explainer_geo.uncertainty_aware_explain(
    x_test     = X_test,
    coord_test = loc_test.values
)
acc = results_test.accuracy_summary()
print(acc[['coverage_probability', 'R2', 'RMSE']].to_string())

# ─────────────────────────────────────────────
# 7. Build output JSON
# ─────────────────────────────────────────────
print("\nStep 6: Building output JSON...")

result_df = results.result     # all 2110 points
split_map = {}
for idx in X_train.index: split_map[idx] = 'train'
for idx in X_calib.index:  split_map[idx] = 'calib'
for idx in X_test.index:   split_map[idx] = 'test'

output = {
    "meta": {
        "dataset":          "NYC Census Tract Median Income",
        "n_total":          int(len(X)),
        "n_train":          int(len(X_train)),
        "n_calib":          int(len(X_calib)),
        "n_test":           int(len(X_test)),
        "miscoverage_level": 0.1,
        "target_coverage":  0.9,
        "bandwidth":        0.15,
        "train_r2":         float(train_r2),
        "test_r2":          float(test_r2),
        "test_rmse":        float(test_rmse),
        "income_min":       float(y.min()),
        "income_max":       float(y.max()),
        "model":            "XGBoost (n_estimators=500, max_depth=3)",
        "shap_method":      "TreeSHAP (TreeExplainer)",
        "shap_predictor":   "MLP (single model, early stopping)"
    },
    "features":       FEATURES,
    "feature_labels": FEATURE_LABELS,
    "accuracy":       {},
    "mean_abs_shap":  {},
    "global_uncertainty": {},
    "points":         []
}

# Per-feature accuracy (from test set)
for feature in FEATURES:
    label = FEATURE_LABELS[feature]
    if label in acc.index:
        row = acc.loc[label]
        output["accuracy"][feature] = {
            "coverage_probability": float(row['coverage_probability']),
            "R2":   float(row['R2']),
            "RMSE": float(row['RMSE']),
        }
    else:
        output["accuracy"][feature] = {"coverage_probability": None, "R2": None, "RMSE": None}

# Global stats (from all points)
for j, feature in enumerate(FEATURES):
    output["mean_abs_shap"][feature] = float(
        np.mean(np.abs(results.explanation_values[:, j]))
    )
    output["global_uncertainty"][feature] = float(
        results.geocp_results[j].uncertainty
    ) if hasattr(results.geocp_results[j], 'uncertainty') else float(
        np.mean(results.geocp_results[j].geo_uncertainty)
    )

# Per-point data (all 2110 tracts)
all_indices = X.index.tolist()
for i in range(len(result_df)):
    row  = result_df.iloc[i]
    orig_idx = all_indices[i]
    point = {
        "lon":    float(row['x']),
        "lat":    float(row['y']),
        "income": float(y.iloc[i]),
        "split":  split_map.get(orig_idx, 'test'),
        "features": {}
    }
    for feature in FEATURES:
        point["features"][feature] = {
            "shap":            float(row[f'{feature}_shap']),
            "pred":            float(row[f'{feature}_pred']),
            "geo_uncertainty": float(row[f'{feature}_geo_uncertainty']),
        }
    output["points"].append(point)

# ─────────────────────────────────────────────
# 8. Save
# ─────────────────────────────────────────────
os.makedirs(os.path.join(PROJECT_ROOT, 'data'), exist_ok=True)
out_js   = os.path.join(PROJECT_ROOT, 'data', 'results_nyc.js')
out_json = os.path.join(PROJECT_ROOT, 'data', 'results_nyc.json')

with open(out_js, 'w', encoding='utf-8') as f:
    f.write('// GeoXCP NYC Income Results\n')
    f.write('// Generated by generate_data_nyc.py\n')
    f.write('window.GeoXCPDataNYC = ')
    json.dump(output, f, separators=(',', ':'))
    f.write(';\n')

with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

size_kb = os.path.getsize(out_js) / 1024
print(f"  Saved: {out_js} ({size_kb:.1f} KB)")
print(f"  {len(output['points'])} census tracts × {len(FEATURES)} features")

total = time.time() - t_start
print(f"\n{'='*60}")
print(f"Done! Total time: {total/60:.1f} min ({total:.0f}s)")
print(f"{'='*60}")
print(f"\nNext step: open nyc.html in a browser (or run: python3 -m http.server 8765)")
