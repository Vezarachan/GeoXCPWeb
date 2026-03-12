#!/usr/bin/env python
"""
GeoXCP Data Generation - US Politics Voting
============================================
Generates pre-computed results for the US County-level Politics voting dataset.

Output: data/results_politics.js  (and .json)

Usage:
    /opt/anaconda3/envs/UncertaintyGeoXAI/bin/python generate_data_politics.py
"""

import sys
import os
import json
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
import shap
import pyproj
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from GeoConformalizedExplainer import GeoConformalizedExplainer

t_start = time.time()

# ==============================================================
# 1. Load and prepare data
# ==============================================================
print("=" * 60)
print("Step 1: Loading US Politics Voting data...")
print("=" * 60)

data_path = os.path.join(PROJECT_ROOT, 'datasets', 'US_Politics_Voting.csv')
data = pd.read_csv(data_path, index_col=0)

# Convert projected coordinates (EPSG:5070 NAD83/Conus Albers) to lat/lon
transformer = pyproj.Transformer.from_crs('EPSG:5070', 'EPSG:4326', always_xy=True)
lons, lats = transformer.transform(data.proj_x.values, data.proj_y.values)
data['lon'] = lons
data['lat'] = lats

feature_cols = [f'X{i}' for i in range(1, 15)]
feature_labels = {
    'X1':  'Sex Ratio',
    'X2':  'Black Pop. %',
    'X3':  'Hispanic Pop. %',
    'X4':  'Bach. Degree %',
    'X5':  'Median Income',
    'X6':  'Pop. 65+ %',
    'X7':  'Pop. 18-29 %',
    'X8':  'Gini Index',
    'X9':  'Manuf. Employ. %',
    'X10': 'Log Pop. Density',
    'X11': '3rd Party Vote %',
    'X12': 'Voter Turnout',
    'X13': 'Foreign Born %',
    'X14': 'Uninsured %',
}

y = data['Y']
X = data[feature_cols]
loc = data[['lon', 'lat']]

print(f"  Data loaded: {len(data)} counties, {len(feature_cols)} features")
print(f"  Y (Dem. Vote %): min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
print(f"  Lat range: {data.lat.min():.2f} – {data.lat.max():.2f}")
print(f"  Lon range: {data.lon.min():.2f} – {data.lon.max():.2f}")

# ==============================================================
# 2. Train / Calibration / Test split (80/10/10)
# ==============================================================
print("\nStep 2: Train/Calibration/Test split (80/10/10)...")

X_train, X_temp, y_train, y_temp, loc_train, loc_temp = train_test_split(
    X, y, loc, train_size=0.8, random_state=42
)
X_calib, X_test, y_calib, y_test, loc_calib, loc_test = train_test_split(
    X_temp, y_temp, loc_temp, train_size=0.5, random_state=42
)
print(f"  Train: {len(X_train)}  |  Calib: {len(X_calib)}  |  Test: {len(X_test)}")

# All points for map display
train_idx = set(X_train.index)
calib_idx = set(X_calib.index)
all_points_meta = []
for i in data.index:
    split = 'train' if i in train_idx else ('calib' if i in calib_idx else 'test')
    all_points_meta.append({
        'idx': int(i),
        'lon': float(data.loc[i, 'lon']),
        'lat': float(data.loc[i, 'lat']),
        'y_val': float(data.loc[i, 'Y']),
        'split': split
    })

# ==============================================================
# 3. Train XGBoost model
# ==============================================================
print("\nStep 3: Training XGBoost model (n_estimators=500, max_depth=3)...")
t0 = time.time()

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=3,
    min_child_weight=1.0,
    colsample_bytree=1.0,
    random_state=42
)
model.fit(X_train.values, y_train.values)

train_preds = model.predict(X_train.values)
test_preds  = model.predict(X_test.values)
train_r2    = r2_score(y_train, train_preds)
test_r2     = r2_score(y_test, test_preds)
test_rmse   = root_mean_squared_error(y_test, test_preds)

print(f"  XGBoost trained in {time.time()-t0:.1f}s")
print(f"  Train R²={train_r2:.4f} | Test R²={test_r2:.4f} | Test RMSE={test_rmse:.4f}")

# ==============================================================
# 4. TreeSHAP
# ==============================================================
print("\nStep 4: Setting up TreeSHAP...")
t0 = time.time()

tree_explainer = shap.TreeExplainer(model)
def tree_shap_f(x):
    return tree_explainer.shap_values(x)

sample_shap = tree_shap_f(X_test.values[:5])
print(f"  TreeSHAP ready. Shape: {sample_shap.shape}, Setup: {time.time()-t0:.2f}s")

# ==============================================================
# 5. GeoXCP pipeline
# ==============================================================
print("\nStep 5: Running GeoXCP pipeline (bandwidth=2.0°, national scale)...")
t0 = time.time()

explainer = GeoConformalizedExplainer(
    prediction_f=model.predict,
    x_train=X_train,
    x_calib=X_calib,
    coord_calib=loc_calib.values,
    shap_value_f=tree_shap_f,
    miscoverage_level=0.1,
    band_width=2.0,          # 2 degrees ≈ 220 km, appropriate for national data
    feature_names=feature_cols,
    is_single_model=True
)

results = explainer.uncertainty_aware_explain(
    x_test=X_test,
    coord_test=loc_test.values
)
print(f"  GeoXCP completed in {time.time()-t0:.1f}s")

acc_summary = results.accuracy_summary()
print(acc_summary.to_string())

# ==============================================================
# 6. Build output
# ==============================================================
print("\nStep 6: Building output structure...")

result_df = results.result

# Map test indices back to original data indices
test_original_idx = list(X_test.index)

# Build test point lookup
test_lookup = {}
for i in range(len(result_df)):
    orig_idx = test_original_idx[i]
    row = result_df.iloc[i]
    feat_dict = {}
    for feat in feature_cols:
        feat_dict[feat] = {
            'shap':            float(row[f'{feat}_shap']),
            'pred':            float(row[f'{feat}_pred']),
            'geo_uncertainty': float(row[f'{feat}_geo_uncertainty']),
            'upper_bound':     float(row[f'{feat}_upper_bound']),
            'lower_bound':     float(row[f'{feat}_lower_bound']),
            'shap_abs':        float(row[f'{feat}_shap_abs']),
            'value':           float(row[f'{feat}_value']),
        }
    test_lookup[orig_idx] = feat_dict

# All points with features only for test points
points = []
for pm in all_points_meta:
    pt = {
        'lon':   pm['lon'],
        'lat':   pm['lat'],
        'y_val': pm['y_val'],
        'split': pm['split'],
    }
    if pm['idx'] in test_lookup:
        pt['features'] = test_lookup[pm['idx']]
    points.append(pt)

# Mean abs SHAP and global uncertainty
mean_abs_shap = {}
global_uncertainty = {}
for j, feat in enumerate(feature_cols):
    mean_abs_shap[feat]      = float(np.mean(np.abs(results.explanation_values[:, j])))
    global_uncertainty[feat] = float(results.geocp_results[j].uncertainty)

# Accuracy
accuracy = {}
for feat in feature_cols:
    row = acc_summary.loc[feat]
    accuracy[feat] = {
        'coverage_probability': float(row['coverage_probability']),
        'R2':   float(row['R2']),
        'RMSE': float(row['RMSE']),
        'SHAP_Var':      float(row['SHAP_Var']),
        'Pred_SHAP_Var': float(row['Pred_SHAP_Var']),
    }

output = {
    'meta': {
        'dataset':        'US County Politics Voting',
        'y_field':        'y_val',
        'y_label':        'Dem. Vote %',
        'y_unit':         '%',
        'n_total':        int(len(data)),
        'n_train':        int(len(X_train)),
        'n_calib':        int(len(X_calib)),
        'n_test':         int(len(X_test)),
        'miscoverage_level': 0.1,
        'target_coverage':   0.9,
        'bandwidth':         2.0,
        'train_r2':  float(train_r2),
        'test_r2':   float(test_r2),
        'test_rmse': float(test_rmse),
        'y_min': float(y.min()),
        'y_max': float(y.max()),
        'lat_min': float(data.lat.min()),
        'lat_max': float(data.lat.max()),
        'lon_min': float(data.lon.min()),
        'lon_max': float(data.lon.max()),
        'model':          'XGBoost (n_estimators=500, max_depth=3)',
        'shap_method':    'TreeSHAP (TreeExplainer)',
        'shap_predictor': 'MLP (2048×2048×2048, early stopping)',
    },
    'features':          feature_cols,
    'feature_labels':    feature_labels,
    'accuracy':          accuracy,
    'mean_abs_shap':     mean_abs_shap,
    'global_uncertainty':global_uncertainty,
    'points':            points,
}

# ==============================================================
# 7. Save
# ==============================================================
output_js   = os.path.join(PROJECT_ROOT, 'data', 'results_politics.js')
output_json = os.path.join(PROJECT_ROOT, 'data', 'results_politics.json')
os.makedirs(os.path.dirname(output_js), exist_ok=True)

with open(output_js, 'w', encoding='utf-8') as f:
    f.write('// GeoXCP US Politics Voting Results\n')
    f.write('// Generated by generate_data_politics.py\n')
    f.write('window.GeoXCPDataPolitics = ')
    json.dump(output, f, separators=(',', ':'))
    f.write(';\n')

with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)

sz = os.path.getsize(output_js) / 1024
print(f"\nSaved: {output_js} ({sz:.1f} KB)")
print(f"Points: {len(points)} | Test w/ features: {sum(1 for p in points if 'features' in p)}")
print(f"Total time: {(time.time()-t_start)/60:.1f} min")
