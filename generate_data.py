#!/usr/bin/env python
"""
GeoXCP Data Generation Script
==============================
Generates pre-computed results for the GeoXCP interactive web visualization.

Uses TreeSHAP (XGBoost TreeExplainer) instead of KernelSHAP for fast computation.
- KernelSHAP: ~18 minutes for 3000 samples
- TreeSHAP:   ~5 seconds for 3000 samples

Output: data/results.js  (loadable as <script src="data/results.js">)

Usage:
    python generate_data.py
    # or with the project conda env:
    /opt/anaconda3/envs/UncertaintyGeoXAI/bin/python generate_data.py
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
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from GeoConformalizedExplainer import GeoConformalizedExplainer

t_start = time.time()

# ==============================================================
# 1. Load and prepare data
# ==============================================================
print("=" * 60)
print("Step 1: Loading Seattle House Price data...")
print("=" * 60)

data_path = os.path.join(PROJECT_ROOT, 'datasets', 'seattle_sample_3k.csv')
data = pd.read_csv(data_path)

data = gpd.GeoDataFrame(
    data, crs="EPSG:32610",
    geometry=gpd.points_from_xy(x=data.UTM_X, y=data.UTM_Y)
)
data = data.to_crs(4326)
data['lon'] = data.geometry.get_coordinates()['x']
data['lat'] = data.geometry.get_coordinates()['y']
data['price'] = np.power(10, data['log_price']) / 10000

y = data.price
X = data[['bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'condition',
           'waterfront', 'view', 'age', 'lon', 'lat']]
loc = data[['lon', 'lat']]
feature_names = X.columns.tolist()

print(f"  Data loaded: {len(data)} samples, {len(feature_names)} features")
print(f"  Features: {feature_names}")
print(f"  Price range: ${data.price.min():.1f}k - ${data.price.max():.1f}k (×$10,000)")

# ==============================================================
# 2. Train / Calibration / Test split
# ==============================================================
print("\nStep 2: Train/Calibration/Test split (80/10/10)...")

X_train, X_temp, y_train, y_temp, loc_train, loc_temp = train_test_split(
    X, y, loc, train_size=0.8, random_state=42
)
X_calib, X_test, y_calib, y_test, loc_calib, loc_test = train_test_split(
    X_temp, y_temp, loc_temp, train_size=0.5, random_state=42
)

print(f"  Train: {len(X_train)}  |  Calib: {len(X_calib)}  |  Test: {len(X_test)}")

# Also keep the full dataset points for background map display
all_points_for_map = []
for i in range(len(data)):
    all_points_for_map.append({
        "lon": float(data.iloc[i]['lon']),
        "lat": float(data.iloc[i]['lat']),
        "price": float(data.iloc[i]['price']),
        "split": "train" if data.index[i] in X_train.index else
                 ("calib" if data.index[i] in X_calib.index else "test")
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
test_preds = model.predict(X_test.values)
from sklearn.metrics import r2_score, root_mean_squared_error
train_r2 = r2_score(y_train, train_preds)
test_r2 = r2_score(y_test, test_preds)
test_rmse = root_mean_squared_error(y_test, test_preds)

print(f"  XGBoost trained in {time.time()-t0:.1f}s")
print(f"  Train R²={train_r2:.4f} | Test R²={test_r2:.4f} | Test RMSE={test_rmse:.4f}")

# ==============================================================
# 4. Set up TreeSHAP (replaces slow KernelSHAP)
# ==============================================================
print("\nStep 4: Setting up TreeSHAP (fast alternative to KernelSHAP)...")
t0 = time.time()

tree_explainer = shap.TreeExplainer(model)

def tree_shap_f(x):
    """TreeSHAP: O(TLD^2) vs KernelSHAP: O(TKS^2) - much faster."""
    return tree_explainer.shap_values(x)

# Quick validation
sample_shap = tree_shap_f(X_test.values[:5])
print(f"  TreeSHAP ready. Sample shape: {sample_shap.shape}")
print(f"  Setup time: {time.time()-t0:.2f}s")

# ==============================================================
# 5. Run GeoXCP pipeline
# ==============================================================
print("\nStep 5: Running GeoXCP pipeline...")
print("  (MLP fitting with early stopping - ~2-8 min)")

explainer = GeoConformalizedExplainer(
    prediction_f=model.predict,
    x_train=X_train,
    x_calib=X_calib,
    coord_calib=loc_calib.values,
    shap_value_f=tree_shap_f,   # Use TreeSHAP instead of KernelSHAP
    miscoverage_level=0.1,       # 90% coverage target
    band_width=0.15,             # Geographic bandwidth
    feature_names=feature_names,
    is_single_model=True         # Single MLP for all features
)

t0 = time.time()
results = explainer.uncertainty_aware_explain(
    x_test=X_test,
    coord_test=loc_test.values
)
print(f"  GeoXCP pipeline completed in {time.time()-t0:.1f}s")

# ==============================================================
# 6. Print accuracy summary
# ==============================================================
print("\nAccuracy Summary:")
acc_summary = results.accuracy_summary()
print(acc_summary.to_string())

# ==============================================================
# 7. Export results to data/results.js
# ==============================================================
print("\nStep 6: Exporting results to data/results.js...")

result_df = results.result

# Build output structure
output = {
    "meta": {
        "dataset": "Seattle House Price",
        "n_train": int(len(X_train)),
        "n_calib": int(len(X_calib)),
        "n_test": int(len(X_test)),
        "miscoverage_level": 0.1,
        "target_coverage": 0.9,
        "bandwidth": 0.15,
        "train_r2": float(train_r2),
        "test_r2": float(test_r2),
        "test_rmse": float(test_rmse),
        "model": "XGBoost (n_estimators=500, max_depth=3)",
        "shap_method": "TreeSHAP (TreeExplainer)",
        "shap_predictor": "MLP (2048×2048×2048, early stopping)"
    },
    "features": feature_names,
    "feature_labels": {
        "bathrooms": "Bathrooms",
        "sqft_living": "Living Area (sqft)",
        "sqft_lot": "Lot Area (sqft)",
        "grade": "Grade",
        "condition": "Condition",
        "waterfront": "Waterfront",
        "view": "View",
        "age": "Age (years)",
        "lon": "Longitude",
        "lat": "Latitude"
    },
    "accuracy": {},
    "mean_abs_shap": {},
    "global_uncertainty": {},
    "points": []
}

# Accuracy per feature
for feature in feature_names:
    row = acc_summary.loc[feature]
    output["accuracy"][feature] = {
        "coverage_probability": float(row['coverage_probability']),
        "R2": float(row['R2']),
        "RMSE": float(row['RMSE']),
        "SHAP_Var": float(row['SHAP_Var']),
        "Pred_SHAP_Var": float(row['Pred_SHAP_Var'])
    }

# Mean absolute SHAP per feature
for j, feature in enumerate(feature_names):
    output["mean_abs_shap"][feature] = float(
        np.mean(np.abs(results.explanation_values[:, j]))
    )
    output["global_uncertainty"][feature] = float(
        results.geocp_results[j].uncertainty
    )

# Per-point data
for i in range(len(result_df)):
    row = result_df.iloc[i]
    point = {
        "lon": float(row['x']),
        "lat": float(row['y']),
        "price": float(y_test.iloc[i]),
        "features": {}
    }
    for feature in feature_names:
        point["features"][feature] = {
            "shap": float(row[f'{feature}_shap']),
            "value": float(row[f'{feature}_value']),
            "geo_uncertainty": float(row[f'{feature}_geo_uncertainty']),
            "upper_bound": float(row[f'{feature}_upper_bound']),
            "lower_bound": float(row[f'{feature}_lower_bound']),
            "pred": float(row[f'{feature}_pred']),
            "shap_abs": float(row[f'{feature}_shap_abs'])
        }
    output["points"].append(point)

# Save as JavaScript (works with file:// protocol, no server needed)
output_path = os.path.join(PROJECT_ROOT, 'data', 'results.js')
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w', encoding='utf-8') as f:
    f.write('// GeoXCP Pre-computed Results\n')
    f.write('// Generated by generate_data.py\n')
    f.write('window.GeoXCPData = ')
    json.dump(output, f, separators=(',', ':'))
    f.write(';\n')

file_size = os.path.getsize(output_path) / 1024
print(f"  Saved: {output_path} ({file_size:.1f} KB)")
print(f"  Contains: {len(output['points'])} test points × {len(feature_names)} features")

# Also save a JSON version for inspection
json_path = output_path.replace('.js', '.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2)
print(f"  Also saved: {json_path}")

total_time = time.time() - t_start
print(f"\n{'='*60}")
print(f"Done! Total time: {total_time/60:.1f} min ({total_time:.0f}s)")
print(f"{'='*60}")
print(f"\nNext step: open index.html in a browser.")
