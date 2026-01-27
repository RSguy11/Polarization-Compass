#!/usr/bin/env python
"""Quick test of training pipeline - skip full data loading"""
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Models.L2_Linear_reg.L2_pipeline import create_baseline_model
from Models.SVR_reg.SVR_pipeline import create_svr_model  
from Models.Random_Forest_reg.Random_Forest_pipeline import create_random_forest_model

# Mock data instead of loading from disk
print("=" * 60)
print("QUICK TEST: Training with synthetic data")
print("=" * 60)
print()

# Create synthetic training data
np.random.seed(42)
n_train = 1960
n_test = 490
n_features = 1000  # Reduced from 80k for speed

print(f"Creating synthetic data...")
X_train = np.random.randn(n_train, n_features) * 100
y_train = np.deg2rad(np.random.uniform(0, 360, n_train))

X_test = np.random.randn(n_test, n_features) * 100
y_test = np.deg2rad(np.random.uniform(0, 360, n_test))

print(f"[OK] X_train: {X_train.shape}")
print(f"[OK] y_train: {y_train.shape}")
print(f"[OK] X_test: {X_test.shape}")
print(f"[OK] y_test: {y_test.shape}")
print()

# Initialize models with tuned hyperparameters
print("STEP 2: Training Models")
print("=" * 60)

models = {
    'L2_Baseline': create_baseline_model(alpha=0.001),
    'SVR_RBF': create_svr_model(C=100.0, gamma=0.001, epsilon=0.1),
    'Random_Forest': create_random_forest_model(
        n_estimators=200,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5
    )
}

results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    try:
        print(f"  Training final model...")
        train_metrics = model.fit_from_features(X_train, y_train)
        
        print(f"  Evaluating on test set...")
        test_predictions = model.predict_from_features(X_test)
        test_mae = np.rad2deg(np.mean(np.abs(test_predictions - y_test)))
        test_rmse = np.rad2deg(np.sqrt(np.mean((test_predictions - y_test) ** 2)))
        
        print(f"  Cross-validation...")
        cv_metrics = model.cross_validate_from_features(X_train, y_train, cv_folds=5)
        
        results[model_name] = {
            'training_mae': float(train_metrics['mae']),
            'cv_mae': float(cv_metrics['mae_mean']),
            'cv_rmse': float(cv_metrics['rmse_mean']),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse),
            'meets_requirements': bool(test_mae < 5.0)
        }
        
        print(f"✓ {model_name}")
        print(f"  Train MAE: {train_metrics['mae']:.3f} deg")
        print(f"  CV MAE: {cv_metrics['mae_mean']:.3f} deg")
        print(f"  Test MAE: {test_mae:.3f} deg (held-out)")
        
    except Exception as e:
        print(f"✗ {model_name} failed: {str(e)}")
        results[model_name] = {'error': str(e)}
        import traceback
        traceback.print_exc()

print(f"\n" + "=" * 60)
print(f"RESULTS SUMMARY:")
print("=" * 60)
for name, result in results.items():
    if 'error' not in result:
        print(f"{name}:")
        print(f"  CV MAE: {result['cv_mae']:.3f} deg")
        print(f"  Test MAE: {result['test_mae']:.3f} deg - {'PASS' if result['meets_requirements'] else 'FAIL'}")

print(f"\nTest completed successfully!")
