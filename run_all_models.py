"""
Complete Model Training Pipeline

This script runs all three models (L2, SVR, Random Forest) with the same dataset
and provides a complete comparison as specified in the blueprint.
"""

import os
import sys
import numpy as np
from datetime import datetime
import json

sys.path.append('.')

from L2_Linear_reg.L2_pipeline import create_baseline_model
from SVR_reg.SVR_pipeline import create_svr_model  
from Random_Forest_reg.Random_Forest_pipeline import create_random_forest_model
from solar_azimuth_generator import SolarPositionCalculator

def run_complete_pipeline():
    """Run the complete model training pipeline for all three models."""
    
    print("POLARIZATION COMPASS - COMPLETE MODEL PIPELINE")
    print("=" * 60)
    print("Training L2, SVR, and Random Forest models")
    print()
    
    # TODO: Replace with actual data loading
    # For now, create minimal test data
    print("Loading data...")
    n_samples = 200
    np.random.seed(42)
    
    dolp = np.random.uniform(0, 1, (n_samples, 32, 32)).astype(np.float32)
    aolp = np.random.uniform(0, 180, (n_samples, 32, 32)).astype(np.float32)
    azimuth = np.random.uniform(0, 360, n_samples).astype(np.float32)
    
    results = {}
    
    # Test each model
    models = {
        'L2_Baseline': create_baseline_model(alpha=1.0),
        'Random_Forest': create_random_forest_model(n_estimators=50, max_depth=10)
        # SVR commented out due to memory issues - fix and uncomment
        # 'SVR_RBF': create_svr_model(C=1.0, gamma='scale', epsilon=0.1)
    }
    
    for model_name, model in models.items():
        print(f"\\nTraining {model_name}...")
        try:
            # Train
            train_metrics = model.fit(dolp, aolp, azimuth)
            
            # Cross-validate
            cv_metrics = model.cross_validate(dolp, aolp, azimuth, cv_folds=3)
            
            results[model_name] = {
                'training_mae': train_metrics['mae'],
                'cv_mae': cv_metrics['mae_mean'],
                'cv_rmse': cv_metrics['rmse_mean'],
                'meets_requirements': cv_metrics['mae_mean'] < 5.0
            }
            
            print(f"✓ {model_name} - CV MAE: {cv_metrics['mae_mean']:.3f}°")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)}")
            results[model_name] = {'error': str(e)}
    
    # Save results
    os.makedirs('pipeline_results', exist_ok=True)
    with open('pipeline_results/complete_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n📊 RESULTS SUMMARY:")
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name}: MAE {result['cv_mae']:.3f}° - {'✓' if result['meets_requirements'] else '✗'}")
    
    print(f"\\n💾 Results saved to pipeline_results/complete_results.json")

if __name__ == "__main__":
    run_complete_pipeline()