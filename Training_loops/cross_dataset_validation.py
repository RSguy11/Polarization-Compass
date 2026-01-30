"""
Cross-Dataset Validation: Test models on completely separate dataset
This tests for overfitting by evaluating on data from a different session

Training Data: 2025-11-24 (rmc dataset, ~10 min drive)
Test Data: 2024-10-08 (parade square dataset, 360 angles)
"""

import os
import sys
import numpy as np
import cv2
import joblib
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Bens_Data_Import.Image_data_loaders.Spatial_Stoke_Loader.SpatialStokeLoader import SpatialStokeDataLoader

def circular_error(pred, actual):
    """Calculate signed circular error in degrees"""
    diff = pred - actual
    # Wrap to [-180, 180]
    while diff > 180:
        diff -= 360
    while diff < -180:
        diff += 360
    return diff

def predict_from_model_dict(model_dict, features):
    """Handle prediction from our saved model dict format"""
    if isinstance(model_dict, dict):
        scaler = model_dict.get('scaler')
        sin_model = model_dict.get('sin_model')
        cos_model = model_dict.get('cos_model')
        
        if scaler is not None and sin_model is not None and cos_model is not None:
            # Circular regression model
            features_scaled = scaler.transform(features)
            sin_pred = sin_model.predict(features_scaled)
            cos_pred = cos_model.predict(features_scaled)
            return np.arctan2(sin_pred, cos_pred)
        
        # Direct regressor (like Gradient Boosting)
        regressor = model_dict.get('regressor')
        if regressor is not None:
            if scaler is not None:
                features_scaled = scaler.transform(features)
                return regressor.predict(features_scaled)
            return regressor.predict(features)
    
    raise ValueError(f"Unknown model format: {type(model_dict)}")

def load_parade_square_data(data_dir):
    """Load the parade square dataset with angles from filenames"""
    data_dir = Path(data_dir)
    samples = []
    
    for img_file in sorted(data_dir.glob("*.png")):
        # Extract angle from filename: 2024-10-08-19-31-33_angle_0.png
        match = re.search(r'_angle_(\d+)\.png$', img_file.name)
        if match:
            angle = int(match.group(1))
            samples.append({
                'path': str(img_file),
                'angle': angle,
                'filename': img_file.name
            })
    
    print(f"Loaded {len(samples)} images from parade square dataset")
    print(f"  Angle range: {min(s['angle'] for s in samples)}° to {max(s['angle'] for s in samples)}°")
    return samples

def main():
    print("=" * 60)
    print("CROSS-DATASET VALIDATION (Out-of-Distribution Test)")
    print("=" * 60)
    print()
    print("Training data: 2025-11-24 (rmc rosbag)")
    print("Test data:     2024-10-08 (parade square rotation)")
    print()
    
    # Paths
    workspace = Path(r"C:\Users\naesl\Polarization-Compass")
    parade_dir = workspace / "Bens_Data_Import" / "24-10-08-t000-forward-paradesquare"
    models_dir = workspace / "saved_models" / "2026-01-30"
    
    if not models_dir.exists():
        # Fall back to most recent
        models_base = workspace / "saved_models"
        model_dirs = sorted([d for d in models_base.iterdir() if d.is_dir()])
        if model_dirs:
            models_dir = model_dirs[-1]
    
    print(f"Using models from: {models_dir.name}")
    
    # Load models (try both .joblib and .pkl)
    models = {}
    for ext in ["*.joblib", "*_final.pkl"]:
        for model_file in models_dir.glob(ext):
            name = model_file.stem.replace('_final', '')
            try:
                models[name] = joblib.load(model_file)
                print(f"  Loaded: {name}")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
    
    if not models:
        print("ERROR: No models found!")
        return
    
    # Load parade square data
    print()
    samples = load_parade_square_data(parade_dir)
    
    if not samples:
        print("ERROR: No data found in parade square directory!")
        return
    
    # Process samples
    print()
    print("Extracting features from parade square images...")
    
    # Collect results
    results = []
    
    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{len(samples)}...")
        
        # Load image
        img = cv2.imread(sample['path'], cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  Warning: Could not load {sample['filename']}")
            continue
        
        # Create extractor for this image and extract features
        try:
            extractor = SpatialStokeDataLoader(img)
            # Get scalar features for ML
            feature_dict = extractor.extract_scalar_features()
            if feature_dict is None:
                print(f"  Warning: Feature extraction failed for {sample['filename']}")
                continue
            
            # Convert feature dict to array
            features = np.array(list(feature_dict.values())).reshape(1, -1)
        except Exception as e:
            print(f"  Warning: Failed to process {sample['filename']}: {e}")
            continue
        
        actual_az = sample['angle']
        
        row = {
            'filename': sample['filename'],
            'actual_azimuth': actual_az
        }
        
        # Get predictions from each model
        for model_name, model in models.items():
            try:
                if isinstance(model, dict):
                    pred_rad = predict_from_model_dict(model, features)
                    pred = np.rad2deg(pred_rad[0]) % 360
                elif hasattr(model, 'predict'):
                    pred = model.predict(features)[0]
                    if abs(pred) < 10:
                        pred = np.rad2deg(pred) % 360
                    else:
                        pred = pred % 360
                else:
                    continue
                
                error = circular_error(pred, actual_az)
                row[f'{model_name}_pred'] = round(pred, 2)
                row[f'{model_name}_error'] = round(error, 2)
            except Exception as e:
                if i == 0:  # Only print first error
                    print(f"    {model_name} error: {e}")
                row[f'{model_name}_pred'] = None
                row[f'{model_name}_error'] = None
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    output_dir = workspace / "Training_loops" / "validation_output"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "cross_dataset_validation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved results to: {csv_path}")
    
    # Print summary statistics
    print()
    print("=" * 60)
    print("CROSS-DATASET VALIDATION RESULTS")
    print("=" * 60)
    print()
    
    model_stats = {}
    for model_name in models.keys():
        error_col = f'{model_name}_error'
        if error_col in df.columns:
            errors = df[error_col].dropna().abs()
            if len(errors) > 0:
                mae = errors.mean()
                max_err = errors.max()
                within_5 = (errors <= 5).sum()
                within_10 = (errors <= 10).sum()
                within_45 = (errors <= 45).sum()
                
                model_stats[model_name] = {
                    'mae': mae,
                    'max': max_err,
                    'within_5': within_5,
                    'within_10': within_10,
                    'within_45': within_45,
                    'total': len(errors)
                }
                
                status = "PASS" if mae < 5 else "MARGINAL" if mae < 10 else "FAIL"
                print(f"{model_name}:")
                print(f"  MAE: {mae:.2f}° [{status}]")
                print(f"  Max Error: {max_err:.2f}°")
                print(f"  Within 5°: {within_5}/{len(errors)} ({100*within_5/len(errors):.1f}%)")
                print(f"  Within 10°: {within_10}/{len(errors)} ({100*within_10/len(errors):.1f}%)")
                print(f"  Within 45°: {within_45}/{len(errors)} ({100*within_45/len(errors):.1f}%)")
                print()
    
    # Generate plots
    print("Generating comparison plots...")
    
    # 1. Scatter plots: Predicted vs Actual for each model
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (model_name, stats) in enumerate(model_stats.items()):
        if i >= len(axes):
            break
        
        ax = axes[i]
        pred_col = f'{model_name}_pred'
        
        if pred_col in df.columns:
            valid = df[[pred_col, 'actual_azimuth']].dropna()
            if len(valid) > 0:
                ax.scatter(valid['actual_azimuth'], valid[pred_col], alpha=0.5, s=10)
                ax.plot([0, 360], [0, 360], 'r--', linewidth=2, label='Perfect')
                ax.set_xlabel('Actual Azimuth (°)')
                ax.set_ylabel('Predicted Azimuth (°)')
                ax.set_title(f'{model_name}\nMAE: {stats["mae"]:.2f}°')
                ax.set_xlim(0, 360)
                ax.set_ylim(0, 360)
                ax.legend()
    
    # Hide unused subplots
    for j in range(len(model_stats), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Cross-Dataset Validation: Predicted vs Actual Azimuth\n(Trained on 2025-11 data, Tested on 2024-10 data)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / "cross_dataset_scatter.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[OK] Saved scatter plots to: {plot_path}")
    plt.close()
    
    # 2. Error vs Azimuth plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (model_name, stats) in enumerate(model_stats.items()):
        if i >= len(axes):
            break
        
        ax = axes[i]
        error_col = f'{model_name}_error'
        
        if error_col in df.columns:
            valid = df[['actual_azimuth', error_col]].dropna()
            if len(valid) > 0:
                ax.scatter(valid['actual_azimuth'], valid[error_col], alpha=0.5, s=10)
                ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
                ax.axhline(y=5, color='orange', linestyle=':', alpha=0.5)
                ax.axhline(y=-5, color='orange', linestyle=':', alpha=0.5)
                ax.set_xlabel('Actual Azimuth (°)')
                ax.set_ylabel('Prediction Error (°)')
                ax.set_title(f'{model_name}')
                ax.set_xlim(0, 360)
                ax.set_ylim(-180, 180)
    
    for j in range(len(model_stats), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Cross-Dataset Validation: Error vs Azimuth\n(Orange lines = ±5° target)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = output_dir / "cross_dataset_error_vs_azimuth.png"
    plt.savefig(plot_path, dpi=150)
    print(f"[OK] Saved error plots to: {plot_path}")
    plt.close()
    
    # Final verdict
    print()
    print("=" * 60)
    print("OVERFITTING ASSESSMENT")
    print("=" * 60)
    
    # Compare training vs cross-dataset performance
    print("""
If models are overfitted to the training dataset:
  - Cross-dataset MAE will be MUCH higher than training MAE
  - Expect 50-100°+ errors instead of <1° errors
  
If models learned generalizable features:
  - Cross-dataset MAE should be similar to training MAE
  - Some degradation expected due to different conditions
    """)
    
    # Summary table
    print("\nSummary Comparison:")
    print("-" * 50)
    print(f"{'Model':<20} {'Training MAE':<15} {'Cross-Dataset MAE':<15}")
    print("-" * 50)
    
    # Training MAEs from our validation run
    training_maes = {
        'RF_Enhanced': 0.29,
        'SVR_Circular': 0.31,
        'Gradient_Boosting': 0.62,
        'Ensemble': None,
        'L2_PCA': None
    }
    
    for model_name in model_stats:
        train_mae = training_maes.get(model_name, '?')
        cross_mae = model_stats[model_name]['mae']
        
        if train_mae is not None:
            ratio = cross_mae / train_mae if train_mae > 0 else float('inf')
            print(f"{model_name:<20} {train_mae:<15} {cross_mae:<15.2f} (×{ratio:.1f})")
        else:
            print(f"{model_name:<20} {'N/A':<15} {cross_mae:<15.2f}")
    
    print("-" * 50)

if __name__ == "__main__":
    main()
