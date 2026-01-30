"""
Validation Script: Manual inspection of model predictions
Outputs CSV with image paths, actual azimuths, predicted azimuths, and errors
"""

import numpy as np
import pandas as pd
import joblib
import sys
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from Bens_Data_Import.Polarization_DataLoader.PolarizationDataLoader import PolarizationDataLoader

# Import model classes needed for unpickling
from Training_loops.run_all_models import (
    CircularRegressionWrapper, 
    EnsembleCircularModel,
    GradientBoostingWrapper,
    extract_statistical_features_from_single_image
)


def circular_error(pred_deg, actual_deg):
    """Calculate circular error (handles 0/360 wraparound)"""
    diff = pred_deg - actual_deg
    # Wrap to [-180, 180]
    diff = (diff + 180) % 360 - 180
    return diff


def load_model(model_path):
    """Load a saved model"""
    return joblib.load(model_path)


def predict_from_model_dict(model_dict, features):
    """
    Make prediction from a saved model dictionary.
    Handles CircularRegressionWrapper format (scaler, sin_model, cos_model)
    and GradientBoostingWrapper format (scaler, regressor)
    """
    # Scale features
    scaler = model_dict.get('scaler')
    if scaler is not None:
        features_scaled = scaler.transform(features)
    else:
        features_scaled = features
    
    # Check if it's a circular model (sin/cos) or direct regressor
    if 'sin_model' in model_dict and 'cos_model' in model_dict:
        # Circular regression: predict sin and cos, then combine
        sin_pred = model_dict['sin_model'].predict(features_scaled)
        cos_pred = model_dict['cos_model'].predict(features_scaled)
        # Convert back to angle using atan2
        azimuth_rad = np.arctan2(sin_pred, cos_pred)
        return azimuth_rad  # Returns radians
    elif 'regressor' in model_dict:
        # Direct regressor (e.g., GradientBoosting)
        pred = model_dict['regressor'].predict(features_scaled)
        return pred  # Returns radians
    elif 'models' in model_dict:
        # Ensemble model
        weights = model_dict.get('weights', None)
        models = model_dict['models']
        preds = []
        for m in models:
            p = predict_from_model_dict(m, features)
            preds.append(p)
        preds = np.array(preds)
        if weights is not None:
            weights = np.array(weights).reshape(-1, 1)
            weighted_pred = np.sum(preds * weights, axis=0) / np.sum(weights)
            return weighted_pred
        else:
            return np.mean(preds, axis=0)
    else:
        raise ValueError(f"Unknown model format with keys: {model_dict.keys()}")


def extract_features_for_sample(loader, idx):
    """Extract 468 features for a single sample"""
    sample = loader.get_item(idx)
    if sample is None:
        return None
    
    features = sample.get('features', {})
    aolp = features.get('aolp')
    dolp = features.get('dolp')
    
    if aolp is None or dolp is None:
        return None
    
    return extract_statistical_features_from_single_image(dolp, aolp)


def main():
    print("=" * 60)
    print("MODEL PREDICTION VALIDATION")
    print("=" * 60)
    
    # Setup paths
    rmc_folder = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/Polarization_DataLoader/rmc")
    model_dir = Path("C:/Users/naesl/Polarization-Compass/saved_models")
    output_dir = Path("C:/Users/naesl/Polarization-Compass/Training_loops/validation_output")
    output_dir.mkdir(exist_ok=True)
    
    # Find latest model folder
    model_folders = sorted([d for d in model_dir.iterdir() if d.is_dir()])
    if not model_folders:
        print("No saved models found!")
        return
    latest_model_dir = model_folders[-1]
    print(f"Using models from: {latest_model_dir.name}")
    
    # Load models
    models = {}
    model_files = ['RF_Enhanced_final.pkl', 'SVR_Circular_final.pkl', 'Ensemble_final.pkl', 
                   'Gradient_Boosting_final.pkl', 'L2_PCA_final.pkl']
    
    for mf in model_files:
        model_path = latest_model_dir / mf
        if model_path.exists():
            model_name = mf.replace('_final.pkl', '')
            try:
                models[model_name] = load_model(model_path)
                print(f"  Loaded: {model_name}")
            except Exception as e:
                print(f"  Failed to load {model_name}: {e}")
    
    if not models:
        print("No models loaded!")
        return
    
    # Load data
    print("\nLoading polarization data...")
    loader = PolarizationDataLoader(rmc_folder)
    n_samples = len(loader)
    
    # Get all azimuths
    all_azimuths = [loader.labels_df.iloc[i]['azimuth'] for i in range(n_samples)]
    
    # Stratified sampling: pick ~200 samples across all azimuth bins
    print("\nSelecting stratified validation samples...")
    n_bins = 18  # 20-degree bins
    samples_per_bin = 12  # ~216 total samples
    bins = np.linspace(0, 360, n_bins + 1)
    
    selected_indices = []
    rng = np.random.default_rng(123)  # Different seed than training
    
    for b in range(n_bins):
        bin_min, bin_max = bins[b], bins[b+1]
        bin_indices = [i for i, az in enumerate(all_azimuths) if bin_min <= az < bin_max]
        if bin_indices:
            n_select = min(samples_per_bin, len(bin_indices))
            selected = rng.choice(bin_indices, size=n_select, replace=False)
            selected_indices.extend(selected)
    
    print(f"Selected {len(selected_indices)} validation samples")
    
    # Extract features and get predictions
    print("\nExtracting features and generating predictions...")
    results = []
    
    for i, idx in enumerate(selected_indices):
        if (i + 1) % 20 == 0:
            print(f"  Processing {i+1}/{len(selected_indices)}...")
        
        # Get actual azimuth
        actual_az = all_azimuths[idx]
        
        # Get image path
        image_path = loader.image_files[idx]
        
        # Extract features
        features = extract_features_for_sample(loader, idx)
        if features is None:
            continue
        
        features = features.reshape(1, -1)
        
        # Get predictions from each model
        row = {
            'index': idx,
            'image_path': str(image_path.name),
            'actual_azimuth': round(actual_az, 2),
        }
        
        for model_name, model in models.items():
            try:
                # Use our custom predict function for dict-format models
                if isinstance(model, dict):
                    pred_rad = predict_from_model_dict(model, features)
                    pred = np.rad2deg(pred_rad[0]) % 360
                elif hasattr(model, 'predict_from_features'):
                    pred = model.predict_from_features(features)[0]
                    pred = np.rad2deg(pred) % 360
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
                print(f"    Warning: {model_name} failed on sample {idx}: {e}")
                row[f'{model_name}_pred'] = None
                row[f'{model_name}_error'] = None
        
        results.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by actual azimuth for easier inspection
    df = df.sort_values('actual_azimuth').reset_index(drop=True)
    
    # Save to CSV
    csv_path = output_dir / 'validation_predictions.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Saved predictions to: {csv_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    for model_name in models.keys():
        error_col = f'{model_name}_error'
        if error_col in df.columns:
            errors = df[error_col].dropna().abs()
            print(f"\n{model_name}:")
            print(f"  MAE: {errors.mean():.2f}°")
            print(f"  Max Error: {errors.max():.2f}°")
            print(f"  Samples within 1°: {(errors <= 1).sum()}/{len(errors)} ({100*(errors <= 1).mean():.1f}%)")
            print(f"  Samples within 5°: {(errors <= 5).sum()}/{len(errors)} ({100*(errors <= 5).mean():.1f}%)")
    
    # Create scatter plots
    print("\nGenerating validation plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Model Predictions vs Actual Azimuth (Validation Set)', fontsize=14)
    
    model_names = list(models.keys())
    for i, model_name in enumerate(model_names[:5]):
        ax = axes.flat[i]
        pred_col = f'{model_name}_pred'
        
        if pred_col in df.columns:
            valid = df[['actual_azimuth', pred_col]].dropna()
            ax.scatter(valid['actual_azimuth'], valid[pred_col], alpha=0.5, s=20)
            ax.plot([0, 360], [0, 360], 'r--', linewidth=2, label='Perfect prediction')
            ax.set_xlabel('Actual Azimuth (°)')
            ax.set_ylabel('Predicted Azimuth (°)')
            ax.set_title(model_name)
            ax.set_xlim(0, 360)
            ax.set_ylim(0, 360)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    if len(model_names) < 6:
        axes.flat[5].axis('off')
    
    plt.tight_layout()
    plot_path = output_dir / 'validation_scatter_plots.png'
    plt.savefig(plot_path, dpi=150)
    print(f"[OK] Saved scatter plots to: {plot_path}")
    
    # Create error distribution histogram
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Prediction Error Distribution (Validation Set)', fontsize=14)
    
    for i, model_name in enumerate(model_names[:5]):
        ax = axes.flat[i]
        error_col = f'{model_name}_error'
        
        if error_col in df.columns:
            errors = df[error_col].dropna()
            ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
            ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
            ax.axvline(x=errors.mean(), color='g', linestyle='-', linewidth=2, label=f'Mean: {errors.mean():.2f}°')
            ax.set_xlabel('Prediction Error (°)')
            ax.set_ylabel('Count')
            ax.set_title(model_name)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    if len(model_names) < 6:
        axes.flat[5].axis('off')
    
    plt.tight_layout()
    hist_path = output_dir / 'validation_error_histograms.png'
    plt.savefig(hist_path, dpi=150)
    print(f"[OK] Saved error histograms to: {hist_path}")
    
    # Show worst predictions for manual inspection
    print("\n" + "=" * 60)
    print("WORST PREDICTIONS (for manual inspection)")
    print("=" * 60)
    
    # Use RF_Enhanced as reference
    if 'RF_Enhanced_error' in df.columns:
        df['abs_error'] = df['RF_Enhanced_error'].abs()
        worst = df.nlargest(10, 'abs_error')[['image_path', 'actual_azimuth', 'RF_Enhanced_pred', 'RF_Enhanced_error']]
        print("\nTop 10 worst RF_Enhanced predictions:")
        print(worst.to_string(index=False))
    
    print(f"\n[OK] Full results saved to: {csv_path}")
    print("Open the CSV to inspect individual predictions!")


if __name__ == "__main__":
    main()
