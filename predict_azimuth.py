"""
Inference Script for Solar Azimuth Prediction

This script loads trained models and makes predictions on new polarization data.
Use this for real-time azimuth estimation after training is complete.

Usage:
    python predict_azimuth.py --model saved_models/2026-01-12/Random_Forest.pkl --data path/to/images
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
from datetime import datetime

# Add Models to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Models.L2_Linear_reg.L2_pipeline import L2PolarizationRegressor
from Models.Random_Forest_reg.Random_Forest_pipeline import RandomForestPolarizationRegressor
from Bens_Data_Import.Image_data_loaders.Spatial_Gradient_Loader.SpatialPolarizationLoader import SpatialPolarizationLoader


def load_model(model_path: str):
    """
    Load a trained model from pickle file.
    
    Args:
        model_path: Path to the saved .pkl model file
        
    Returns:
        Loaded model instance (L2 or RandomForest)
    """
    print(f"Loading model from: {model_path}")
    
    # Determine model type from path
    if 'L2' in model_path or 'Ridge' in model_path:
        model = L2PolarizationRegressor.load_model(model_path)
        print("✓ Loaded L2 Ridge Regression model")
    elif 'Random_Forest' in model_path or 'RF' in model_path:
        model = RandomForestPolarizationRegressor.load_model(model_path)
        print("✓ Loaded Random Forest model")
    else:
        # Try to load and infer type
        import joblib
        model_data = joblib.load(model_path)
        if 'n_estimators' in model_data.get('hyperparameters', {}):
            model = RandomForestPolarizationRegressor.load_model(model_path)
            print("✓ Loaded Random Forest model")
        else:
            model = L2PolarizationRegressor.load_model(model_path)
            print("✓ Loaded L2 Ridge Regression model")
    
    # Display model info
    if hasattr(model, 'training_metrics') and model.training_metrics:
        print(f"  Training MAE: {model.training_metrics.get('mae', 'N/A'):.3f}°")
    
    return model


def predict_from_images(model, image_path: Path, target_size=(64, 64)):
    """
    Load images and predict solar azimuth.
    
    Args:
        model: Trained model instance
        image_path: Path to directory containing polarization images
        target_size: Resolution for processing (default 64x64)
        
    Returns:
        Array of predicted azimuth angles in degrees
    """
    print(f"\nLoading images from: {image_path}")
    
    # Load polarization data
    loader = SpatialPolarizationLoader(
        data_path=image_path,
        start_deg=0.0,
        step_deg=1.0,
        target_size=target_size
    )
    
    dolp, aolp, _ = loader.get_spatial_data(max_samples=None)
    print(f"✓ Loaded {len(dolp)} samples")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions_rad = model.predict(dolp, aolp)
    predictions_deg = np.rad2deg(predictions_rad)
    
    return predictions_deg


def predict_from_arrays(model, dolp: np.ndarray, aolp: np.ndarray):
    """
    Predict azimuth from DoLP and AoLP arrays.
    
    Args:
        model: Trained model instance
        dolp: DoLP array (N, H, W)
        aolp: AoLP array (N, H, W)
        
    Returns:
        Array of predicted azimuth angles in degrees
    """
    print(f"\nMaking predictions on {len(dolp)} samples...")
    predictions_rad = model.predict(dolp, aolp)
    predictions_deg = np.rad2deg(predictions_rad)
    return predictions_deg


def main():
    parser = argparse.ArgumentParser(description='Predict solar azimuth from polarization data')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved model (.pkl file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to directory containing polarization images')
    parser.add_argument('--resolution', type=int, default=64,
                       help='Processing resolution (default: 64x64)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predictions (CSV format)')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Load data and predict
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data path not found: {data_path}")
        return
    
    target_size = (args.resolution, args.resolution)
    predictions = predict_from_images(model, data_path, target_size)
    
    # Display results
    print(f"\n{'='*60}")
    print("PREDICTION RESULTS")
    print(f"{'='*60}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Mean azimuth: {predictions.mean():.1f}°")
    print(f"Azimuth range: [{predictions.min():.1f}°, {predictions.max():.1f}°]")
    print(f"Std deviation: {predictions.std():.1f}°")
    
    # Show first few predictions
    print(f"\nFirst 10 predictions:")
    for i, pred in enumerate(predictions[:10]):
        print(f"  Sample {i+1}: {pred:.2f}°")
    
    # Save to file if requested
    if args.output:
        import pandas as pd
        df = pd.DataFrame({
            'sample_index': range(len(predictions)),
            'predicted_azimuth_degrees': predictions
        })
        df.to_csv(args.output, index=False)
        print(f"\n💾 Predictions saved to: {args.output}")


if __name__ == "__main__":
    # Example usage without command line args
    if len(sys.argv) == 1:
        print("INFERENCE SCRIPT - Solar Azimuth Prediction")
        print("=" * 60)
        print("\nUsage:")
        print("  python predict_azimuth.py --model <model.pkl> --data <image_dir>")
        print("\nExample:")
        print("  python predict_azimuth.py \\")
        print("    --model saved_models/2026-01-12/Random_Forest.pkl \\")
        print("    --data Bens_Data_Import/24-10-08-t000-forward-paradesquare \\")
        print("    --output predictions.csv")
        print("\nFor programmatic use:")
        print("  from predict_azimuth import load_model, predict_from_arrays")
        print("  model = load_model('saved_models/2026-01-12/Random_Forest.pkl')")
        print("  predictions = predict_from_arrays(model, dolp, aolp)")
    else:
        main()
