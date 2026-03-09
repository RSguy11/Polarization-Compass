"""
Quick SVM Classification Test
============================

Simple test script to validate SVM classification implementation
without running the full pipeline.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Models.SVM_classification.svm_classification_wrapper import SVMClassificationWrapper
from Underwater_testing.UnderwaterDataLoader import UnderwaterDataLoader
from Underwater_testing.run_all_models import load_batch_features
import numpy as np


def quick_svm_test():
    print("QUICK SVM CLASSIFICATION TEST")
    print("=" * 40)
    
    # Load small dataset
    loader = UnderwaterDataLoader()
    
    # Get first 200 samples for speed
    indices = []
    labels = []
    
    print("Loading 200 samples...")
    for i in range(min(500, len(loader))):
        try:
            label = loader._get_labels(i)
            indices.append(i)
            labels.append(label["azimuth"])
            if len(indices) >= 200:
                break
        except:
            continue
    
    print(f"Loaded {len(indices)} samples")
    azimuth_deg = np.array(labels)
    
    # Split data 
    n_train = int(0.8 * len(indices))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    azimuth_train = azimuth_deg[:n_train] 
    azimuth_test = azimuth_deg[n_train:]
    
    print(f"Train: {len(train_indices)}, Test: {len(test_indices)}")
    print(f"Azimuth range: {azimuth_deg.min():.1f}° - {azimuth_deg.max():.1f}°")
    
    # Extract features
    print("Extracting features...")
    X_train = load_batch_features(loader, train_indices)
    X_test = load_batch_features(loader, test_indices)
    
    print(f"Feature shapes: Train {X_train.shape}, Test {X_test.shape}")
    
    # Test SVM classification
    print("\\nTesting SVM Classification...")
    
    svm_model = SVMClassificationWrapper(
        n_bins=16,
        C=10.0,
        gamma='scale',
        feature_selection=100
    )
    
    # Convert to radians for interface compatibility
    azimuth_train_rad = np.deg2rad(azimuth_train)
    azimuth_test_rad = np.deg2rad(azimuth_test)
    
    print("Training...")
    svm_model.fit(X_train, azimuth_train_rad)
    
    print("Predicting...")
    pred_train_rad = svm_model.predict(X_train)
    pred_test_rad = svm_model.predict(X_test)
    
    # Convert back to degrees
    pred_train_deg = np.rad2deg(pred_train_rad) % 360  
    pred_test_deg = np.rad2deg(pred_test_rad) % 360
    
    # Calculate circular errors
    def circular_error(true_deg, pred_deg):
        diff = np.angle(np.exp(1j * np.deg2rad(pred_deg - true_deg)))
        return np.mean(np.abs(np.rad2deg(diff)))
    
    train_mae = circular_error(azimuth_train, pred_train_deg)
    test_mae = circular_error(azimuth_test, pred_test_deg)
    
    print(f"\\nResults:")
    print(f"  Train MAE: {train_mae:.2f}°")
    print(f"  Test MAE:  {test_mae:.2f}°")
    print(f"  Overfitting: {test_mae - train_mae:.2f}°")
    
    # Show some predictions
    print(f"\\nSample predictions:")
    for i in range(min(5, len(azimuth_test))):
        true_angle = azimuth_test[i]
        pred_angle = pred_test_deg[i]
        error = abs(np.rad2deg(np.angle(np.exp(1j * np.deg2rad(pred_angle - true_angle)))))
        print(f"  True: {true_angle:6.1f}°, Pred: {pred_angle:6.1f}°, Error: {error:5.1f}°")
    
    print("\\nSVM Classification test complete!")
    return train_mae, test_mae


if __name__ == "__main__":
    quick_svm_test()