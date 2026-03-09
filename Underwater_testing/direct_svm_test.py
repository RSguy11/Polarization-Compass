#!/usr/bin/env python3
"""
Direct SVM Test with Timelapse Data
==================================

Test SVM classification directly on timelapse polarization data
without requiring the full UnderwaterDataLoader structure.
"""

import numpy as np
import pandas as pd
import sys
import cv2
import polanalyser as pa
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Models.SVM_classification.svm_classification_wrapper import (
    CircularSVMClassifier, 
    SVMClassificationWrapper,
    calculate_circular_error
)
from Training_loops.run_all_models import extract_statistical_features_from_single_image


def process_polarization_image(image_path):
    """Process polarization image to extract AoLP and DoLP."""
    try:
        img_raw = cv2.imread(str(image_path), 0)
        if img_raw is None:
            return None, None
        
        # Demosaic using polanalyser
        img_000, img_045, img_090, img_135 = pa.demosaicing(img_raw, pa.COLOR_PolarMono)
        
        # Calculate Stokes vectors
        image_list = [img_000, img_045, img_090, img_135]
        angles = np.deg2rad([0, 45, 90, 135])
        img_stokes = pa.calcStokes(image_list, angles)
        
        # Extract DoLP and AoLP
        img_dolp = pa.cvtStokesToDoLP(img_stokes)
        img_aolp = pa.cvtStokesToAoLP(img_stokes)
        
        # Convert AoLP to degrees
        img_aolp = np.rad2deg(img_aolp)
        
        return img_dolp, img_aolp
        
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return None, None


def test_svm_on_timelapse_data():
    """Test SVM on timelapse data directly."""
    print("DIRECT SVM TEST ON TIMELAPSE DATA")
    print("=" * 50)
    
    # Load the dataset
    parquet_path = Path("Capstone_live_data/solar_labels.parquet")
    if not parquet_path.exists():
        print(f"❌ Dataset not found: {parquet_path}")
        print("Run generate_labels_from_images.py first")
        return
    
    df = pd.read_parquet(parquet_path)
    print(f"📊 Loaded dataset: {len(df)} samples")
    
    # Show session distribution
    session_counts = df['session'].value_counts()
    print(f"Sessions: {session_counts.to_dict()}")
    
    # Extract features and labels for a subset
    max_samples_per_session = 20  # Limit for speed
    
    features_list = []
    labels_list = []
    sessions_list = []
    processed = 0
    failed = 0
    
    print(f"\\n🔬 Processing images (max {max_samples_per_session} per session)...")
    
    for session in ["Day_12", "Day_23"]:
        session_data = df[df['session'] == session].head(max_samples_per_session)
        print(f"\\n  {session}: processing {len(session_data)} images...")
        
        for idx, row in session_data.iterrows():
            # Build full path to image
            img_path = Path("C:/") / row['image_path']
            
            if not img_path.exists():
                print(f"    ⚠️  Image not found: {img_path}")
                failed += 1
                continue
            
            # Process image
            dolp, aolp = process_polarization_image(img_path)
            
            if dolp is not None and aolp is not None:
                try:
                    # Extract statistical features (same as diagnostic tests)
                    features = extract_statistical_features_from_single_image(dolp, aolp)
                    
                    features_list.append(features)
                    labels_list.append(row['solar_azimuth'])
                    sessions_list.append(session)
                    processed += 1
                    
                    if processed % 5 == 0:
                        print(f"    Processed: {processed}, Failed: {failed}")
                        
                except Exception as e:
                    print(f"    Feature extraction error: {e}")
                    failed += 1
            else:
                failed += 1
    
    if len(features_list) < 10:
        print(f"❌ Insufficient data: only {len(features_list)} valid samples")
        return
    
    # Convert to arrays
    X = np.array(features_list, dtype=np.float32)
    y = np.array(labels_list)
    sessions = np.array(sessions_list)
    
    print(f"\\n✅ Feature extraction complete:")
    print(f"   Total features: {X.shape}")
    print(f"   Successfully processed: {processed}")
    print(f"   Failed: {failed}")
    
    # Split by session (train on Day_12, test on Day_23)
    train_mask = sessions == "Day_12"
    test_mask = sessions == "Day_23"
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    
    print(f"\\n📈 Cross-session split:")
    print(f"   Training (Day_12): {len(X_train)} samples, azimuth {y_train.min():.1f}°-{y_train.max():.1f}°")
    print(f"   Testing (Day_23):  {len(X_test)} samples, azimuth {y_test.min():.1f}°-{y_test.max():.1f}°")
    
    if len(X_train) < 3 or len(X_test) < 3:
        print("❌ Insufficient data for cross-session testing")
        return
    
    # Test SVM configurations
    print(f"\\n🧠 Training SVM models...")
    
    svm_configs = [
        {"name": "SVM_8bins", "n_bins": 8, "C": 10},
        {"name": "SVM_16bins", "n_bins": 16, "C": 50},
    ]
    
    results = {}
    
    for config in svm_configs:
        print(f"\\n  {config['name']}: {config['n_bins']} bins")
        
        try:
            # Train SVM
            svm = CircularSVMClassifier(
                n_bins=config["n_bins"],
                C=config["C"],
                gamma="scale",
                probability=True,
                feature_selection=min(50, X_train.shape[1]),  # Limit features
                class_weight="balanced"
            )
            
            svm.fit(X_train, y_train)
            
            # Predict
            pred_train = svm.predict(X_train)
            pred_test = svm.predict(X_test)
            
            # Calculate errors
            train_mae, _ = calculate_circular_error(y_train, pred_train)
            test_mae, _ = calculate_circular_error(y_test, pred_test)
            
            results[config["name"]] = {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "gap": test_mae - train_mae
            }
            
            print(f"    Train MAE: {train_mae:.2f}°")
            print(f"    Test MAE:  {test_mae:.2f}°")
            print(f"    Gap:       {test_mae - train_mae:.2f}°")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results[config["name"]] = {"error": str(e)}
    
    # Summary
    print(f"\\n🎯 RESULTS SUMMARY")
    print("=" * 40)
    
    valid_results = [(k, v) for k, v in results.items() if "error" not in v]
    if valid_results:
        best_model = min(valid_results, key=lambda x: x[1]["test_mae"])
        
        print(f"Best SVM: {best_model[0]}")
        print(f"Test MAE: {best_model[1]['test_mae']:.2f}°")
        print(f"Gap:      {best_model[1]['gap']:.2f}°")
        
        # Compare to baseline
        baseline_mae = 23.1  # From diagnostic report
        improvement = baseline_mae - best_model[1]['test_mae']
        improvement_pct = improvement / baseline_mae * 100
        
        print(f"\\nBaseline (RandomForest): {baseline_mae:.1f}°")
        print(f"Improvement: {improvement:.1f}° ({improvement_pct:.1f}%)")
        
        if best_model[1]['test_mae'] < baseline_mae:
            print("✅ SVM improves cross-session performance!")
        else:
            print("⚠️  SVM similar to baseline - consider tuning")
    
    else:
        print("❌ No valid SVM results")
    
    return results


if __name__ == "__main__":
    test_svm_on_timelapse_data()