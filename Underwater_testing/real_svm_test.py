"""
Real Data SVM Classification Test
=================================

Test SVM classification implementation with real underwater polarization data.
Uses the same data structure as the diagnostic tests for cross-session validation.
"""

import numpy as np
import sys
import gc
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Models.SVM_classification.svm_classification_wrapper import (
    CircularSVMClassifier, 
    SVMClassificationWrapper,
    calculate_circular_error
)
from Models.SVM_classification.svr_regression_wrapper import (
    CircularSVR,
    SVRWrapper, 
    calculate_circular_mae_svr
)
from Training_loops.run_all_models import extract_statistical_features_from_single_image
from Underwater_testing.UnderwaterDataLoader import UnderwaterDataLoader


def extract_features(loader, indices, label=""):
    """Extract features from underwater data (same as diagnostic tests)."""
    features, valid_indices = [], []
    failed = 0
    
    for num, idx in enumerate(indices):
        if (num + 1) % 50 == 0:
            print(f"    {label} {num+1}/{len(indices)} (failed: {failed})...")
            sys.stdout.flush()
        
        try:
            sample = loader.get_item(idx)
            if sample is not None:
                feat = extract_statistical_features_from_single_image(
                    sample["features"]["dolp"], 
                    sample["features"]["aolp"]
                )
                features.append(feat)
                valid_indices.append(idx)
            else:
                failed += 1
        except Exception:
            failed += 1
        
        # Memory cleanup
        if (num + 1) % 50 == 0:
            gc.collect()
    
    print(f"  Successfully extracted {len(features)} features ({failed} failed)")
    return np.array(features, dtype=np.float32), np.array(valid_indices)


def test_svm_classification():
    """Test SVM classification with real underwater polarization data."""
    
    # SAMPLING MODE SWITCH
    SAMPLING_MODE = "first"  # Options: "first", "advanced", "last"
    
    print("REAL DATA SVM CLASSIFICATION TEST")
    print("=" * 60)
    print("Cross-Session Testing: June_23 → June_24 (Correct Setup)")
    if SAMPLING_MODE == "advanced":
        print("Sampling: Advanced (5th-8th frames per burst)")
    elif SAMPLING_MODE == "last":
        print("Sampling: Last frame per burst")
    else:
        print("Sampling: Simple (first frame per burst)")
    print("=" * 60)
    
    # Initialize UnderwaterDataLoader (point to actual image location)
    data_root = Path("C:/Queens/ELEC498/Capstone_live_data").resolve()  # Where your images are actually located
    loader = UnderwaterDataLoader(data_root=data_root)
    n = len(loader)
    print()
    
    # Load all labels and sessions
    all_labels = np.array([loader._get_labels(i)["azimuth"] for i in range(n)])
    all_sessions = np.array([loader._get_labels(i)["session"] for i in range(n)])
    
    print(f"Dataset Overview:")
    print(f"  Total samples: {n:,}")
    print(f"  Azimuth range: {all_labels.min():.1f}° - {all_labels.max():.1f}°")
    print(f"  Coverage: {all_labels.max() - all_labels.min():.1f}° of 360°")
    
    for sess in ["June_23", "June_24", "Mar_09"]:
        mask = all_sessions == sess
        if mask.sum() > 0:
            az = all_labels[mask]
            print(f"  {sess}: {az.min():.1f}° - {az.max():.1f}° (n={mask.sum():,})")
        else:
            print(f"  {sess}: No data found")
    
    # Show all available sessions
    unique_sessions = np.unique(all_sessions)
    print(f"  All available sessions: {list(unique_sessions)}")
    print()
    
    # Use burst-based sampling with configurable method
    if SAMPLING_MODE == "advanced":
        print("Collecting burst-based samples (prefer 5th-8th frames per burst)...")
        burst_indices = {}
        
        # First pass: collect all indices for each burst
        for i in range(n):
            try:
                labels = loader._get_labels(i)
                burst_key = f"{labels['session']}_{labels['run']}_{labels['burst']}"
                if burst_key not in burst_indices:
                    burst_indices[burst_key] = []
                burst_indices[burst_key].append(i)
            except:
                continue
        
        # Second pass: select preferred sample from each burst
        subsample = []
        for burst_key, indices in burst_indices.items():
            if len(indices) >= 8:
                # Take 5th sample (index 4) if 8+ samples available
                subsample.append(indices[4])
            elif len(indices) >= 5:
                # Take 5th sample if 5-7 samples available  
                subsample.append(indices[4])
            else:
                # Take first sample if less than 5 samples
                subsample.append(indices[0])
                
    elif SAMPLING_MODE == "last":
        print("Collecting burst-based samples (last frame per burst)...")
        burst_samples = {}
        for i in range(n):
            try:
                labels = loader._get_labels(i)
                burst_key = f"{labels['session']}_{labels['run']}_{labels['burst']}"
                # Always update with the latest index (will end up with last frame)
                burst_samples[burst_key] = i
            except:
                continue
        
        subsample = list(burst_samples.values())
        
    else:
        print("Collecting burst-based samples (first frame per burst)...")
        burst_samples = {}
        for i in range(n):
            try:
                labels = loader._get_labels(i)
                burst_key = f"{labels['session']}_{labels['run']}_{labels['burst']}"
                if burst_key not in burst_samples:
                    burst_samples[burst_key] = i  # Take first sample from each burst
            except:
                continue
        
        subsample = list(burst_samples.values())
    print(f"Using {len(subsample)} burst-based samples (1 per burst)\n")
    
    # Extract features for burst samples
    print("Extracting features for burst samples...")
    sub_feat, sub_valid = extract_features(loader, subsample, "burst samples")
    sub_labels = all_labels[sub_valid]
    sub_sessions = all_sessions[sub_valid]
    
    print(f"Feature extraction complete: {sub_feat.shape}\\n")
    
    # ═════════════════════════════════════════════════════════════════════
    # CROSS-SESSION SPLIT (Train June_23, Test June_24) - CORRECTED DIRECTION
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("CROSS-SESSION SVM TESTING (Train June_23 → Test June_24)")
    print("=" * 60)
    
    train_mask = sub_sessions == "June_23"  # Training data (was Day_2)
    test_mask = sub_sessions == "June_24"   # Test data (was Day_1)
    
    # Debug session information
    print(f"Available sessions: {np.unique(sub_sessions)}")
    print(f"Session counts: {[(sess, (sub_sessions == sess).sum()) for sess in np.unique(sub_sessions)]}")
    print(f"Training data found: {train_mask.sum()} samples")
    print(f"Test data found: {test_mask.sum()} samples")
    
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("ERROR: Insufficient data for cross-session testing")
        print(f"  - Train (June_23): {train_mask.sum()} samples")
        print(f"  - Test (June_24): {test_mask.sum()} samples")
        return {}
    
    X_train = sub_feat[train_mask]
    X_test = sub_feat[test_mask] 
    y_train = sub_labels[train_mask]
    y_test = sub_labels[test_mask]
    
    print(f"  Training (June_23):    {X_train.shape[0]:3d} samples")
    print(f"  Testing (June_24):     {X_test.shape[0]:3d} samples")
    print(f"  Feature dimensions:   {X_train.shape[1]} features")
    print(f"  Train azimuth range:  {y_train.min():.1f}° - {y_train.max():.1f}°")
    print(f"  Test azimuth range:   {y_test.min():.1f}° - {y_test.max():.1f}°")
    print()
    
    # Test multiple SVM configurations (AGGRESSIVE feature reduction)
    svm_configs = [
        {"name": "SVM_8bins_minimal", "n_bins": 8, "C": 1.0, "feature_selection": 15},
        {"name": "SVM_16bins_minimal", "n_bins": 16, "C": 10, "feature_selection": 12},
        {"name": "SVM_24bins_minimal", "n_bins": 24, "C": 50, "feature_selection": 10},
        {"name": "SVM_32bins_minimal", "n_bins": 32, "C": 100, "feature_selection": 8},
        {"name": "SVM_8bins_medium", "n_bins": 8, "C": 10, "feature_selection": 25},
        {"name": "SVM_16bins_medium", "n_bins": 16, "C": 50, "feature_selection": 20},
    ]
    
    # Test SVR configurations (also with minimal features)
    # svr_configs = [
    #     {"name": "SVR_minimal", "C": 10, "epsilon": 0.1, "feature_selection": 15, "use_complex": True},
    #     {"name": "SVR_tiny", "C": 1.0, "epsilon": 0.2, "feature_selection": 10, "use_complex": True},
    #     {"name": "SVR_micro", "C": 0.1, "epsilon": 0.3, "feature_selection": 5, "use_complex": True},
    # ]
    
    results = {}
    
    print("Testing SVM configurations...")
    print("-" * 60)
    
    for config in svm_configs:
        print(f"\\n{config['name']}: {config['n_bins']} bins ({360/config['n_bins']:.1f}° resolution)")
        print(f"  Parameters: C={config['C']}, features={config['feature_selection']}")
        
        try:
            # Train SVM classifier
            svm = CircularSVMClassifier(
                n_bins=config["n_bins"],
                C=config["C"],
                gamma="scale", 
                probability=True,
                feature_selection=config["feature_selection"],
                class_weight="balanced"
            )
            
            print(f"  Training...")
            svm.fit(X_train, y_train)
            
            print(f"  Predicting...")
            pred_train = svm.predict(X_train)
            pred_test = svm.predict(X_test)
            
            # Calculate circular errors
            train_mae, train_rmse = calculate_circular_error(y_train, pred_train)
            test_mae, test_rmse = calculate_circular_error(y_test, pred_test)
            
            cross_session_gap = test_mae - train_mae
            
            results[config["name"]] = {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "cross_session_gap": cross_session_gap,
                "config": config
            }
            
            print(f"  Train MAE:     {train_mae:6.2f}°")
            print(f"  Test MAE:      {test_mae:6.2f}° (cross-session)")
            print(f"  Session gap:   {cross_session_gap:6.2f}°")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[config["name"]] = {"error": str(e)}
    
    # Test SVR configurations
    # print(f"\\n" + "=" * 60)
    # print("SVR REGRESSION TESTING")
    # print("=" * 60)
    # 
    # for config in svr_configs:
    #     print(f"\\n{config['name']}: SVR regression")
    #     print(f"  Parameters: C={config['C']}, epsilon={config['epsilon']}, features={config['feature_selection']}")
    #     
    #     try:
    #         # Train SVR
    #         svr = CircularSVR(
    #             C=config["C"],
    #             gamma="scale",
    #             epsilon=config["epsilon"], 
    #             feature_selection=config["feature_selection"],
    #             use_complex=config["use_complex"]
    #         )
    #         
    #         print(f"  Training...")
    #         svr.fit(X_train, y_train)
    #         
    #         print(f"  Predicting...")
    #         pred_train = svr.predict(X_train)
    #         pred_test = svr.predict(X_test)
    #         
    #         # Calculate circular errors
    #         train_mae, train_rmse = calculate_circular_mae_svr(y_train, pred_train)
    #         test_mae, test_rmse = calculate_circular_mae_svr(y_test, pred_test)
    #         
    #         cross_session_gap = test_mae - train_mae
    #         
    #         results[config["name"]] = {
    #             "train_mae": train_mae,
    #             "test_mae": test_mae,
    #             "cross_session_gap": cross_session_gap,
    #             "config": config,
    #             "type": "SVR"
    #         }
    #         
    #         print(f"  Train MAE:     {train_mae:6.2f}°")
    #         print(f"  Test MAE:      {test_mae:6.2f}° (cross-session)")
    #         print(f"  Session gap:   {cross_session_gap:6.2f}°")
    #         
    #     except Exception as e:
    #         print(f"  ERROR: {e}")
    #         results[config["name"]] = {"error": str(e)}
    
    # Test wrapper interface (for integration compatibility)
    # print(f"\\nTesting SVMClassificationWrapper interface...")
    # try:
    #     wrapper = SVMClassificationWrapper(
    #         n_bins=16, C=50, feature_selection=100
    #     )
    #     
    #     # Convert to radians (interface compatibility)
    #     y_train_rad = np.deg2rad(y_train)
    #     y_test_rad = np.deg2rad(y_test)
    #     
    #     print("  Training wrapper...")
    #     wrapper.fit(X_train, y_train_rad)
    #     
    #     print("  Predicting...")
    #     pred_test_rad = wrapper.predict(X_test)
    #     pred_test_wrapper = np.rad2deg(pred_test_rad) % 360
    #     
    #     wrapper_mae, _ = calculate_circular_error(y_test, pred_test_wrapper)
    #     print(f"  Wrapper Test MAE: {wrapper_mae:.2f}°")
    #     
    # except Exception as e:
    #     print(f"  Wrapper ERROR: {e}")
    
    # ═════════════════════════════════════════════════════════════════════
    # RESULTS ANALYSIS
    # ═════════════════════════════════════════════════════════════════════
    print(f"\\n" + "=" * 60) 
    print("CROSS-SESSION RESULTS: SVM ONLY")
    print("=" * 60)
    
    print(f"{'Model':<15} {'Train':<8} {'Test':<8} {'Gap':<8} {'Type'}")
    print("-" * 55)
    
    best_model = None
    best_test_mae = float('inf')
    
    for name, result in results.items():
        if "error" not in result:
            train_mae = result["train_mae"]
            test_mae = result["test_mae"]
            gap = result["cross_session_gap"]
            model_type = result.get("type", "SVM")
            
            if "n_bins" in result["config"]:
                n_bins = result["config"]["n_bins"]
                resolution = 360 / n_bins
                print(f"{name:<15} {train_mae:<8.2f} {test_mae:<8.2f} {gap:<8.2f} {model_type} ({resolution:.1f}°)")
            else:
                print(f"{name:<15} {train_mae:<8.2f} {test_mae:<8.2f} {gap:<8.2f} {model_type}")
            
            if test_mae < best_test_mae:
                best_test_mae = test_mae
                best_model = name
    
    # Compare with your baseline problem
    print(f"\\nPERFORMANCE COMPARISON:")
    print("-" * 30)
    
    current_baseline = 23.1  # From diagnostic report cross-session MAE
    target_improvement = 20.0  # Target MAE
    
    if best_model and best_test_mae < float('inf'):
        improvement_pct = (current_baseline - best_test_mae) / current_baseline * 100
        
        print(f"Best SVM model:     {best_model}")
        print(f"Cross-session MAE:  {best_test_mae:.2f}°")
        print(f"Current baseline:   {current_baseline:.1f}° (RandomForest)")
        print(f"Improvement:        {improvement_pct:.1f}%")
        
        if best_test_mae < target_improvement:
            print("\\n✅ EXCELLENT: Model significantly improves cross-session generalization!")
            print("   This solves the meridian detection and domain shift problems!")
        elif best_test_mae < current_baseline:
            print("\\n✅ GOOD: Model improves cross-session performance")
            print("   Consider fine-tuning parameters for further improvement")
        else:
            print("\\n⚠️  NEEDS WORK: Model performance similar to baseline")
            print("   Try different approaches, feature engineering, or domain adaptation")
            
        # Show which configurations work best
        print(f"\\nRECOMMENDATIONS:")
        print(f"- Best overall: {best_model}")
        
        # Find best by different criteria
        valid_results = [(k, v) for k, v in results.items() if "error" not in v]
        if valid_results:
            best_gap = min(valid_results, key=lambda x: x[1]["cross_session_gap"])
            print(f"- Best generalization: {best_gap[0]} (gap: {best_gap[1]['cross_session_gap']:.2f}°)")
            
            # Show best SVM vs SVR
            svm_results = [(k, v) for k, v in valid_results if v.get("type", "SVM") == "SVM"]
            # svr_results = [(k, v) for k, v in valid_results if v.get("type") == "SVR"]
            
            if svm_results:
                best_svm = min(svm_results, key=lambda x: x[1]["test_mae"])
                print(f"- Best SVM: {best_svm[0]} ({best_svm[1]['test_mae']:.2f}° MAE)")
                
            # if svr_results:
            #     best_svr = min(svr_results, key=lambda x: x[1]["test_mae"])
            #     print(f"- Best SVR: {best_svr[0]} ({best_svr[1]['test_mae']:.2f}° MAE)")
        
        # SVR vs SVM comparison
        # if len(svr_results) > 0 and len(svm_results) > 0:
        #     avg_svr = np.mean([v["test_mae"] for k, v in svr_results])
        #     avg_svm = np.mean([v["test_mae"] for k, v in svm_results])
        #     print(f"\\n📊 SVR vs SVM Comparison:")
        #     print(f"   Average SVR MAE: {avg_svr:.2f}°")
        #     print(f"   Average SVM MAE: {avg_svm:.2f}°")
        #     print(f"   SVR advantage: {avg_svm - avg_svr:.2f}° better" if avg_svr < avg_svm else f"   SVM advantage: {avg_svr - avg_svm:.2f}° better")
            
    else:
        print("\\n❌ ERROR: No valid SVM results obtained")
    
    print(f"\\nCross-session SVM evaluation complete!")
    
    # Save results summary
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(__file__).parent / "svm_analysis_results"
    output_dir.mkdir(exist_ok=True)
    
    summary_file = output_dir / f"svm_cross_session_{timestamp}.txt"
    with open(summary_file, "w") as f:
        f.write(f"SVM Cross-Session Results - {timestamp}\\n")
        f.write("=" * 50 + "\\n")
        
        if best_model and best_test_mae < float('inf'):
            improvement_pct = (current_baseline - best_test_mae) / current_baseline * 100
            f.write(f"Best model: {best_model}\\n")
            f.write(f"Best MAE: {best_test_mae:.2f}°\\n")
            f.write(f"Baseline: {current_baseline:.1f}°\\n")
            f.write(f"Improvement: {improvement_pct:.1f}%\\n\\n")
        else:
            f.write("No valid SVM results obtained\\n")
            f.write("Issues: Insufficient training data diversity\\n\\n")
        
        for name, result in results.items():
            if "error" not in result:
                f.write(f"{name}: {result['test_mae']:.2f}° MAE\\n")
            else:
                f.write(f"{name}: ERROR - {result['error']}\\n")
    
    print(f"\\nResults saved to: {summary_file}")
    
    return results


if __name__ == "__main__":
    test_svm_classification()