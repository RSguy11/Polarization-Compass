"""
Real Data SVM Classification Test
=================================

Test SVM classification implementation with real underwater polarization data.
Uses the same data structure as the diagnostic tests for cross-session validation.
"""

import numpy as np
import sys
import gc
import pickle
import os
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


# Model cache directory
MODEL_CACHE_DIR = Path(__file__).parent / "model_cache"
MODEL_CACHE_DIR.mkdir(exist_ok=True)


def get_model_cache_path(config, sampling_mode="first"):
    """Generate cache file path for a specific model configuration."""
    if "n_bins" in config:
        # SVM configuration
        cache_name = f"svm_{config['n_bins']}bins_C{config['C']}_fs{config['feature_selection']}_{sampling_mode}.pkl"
    else:
        # SVR configuration  
        cache_name = f"svr_C{config['C']}_eps{config['epsilon']}_fs{config['feature_selection']}_{sampling_mode}.pkl"
    return MODEL_CACHE_DIR / cache_name


def save_model(model, config, sampling_mode="first", additional_data=None):
    """Save trained model and metadata to cache."""
    cache_path = get_model_cache_path(config, sampling_mode)
    cache_data = {
        'model': model,
        'config': config,
        'sampling_mode': sampling_mode,
        'timestamp': datetime.now().isoformat(),
        'additional_data': additional_data
    }
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"  Model saved to: {cache_path.name}")
        return True
    except Exception as e:
        print(f"  Warning: Failed to save model - {e}")
        return False


def load_model(config, sampling_mode="first"):
    """Load trained model from cache if it exists."""
    cache_path = get_model_cache_path(config, sampling_mode)
    
    if not cache_path.exists():
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Verify config matches
        if cache_data['config'] == config and cache_data['sampling_mode'] == sampling_mode:
            print(f"  Loaded cached model: {cache_path.name}")
            return cache_data['model'], cache_data.get('additional_data')
        else:
            print(f"  Cache config mismatch, will retrain")
            return None
            
    except Exception as e:
        print(f"  Warning: Failed to load model - {e}")
        return None


def clear_model_cache():
    """Clear all cached models."""
    if MODEL_CACHE_DIR.exists():
        for cache_file in MODEL_CACHE_DIR.glob("*.pkl"):
            try:
                cache_file.unlink()
                print(f"Deleted: {cache_file.name}")
            except Exception as e:
                print(f"Failed to delete {cache_file.name}: {e}")
        print(f"Model cache cleared: {MODEL_CACHE_DIR}")
    else:
        print("No model cache directory found")


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
    print("Cross-Session Testing: feb_23+feb_24 → Mar_09 (Combined Training)")
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
    
    for sess in ["feb_23", "feb_24", "Mar_09"]:
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
    # CROSS-SESSION SPLIT (Train feb_23, Test Mar_09) - CORRECTED DIRECTION
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("CROSS-SESSION SVM TESTING (Train feb_23+feb_24 → Test Mar_09)")
    print("=" * 60)
    
    train_mask = (sub_sessions == "Mar_09")  # Training data 
    test_mask = sub_sessions == "feb_23"   # Test data 
    
    # Debug session information
    print(f"Available sessions: {np.unique(sub_sessions)}")
    print(f"Session counts: {[(sess, (sub_sessions == sess).sum()) for sess in np.unique(sub_sessions)]}")
    print(f"Training data found: {train_mask.sum()} samples")
    print(f"Test data found: {test_mask.sum()} samples")
    
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        print("ERROR: Insufficient data for cross-session testing")
        print(f"  - Train (feb_23+feb_24): {train_mask.sum()} samples")
        print(f"  - Test (Mar_09): {test_mask.sum()} samples")
        return {}
    
    X_train = sub_feat[train_mask]
    X_test = sub_feat[test_mask] 
    y_train = sub_labels[train_mask]
    y_test = sub_labels[test_mask]
    
    print(f"  Training (feb_23+feb_24): {X_train.shape[0]:3d} samples")
    print(f"  Testing (Mar_09):       {X_test.shape[0]:3d} samples")
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
            # Try to load cached model first
            cached_result = load_model(config, SAMPLING_MODE)
            
            if cached_result is not None:
                svm, additional_data = cached_result
                print(f"  Using cached model (skip training)")
                
                # Get cached predictions if available
                if additional_data and 'pred_train' in additional_data and 'pred_test' in additional_data:
                    pred_train = additional_data['pred_train']
                    pred_test = additional_data['pred_test']
                    print(f"  Using cached predictions")
                else:
                    print(f"  Generating predictions...")
                    pred_train = svm.predict(X_train)
                    pred_test = svm.predict(X_test)
            else:
                # Train new SVM classifier
                svm = CircularSVMClassifier(
                    n_bins=config["n_bins"],
                    C=config["C"],
                    gamma="scale", 
                    probability=True,
                    feature_selection=config["feature_selection"],
                    class_weight="balanced"
                )
                
                print(f"  Training new model...")
                svm.fit(X_train, y_train)
                
                print(f"  Predicting...")
                pred_train = svm.predict(X_train)
                pred_test = svm.predict(X_test)
                
                # Save the trained model and predictions
                print(f"  [DEBUG] About to save model...")
                additional_data = {
                    'pred_train': pred_train,
                    'pred_test': pred_test,
                    'X_train_shape': X_train.shape,
                    'X_test_shape': X_test.shape
                }
                save_success = save_model(svm, config, SAMPLING_MODE, additional_data)
                print(f"  [DEBUG] Save result: {save_success}")
            
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
            print(f"  [DEBUG] Exception occurred, model not saved")
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


# Global cache for features to avoid re-extraction
_FEATURE_CACHE = None


def get_best_svm_for_visualization(use_cache=True, force_retrain=False):
    """Get trained SVM model and data for visualization dashboard."""
    global _FEATURE_CACHE
    
    print("[DASHBOARD] Loading SVM model and data for visualization...")
    
    # Best performing configuration from testing results - use EXACT config that was saved
    best_config = {"name": "SVM_32bins_minimal", "n_bins": 32, "C": 100, "feature_selection": 8}  # Best MAE: 8.84°
    sampling_mode = "first"
    
    # Try to load cached model first (unless forced to retrain)
    if not force_retrain:
        print("[DASHBOARD] Checking for cached model...")
        cached_result = load_model(best_config, sampling_mode)
        
        if cached_result is not None:
            svm, cached_data = cached_result
            print("[DASHBOARD] ✅ Using cached SVM model (no feature extraction needed)!")
            
            # Check if we have cached features and predictions
            if use_cache and _FEATURE_CACHE is not None:
                print("[DASHBOARD] ✅ Using cached features")
                y_test = _FEATURE_CACHE['y_test']
                
                # Use cached predictions if available and shapes match
                if (cached_data and 'pred_test' in cached_data):
                    pred_test = cached_data['pred_test']
                    print("[DASHBOARD] ✅ Using cached predictions - INSTANT RESULTS!")
                else:
                    print("[DASHBOARD] Generating fresh predictions...")
                    X_test = _FEATURE_CACHE['X_test'] 
                    pred_test = svm.predict(X_test)
                
                # Calculate error
                from Models.SVM_classification.svm_classification_wrapper import calculate_circular_error
                test_mae, _ = calculate_circular_error(y_test, pred_test)
                
                return svm, pred_test, y_test, test_mae, {
                    'train_predictions': cached_data.get('pred_train') if cached_data else None,
                    'train_labels': _FEATURE_CACHE.get('y_train'),
                    'train_mae': None,  # Would need to recalculate
                    'test_mae': test_mae,
                    'config': best_config
                }
            else:
                print("[DASHBOARD] Need to extract features (cached model but no feature cache)")
        else:
            print("[DASHBOARD] No cached model found, will train new one")
    else:
        print("[DASHBOARD] Force retrain requested, skipping cache")
    
    # Check if we have cached features
    if use_cache and _FEATURE_CACHE is not None:
        print("[DASHBOARD] Using cached features (much faster!)")
        X_train = _FEATURE_CACHE['X_train']
        X_test = _FEATURE_CACHE['X_test']
        y_train = _FEATURE_CACHE['y_train']
        y_test = _FEATURE_CACHE['y_test']
        print(f"[DASHBOARD] Cached: Train {X_train.shape[0]}, Test {X_test.shape[0]} samples")
    else:
        print("[DASHBOARD] Extracting features (first time or cache disabled)...")
        
        # Initialize data loader
        data_root = Path("C:/Queens/ELEC498/Capstone_live_data").resolve()
        loader = UnderwaterDataLoader(data_root=data_root)
        n = len(loader)
        
        # Load all labels and sessions
        all_labels = np.array([loader._get_labels(i)["azimuth"] for i in range(n)])
        all_sessions = np.array([loader._get_labels(i)["session"] for i in range(n)])
        
        print(f"[DASHBOARD] Dataset: {n:,} total samples")
        
        # Use burst-based sampling (same as main test)
        print("[DASHBOARD] Collecting burst-based samples...")
        burst_samples = {}
        for i in range(n):
            try:
                labels = loader._get_labels(i)
                burst_key = f"{labels['session']}_{labels['run']}_{labels['burst']}"
                if burst_key not in burst_samples:
                    burst_samples[burst_key] = i
            except:
                continue
        
        subsample = list(burst_samples.values())
        print(f"[DASHBOARD] Using {len(subsample)} burst samples")
        
        # Extract features
        print("[DASHBOARD] Extracting features...")
        sub_feat, sub_valid = extract_features(loader, subsample, "dashboard")
        sub_labels = all_labels[sub_valid]
        sub_sessions = all_sessions[sub_valid]
        
        # Create train/test split (feb_23+feb_24 train, Mar_09 test)
        train_mask = (sub_sessions == "feb_23") | (sub_sessions == "feb_24")
        test_mask = sub_sessions == "Mar_09"
        
        if train_mask.sum() == 0 or test_mask.sum() == 0:
            print("[DASHBOARD] ERROR: Insufficient real data")
            return None, None, None, None, None
        
        X_train = sub_feat[train_mask]
        X_test = sub_feat[test_mask]
        y_train = sub_labels[train_mask]
        y_test = sub_labels[test_mask]
        
        print(f"[DASHBOARD] Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        # Cache the features for future use
        if use_cache:
            _FEATURE_CACHE = {
                'X_train': X_train,
                'X_test': X_test, 
                'y_train': y_train,
                'y_test': y_test
            }
            print("[DASHBOARD] Features cached for future use")
    
    # Train the best performing SVM configuration (or use cached)
    print(f"[DASHBOARD] Training/Loading SVM with config: {best_config}")
    
    # Check for cached model first
    if not force_retrain:
        cached_result = load_model(best_config, sampling_mode)
        if cached_result is not None:
            svm, cached_data = cached_result
            print("[DASHBOARD] Using cached model")
            
            # Use cached predictions if shapes match
            if (cached_data and 'pred_train' in cached_data and 'pred_test' in cached_data and
                cached_data.get('X_train_shape') == X_train.shape and 
                cached_data.get('X_test_shape') == X_test.shape):
                pred_train = cached_data['pred_train']
                pred_test = cached_data['pred_test']
                print("[DASHBOARD] Using cached predictions")
            else:
                print("[DASHBOARD] Generating fresh predictions...")
                pred_train = svm.predict(X_train)
                pred_test = svm.predict(X_test)
        else:
            # Train new model
            print("[DASHBOARD] Training new SVM model...")
            svm = CircularSVMClassifier(
                n_bins=best_config["n_bins"],
                C=best_config["C"],
                gamma="scale",
                probability=True,
                feature_selection=best_config["feature_selection"],
                class_weight="balanced"
            )
            
            svm.fit(X_train, y_train)
            
            # Generate predictions
            print("[DASHBOARD] Generating predictions...")
            pred_train = svm.predict(X_train)
            pred_test = svm.predict(X_test)
            
            # Save the model and predictions
            additional_data = {
                'pred_train': pred_train,
                'pred_test': pred_test,
                'X_train_shape': X_train.shape,
                'X_test_shape': X_test.shape
            }
            save_model(svm, best_config, sampling_mode, additional_data)
            print("[DASHBOARD] Model saved for future use")
    else:
        # Force retrain
        print("[DASHBOARD] Force training new SVM model...")
        svm = CircularSVMClassifier(
            n_bins=best_config["n_bins"],
            C=best_config["C"],
            gamma="scale",
            probability=True,
            feature_selection=best_config["feature_selection"],
            class_weight="balanced"
        )
        
        svm.fit(X_train, y_train)
        
        # Generate predictions
        print("[DASHBOARD] Generating predictions...")
        pred_train = svm.predict(X_train)
        pred_test = svm.predict(X_test)
        
        # Save the model and predictions
        additional_data = {
            'pred_train': pred_train,
            'pred_test': pred_test,
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape
        }
        save_model(svm, best_config, sampling_mode, additional_data)
        print("[DASHBOARD] New model saved")
    
    # Calculate errors
    train_mae, _ = calculate_circular_error(y_train, pred_train)
    test_mae, _ = calculate_circular_error(y_test, pred_test)
    
    print(f"[DASHBOARD] Train MAE: {train_mae:.2f}°, Test MAE: {test_mae:.2f}°")
    print(f"[DASHBOARD] Ready for visualization with {len(pred_test)} test predictions")
    
    return svm, pred_test, y_test, test_mae, {
        'train_predictions': pred_train,
        'train_labels': y_train,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'config': best_config
    }


def clear_feature_cache():
    """Clear the feature cache to force re-extraction."""
    global _FEATURE_CACHE
    _FEATURE_CACHE = None
    print("[DASHBOARD] Feature cache cleared")


if __name__ == "__main__":
    # Optional: Clear cache and force retrain
    # clear_model_cache()
    
    test_svm_classification()
    
    # Show cache status
    print(f"\nModel cache directory: {MODEL_CACHE_DIR}")
    cache_files = list(MODEL_CACHE_DIR.glob("*.pkl"))
    if cache_files:
        print(f"Cached models: {len(cache_files)}")
        for cache_file in cache_files:
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"  - {cache_file.name} ({size_mb:.1f} MB)")
    else:
        print("No cached models found")
    
    print("\nTo clear cache: uncomment clear_model_cache() call")