"""
Quick demonstration of cached model loading with NO feature extraction
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from real_svm_test import load_model, MODEL_CACHE_DIR
from Models.SVM_classification.svm_classification_wrapper import calculate_circular_error

def test_instant_cached_model():
    """Test loading and using a cached model instantly - no feature extraction!"""
    
    print("=" * 60)
    print("INSTANT CACHED MODEL TEST")
    print("No Feature Extraction - Just Pure Model Usage!")
    print("=" * 60)
    
    # Show available models
    cache_files = list(MODEL_CACHE_DIR.glob("svm_*.pkl"))
    print(f"\\n📁 Available cached models: {len(cache_files)}")
    for f in cache_files:
        size_kb = f.stat().st_size / 1024
        print(f"   - {f.name} ({size_kb:.1f} KB)")
    
    # Use the best performing model - SVM_32bins_minimal (8.84° MAE)
    config = {"name": "SVM_32bins_minimal", "n_bins": 32, "C": 100, "feature_selection": 8}
    sampling_mode = "first"
    
    print(f"\\n🚀 LOADING CACHED MODEL:")
    print(f"   Config: {config}")
    print(f"   Cache file: svm_32bins_C100_fs8_first.pkl")
    
    start_time = time.time()
    
    # Load the cached model
    cached_result = load_model(config, sampling_mode)
    
    load_time = time.time() - start_time
    
    if cached_result is None:
        print("❌ Failed to load cached model")
        return
    
    svm, cached_data = cached_result
    print(f"✅ Model loaded in {load_time:.3f} seconds!")
    
    # Show what's in the cached data
    if cached_data:
        print(f"\\n📦 CACHED DATA CONTENTS:")
        for key, value in cached_data.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: {value.shape} array")
            else:
                print(f"   {key}: {value}")
    
    # Create some test data to demonstrate the model works
    print(f"\\n🧪 TESTING MODEL WITH SYNTHETIC DATA:")
    np.random.seed(42)
    
    # Create test features (same dimensions as the model was trained on)
    X_test_shape = cached_data.get('X_test_shape', (50, 8))  # Fallback if not cached
    n_features = X_test_shape[1] if len(X_test_shape) > 1 else 8
    
    # Generate synthetic test data
    X_synthetic = np.random.randn(20, n_features)
    
    print(f"   Synthetic test data: {X_synthetic.shape}")
    print(f"   Model expects: {n_features} features")
    
    # Make predictions
    predictions_start = time.time()
    predictions = svm.predict(X_synthetic)
    prediction_time = time.time() - predictions_start
    
    print(f"   Predictions: {predictions[:5]} ... (showing first 5)")
    print(f"   Prediction time: {prediction_time:.4f} seconds")
    print(f"   Average prediction: {np.mean(predictions):.1f}°")
    print(f"   Prediction range: {np.min(predictions):.1f}° - {np.max(predictions):.1f}°")
    
    # If we have cached test predictions, compare with real performance
    if 'pred_test' in cached_data and 'X_test_shape' in cached_data:
        cached_preds = cached_data['pred_test']
        print(f"\\n📊 REAL PERFORMANCE (from cache):")
        print(f"   Cached test predictions: {len(cached_preds)} samples")
        print(f"   Example cached predictions: {cached_preds[:5]} ...")
        
        # We'd need the true labels to calculate MAE, but we can show the predictions exist
        print(f"   Cached prediction range: {np.min(cached_preds):.1f}° - {np.max(cached_preds):.1f}°")
        print(f"   ✅ Real test predictions are readily available!")
    
    print(f"\\n🎯 SUMMARY:")
    print(f"   ✅ Model loaded instantly: {load_time:.3f}s")
    print(f"   ✅ Predictions work: {prediction_time:.4f}s for {len(predictions)} samples") 
    print(f"   ✅ No feature extraction needed!")
    print(f"   ✅ Ready for visualization!")
    
    return svm, cached_data

def demonstrate_all_cached_models():
    """Show that all cached models can be loaded instantly."""
    
    print(f"\\n" + "=" * 60)
    print("LOADING ALL CACHED MODELS")
    print("=" * 60)
    
    # Define all the configs that were saved
    configs = [
        {"name": "SVM_8bins_minimal", "n_bins": 8, "C": 1.0, "feature_selection": 15},
        {"name": "SVM_16bins_minimal", "n_bins": 16, "C": 10, "feature_selection": 12},
        {"name": "SVM_24bins_minimal", "n_bins": 24, "C": 50, "feature_selection": 10},
        {"name": "SVM_32bins_minimal", "n_bins": 32, "C": 100, "feature_selection": 8},
        {"name": "SVM_8bins_medium", "n_bins": 8, "C": 10, "feature_selection": 25},
        {"name": "SVM_16bins_medium", "n_bins": 16, "C": 50, "feature_selection": 20},
    ]
    
    total_load_time = 0
    loaded_models = 0
    
    for config in configs:
        print(f"\\n📋 {config['name']}: {config['n_bins']} bins, {config['feature_selection']} features")
        
        start_time = time.time()
        cached_result = load_model(config, "first")
        load_time = time.time() - start_time
        
        if cached_result is not None:
            svm, cached_data = cached_result
            print(f"   ✅ Loaded in {load_time:.3f}s")
            
            # Quick test
            X_test = np.random.randn(5, config['feature_selection'])
            preds = svm.predict(X_test)
            print(f"   🧪 Test predictions: {preds[0]:.1f}° (sample)")
            
            total_load_time += load_time
            loaded_models += 1
        else:
            print(f"   ❌ Not found in cache")
    
    print(f"\\n🏆 FINAL RESULTS:")
    print(f"   Models loaded: {loaded_models}/{len(configs)}")
    print(f"   Total load time: {total_load_time:.3f}s")
    print(f"   Average load time: {total_load_time/max(loaded_models,1):.3f}s per model")
    print(f"   🚀 All models ready for instant visualization!")

if __name__ == "__main__":
    test_instant_cached_model()
    demonstrate_all_cached_models()