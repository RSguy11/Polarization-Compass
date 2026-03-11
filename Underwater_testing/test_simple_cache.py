"""
Simplified test to isolate the caching issue
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Models.SVM_classification.svm_classification_wrapper import CircularSVMClassifier
from real_svm_test import save_model, load_model, MODEL_CACHE_DIR

def test_single_svm_with_cache():
    """Test training and caching a single SVM."""
    print("Testing single SVM with caching...")
    
    # Create simple test data
    np.random.seed(42)
    X_train = np.random.randn(100, 20)
    y_train = np.random.uniform(0, 360, 100)
    X_test = np.random.randn(50, 20)
    y_test = np.random.uniform(0, 360, 50)
    
    # Test config
    config = {"name": "test_svm", "n_bins": 8, "C": 1.0, "feature_selection": 10}
    sampling_mode = "first"
    
    print(f"Config: {config}")
    print(f"Cache dir: {MODEL_CACHE_DIR}")
    
    # Step 1: Check if model exists
    print("\n1. Checking for existing model...")
    cached_result = load_model(config, sampling_mode)
    if cached_result is not None:
        print("   ✅ Found cached model!")
        svm, additional_data = cached_result
    else:
        print("   ❌ No cached model found, training new one...")
        
        # Step 2: Train new model
        print("\n2. Training new SVM...")
        svm = CircularSVMClassifier(
            n_bins=config["n_bins"],
            C=config["C"],
            feature_selection=config["feature_selection"]
        )
        
        print("   Training...")
        svm.fit(X_train, y_train)
        
        print("   Predicting...")
        pred_train = svm.predict(X_train)
        pred_test = svm.predict(X_test)
        
        # Step 3: Save model
        print("\n3. Saving model...")
        additional_data = {
            'pred_train': pred_train,
            'pred_test': pred_test,
            'X_train_shape': X_train.shape,
            'X_test_shape': X_test.shape
        }
        
        print(f"   About to call save_model...")
        save_success = save_model(svm, config, sampling_mode, additional_data)
        print(f"   Save result: {save_success}")
        
        if save_success:
            print("   ✅ Model saved successfully!")
        else:
            print("   ❌ Model save failed!")
    
    # Step 4: Test loading the saved model
    print("\n4. Testing model loading...")
    cached_result = load_model(config, sampling_mode)
    if cached_result is not None:
        print("   ✅ Model loaded successfully from cache!")
        loaded_svm, loaded_data = cached_result
        
        # Test if predictions match
        test_pred_original = svm.predict(X_test[:10])  # Test with first 10 samples
        test_pred_loaded = loaded_svm.predict(X_test[:10])
        
        if np.allclose(test_pred_original, test_pred_loaded):
            print("   ✅ Loaded model predictions match original!")
        else:
            print("   ⚠️  Loaded model predictions differ from original")
            
    else:
        print("   ❌ Failed to load model from cache")
    
    # Step 5: Check cache directory
    print("\n5. Cache directory contents:")
    cache_files = list(MODEL_CACHE_DIR.glob("*.pkl"))
    if cache_files:
        for cache_file in cache_files:
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"   - {cache_file.name} ({size_mb:.1f} MB)")
    else:
        print("   No files found in cache")

if __name__ == "__main__":
    test_single_svm_with_cache()