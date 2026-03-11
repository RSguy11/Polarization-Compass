"""
Debug script to test pickle functionality and find the issue
"""

import sys
import pickle
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Models.SVM_classification.svm_classification_wrapper import CircularSVMClassifier

def test_pickle_svm():
    """Test if we can pickle the SVM model directly."""
    print("Testing SVM pickle functionality...")
    
    # Simple test data
    X = np.random.randn(50, 10)
    y = np.random.uniform(0, 360, 50)
    
    # Create and train a simple SVM
    print("1. Creating SVM...")
    svm = CircularSVMClassifier(n_bins=8, C=1.0, feature_selection=5)
    
    print("2. Training SVM...")
    svm.fit(X, y)
    
    print("3. Testing pickle...")
    try:
        # Test pickling
        cache_path = Path(__file__).parent / "model_cache" / "test_svm.pkl"
        cache_path.parent.mkdir(exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(svm, f)
        print(f"✅ SVM pickled successfully to: {cache_path}")
        
        # Test unpickling
        with open(cache_path, 'rb') as f:
            loaded_svm = pickle.load(f)
        print("✅ SVM unpickled successfully")
        
        # Test predictions match
        pred_original = svm.predict(X)
        pred_loaded = loaded_svm.predict(X)
        
        if np.allclose(pred_original, pred_loaded):
            print("✅ Predictions match after pickling")
        else:
            print("❌ Predictions don't match after pickling")
            
        return True
        
    except Exception as e:
        print(f"❌ Pickle failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_complex_pickle():
    """Test pickling with additional data like we do in real_svm_test."""
    print("\nTesting complex data structure pickle...")
    
    try:
        # Test the full cache structure
        cache_data = {
            'model': "test_model",
            'config': {"n_bins": 16, "C": 10, "feature_selection": 12},
            'sampling_mode': "first",
            'timestamp': "2026-03-10",
            'additional_data': {
                'pred_train': np.array([1, 2, 3]),
                'pred_test': np.array([4, 5, 6]),
                'X_train_shape': (100, 20),
                'X_test_shape': (50, 20)
            }
        }
        
        cache_path = Path(__file__).parent / "model_cache" / "test_complex.pkl"
        cache_path.parent.mkdir(exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print("✅ Complex structure pickled successfully")
        
        with open(cache_path, 'rb') as f:
            loaded_data = pickle.load(f)
        print("✅ Complex structure unpickled successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Complex pickle failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("PICKLE DEBUGGING")
    print("=" * 50)
    
    # Test basic SVM pickle
    svm_ok = test_pickle_svm()
    
    # Test complex structure
    complex_ok = test_complex_pickle()
    
    if svm_ok and complex_ok:
        print("\n✅ All pickle tests passed - the issue is elsewhere")
    else:
        print("\n❌ Pickle tests failed - found the problem!")
        
    # Check model cache directory
    cache_dir = Path(__file__).parent / "model_cache"
    print(f"\nModel cache directory: {cache_dir}")
    if cache_dir.exists():
        files = list(cache_dir.glob("*.pkl"))
        print(f"Files in cache: {len(files)}")
        for f in files:
            size = f.stat().st_size
            print(f"  - {f.name} ({size} bytes)")
    else:
        print("Cache directory doesn't exist")