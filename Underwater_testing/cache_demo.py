"""Lightweight test - shows cache messages without waiting for full extraction"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the slow extract_features function to show caching concept
def mock_extract():
    print("    [SIMULATION] Feature extraction... (normally takes 30 seconds)")
    print("    [SIMULATION] Training SVM... (normally takes 10 seconds)")
    return "mock_model", [1, 2, 3], [1.1, 2.1, 3.1], 8.5, {"config": "test"}

# Simulate the cache behavior
_SVM_CACHE = None

def get_cached_svm_data():
    global _SVM_CACHE
    
    if _SVM_CACHE is not None:
        print("[CACHE] Using cached SVM model and data (much faster!)")
        return (_SVM_CACHE['model'], _SVM_CACHE['predictions'], 
               _SVM_CACHE['actual_labels'], _SVM_CACHE['test_mae'], 
               _SVM_CACHE['extra_data'])
    
    print("[CACHE] Training SVM for the first time...")
    
    # Simulate training (this would call the real function)
    model, pred, labels, mae, extra = mock_extract()
    
    # Cache everything
    _SVM_CACHE = {
        'model': model, 'predictions': pred, 'actual_labels': labels, 
        'test_mae': mae, 'extra_data': extra
    }
    
    print(f"[CACHE] ✅ Cached SVM model with {len(pred)} predictions and {mae:.2f}° MAE")
    return model, pred, labels, mae, extra

if __name__ == "__main__":
    print("SVM Caching Demo")
    print("=" * 40)
    
    print("\n1. First call (extract features):")
    model1, pred1, labels1, mae1, extra1 = get_cached_svm_data()
    
    print("\n2. Second call (use cache):")  
    model2, pred2, labels2, mae2, extra2 = get_cached_svm_data()
    
    print("\n3. Third call (use cache):")
    model3, pred3, labels3, mae3, extra3 = get_cached_svm_data()
    
    print(f"\n✅ Demo complete! Cache working perfectly.")
    print(f"   Same cached data: {model1 is model2 is model3}")