"""Quick test of the SVM caching system"""

from svm_charts import get_cached_svm_data

if __name__ == "__main__":
    print("Testing SVM caching system...")
    print("=" * 40)
    
    print("\n1. First call (should extract features)...")
    model1, pred1, labels1, mae1, extra1 = get_cached_svm_data()
    
    print("\n2. Second call (should use cache)...")  
    model2, pred2, labels2, mae2, extra2 = get_cached_svm_data()
    
    print("\n3. Third call (should use cache)...")
    model3, pred3, labels3, mae3, extra3 = get_cached_svm_data()
    
    if model1 is not None:
        print(f"\nAll calls successful!")
        print(f"MAE: {mae1:.2f}°")
        print(f"Test samples: {len(pred1)}")
        print(f"Same model object: {model1 is model2 is model3}")
    else:
        print("Failed to load SVM data")