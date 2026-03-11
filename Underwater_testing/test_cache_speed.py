"""
Quick test to demonstrate fast cached model loading
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from real_svm_test import get_best_svm_for_visualization, MODEL_CACHE_DIR

def test_cached_vs_fresh_training():
    """Compare loading cached model vs training from scratch."""
    
    print("=" * 60)
    print("CACHED MODEL SPEED TEST")
    print("=" * 60)
    
    # Show available cached models
    cache_files = list(MODEL_CACHE_DIR.glob("svm_*.pkl"))
    print(f"\n📁 Available cached SVM models: {len(cache_files)}")
    for cache_file in cache_files:
        size_kb = cache_file.stat().st_size / 1024
        print(f"   - {cache_file.name} ({size_kb:.1f} KB)")
    
    # Test 1: Load from cache
    print(f"\n🚀 TEST 1: Loading from cache")
    print("-" * 30)
    start_time = time.time()
    
    result = get_best_svm_for_visualization(use_cache=True, force_retrain=False)
    
    cache_time = time.time() - start_time
    
    if result[0] is not None:
        svm, pred_test, y_test, test_mae, extras = result
        print(f"✅ Success! Loaded in {cache_time:.2f} seconds")
        print(f"   Model: {extras.get('config', {})}")
        print(f"   Test MAE: {test_mae:.2f}°")
        print(f"   Test samples: {len(pred_test)}")
    else:
        print("❌ Failed to load cached model")
        return
    
    # Test 2: Force retrain (if you want to compare speed)
    print(f"\n🔄 TEST 2: Force retrain (for comparison)")
    print("-" * 30)
    start_time = time.time()
    
    result2 = get_best_svm_for_visualization(use_cache=True, force_retrain=True)
    
    retrain_time = time.time() - start_time
    
    if result2[0] is not None:
        svm2, pred_test2, y_test2, test_mae2, extras2 = result2
        print(f"✅ Success! Retrained in {retrain_time:.2f} seconds")
        print(f"   Model: {extras2.get('config', {})}")
        print(f"   Test MAE: {test_mae2:.2f}°")
        print(f"   Test samples: {len(pred_test2)}")
    else:
        print("❌ Failed to retrain model")
        return
    
    # Compare performance
    print(f"\n📊 SPEED COMPARISON:")
    print(f"   Cached loading:  {cache_time:.2f} seconds")
    print(f"   Fresh training:  {retrain_time:.2f} seconds")
    print(f"   Speedup:         {retrain_time/cache_time:.1f}x faster with cache!")
    
    print(f"\n🎯 MODEL VERIFICATION:")
    print(f"   Cached MAE:      {test_mae:.3f}°")
    print(f"   Retrained MAE:   {test_mae2:.3f}°")
    print(f"   Difference:      {abs(test_mae - test_mae2):.3f}°")
    
    if abs(test_mae - test_mae2) < 0.001:
        print("   ✅ Models are identical!")
    else:
        print("   ⚠️  Models differ slightly (expected due to randomness)")

def show_cache_summary():
    """Show summary of all cached models."""
    
    print(f"\n📂 CACHE SUMMARY:")
    print(f"   Cache directory: {MODEL_CACHE_DIR}")
    
    cache_files = list(MODEL_CACHE_DIR.glob("svm_*.pkl"))
    if not cache_files:
        print("   No SVM models cached yet")
        return
    
    total_size = sum(f.stat().st_size for f in cache_files)
    print(f"   Total models: {len(cache_files)}")
    print(f"   Total size: {total_size/1024:.1f} KB")
    
    print(f"\n   Model Details:")
    for cache_file in sorted(cache_files):
        size_kb = cache_file.stat().st_size / 1024
        # Parse config from filename
        name = cache_file.stem.replace("_first", "")
        print(f"     {name} ({size_kb:.1f} KB)")
    
    print(f"\n💡 USAGE TIP:")
    print(f"   Use get_best_svm_for_visualization(use_cache=True) for instant loading!")
    print(f"   Use force_retrain=True only when you change the algorithm")

if __name__ == "__main__":
    show_cache_summary()
    test_cached_vs_fresh_training()