"""
Test script to generate cached models and create visualizations
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from real_svm_test import get_best_svm_for_visualization, clear_feature_cache, clear_model_cache
import matplotlib.pyplot as plt
import numpy as np

def circular_error(y_true, y_pred):
    """Calculate circular error between true and predicted angles."""
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 360 - diff)
    return diff

def test_cached_model_visualization():
    """Test the cached model system and create visualizations."""
    
    print("=" * 60)
    print("TESTING CACHED MODEL VISUALIZATION")
    print("=" * 60)
    
    # First run: This should train and cache the model
    print("\n1. FIRST RUN - Training and Caching Model")
    print("-" * 50)
    
    result = get_best_svm_for_visualization(use_cache=True, force_retrain=False)
    
    if result[0] is None:
        print("❌ ERROR: Failed to get model")
        return
    
    svm, pred_test, y_test, test_mae, extras = result
    
    print(f"✅ Model loaded successfully")
    print(f"   Test MAE: {test_mae:.2f}°")
    print(f"   Test samples: {len(pred_test)}")
    
    # Check if model was cached
    print(f"\n📁 Model cache status:")
    cache_dir = Path(__file__).parent / "model_cache"
    cache_files = list(cache_dir.glob("*.pkl"))
    if cache_files:
        for cache_file in cache_files:
            size_mb = cache_file.stat().st_size / (1024 * 1024)
            print(f"   ✅ {cache_file.name} ({size_mb:.1f} MB)")
    else:
        print("   ⚠️  No cached models found")
    
    # Second run: This should use the cached model
    print(f"\n2. SECOND RUN - Using Cached Model (if available)")
    print("-" * 50)
    
    result2 = get_best_svm_for_visualization(use_cache=True, force_retrain=False)
    if result2[0] is not None:
        svm2, pred_test2, y_test2, test_mae2, extras2 = result2
        print(f"✅ Cached model used successfully")
        print(f"   Test MAE: {test_mae2:.2f}°")
        
        # Verify predictions are identical (if using cached predictions)
        if np.array_equal(pred_test, pred_test2):
            print(f"   ✅ Predictions are identical (using cached)")
        else:
            print(f"   ⚠️  Predictions differ (regenerated)")
    else:
        print(f"   ❌ Failed to load cached model")
    
    # Create visualizations
    print(f"\n3. CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Calculate errors
    errors = circular_error(y_test, pred_test)
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Predicted vs Actual scatter plot
    ax1.scatter(y_test, pred_test, alpha=0.6, s=20)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('True Azimuth (°)')
    ax1.set_ylabel('Predicted Azimuth (°)')
    ax1.set_title(f'SVM Predictions vs Ground Truth\\nMAE: {test_mae:.2f}°')
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution histogram
    ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Circular Error (°)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Error Distribution\\nMean: {np.mean(errors):.2f}°, Std: {np.std(errors):.2f}°')
    ax2.axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.2f}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals plot
    ax3.scatter(y_test, errors, alpha=0.6, s=20)
    ax3.set_xlabel('True Azimuth (°)')
    ax3.set_ylabel('Absolute Error (°)')
    ax3.set_title('Residuals Plot')
    ax3.axhline(np.mean(errors), color='red', linestyle='--', label=f'Mean Error: {np.mean(errors):.2f}°')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by azimuth range
    bins = np.linspace(y_test.min(), y_test.max(), 10)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_errors = []
    
    for i in range(len(bins)-1):
        mask = (y_test >= bins[i]) & (y_test < bins[i+1])
        if mask.sum() > 0:
            bin_errors.append(np.mean(errors[mask]))
        else:
            bin_errors.append(np.nan)
    
    ax4.plot(bin_centers, bin_errors, 'bo-', linewidth=2, markersize=8)
    ax4.set_xlabel('Azimuth Range (°)')
    ax4.set_ylabel('Mean Absolute Error (°)')
    ax4.set_title('Performance by Azimuth Range')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_path = Path(__file__).parent / "svm_analysis_results" / "cached_model_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved: {output_path}")
    
    # Show performance summary
    print(f"\n📊 PERFORMANCE SUMMARY:")
    print(f"   Cross-session MAE:    {test_mae:.2f}°")
    print(f"   Min error:           {np.min(errors):.2f}°")
    print(f"   Max error:           {np.max(errors):.2f}°")
    print(f"   Std deviation:       {np.std(errors):.2f}°")
    print(f"   Errors < 10°:        {(errors < 10).sum()}/{len(errors)} ({(errors < 10).mean()*100:.1f}%)")
    print(f"   Errors < 5°:         {(errors < 5).sum()}/{len(errors)} ({(errors < 5).mean()*100:.1f}%)")
    
    plt.show()

def test_force_retrain():
    """Test force retraining functionality."""
    print(f"\n4. TESTING FORCE RETRAIN")
    print("-" * 50)
    
    result = get_best_svm_for_visualization(use_cache=True, force_retrain=True)
    if result[0] is not None:
        svm, pred_test, y_test, test_mae, extras = result
        print(f"✅ Force retrain completed")
        print(f"   Test MAE: {test_mae:.2f}°")
    else:
        print(f"   ❌ Force retrain failed")

if __name__ == "__main__":
    # Optional: Clear all caches to start fresh 
    # clear_model_cache()
    # clear_feature_cache()
    
    test_cached_model_visualization() 
    test_force_retrain()
    
    print(f"\n🎉 Visualization testing complete!")