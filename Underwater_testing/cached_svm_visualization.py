"""
Visualization script using cached SVM models
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from real_svm_test import get_best_svm_for_visualization

def circular_error(y_true, y_pred):
    """Calculate circular error between true and predicted angles in degrees."""
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, 360 - diff)
    return diff

def create_svm_visualization_dashboard():
    """Create comprehensive visualization dashboard using cached SVM models."""
    
    print("=" * 60)
    print("SVM MODEL VISUALIZATION DASHBOARD")
    print("Using Cached Models for Instant Results!")
    print("=" * 60)
    
    # Load the best model and data (should use cache now!)
    print("\n📥 Loading best SVM model and data...")
    result = get_best_svm_for_visualization(use_cache=True, force_retrain=False)
    
    if result[0] is None:
        print("❌ ERROR: Failed to load model for visualization")
        return
    
    svm, pred_test, y_test, test_mae, extras = result
    
    print(f"✅ Model loaded successfully!")
    print(f"   Model type: CircularSVM")
    print(f"   Test samples: {len(pred_test):,}")
    print(f"   Test MAE: {test_mae:.2f}°")
    print(f"   Config: {extras.get('config', 'Unknown')}")
    
    # Calculate detailed error metrics
    errors = circular_error(y_test, pred_test)
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"   Mean Absolute Error: {np.mean(errors):.2f}°")
    print(f"   Median Error:       {np.median(errors):.2f}°") 
    print(f"   Standard Deviation: {np.std(errors):.2f}°")
    print(f"   Max Error:          {np.max(errors):.2f}°")
    print(f"   Min Error:          {np.min(errors):.2f}°")
    print(f"   Errors < 5°:        {(errors < 5).sum()}/{len(errors)} ({(errors < 5).mean()*100:.1f}%)")
    print(f"   Errors < 10°:       {(errors < 10).sum()}/{len(errors)} ({(errors < 10).mean()*100:.1f}%)")
    print(f"   Errors < 15°:       {(errors < 15).sum()}/{len(errors)} ({(errors < 15).mean()*100:.1f}%)")
    
    # Create comprehensive visualization
    print(f"\n🎨 Creating visualization dashboard...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Main scatter plot: Predicted vs Actual
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(y_test, pred_test, c=errors, cmap='viridis', alpha=0.7, s=30)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel('True Azimuth (°)', fontsize=12)
    ax1.set_ylabel('Predicted Azimuth (°)', fontsize=12)
    ax1.set_title(f'SVM Predictions vs Ground Truth\\nMAE: {test_mae:.2f}° | Samples: {len(pred_test)}', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, label='Error (°)')
    
    # 2. Error distribution
    ax2 = plt.subplot(2, 3, 2)
    n, bins, patches = ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.set_xlabel('Circular Error (°)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Error Distribution\\nMean: {np.mean(errors):.2f}° ± {np.std(errors):.2f}°', fontsize=12)
    ax2.axvline(np.mean(errors), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}°')
    ax2.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.2f}°')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Residuals plot
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(y_test, errors, alpha=0.6, s=20, color='coral')
    ax3.set_xlabel('True Azimuth (°)', fontsize=12)
    ax3.set_ylabel('Absolute Error (°)', fontsize=12)
    ax3.set_title('Residuals: Error vs True Azimuth', fontsize=12)
    ax3.axhline(np.mean(errors), color='red', linestyle='--', label=f'Mean Error: {np.mean(errors):.2f}°')
    # Add trend line
    z = np.polyfit(y_test, errors, 1)
    p = np.poly1d(z)
    ax3.plot(sorted(y_test), p(sorted(y_test)), "g--", alpha=0.8, label=f'Trend: {z[0]:.3f}x + {z[1]:.2f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Performance by azimuth range (binned analysis)
    ax4 = plt.subplot(2, 3, 4)
    n_bins = 8
    bins = np.linspace(y_test.min(), y_test.max(), n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_errors = []
    bin_counts = []
    
    for i in range(len(bins)-1):
        mask = (y_test >= bins[i]) & (y_test < bins[i+1])
        if mask.sum() > 0:
            bin_errors.append(np.mean(errors[mask]))
            bin_counts.append(mask.sum())
        else:
            bin_errors.append(np.nan)
            bin_counts.append(0)
    
    bars = ax4.bar(bin_centers, bin_errors, width=(bin_centers[1]-bin_centers[0])*0.8, 
                   alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    ax4.set_xlabel('Azimuth Range (°)', fontsize=12)
    ax4.set_ylabel('Mean Absolute Error (°)', fontsize=12)
    ax4.set_title('Performance by Azimuth Range', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        if count > 0:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
    
    # 5. Cumulative error distribution
    ax5 = plt.subplot(2, 3, 5)
    sorted_errors = np.sort(errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax5.plot(sorted_errors, cumulative, 'b-', linewidth=2)
    ax5.set_xlabel('Error Threshold (°)', fontsize=12)
    ax5.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax5.set_title('Cumulative Error Distribution', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    # Add some key thresholds
    for threshold in [5, 10, 15, 20]:
        pct = (errors <= threshold).mean() * 100
        if pct > 0:
            ax5.axvline(threshold, color='red', linestyle='--', alpha=0.7)
            ax5.text(threshold + 0.5, pct - 5, f'{pct:.1f}%\\nat {threshold}°', 
                    rotation=90, va='top', ha='left', fontsize=9)
    
    # 6. Model comparison summary (if we have training data)
    ax6 = plt.subplot(2, 3, 6)
    if 'train_predictions' in extras and extras['train_predictions'] is not None:
        train_pred = extras['train_predictions']
        train_labels = extras['train_labels']
        train_errors = circular_error(train_labels, train_pred)
        train_mae = np.mean(train_errors)
        
        # Create comparison bar chart
        categories = ['Training', 'Cross-Session\\nTesting']
        maes = [train_mae, test_mae]
        colors = ['lightblue', 'lightcoral']
        
        bars = ax6.bar(categories, maes, color=colors, alpha=0.7, edgecolor='black')
        ax6.set_ylabel('Mean Absolute Error (°)', fontsize=12)
        ax6.set_title('Training vs Cross-Session Performance', fontsize=12)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mae in zip(bars, maes):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{mae:.2f}°', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Add gap annotation
        gap = test_mae - train_mae
        ax6.text(0.5, max(maes) * 0.8, f'Cross-session gap:\\n{gap:.2f}°', 
                ha='center', va='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5))
    else:
        ax6.text(0.5, 0.5, 'Training data\\nnot available', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
        ax6.set_title('Model Performance Summary', fontsize=12)
    
    plt.tight_layout()
    
    # Save the visualization
    output_dir = Path(__file__).parent / "svm_analysis_results"
    output_path = output_dir / "cached_svm_dashboard.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Visualization saved: {output_path}")
    
    # Show summary statistics
    print(f"\n🎯 FINAL SUMMARY:")
    print(f"   Best Model: CircularSVM with {extras.get('config', {}).get('n_bins', 'N/A')} bins")
    print(f"   Cross-Session MAE: {test_mae:.2f}°")
    print(f"   Excellent predictions (< 5°):  {(errors < 5).mean()*100:.1f}%")
    print(f"   Good predictions (< 10°):      {(errors < 10).mean()*100:.1f}%")
    print(f"   Acceptable predictions (< 15°): {(errors < 15).mean()*100:.1f}%")
    
    if test_mae < 10:
        print(f"   🎉 EXCELLENT: Model achieves single-digit cross-session error!")
    elif test_mae < 15:
        print(f"   ✅ GOOD: Strong cross-session performance")
    else:
        print(f"   ⚠️  FAIR: Room for improvement in cross-session generalization")
    
    plt.show()

if __name__ == "__main__":
    create_svm_visualization_dashboard()