"""
SVM Analysis Dashboard
=====================

Comprehensive visualizations for SVM model analysis including:
- Feature importance by group
- Error analysis and performance metrics  
- Learning curves and model diagnostics
- Cardinal direction error analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Models.SVM_classification.svm_classification_wrapper import CircularSVMClassifier
from Training_loops.run_all_models import extract_statistical_features_from_single_image

# Style configuration (borrowed from visualization.py)
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)


def get_feature_groups():
    """Define feature groups for better visualization."""
    feature_groups = {
        'DoLP_Mean': ['dolp_mean', 'dolp_q25', 'dolp_median', 'dolp_q75'],
        'DoLP_Spread': ['dolp_std', 'dolp_var', 'dolp_range', 'dolp_iqr'],
        'DoLP_Shape': ['dolp_skewness', 'dolp_kurtosis'],
        'DoLP_Extremes': ['dolp_min', 'dolp_max'],
        'DoLP_Percentiles': ['dolp_p10', 'dolp_p90', 'dolp_p95', 'dolp_p99'],
        
        'AoLP_Mean': ['aolp_mean', 'aolp_q25', 'aolp_median', 'aolp_q75'],
        'AoLP_Spread': ['aolp_std', 'aolp_var', 'aolp_range', 'aolp_iqr'],
        'AoLP_Shape': ['aolp_skewness', 'aolp_kurtosis'],
        'AoLP_Extremes': ['aolp_min', 'aolp_max'],
        'AoLP_Percentiles': ['aolp_p10', 'aolp_p90', 'aolp_p95', 'aolp_p99'],
        
        'DoLP_Histogram': [f'dolp_hist_{i}' for i in range(32)],  # Assuming 32 bins
        'AoLP_Histogram': [f'aolp_hist_{i}' for i in range(32)],
    }
    return feature_groups


def get_feature_names():
    """Get all feature names in the expected order."""
    # This matches the order from extract_statistical_features_from_single_image
    dolp_features = [
        'dolp_mean', 'dolp_std', 'dolp_var', 'dolp_min', 'dolp_max', 'dolp_range',
        'dolp_q25', 'dolp_median', 'dolp_q75', 'dolp_iqr',
        'dolp_skewness', 'dolp_kurtosis',
        'dolp_p10', 'dolp_p90', 'dolp_p95', 'dolp_p99'
    ]
    
    aolp_features = [
        'aolp_mean', 'aolp_std', 'aolp_var', 'aolp_min', 'aolp_max', 'aolp_range',
        'aolp_q25', 'aolp_median', 'aolp_q75', 'aolp_iqr',
        'aolp_skewness', 'aolp_kurtosis',
        'aolp_p10', 'aolp_p90', 'aolp_p95', 'aolp_p99'
    ]
    
    # Add histogram features (32 bins each)
    dolp_hist = [f'dolp_hist_{i}' for i in range(32)]
    aolp_hist = [f'aolp_hist_{i}' for i in range(32)]
    
    return dolp_features + aolp_features + dolp_hist + aolp_hist


def create_svm_dashboard(model, predictions=None, actual_labels=None, 
                        training_history=None, cv_scores=None, output_dir=None):
    """
    Create comprehensive SVM analysis dashboard.
    
    Args:
        model: Trained SVM model
        predictions: Array of predicted azimuths (degrees) 
        actual_labels: Array of actual azimuths (degrees)
        training_history: Dict with 'sample_sizes', 'train_errors', 'val_errors' (optional)
        cv_scores: Array of cross-validation MAE scores (optional)
        output_dir: Directory to save plots (optional)
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1], hspace=0.4, wspace=0.3)
    
    # 1. Feature Importance (top-left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    create_feature_importance_subplot(model, ax1)
    
    # 2. Error vs Azimuth (top-right, spans 2 columns)  
    ax2 = fig.add_subplot(gs[0, 2:])
    if predictions is not None and actual_labels is not None:
        create_error_vs_azimuth_subplot(predictions, actual_labels, ax2)
    else:
        ax2.text(0.5, 0.5, 'Error vs Azimuth\n(Requires predictions & labels)', 
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Error vs Azimuth', fontweight='bold', fontsize=13)
    
    # 3. Learning Curves (middle-left, spans 2 columns)
    ax3 = fig.add_subplot(gs[1, :2])  
    if training_history:
        create_learning_curves_subplot(training_history, ax3)
    else:
        ax3.text(0.5, 0.5, 'Learning Curves\n(Requires training_history)', 
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_title('Learning Curves', fontweight='bold', fontsize=13)
    
    # 4. Cardinal Direction Error Analysis (middle-right, spans 2 columns)
    ax4 = fig.add_subplot(gs[1, 2:])
    if predictions is not None and actual_labels is not None:
        create_cardinal_error_subplot(predictions, actual_labels, ax4)
    else:
        ax4.text(0.5, 0.5, 'Cardinal Direction Error\n(Requires predictions & labels)', 
                ha='center', va='center', fontsize=12, transform=ax4.transAxes)
        ax4.set_title('Cardinal Direction Error', fontweight='bold', fontsize=13)
    
    # 5. Error Distribution (bottom-left)
    ax5 = fig.add_subplot(gs[2, 0])
    if predictions is not None and actual_labels is not None:
        create_error_distribution_subplot(predictions, actual_labels, ax5)
    else:
        ax5.text(0.5, 0.5, 'Error Distribution\n(Requires data)', 
                ha='center', va='center', fontsize=10, transform=ax5.transAxes)
        ax5.set_title('Error Distribution', fontweight='bold', fontsize=11)
    
    # 6. Cross-Validation Scores (bottom-center-left)
    ax6 = fig.add_subplot(gs[2, 1])
    if cv_scores is not None:
        create_cv_scores_subplot(cv_scores, ax6) 
    else:
        ax6.text(0.5, 0.5, 'CV Scores\n(Requires cv_scores)', 
                ha='center', va='center', fontsize=10, transform=ax6.transAxes)
        ax6.set_title('Cross-Validation', fontweight='bold', fontsize=11)
    
    # 7. Performance Summary (bottom-center-right)
    ax7 = fig.add_subplot(gs[2, 2])
    if predictions is not None and actual_labels is not None:
        create_performance_summary_subplot(predictions, actual_labels, ax7)
    else:
        ax7.text(0.5, 0.5, 'Performance\nSummary\n(Requires data)', 
                ha='center', va='center', fontsize=10, transform=ax7.transAxes)
        ax7.set_title('Performance Summary', fontweight='bold', fontsize=11)
    
    # 8. Blueprint Compliance (bottom-right)
    ax8 = fig.add_subplot(gs[2, 3])
    if predictions is not None and actual_labels is not None:
        create_compliance_subplot(predictions, actual_labels, ax8)
    else:
        ax8.text(0.5, 0.5, 'Blueprint\nCompliance\n(Requires data)', 
                ha='center', va='center', fontsize=10, transform=ax8.transAxes)
        ax8.set_title('Compliance', fontweight='bold', fontsize=11)
    
    # Main title
    plt.suptitle('SVM Analysis Dashboard: Performance & Feature Analysis', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save if output directory provided
    if output_dir:
        save_path = output_dir / 'svm_analysis_dashboard.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] SVM dashboard saved to: {save_path}")
    
    # Only show plot if not running in script mode
    try:
        plt.show(block=False)
    except:
        print("[INFO] Plot saved to file (GUI display not available)")
        
    plt.close()  # Clean up memory


def create_feature_importance_subplot(model, ax):
    """Create feature importance chart as subplot.""" 
    feature_names = get_feature_names()
    feature_groups = get_feature_groups()
    
    # Extract feature weights
    if hasattr(model, 'svm') and hasattr(model.svm, 'coef_'):
        weights = np.abs(model.svm.coef_[0])
    elif hasattr(model, 'coef_'):
        weights = np.abs(model.coef_[0])
    else:
        ax.text(0.5, 0.5, 'No feature weights\navailable', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Feature Importance', fontweight='bold', fontsize=13)
        return
    
    # Calculate group importances
    group_importances = {}
    for group_name, group_features in feature_groups.items():
        indices = [feature_names.index(feat) for feat in group_features if feat in feature_names]
        if indices:
            group_importances[group_name] = np.sum(weights[indices])
    
    # Sort and plot
    sorted_groups = sorted(group_importances.items(), key=lambda x: x[1], reverse=True)
    group_names = [item[0] for item in sorted_groups]
    importances = [item[1] for item in sorted_groups]
    
    colors = ['#1f77b4' if 'DoLP' in name else '#ff7f0e' for name in group_names]
    bars = ax.bar(range(len(group_names)), importances, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Feature Groups', fontweight='bold', fontsize=10)
    ax.set_ylabel('Total Importance', fontweight='bold', fontsize=10)
    ax.set_title('Feature Importance by Group', fontweight='bold', fontsize=13)
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.3)


def create_error_vs_azimuth_subplot(predictions, actual_labels, ax):
    """Create error vs azimuth scatter plot."""
    # Ensure arrays are same size and not None
    if predictions is None or actual_labels is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Error vs Azimuth', fontweight='bold', fontsize=13)
        return
        
    # Convert to numpy arrays and ensure same size
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    if len(predictions) != len(actual_labels):
        ax.text(0.5, 0.5, f'Size mismatch\nPred: {len(predictions)}\nActual: {len(actual_labels)}', 
               ha='center', va='center', fontsize=10, transform=ax.transAxes)
        ax.set_title('Error vs Azimuth', fontweight='bold', fontsize=13)
        return
    
    # Calculate circular errors
    errors = circular_error(predictions, actual_labels)
    
    if errors is None or len(errors) == 0:
        ax.text(0.5, 0.5, 'Error calculation failed', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Error vs Azimuth', fontweight='bold', fontsize=13)
        return
    
    ax.scatter(actual_labels, errors, alpha=0.6, s=15, color='#3498db')
    ax.axhline(y=0, color='red', linestyle='-', linewidth=2, alpha=0.8)
    ax.axhline(y=5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axhline(y=-5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    
    ax.set_xlabel('True Azimuth (°)', fontweight='bold', fontsize=10)
    ax.set_ylabel('Prediction Error (°)', fontweight='bold', fontsize=10) 
    ax.set_title('Error vs True Azimuth', fontweight='bold', fontsize=13)
    ax.set_xlim(0, 360)
    ax.set_ylim(-180, 180)
    ax.grid(True, alpha=0.3)


def create_learning_curves_subplot(training_history, ax):
    """Create learning curves subplot."""
    sample_sizes = training_history.get('sample_sizes', [])
    train_errors = training_history.get('train_errors', [])
    val_errors = training_history.get('val_errors', [])
    
    if len(sample_sizes) == 0:
        ax.text(0.5, 0.5, 'No learning curve\ndata available', 
               ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title('Learning Curves', fontweight='bold', fontsize=13)
        return
    
    ax.plot(sample_sizes, train_errors, 'o-', label='Training', 
           linewidth=2, markersize=6, color='#2ecc71')
    ax.plot(sample_sizes, val_errors, 's--', label='Validation', 
           linewidth=2, markersize=6, color='#e74c3c', alpha=0.8)
    
    # Mark best point if available
    best_size = training_history.get('best_sample_size')
    if best_size and best_size in sample_sizes:
        best_idx = sample_sizes.index(best_size)
        ax.plot(best_size, val_errors[best_idx], '*', markersize=15, 
               color='gold', markeredgecolor='black', markeredgewidth=1, zorder=10)
    
    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target (5°)')
    ax.set_xlabel('Training Set Size', fontweight='bold', fontsize=10)
    ax.set_ylabel('MAE (degrees)', fontweight='bold', fontsize=10)
    ax.set_title('Learning Curves', fontweight='bold', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)


def create_cardinal_error_subplot(predictions, actual_labels, ax):
    """Create cardinal direction error analysis."""
    # Ensure arrays are valid
    if predictions is None or actual_labels is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Cardinal Direction Error', fontweight='bold', fontsize=13)
        return
        
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    if len(predictions) != len(actual_labels) or len(predictions) == 0:
        ax.text(0.5, 0.5, 'Invalid data size', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Cardinal Direction Error', fontweight='bold', fontsize=13)
        return
    
    error_values = circular_error(predictions, actual_labels)
    
    if error_values is None:
        ax.text(0.5, 0.5, 'Error calculation failed', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Cardinal Direction Error', fontweight='bold', fontsize=13)
        return
    
    errors = np.abs(error_values)
    
    # Define cardinal directions (45° sectors)
    dir_names = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    dir_errors = []
    
    for i, name in enumerate(dir_names):
        # Each sector is 45° wide, centered on cardinal/intercardinal directions
        center = i * 45
        # Create mask for this sector (handling wraparound)
        if center == 0:  # North sector (337.5° to 22.5°)
            mask = (actual_labels >= 337.5) | (actual_labels < 22.5)
        else:
            start = center - 22.5
            end = center + 22.5
            mask = (actual_labels >= start) & (actual_labels < end)
        
        if np.sum(mask) > 0:
            dir_errors.append(np.mean(errors[mask]))
        else:
            dir_errors.append(0)
    
    colors = ['#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#2ecc71', '#1abc9c', '#34495e', '#95a5a6']
    ax.bar(range(len(dir_names)), dir_errors, color=colors, alpha=0.8, edgecolor='black')
    
    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target')
    ax.set_xlabel('Cardinal Direction', fontweight='bold', fontsize=10)
    ax.set_ylabel('Mean Absolute Error (°)', fontweight='bold', fontsize=10)
    ax.set_title('Error by Cardinal Direction', fontweight='bold', fontsize=13)
    ax.set_xticks(range(len(dir_names)))
    ax.set_xticklabels(dir_names, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.legend(fontsize=9)


def create_error_distribution_subplot(predictions, actual_labels, ax):
    """Create error distribution histogram."""
    # Ensure arrays are valid
    if predictions is None or actual_labels is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Error Distribution', fontweight='bold', fontsize=11)
        return
        
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    if len(predictions) != len(actual_labels) or len(predictions) == 0:
        ax.text(0.5, 0.5, 'Invalid data size', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Error Distribution', fontweight='bold', fontsize=11)
        return
    
    error_values = circular_error(predictions, actual_labels)
    
    if error_values is None:
        ax.text(0.5, 0.5, 'Error calculation failed', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Error Distribution', fontweight='bold', fontsize=11)
        return
        
    errors = np.abs(error_values)
        
    mae = np.mean(errors)
    
    ax.hist(errors, bins=20, alpha=0.7, edgecolor='black', color='#3498db')
    ax.axvline(x=mae, color='red', linestyle='-', linewidth=2, label=f'MAE: {mae:.2f}°')
    ax.axvline(x=5.0, color='green', linestyle='--', linewidth=2, label='Target: 5°')
    
    ax.set_xlabel('Absolute Error (°)', fontweight='bold', fontsize=10)
    ax.set_ylabel('Count', fontweight='bold', fontsize=10)
    ax.set_title('Error Distribution', fontweight='bold', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def create_cv_scores_subplot(cv_scores, ax):
    """Create cross-validation scores visualization."""
    mean_cv = np.mean(cv_scores)
    std_cv = np.std(cv_scores)
    
    ax.bar(['CV MAE'], [mean_cv], color='#3498db', alpha=0.8, edgecolor='black', 
          yerr=[std_cv], capsize=5)
    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    
    ax.set_ylabel('MAE (degrees)', fontweight='bold', fontsize=10)
    ax.set_title('Cross-Validation', fontweight='bold', fontsize=11)
    ax.text(0, mean_cv + std_cv + 0.2, f'{mean_cv:.2f}° ± {std_cv:.2f}°', 
           ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)


def create_performance_summary_subplot(predictions, actual_labels, ax):
    """Create performance summary metrics."""
    # Ensure arrays are valid
    if predictions is None or actual_labels is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Performance Summary', fontweight='bold', fontsize=11)
        return
        
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    if len(predictions) != len(actual_labels) or len(predictions) == 0:
        ax.text(0.5, 0.5, 'Invalid data size', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Performance Summary', fontweight='bold', fontsize=11)
        return
    
    error_values = circular_error(predictions, actual_labels)
    
    if error_values is None:
        ax.text(0.5, 0.5, 'Error calculation failed', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Performance Summary', fontweight='bold', fontsize=11)
        return
        
    errors = np.abs(error_values)
        
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(errors**2))
    median_error = np.median(errors)
    p95_error = np.percentile(errors, 95)
    
    metrics = ['MAE', 'RMSE', 'Median', 'P95']
    values = [mae, rmse, median_error, p95_error]
    colors = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6']
    
    bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    ax.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Target')
    
    ax.set_ylabel('Error (degrees)', fontweight='bold', fontsize=10)
    ax.set_title('Performance Summary', fontweight='bold', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.2f}°', ha='center', va='bottom', fontsize=8, fontweight='bold')


def create_compliance_subplot(predictions, actual_labels, ax):
    """Create blueprint compliance visualization."""
    # Ensure arrays are valid
    if predictions is None or actual_labels is None:
        ax.text(0.5, 0.5, 'Data not available', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Blueprint\nCompliance', fontweight='bold', fontsize=11)
        return
        
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    if len(predictions) != len(actual_labels) or len(predictions) == 0:
        ax.text(0.5, 0.5, 'Invalid data size', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Blueprint\nCompliance', fontweight='bold', fontsize=11)
        return
    
    error_values = circular_error(predictions, actual_labels)
    
    if error_values is None:
        ax.text(0.5, 0.5, 'Error calculation failed', ha='center', va='center', 
               fontsize=12, transform=ax.transAxes)
        ax.set_title('Blueprint\nCompliance', fontweight='bold', fontsize=11)
        return
    
    errors = np.abs(error_values)
    
    meets_target = np.mean(errors) < 5.0
    pct_under_5 = np.mean(errors < 5.0) * 100
    
    colors = ['#2ecc71' if meets_target else '#e74c3c']
    ax.bar(['Meets Target'], [1 if meets_target else 0], color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Compliance', fontweight='bold', fontsize=10)
    ax.set_title('Blueprint\nCompliance', fontweight='bold', fontsize=11)
    ax.set_ylim([0, 1.2])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.grid(axis='y', alpha=0.3)
    
    # Add percentage text
    ax.text(0, 0.5, f'{pct_under_5:.1f}%\nof predictions\n< 5° error', 
           ha='center', va='center', fontsize=8, fontweight='bold')


def circular_error(pred_deg, actual_deg):
    """Calculate circular error (handles 0/360 wraparound)."""
    try:
        pred_deg = np.array(pred_deg, dtype=float)
        actual_deg = np.array(actual_deg, dtype=float)
        
        if len(pred_deg) != len(actual_deg):
            print(f"WARNING: Array size mismatch in circular_error: {len(pred_deg)} vs {len(actual_deg)}")
            min_len = min(len(pred_deg), len(actual_deg))
            pred_deg = pred_deg[:min_len]
            actual_deg = actual_deg[:min_len]
        
        diff = pred_deg - actual_deg
        # Wrap to [-180, 180]
        diff = (diff + 180) % 360 - 180
        return diff
    except Exception as e:
        print(f"Error in circular_error calculation: {e}")
        return None


# Global cache to avoid re-training SVM and re-extracting features
_SVM_CACHE = None


def get_cached_svm_data():
    """Get cached SVM model and data, training only once."""
    global _SVM_CACHE
    
    if _SVM_CACHE is not None:
        print("[CACHE] Using cached SVM model and data (much faster!)")
        return (
            _SVM_CACHE['model'],
            _SVM_CACHE['predictions'], 
            _SVM_CACHE['actual_labels'],
            _SVM_CACHE['test_mae'],
            _SVM_CACHE['extra_data']
        )
    
    print("[CACHE] Training SVM for the first time...")
    
    # Import and get real SVM data
    from real_svm_test import get_best_svm_for_visualization
    
    # Train the model once
    svm_model, predictions, actual_labels, test_mae, extra_data = get_best_svm_for_visualization()
    
    if svm_model is None:
        print("[CACHE] ERROR: Failed to train SVM model")
        return None, None, None, None, None
    
    # Cache everything
    _SVM_CACHE = {
        'model': svm_model,
        'predictions': predictions,
        'actual_labels': actual_labels, 
        'test_mae': test_mae,
        'extra_data': extra_data
    }
    
    predictions_count = len(predictions) if predictions is not None else 0
    print(f"[CACHE] ✅ Cached SVM model with {predictions_count} predictions and {test_mae:.2f}° MAE")
    
    return svm_model, predictions, actual_labels, test_mae, extra_data


def clear_svm_cache():
    """Clear the SVM cache to force re-training."""
    global _SVM_CACHE
    _SVM_CACHE = None
    print("[CACHE] SVM cache cleared")


def demo_feature_importance():
    """Create feature importance using REAL trained SVM model."""
    try:
        print("Loading REAL SVM model for feature importance analysis...")
        
        # Use cached SVM data
        svm_model, predictions, actual_labels, test_mae, extra_data = get_cached_svm_data()
        
        if svm_model is None:
            print("ERROR: Failed to load real SVM model")
            return
            
        print(f"SUCCESS: Loaded real SVM with {test_mae:.2f}° test MAE")
        
        # Create figure with REAL model
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(14, 8))
        
        create_feature_importance_subplot(svm_model, ax)
        
        plt.tight_layout()
        save_path = Path(__file__).parent / "real_svm_feature_importance.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        try:
            plt.show(block=False)
        except:
            print("[INFO] Plot saved to file (GUI display not available)")
            
        plt.close()  # Clean up memory
        
        print(f"REAL SVM feature importance saved to: {save_path}")
        
    except Exception as e:
        print(f"ERROR loading real SVM data: {e}")
        import traceback
        traceback.print_exc()


def demo_svm_dashboard():
    """Create comprehensive dashboard using REAL SVM model and data."""
    try:
        print("Creating comprehensive dashboard with REAL SVM data...")
        
        # Use cached SVM data  
        svm_model, predictions, actual_labels, test_mae, extra_data = get_cached_svm_data()
        
        if svm_model is None:
            print("ERROR: Failed to load real SVM model and data")
            return
        
        predictions_count = len(predictions) if predictions is not None else 0
        print(f"SUCCESS: Using real SVM model with {predictions_count} predictions")
        print(f"Real test MAE: {test_mae:.2f}°")
        
        # Create output directory
        output_dir = Path(__file__).parent / "real_svm_analysis_results"
        output_dir.mkdir(exist_ok=True)
        
        # Create comprehensive dashboard with REAL data
        create_svm_dashboard(
            model=svm_model,
            predictions=predictions,
            actual_labels=actual_labels,
            training_history=None,  # Could add learning curves later if needed
            cv_scores=None,  # Could add if we run CV
            output_dir=output_dir
        )
        
        print(f"Real SVM dashboard created and saved to: {output_dir}")
        
    except Exception as e:
        print(f"ERROR creating real dashboard: {e}")
        import traceback
        traceback.print_exc()


def analyze_real_svm_performance():
    """Analyze real SVM performance in detail."""
    try:
        print("Analyzing REAL SVM performance...")
        
        # Use cached SVM data
        svm_model, predictions, actual_labels, test_mae, extra_data = get_cached_svm_data()
        
        if svm_model is None or predictions is None or actual_labels is None:
            print("ERROR: No real SVM data available")
            return
            
        # Ensure arrays are the same size and not None
        predictions_len = len(predictions) if predictions is not None else 0
        actual_len = len(actual_labels) if actual_labels is not None else 0
        
        if predictions_len != actual_len:
            print(f"ERROR: Size mismatch - predictions: {predictions_len}, labels: {actual_len}")
            return
            
        # Calculate detailed error metrics using the same function from real_svm_test
        from Models.SVM_classification.svm_classification_wrapper import calculate_circular_error
        test_mae_calc, test_rmse = calculate_circular_error(actual_labels, predictions)
        
        # Also calculate our own errors for detailed analysis
        error_values = circular_error(predictions, actual_labels)
        
        if error_values is None:
            print("ERROR: Failed to calculate circular errors")
            return
            
        errors = np.abs(error_values)
        
        print(f"\nREAL SVM PERFORMANCE ANALYSIS")
        print(f"=" * 40)
        print(f"Test samples: {predictions_len:,}")
        print(f"MAE (calculated): {test_mae_calc:.2f}°")
        print(f"MAE (reported): {test_mae:.2f}°")
        print(f"RMSE: {test_rmse:.2f}°")
        print(f"Median error: {np.median(errors):.2f}°")
        print(f"95th percentile: {np.percentile(errors, 95):.2f}°")
        print(f"Max error: {np.max(errors):.2f}°")
        print(f"% under 5°: {np.mean(errors < 5) * 100:.1f}%")
        print(f"% under 10°: {np.mean(errors < 10) * 100:.1f}%")
        
        # Azimuth range coverage
        print(f"\nDATA COVERAGE:")
        print(f"Azimuth range: {actual_labels.min():.1f}° to {actual_labels.max():.1f}°")
        print(f"Coverage: {actual_labels.max() - actual_labels.min():.1f}° of 360°")
        
        # Feature importance summary
        if hasattr(svm_model, 'svm') and svm_model.svm is not None and hasattr(svm_model.svm, 'coef_'):
            weights = np.abs(svm_model.svm.coef_[0])
            print(f"\nMODEL INFO:")
            print(f"Features used: {len(weights)}")
            print(f"Max feature weight: {np.max(weights):.4f}")
            print(f"Mean feature weight: {np.mean(weights):.4f}")
            
        if extra_data is not None and isinstance(extra_data, dict):
            config_info = extra_data.get('config', 'Unknown')
            print(f"\nConfiguration: {config_info}")
        else:
            print("\nConfiguration: Unknown (extra_data not available)")
        
    except Exception as e:
        print(f"ERROR analyzing performance: {e}")
        import traceback
        traceback.print_exc()


def create_predicted_vs_actual_chart(predictions, actual_labels, model_name="SVM", output_dir=None):
    """
    Create standalone predicted vs actual azimuth scatter plot (like cross_dataset_validation.py).
    
    Args:
        predictions: Array of predicted azimuths (degrees)
        actual_labels: Array of actual azimuths (degrees) 
        model_name: Name of the model for title
        output_dir: Directory to save plot (optional)
    """
    if predictions is None or actual_labels is None:
        print("[ERROR] Cannot create predicted vs actual chart: missing data")
        return
        
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    if len(predictions) != len(actual_labels) or len(predictions) == 0:
        print(f"[ERROR] Data size mismatch: pred={len(predictions)}, actual={len(actual_labels)}")
        return
    
    # Calculate MAE for title
    error_values = circular_error(predictions, actual_labels)
    if error_values is not None:
        mae = np.mean(np.abs(error_values))
    else:
        mae = float('inf')
    
    # Create figure
    plt.figure(figsize=(8, 8))
    
    # Scatter plot
    plt.scatter(actual_labels, predictions, alpha=0.6, s=15, color='#3498db', label='Predictions')
    
    # Formatting with dynamic axis limits based on data
    plt.xlabel('Actual Azimuth (°)', fontweight='bold', fontsize=12)
    plt.ylabel('Predicted Azimuth (°)', fontweight='bold', fontsize=12)
    plt.title(f'{model_name}: Predicted vs Actual Azimuth\nMAE: {mae:.2f}°', fontweight='bold', fontsize=14)
    
    # Set axis limits based on data range with some padding
    min_val = min(np.min(actual_labels), np.min(predictions))
    max_val = max(np.max(actual_labels), np.max(predictions))
    padding = (max_val - min_val) * 0.05  # 5% padding
    axis_min = max(0, min_val - padding)
    axis_max = min(360, max_val + padding)
    
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    
    # Perfect prediction line (plotted after setting limits)
    plt.plot([axis_min, axis_max], [axis_min, axis_max], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add text box with statistics
    within_5 = np.mean(np.abs(error_values) <= 5) * 100 if error_values is not None else 0
    within_10 = np.mean(np.abs(error_values) <= 10) * 100 if error_values is not None else 0
    stats_text = f'Within 5°: {within_5:.1f}%\nWithin 10°: {within_10:.1f}%'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f'{model_name.lower()}_predicted_vs_actual.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Predicted vs actual chart saved to: {save_path}")
    
    try:
        plt.show(block=False)
    except:
        print("[INFO] Plot saved to file (GUI display not available)")
        
    plt.close()


def create_error_count_chart(predictions, actual_labels, model_name="SVM", output_dir=None):
    """
    Create standalone error distribution histogram (count vs calibrated error).
    
    Args:
        predictions: Array of predicted azimuths (degrees)
        actual_labels: Array of actual azimuths (degrees)
        model_name: Name of the model for title
        output_dir: Directory to save plot (optional)
    """
    if predictions is None or actual_labels is None:
        print("[ERROR] Cannot create error count chart: missing data")
        return
        
    predictions = np.array(predictions)
    actual_labels = np.array(actual_labels)
    
    if len(predictions) != len(actual_labels) or len(predictions) == 0:
        print(f"[ERROR] Data size mismatch: pred={len(predictions)}, actual={len(actual_labels)}")
        return
    
    # Calculate circular errors
    error_values = circular_error(predictions, actual_labels)
    if error_values is None:
        print("[ERROR] Could not calculate circular errors")
        return
        
    # Use signed errors for the histogram to show direction bias
    signed_errors = error_values
    abs_errors = np.abs(error_values)
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(abs_errors**2))
    median_error = np.median(abs_errors)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    n_bins = min(30, len(signed_errors) // 3)  # Adaptive bin count
    n, bins, patches = plt.hist(signed_errors, bins=n_bins, alpha=0.7, 
                               edgecolor='black', color='#3498db', label='Error Distribution')
    
    # Add vertical lines for key metrics
    plt.axvline(x=0, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Perfect (0° error)')
    plt.axvline(x=float(mae), color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'MAE: {mae:.2f}°')
    plt.axvline(x=float(-mae), color='red', linestyle='--', linewidth=2, alpha=0.8)
    plt.axvline(x=5, color='orange', linestyle=':', linewidth=2, alpha=0.7, label='Target: ±5°')
    plt.axvline(x=-5, color='orange', linestyle=':', linewidth=2, alpha=0.7)
    
    # Formatting
    plt.xlabel('Calibrated Error (°)', fontweight='bold', fontsize=12)
    plt.ylabel('Count', fontweight='bold', fontsize=12)
    plt.title(f'{model_name}: Error Distribution\nMAE: {mae:.2f}°, RMSE: {rmse:.2f}°, Median: {median_error:.2f}°', 
              fontweight='bold', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add statistics text box
    within_5 = np.mean(abs_errors <= 5) * 100
    within_10 = np.mean(abs_errors <= 10) * 100
    within_45 = np.mean(abs_errors <= 45) * 100
    
    stats_text = f'Within 5°: {within_5:.1f}%\nWithin 10°: {within_10:.1f}%\nWithin 45°: {within_45:.1f}%\nTotal samples: {len(signed_errors)}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / f'{model_name.lower()}_error_distribution.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Error distribution chart saved to: {save_path}")
    
    try:
        plt.show(block=False)
    except:
        print("[INFO] Plot saved to file (GUI display not available)")
        
    plt.close()


def demo_standalone_charts():
    """Demonstrate the standalone chart functions using real SVM data."""
    try:
        # Get cached SVM data
        result = get_cached_svm_data()
        
        # Handle case where function might return None or unexpected format
        if result is None:
            print("[ERROR] No cached SVM data available")
            return
            
        # Defensive unpacking
        if isinstance(result, tuple) and len(result) == 5:
            svm_model, predictions, actual_labels, test_mae, extra_data = result
        else:
            print(f"[ERROR] Unexpected return format from get_cached_svm_data(): {type(result)}")
            return
            
        if svm_model is None or predictions is None:
            print("[ERROR] No cached SVM data available")
            return
            
        # Safe length check
        try:
            predictions_len = len(predictions)
        except (TypeError, AttributeError) as e:
            print(f"[ERROR] Cannot get length of predictions: {e}")
            print(f"[DEBUG] predictions type: {type(predictions)}")
            return
            
        print(f"[DEMO] Creating standalone charts with {predictions_len} predictions...")
        
        # Create output directory
        output_dir = Path(__file__).parent / "svm_analysis_results"
        
        print("\n1. Creating predicted vs actual scatter plot...")
        create_predicted_vs_actual_chart(predictions, actual_labels, "SVM", output_dir)
        
        print("\n2. Creating error distribution histogram...")
        create_error_count_chart(predictions, actual_labels, "SVM", output_dir)
        
        if test_mae is not None:
            print(f"\n[DEMO] ✅ Standalone charts created successfully!")
            print(f"        MAE: {test_mae:.2f}° with {len(predictions)} test samples")
        else:
            print(f"\n[DEMO] ✅ Standalone charts created successfully!")
            print(f"        Charts created with {len(predictions)} test samples")
        
    except Exception as e:
        print(f"[ERROR] Failed to create standalone charts: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("REAL SVM Analysis Visualization Suite")
    print("=" * 50)
    
    print("\n1. Analyzing real SVM performance...")
    analyze_real_svm_performance()
    
    print("\n2. Creating comprehensive dashboard with REAL data...")
    demo_svm_dashboard()
    
    print("\n3. Creating feature importance with REAL model...")
    demo_feature_importance()
    
    print("\n4. Creating standalone charts...")
    demo_standalone_charts()