"""
Complete Model Training Pipeline

This script runs all three models (L2, SVR, Random Forest) with the same dataset
and provides a complete comparison as specified in the blueprint.
"""

import os
import sys
import numpy as np
from datetime import datetime
import json
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.L2_Linear_reg.L2_pipeline import create_baseline_model
from Models.SVR_reg.SVR_pipeline import create_svr_model  
from Models.Random_Forest_reg.Random_Forest_pipeline import create_random_forest_model
from solar_azimuth_generator import SolarPositionCalculator
from Bens_Data_Import.Polarization_DataLoader.PolarizationDataLoader import PolarizationDataLoader
from pathlib import Path

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)


def create_training_plots(results, training_history, output_dir):
    """
    Create comprehensive training visualization plots.
    
    Args:
        results: Dictionary containing model training results
        training_history: Dictionary containing training progression data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for plotting
    model_names = []
    train_mae = []
    best_val_mae = []
    best_val_rmse = []
    
    for name, result in results.items():
        if 'error' not in result:
            model_names.append(name)
            train_mae.append(result['training_mae'])
            best_val_mae.append(result['best_val_mae'])
            best_val_rmse.append(result['best_val_rmse'])
    
    if not model_names:
        print(" No successful results to plot")
        return
    
    # Create figure with GridSpec layout for better spacing
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(2, 4, width_ratios=[2, 1, 1, 1], hspace=0.4, wspace=0.4)
    
    # Plot 1: Learning Curves (Error vs Training Set Size) - takes full left side
    ax1 = fig.add_subplot(gs[:, 0])
    
    if training_history:
        for model_name in model_names:
            if model_name in training_history:
                history = training_history[model_name]
                ax1.plot(history['sample_sizes'], history['train_errors'], 
                        'o-', label=f'{model_name} (Train)', linewidth=2, markersize=6)
                ax1.plot(history['sample_sizes'], history['val_errors'], 
                        's--', label=f'{model_name} (Val)', linewidth=2, markersize=6, alpha=0.7)
                # Mark best validation checkpoint
                best_idx = history['sample_sizes'].index(history['best_sample_size'])
                ax1.plot(history['best_sample_size'], history['val_errors'][best_idx], 
                        '*', markersize=20, color='gold', markeredgecolor='black', 
                        markeredgewidth=1.5, zorder=10)
        
        ax1.axhline(y=5.0, color='green', linestyle='--', linewidth=2, label='Target (5° MAE)', alpha=0.7)
        ax1.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Absolute Error (degrees)', fontsize=12, fontweight='bold')
        ax1.set_title('Learning Curves: Error vs Training Set Size', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
    else:
        ax1.text(0.5, 0.5, 'Learning curves require\nincremental training', 
                ha='center', va='center', fontsize=12, transform=ax1.transAxes)
        ax1.set_title('Learning Curves', fontsize=13, fontweight='bold')
    
    # Plot 2: Metrics Dashboard (3 sub-panels on the right)
    ax2 = fig.add_subplot(gs[0, 1])  # Top left
    ax3 = fig.add_subplot(gs[0, 2])  # Top middle
    ax4 = fig.add_subplot(gs[0, 3])  # Top right
    
    # MAE (Best Validation)
    ax2.bar(range(len(model_names)), best_val_mae, color='#3498db', alpha=0.8, edgecolor='black')
    ax2.axhline(y=5.0, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_ylabel('MAE (degrees)', fontweight='bold', fontsize=10)
    ax2.set_title('Best Validation MAE', fontweight='bold', fontsize=11)
    ax2.set_xticks(range(len(model_names)))
    ax2.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for i, v in enumerate(best_val_mae):
        ax2.text(i, v, f'{v:.1f}°', ha='center', va='bottom', fontsize=8)
    
    # RMSE (Best Validation)
    ax3.bar(range(len(model_names)), best_val_rmse, color='#e74c3c', alpha=0.8, edgecolor='black')
    ax3.set_ylabel('RMSE (degrees)', fontweight='bold', fontsize=10)
    ax3.set_title('Best Validation RMSE', fontweight='bold', fontsize=11)
    ax3.set_xticks(range(len(model_names)))
    ax3.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax3.grid(axis='y', alpha=0.3)
    for i, v in enumerate(best_val_rmse):
        ax3.text(i, v, f'{v:.1f}°', ha='center', va='bottom', fontsize=8)
    
    # Requirements Met
    meets_req = [results[name]['meets_requirements'] for name in model_names]
    colors = ['#2ecc71' if met else '#e74c3c' for met in meets_req]
    ax4.bar(range(len(model_names)), [1 if met else 0 for met in meets_req], color=colors, alpha=0.8, edgecolor='black')
    ax4.set_ylabel('Meets Requirements', fontweight='bold', fontsize=10)
    ax4.set_title('Blueprint Compliance\n(MAE < 5°)', fontweight='bold', fontsize=11)
    ax4.set_ylim([0, 1.2])
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    ax4.set_xticks(range(len(model_names)))
    ax4.set_xticklabels(model_names, rotation=15, ha='right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Training Dashboard: Learning Curves & Performance Metrics', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(output_dir, 'training_dashboard.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training dashboard")
    
    print(f"\n📊 All plots saved to: {output_dir}")


def run_complete_pipeline():
    """Run the complete model training pipeline for all three models."""
    
    print("POLARIZATION COMPASS - COMPLETE MODEL PIPELINE")
    print("=" * 60)
    print("Training L2, SVR, and Random Forest models")
    print()
    
    # Load actual polarization data
    print("Loading polarization data...")
    rmc_folder = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/Polarization_DataLoader/rmc")
    
    loader = PolarizationDataLoader(rmc_folder=rmc_folder)
    
    # Load subset of data (adjust max_samples for memory constraints)
    # Full dataset: ~2947 images. Start with subset for testing, then increase.
    max_samples = min(500, len(loader))  # Start with 500 samples, increase to None for all data
    
    print(f"Extracting DoLP and AoLP features from images...")
    print(f"Loading {max_samples} of {len(loader)} available samples...")
    
    dolp_list = []
    aolp_list = []
    azimuth_list = []
    
    for i in range(max_samples):
        sample = loader.get_item(i)
        if sample is not None:
            dolp_list.append(sample['features']['dolp'])
            aolp_list.append(sample['features']['aolp'])
            azimuth_list.append(sample['label'])
        
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{max_samples} samples...")
    
    # Convert to numpy arrays
    dolp = np.array(dolp_list)
    aolp = np.array(aolp_list)
    azimuth = np.deg2rad(np.array(azimuth_list))  # Convert to radians for consistency with old pipeline
    
    print(f"\n✓ Loaded {len(dolp)} samples")
    print(f"  DoLP shape: {dolp.shape}")
    print(f"  AoLP shape: {aolp.shape}")
    print(f"  Azimuth range: {np.rad2deg(azimuth.min()):.1f}° to {np.rad2deg(azimuth.max()):.1f}°")
    print(f"  DoLP range: [{dolp.min():.3f}, {dolp.max():.3f}]")
    if aolp.size > 0:
        print(f"  AoLP range: [{np.nanmin(aolp):.1f}° to {np.nanmax(aolp):.1f}°]")
    print()
    
    n_samples = len(dolp)
    
    # Shuffle data to prevent sequential bias (data is ordered by azimuth angle)
    print("Shuffling data to ensure random distribution...")
    np.random.seed(42)
    shuffle_idx = np.random.permutation(n_samples)
    dolp = dolp[shuffle_idx]
    aolp = aolp[shuffle_idx]
    azimuth = azimuth[shuffle_idx]
    
    # Create fixed train/test split (80/20)
    test_size = int(n_samples * 0.2)
    train_size = n_samples - test_size
    
    dolp_train, dolp_test = dolp[:train_size], dolp[train_size:]
    aolp_train, aolp_test = aolp[:train_size], aolp[train_size:]
    azimuth_train, azimuth_test = azimuth[:train_size], azimuth[train_size:]
    
    print(f"✓ Split data: {train_size} training, {test_size} test samples\n")
    
    # Create timestamp for saving models and results
    today = datetime.now().strftime('%Y-%m-%d')
    
    results = {}
    training_history = {}
    
    # Test each model
    models = {
        'L2_Baseline': create_baseline_model(alpha=1.0),
        'SVR_RBF': create_svr_model(C=1.0, gamma='scale', epsilon=0.1),
        'Random_Forest': create_random_forest_model(
            n_estimators=20,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )
    }
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        try:
            # Generate learning curve by training on increasing sample sizes
            print(f"  Generating learning curve...")
            
            # Use appropriate sample sizes based on total training data
            if train_size >= 200:
                sample_sizes = [20, 50, 100, 150, 200, min(250, train_size), train_size]
            elif train_size >= 100:
                sample_sizes = [10, 25, 50, 75, 100, train_size]
            elif train_size >= 40:
                sample_sizes = [5, 10, 20, 30, 40, train_size]
            else:
                sample_sizes = [min(5, train_size), min(10, train_size), train_size]
            
            # Remove duplicates and ensure ascending order
            sample_sizes = sorted(list(set(sample_sizes)))
            
            train_errors = []
            val_errors = []
            
            # Track best validation model
            best_val_error = float('inf')
            best_val_rmse = float('inf')
            best_model = None
            best_sample_size = 0
            
            for size in sample_sizes:
                # Create fresh model for each iteration
                if 'L2' in model_name:
                    model_temp = create_baseline_model(alpha=1.0)
                elif 'SVR' in model_name:
                    model_temp = create_svr_model(C=1.0, gamma='scale', epsilon=0.1)
                else:
                    model_temp = create_random_forest_model(
                        n_estimators=20, max_depth=5, min_samples_split=10, min_samples_leaf=5
                    )
                
                # Use 80/20 split within the subset
                subset_train_size = int(size * 0.8)
                subset_val_size = size - subset_train_size
                
                # Train on subset
                train_metrics_temp = model_temp.fit(
                    dolp_train[:subset_train_size], 
                    aolp_train[:subset_train_size], 
                    azimuth_train[:subset_train_size]
                )
                
                # Validate on consistent held-out portion from training data
                if subset_val_size > 0:
                    val_pred = model_temp.predict(
                        dolp_train[subset_train_size:size], 
                        aolp_train[subset_train_size:size]
                    )
                    val_error = np.rad2deg(np.mean(np.abs(val_pred - azimuth_train[subset_train_size:size])))
                    val_rmse = np.rad2deg(np.sqrt(np.mean((val_pred - azimuth_train[subset_train_size:size])**2)))
                else:
                    val_error = train_metrics_temp['mae']
                    val_rmse = train_metrics_temp['rmse']
                
                train_errors.append(train_metrics_temp['mae'])
                val_errors.append(val_error)
                
                # Track best validation performance
                if val_error < best_val_error:
                    best_val_error = val_error
                    best_val_rmse = val_rmse
                    best_model = model_temp  # Save reference to best model
                    best_sample_size = size
            
            print(f"  Best validation MAE: {best_val_error:.3f}° at {best_sample_size} samples")
            
            training_history[model_name] = {
                'sample_sizes': sample_sizes,
                'train_errors': train_errors,
                'val_errors': val_errors,
                'best_val_error': best_val_error,
                'best_val_rmse': best_val_rmse,
                'best_sample_size': best_sample_size
            }
            
            # Train final model on all training data
            train_metrics = model.fit(dolp_train, aolp_train, azimuth_train)
            
            # Evaluate on held-out test set (convert from radians to degrees)
            test_predictions = model.predict(dolp_test, aolp_test)
            test_mae = np.rad2deg(np.mean(np.abs(test_predictions - azimuth_test)))
            test_rmse = np.rad2deg(np.sqrt(np.mean((test_predictions - azimuth_test) ** 2)))
            
            # Cross-validate on training data only
            cv_metrics = model.cross_validate(dolp_train, aolp_train, azimuth_train, cv_folds=5)
            
            results[model_name] = {
                'training_mae': float(train_metrics['mae']),
                'cv_mae': float(cv_metrics['mae_mean']),
                'cv_rmse': float(cv_metrics['rmse_mean']),
                'best_val_mae': float(best_val_error),
                'best_val_rmse': float(best_val_rmse),
                'best_val_samples': int(best_sample_size),
                'test_mae': float(test_mae),
                'test_rmse': float(test_rmse),
                'meets_requirements': bool(test_mae < 5.0)
            }
            
            print(f" {model_name}")
            print(f"  Train MAE: {train_metrics['mae']:.3f}°")
            print(f"  CV MAE: {cv_metrics['mae_mean']:.3f}°")
            print(f"  Test MAE: {test_mae:.3f}° (held-out)")
            
            # Save two versions: final model and best validation model
            model_dir = os.path.join('saved_models', today)
            os.makedirs(model_dir, exist_ok=True)
            
            # Save final model (trained on all data)
            model_path = os.path.join(model_dir, f'{model_name}_final.pkl')
            model.save_model(model_path)
            print(f"  Final model saved: {model_path}")
            
            # Save best validation model (early stopping checkpoint)
            if best_model is not None:
                best_model_path = os.path.join(model_dir, f'{model_name}_best_val.pkl')
                best_model.save_model(best_model_path)
                print(f"  Best validation model saved: {best_model_path} (MAE: {best_val_error:.3f}° at {best_sample_size} samples)")
            
        except Exception as e:
            print(f" {model_name} failed: {str(e)}")
            results[model_name] = {'error': str(e)}
            import traceback
            traceback.print_exc()
    
    # Save results
    today = datetime.now().strftime('%Y-%m-%d')
    results_dir = os.path.join('training_plots', today)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'complete_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n RESULTS SUMMARY:")
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name}:")
            print(f"  CV MAE: {result['cv_mae']:.3f}°")
            print(f"  Test MAE: {result['test_mae']:.3f}° - {'✓' if result['meets_requirements'] else '✗'}")
    
    # Create visualization plots
    print(f"\n📈 Generating training plots...")
    create_training_plots(results, training_history, results_dir)
    
    print(f"\n💾 Results and plots saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    run_complete_pipeline()