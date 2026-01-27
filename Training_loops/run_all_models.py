import os
import sys
import numpy as np
from datetime import datetime
import json
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path


def extract_features_from_single_image(dolp, aolp):
    """Extract fully flattened DoLP and AoLP arrays as features."""
    dolp_flat = dolp.ravel()
    aolp_flat = aolp.ravel()
    features = np.concatenate([dolp_flat, aolp_flat])
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def load_batch_features(loader, indices, verbose=False):
    """Load batch and extract REAL feature vectors from DoLP and AoLP."""
    feature_list = []
    failed_count = 0
    
    for idx_num, idx in enumerate(indices):
        if verbose and (idx_num + 1) % 50 == 0:
            print(f"    Processing {idx_num + 1}/{len(indices)} features (extracted: {len(feature_list)}, failed: {failed_count})...")
            sys.stdout.flush()
        
        try:
            sample = loader.get_item(idx)
            if sample is not None:
                features = extract_features_from_single_image(
                    sample['features']['dolp'],
                    sample['features']['aolp']
                )
                # Convert to float32 to save memory
                feature_list.append(features.astype(np.float32))
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1
        
        # Force garbage collection every 50 samples to free memory from temporary arrays
        if (idx_num + 1) % 50 == 0:
            gc.collect()
    
    print(f"    Successfully extracted {len(feature_list)} features ({failed_count} failed)")
    
    # Convert to numpy array with float32 to save memory
    if len(feature_list) > 0:
        return np.array(feature_list, dtype=np.float32)
    else:
        raise ValueError("No features were successfully extracted!")


def run_complete_pipeline():
    """Run the complete model training pipeline for all three models."""
    
    # IMPORT SKLEARN AND MODELS HERE (after __main__ guard)
    from Models.L2_Linear_reg.L2_pipeline import create_baseline_model
    from Models.SVR_reg.SVR_pipeline import create_svr_model  
    from Models.Random_Forest_reg.Random_Forest_pipeline import create_random_forest_model
    from solar_azimuth_generator import SolarPositionCalculator
    from Bens_Data_Import.Polarization_DataLoader.PolarizationDataLoader import PolarizationDataLoader
    from Training_loops.visualization import create_training_plots
    
    print("POLARIZATION COMPASS - COMPLETE MODEL PIPELINE")
    print("=" * 60)
    print("Training L2, SVR, and Random Forest models")
    print()
    
    print("STEP 1: Loading & Caching Data")
    print("=" * 60)
    
    rmc_folder = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/Polarization_DataLoader/rmc")
    loader = PolarizationDataLoader(rmc_folder=rmc_folder)
    max_samples = len(loader)
    
    batch_size = 100
    all_indices = []
    all_labels = []
    
    print(f"Loading metadata from {max_samples} samples...")
    for i in range(max_samples):
        try:
            all_indices.append(i)
            all_labels.append(loader.labels_df.iloc[i]['azimuth'])
            
            # Progress every 300
            if (i + 1) % 300 == 0:
                print(f"  Loaded {i + 1}/{max_samples} samples...")
        except Exception as e:
            continue
    
    print(f"[OK] Loaded metadata for {len(all_indices)} samples")
    print(f"    Azimuth range: {min(all_labels):.1f}° → {max(all_labels):.1f}°")
    
    azimuth = np.deg2rad(np.array(all_labels))
    n_samples = len(all_indices)
    
    
    print("\nShuffling & splitting data (80% train / 20% test)...")
    np.random.seed(42)
    shuffle_idx = np.random.permutation(n_samples)
    shuffled_indices = [all_indices[i] for i in shuffle_idx]
    azimuth = azimuth[shuffle_idx]
    
    test_size = int(n_samples * 0.2)
    train_size = n_samples - test_size
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]
    azimuth_train = azimuth[:train_size]
    azimuth_test = azimuth[train_size:]
    
    print(f"[OK] Train: {train_size} samples | Test: {test_size} samples")
    
    print("\nCaching training features (fully flattened arrays)...")
    all_train_features = load_batch_features(loader, train_indices, verbose=True)
    print(f"[OK] Cached training features: {all_train_features.shape}")
    
    print("Caching test features...")
    all_test_features = load_batch_features(loader, test_indices, verbose=True)
    print(f"[OK] Cached test features: {all_test_features.shape}\n")
    
    today = datetime.now().strftime('%Y-%m-%d')
    results = {}
    training_history = {}
    
    print("STEP 2: Training Models")
    print("=" * 60)
    
    models = {
        'L2_Baseline': create_baseline_model(alpha=0.001),
        'SVR_RBF': create_svr_model(C=100.0, gamma=0.001, epsilon=0.1),
        'Random_Forest': create_random_forest_model(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5
        )
    }
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        try:
            print(f"  Generating learning curve...")
            
            if train_size >= 200:
                sample_sizes = [20, 50, 100, 150, 200, min(250, train_size), train_size]
            elif train_size >= 100:
                sample_sizes = [10, 25, 50, 75, 100, train_size]
            elif train_size >= 40:
                sample_sizes = [5, 10, 20, 30, 40, train_size]
            else:
                sample_sizes = [min(5, train_size), min(10, train_size), train_size]
            
            sample_sizes = sorted(list(set(sample_sizes)))
            train_errors = []
            val_errors = []
            best_val_error = float('inf')
            best_val_rmse = float('inf')
            best_sample_size = 0
            
            for size in sample_sizes:
                if 'L2' in model_name:
                    model_temp = create_baseline_model(alpha=0.001)
                elif 'SVR' in model_name:
                    model_temp = create_svr_model(C=100.0, gamma=0.001, epsilon=0.1)
                else:
                    model_temp = create_random_forest_model(
                        n_estimators=200, max_depth=10, min_samples_split=10, min_samples_leaf=5
                    )
                
                subset_train_size = int(size * 0.8)
                subset_val_size = size - subset_train_size
                subset_indices = np.random.choice(range(len(all_train_features)), subset_train_size, replace=False)
                train_features = all_train_features[subset_indices]
                
                train_metrics_temp = model_temp.fit_from_features(
                    train_features, 
                    azimuth_train[subset_indices]
                )
                if subset_val_size > 0:
                    remaining_indices = np.setdiff1d(np.arange(len(all_train_features)), subset_indices)
                    val_indices = np.random.choice(remaining_indices, subset_val_size, replace=False)
                    val_features = all_train_features[val_indices]
                    
                    val_pred = model_temp.predict_from_features(val_features)
                    val_error = np.rad2deg(np.mean(np.abs(val_pred - azimuth_train[val_indices])))
                    val_rmse = np.rad2deg(np.sqrt(np.mean((val_pred - azimuth_train[val_indices])**2)))
                else:
                    val_error = train_metrics_temp['mae']
                    val_rmse = train_metrics_temp['rmse']
                
                train_errors.append(train_metrics_temp['mae'])
                val_errors.append(val_error)
                
                if val_error < best_val_error:
                    best_val_error = val_error
                    best_val_rmse = val_rmse
                    best_sample_size = size
            
            print(f"  Best validation MAE: {best_val_error:.3f} deg at {best_sample_size} samples")
            
            training_history[model_name] = {
                'sample_sizes': sample_sizes,
                'train_errors': train_errors,
                'val_errors': val_errors,
                'best_val_error': best_val_error,
                'best_val_rmse': best_val_rmse,
                'best_sample_size': best_sample_size
            }
            
            print(f"  Training final model on full training set ({train_size} samples)...")
            train_metrics = model.fit_from_features(all_train_features, azimuth_train)
            
            print(f"  Evaluating on test set ({test_size} samples)...")
            test_predictions = model.predict_from_features(all_test_features)
            test_mae = np.rad2deg(np.mean(np.abs(test_predictions - azimuth_test)))
            test_rmse = np.rad2deg(np.sqrt(np.mean((test_predictions - azimuth_test) ** 2)))
            
            cv_metrics = model.cross_validate_from_features(all_train_features, azimuth_train, cv_folds=5)
            
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
            print(f"  Train MAE: {train_metrics['mae']:.3f} deg")
            print(f"  CV MAE: {cv_metrics['mae_mean']:.3f} deg")
            print(f"  Test MAE: {test_mae:.3f} deg (held-out)")
            
            model_dir = os.path.join('saved_models', today)
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, f'{model_name}_final.pkl')
            model.save_model(model_path)
            print(f"  Final model saved: {model_path}")
            
        except Exception as e:
            print(f" {model_name} failed: {str(e)}")
            results[model_name] = {'error': str(e)}
            import traceback
            traceback.print_exc()
    
    today = datetime.now().strftime('%Y-%m-%d')
    results_dir = os.path.join('training_plots', today)
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, 'complete_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n RESULTS SUMMARY:")
    for name, result in results.items():
        if 'error' not in result:
            print(f"{name}:")
            print(f"  CV MAE: {result['cv_mae']:.3f} deg")
            print(f"  Test MAE: {result['test_mae']:.3f} deg - {'PASS' if result['meets_requirements'] else 'FAIL'}")
    
    print(f"\nGenerating training plots...")
    create_training_plots(results, training_history, results_dir)
    print(f"\nResults and plots saved to: {results_dir}")
    
    return results

if __name__ == "__main__":
    run_complete_pipeline()