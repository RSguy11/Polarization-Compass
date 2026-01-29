"""
Custom CNN for Polarization-based Solar Azimuth Prediction

Lightweight architecture designed from scratch (NOT pre-trained) to learn
polarization patterns directly from DoLP and AoLP spatial data.

Key differences from pre-trained approach:
- Compact design: 3 conv blocks instead of full MobileNetV2
- Polarization-aware: Processes DoLP and AoLP with dedicated streams
- Regularized: Dropout, early stopping, batch normalization
- Circular loss: Understands azimuth wrapping (0° = 360°)
"""

import os
import sys
import numpy as np
from datetime import datetime
import json
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from Bens_Data_Import.Polarization_DataLoader.PolarizationDataLoader import PolarizationDataLoader

# TensorFlow/Keras imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    TF_AVAILABLE = True
except ImportError:
    print("⚠️  TensorFlow not available - install with: pip install tensorflow")
    TF_AVAILABLE = False


def load_spatial_features(loader, indices, verbose=False):
    """
    Load raw DoLP and AoLP arrays as spatial images for CNN input.
    
    Args:
        loader: PolarizationDataLoader instance
        indices: Sample indices to load
        verbose: Print progress
        
    Returns:
        Tuple of (dolp_array, aolp_array) as spatial tensors
    """
    dolp_list = []
    aolp_list = []
    
    for i, idx in enumerate(indices):
        if verbose and (i + 1) % 50 == 0:
            print(f"  Loading spatial features {i+1}/{len(indices)}...")
        
        try:
            sample = loader.get_item(idx)
            if sample is not None:
                dolp = sample['features']['dolp'].astype(np.float32)
                aolp = sample['features']['aolp'].astype(np.float32)
                
                dolp_list.append(dolp)
                aolp_list.append(aolp)
            else:
                print(f"    Warning: Sample {idx} is None, skipping")
        except Exception as e:
            print(f"    Error loading sample {idx}: {e}")
            gc.collect()
            continue
        
        if (i + 1) % 50 == 0:
            gc.collect()
    
    # Stack into arrays: (n_samples, height, width)
    dolp_array = np.stack(dolp_list, axis=0).astype(np.float32)
    aolp_array = np.stack(aolp_list, axis=0).astype(np.float32)
    
    return dolp_array, aolp_array


def angular_mae(y_true, y_pred):
    """Compute circular MAE for azimuth angles (handles 0°/360° wrapping)."""
    diff = y_pred - y_true
    # Wrap to [-180, 180] range
    diff = (diff + 180) % 360 - 180
    return tf.reduce_mean(tf.abs(diff))


def proper_circular_mse(y_true, y_pred):
    """
    Proper circular MSE loss for angular data.
    
    Computes shortest angular distance BEFORE squaring, not after.
    This is critical for azimuth: treats 359° and 1° as 2° apart, not 358°.
    
    Args:
        y_true: True azimuth angles (degrees, 0-360)
        y_pred: Predicted azimuth angles (degrees, 0-360)
        
    Returns:
        MSE of wrapped angular differences
    """
    # Wrap predictions and true values to [0, 360)
    y_true_wrapped = tf.math.mod(y_true, 360.0)
    y_pred_wrapped = tf.math.mod(y_pred, 360.0)
    
    # Compute angular difference
    diff = y_pred_wrapped - y_true_wrapped
    
    # Wrap to [-180, 180] range (shortest path around circle)
    diff = tf.math.mod(diff + 180, 360) - 180
    
    # MSE on wrapped difference
    return tf.reduce_mean(diff ** 2)


def build_custom_cnn(dolp_shape, aolp_shape):
    """
    Build a lightweight CNN from scratch designed for polarization patterns.
    
    Architecture:
    - Separate input streams for DoLP and AoLP (preserve modality separation)
    - 3 conv blocks per stream (filters: 16 → 32 → 64)
    - Batch norm + ReLU + MaxPool per block
    - Spatial dropout for regularization
    - Concatenate features + dense layers
    - Single output: azimuth angle (0-360°)
    
    Args:
        dolp_shape: Shape of DoLP input (height, width)
        aolp_shape: Shape of AoLP input (height, width)
        
    Returns:
        Keras model for azimuth regression
    """
    print("Building custom CNN for polarization compass...")
    
    # ---- DoLP Stream ----
    dolp_input = layers.Input(shape=(*dolp_shape, 1), name='dolp_input')
    
    # Normalize to [0, 1] range
    dolp_norm = layers.BatchNormalization()(dolp_input)
    
    # Conv block 1: 16 filters
    dolp = layers.Conv2D(16, (3, 3), padding='same', use_bias=False)(dolp_norm)
    dolp = layers.BatchNormalization()(dolp)
    dolp = layers.Activation('relu')(dolp)
    dolp = layers.MaxPooling2D((2, 2))(dolp)
    dolp = layers.SpatialDropout2D(0.2)(dolp)
    
    # Conv block 2: 32 filters
    dolp = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(dolp)
    dolp = layers.BatchNormalization()(dolp)
    dolp = layers.Activation('relu')(dolp)
    dolp = layers.MaxPooling2D((2, 2))(dolp)
    dolp = layers.SpatialDropout2D(0.2)(dolp)
    
    # Conv block 3: 64 filters
    dolp = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(dolp)
    dolp = layers.BatchNormalization()(dolp)
    dolp = layers.Activation('relu')(dolp)
    dolp = layers.MaxPooling2D((2, 2))(dolp)
    dolp = layers.SpatialDropout2D(0.2)(dolp)
    
    # Global average pooling
    dolp_pooled = layers.GlobalAveragePooling2D()(dolp)
    
    # ---- AoLP Stream (Identical architecture) ----
    aolp_input = layers.Input(shape=(*aolp_shape, 1), name='aolp_input')
    
    # Normalize to [0, 360] → [0, 1] range
    aolp_norm = layers.BatchNormalization()(aolp_input)
    
    # Conv block 1: 16 filters
    aolp = layers.Conv2D(16, (3, 3), padding='same', use_bias=False)(aolp_norm)
    aolp = layers.BatchNormalization()(aolp)
    aolp = layers.Activation('relu')(aolp)
    aolp = layers.MaxPooling2D((2, 2))(aolp)
    aolp = layers.SpatialDropout2D(0.2)(aolp)
    
    # Conv block 2: 32 filters
    aolp = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(aolp)
    aolp = layers.BatchNormalization()(aolp)
    aolp = layers.Activation('relu')(aolp)
    aolp = layers.MaxPooling2D((2, 2))(aolp)
    aolp = layers.SpatialDropout2D(0.2)(aolp)
    
    # Conv block 3: 64 filters
    aolp = layers.Conv2D(64, (3, 3), padding='same', use_bias=False)(aolp)
    aolp = layers.BatchNormalization()(aolp)
    aolp = layers.Activation('relu')(aolp)
    aolp = layers.MaxPooling2D((2, 2))(aolp)
    aolp = layers.SpatialDropout2D(0.2)(aolp)
    
    # Global average pooling
    aolp_pooled = layers.GlobalAveragePooling2D()(aolp)
    
    # ---- Fusion and Dense Layers ----
    combined = layers.Concatenate()([dolp_pooled, aolp_pooled])
    
    # Dense layers with strong regularization
    dense1 = layers.Dense(128, use_bias=False)(combined)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Activation('relu')(dense1)
    dense1 = layers.Dropout(0.3)(dense1)
    
    dense2 = layers.Dense(64, use_bias=False)(dense1)
    dense2 = layers.BatchNormalization()(dense2)
    dense2 = layers.Activation('relu')(dense2)
    dense2 = layers.Dropout(0.3)(dense2)
    
    # Output layer: azimuth in [0, 360]
    output = layers.Dense(1, activation='relu', name='azimuth_output')(dense2)
    
    # Create model
    model = models.Model(inputs=[dolp_input, aolp_input], outputs=output)
    
    print(f"✓ Custom CNN built successfully")
    print(f"  Input shapes: DoLP {dolp_shape}, AoLP {aolp_shape}")
    print(f"  Parameters: {model.count_params():,}")
    print(f"  Architecture: 3 conv blocks per stream + fusion + 2 dense layers")
    
    return model


def train_custom_cnn_pipeline():
    """Run the complete custom CNN training pipeline."""
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow not available. Install with: pip install tensorflow")
        return
    
    print("POLARIZATION COMPASS - CUSTOM CNN TRAINING")
    print("=" * 60)
    print("Lightweight CNN designed from scratch for polarization data")
    print()
    
    # ---- STEP 1: Load Data ----
    print("STEP 1: Loading & Caching Data")
    print("=" * 60)
    
    rmc_folder = Path("Bens_Data_Import/rmc")
    loader = PolarizationDataLoader(rmc_folder)
    
    # Load all azimuth labels using loader's built-in labels_df (more reliable)
    print("Loading azimuth labels from GPS/INS data...")
    all_azimuths = loader.labels_df['azimuth'].values.astype(np.float32) % 360
    
    n_samples = len(all_azimuths)
    print(f"[OK] Loaded {n_samples} azimuth labels")
    print(f"    Azimuth range: {all_azimuths.min():.1f}° → {all_azimuths.max():.1f}°")
    
    # Train/test split
    np.random.seed(42)
    shuffled_indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * 0.8)
    test_size = n_samples - train_size
    
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:]
    
    azimuth_train = all_azimuths[train_indices]
    azimuth_test = all_azimuths[test_indices]
    
    print(f"[OK] Train: {train_size} samples | Test: {test_size} samples")
    print()
    
    # ---- STEP 2: Load Spatial Features ----
    print("STEP 2: Loading Spatial Features")
    print("=" * 60)
    
    print("Loading training DoLP and AoLP...")
    dolp_train, aolp_train = load_spatial_features(loader, train_indices, verbose=True)
    print(f"[OK] Train spatial features: DoLP {dolp_train.shape}, AoLP {aolp_train.shape}")
    
    print("Loading test DoLP and AoLP...")
    dolp_test, aolp_test = load_spatial_features(loader, test_indices, verbose=True)
    print(f"[OK] Test spatial features: DoLP {dolp_test.shape}, AoLP {aolp_test.shape}")
    print()
    
    # ---- STEP 3: Build and Train Custom CNN ----
    print("STEP 3: Building and Training Custom CNN")
    print("=" * 60)
    
    # Build model
    dolp_shape = dolp_train.shape[1:]
    aolp_shape = aolp_train.shape[1:]
    
    model = build_custom_cnn(dolp_shape, aolp_shape)
    
    # Compile with PROPER circular MSE loss and moderate learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=proper_circular_mse,  # Critical: proper circular loss that wraps BEFORE squaring
        metrics=[angular_mae]
    )
    
    print()
    print("Training configuration:")
    print(f"  Loss: Proper Circular MSE (wraps [-180,180] BEFORE squaring)")
    print(f"  Metric: Angular MAE")
    print(f"  Optimizer: Adam(lr=1e-4)")
    print(f"  Epochs: 100 with early stopping")
    print(f"  Batch size: 16 (smaller for stability)")
    print(f"  Validation split: 0.2")
    print()
    
    # Define callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,  # Stop if no improvement for 20 epochs
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    # Train
    print("Starting training...")
    history = model.fit(
        [dolp_train, aolp_train], azimuth_train,
        epochs=100,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    print()
    print("=" * 60)
    print("STEP 4: Evaluating on Test Set")
    print("=" * 60)
    
    # Evaluate on test set
    test_loss, test_angular_mae = model.evaluate([dolp_test, aolp_test], azimuth_test, verbose=0)
    print(f"[OK] Test Loss: {test_loss:.4f}")
    print(f"[OK] Test Angular MAE: {test_angular_mae:.3f}°")
    
    # Get predictions
    test_pred = model.predict([dolp_test, aolp_test], verbose=0).flatten()
    
    # Wrap predictions to [0, 360)
    test_pred = np.mod(test_pred, 360)
    
    # Compute detailed metrics
    angular_diff = test_pred - azimuth_test
    angular_diff = np.mod(angular_diff + 180, 360) - 180
    
    angular_mae_val = np.mean(np.abs(angular_diff))
    angular_rmse_val = np.sqrt(np.mean(angular_diff ** 2))
    
    print(f"[OK] Angular MAE (detailed): {angular_mae_val:.3f}°")
    print(f"[OK] Angular RMSE: {angular_rmse_val:.3f}°")
    print()
    
    # ---- STEP 5: Compare to L2 Baseline ----
    print("=" * 60)
    print("COMPARISON TO L2 BASELINE")
    print("=" * 60)
    l2_baseline_test_mae = 12.721
    print(f"L2 Linear Regression Test MAE: {l2_baseline_test_mae:.3f}°")
    print(f"Custom CNN Test MAE:          {angular_mae_val:.3f}°")
    
    improvement = l2_baseline_test_mae - angular_mae_val
    if improvement > 0:
        pct_improvement = (improvement / l2_baseline_test_mae) * 100
        print(f"✅ IMPROVEMENT: {improvement:.3f}° ({pct_improvement:.1f}% better)")
    else:
        pct_degradation = (abs(improvement) / l2_baseline_test_mae) * 100
        print(f"⚠️  DEGRADATION: {abs(improvement):.3f}° ({pct_degradation:.1f}% worse)")
    print()
    
    # ---- STEP 6: Save Results ----
    print("=" * 60)
    print("STEP 6: Saving Results")
    print("=" * 60)
    
    # Create results directory
    results_dir = Path("CNN_Models/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = results_dir / "custom_cnn_model.h5"
    model.save(str(model_path))
    print(f"[OK] Model saved to {model_path}")
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'Custom CNN (from scratch)',
        'architecture': {
            'type': 'Dual-stream 3-conv-block CNN',
            'dolp_input_shape': list(dolp_shape),
            'aolp_input_shape': list(aolp_shape),
            'parameters': int(model.count_params()),
            'loss_function': 'Circular loss (angular-aware)',
            'learning_rate': 1e-5,
            'optimizer': 'Adam'
        },
        'training': {
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'final_train_mae': float(history.history['angular_mae'][-1]),
            'final_val_mae': float(history.history['val_angular_mae'][-1])
        },
        'test_performance': {
            'test_loss': float(test_loss),
            'test_angular_mae': float(test_angular_mae),
            'test_angular_mae_detailed': float(angular_mae_val),
            'test_angular_rmse': float(angular_rmse_val),
            'n_test_samples': int(len(test_pred))
        },
        'comparison_to_l2': {
            'l2_baseline_test_mae': l2_baseline_test_mae,
            'custom_cnn_test_mae': float(angular_mae_val),
            'improvement_degrees': float(improvement),
            'improvement_percent': float(pct_improvement if improvement > 0 else -pct_degradation)
        }
    }
    
    results_path = results_dir / "custom_cnn_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved to {results_path}")
    
    # Save training history
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'angular_mae': [float(x) for x in history.history['angular_mae']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_angular_mae': [float(x) for x in history.history['val_angular_mae']]
    }
    
    history_path = results_dir / "custom_cnn_history.json"
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"[OK] Training history saved to {history_path}")
    
    print()
    print("=" * 60)
    print("CUSTOM CNN TRAINING COMPLETE")
    print("=" * 60)
    print(f"Test Angular MAE: {angular_mae_val:.3f}°")
    if improvement > 0:
        print(f"✅ Beat L2 baseline by {improvement:.3f}°")
    else:
        print(f"⚠️  Did not beat L2 baseline (gap: {abs(improvement):.3f}°)")
    print()
    
    return results


if __name__ == "__main__":
    train_custom_cnn_pipeline()
