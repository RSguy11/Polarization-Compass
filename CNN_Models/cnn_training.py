"""
Pre-trained CNN for Polarization-based Solar Azimuth Prediction

Uses a transfer learning approach with pre-built CNN backbone.
Tests if CNN architecture can improve upon L2's 12.7° test MAE.
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
    from tensorflow.keras.applications import MobileNetV2, ResNet50
    from tensorflow.keras.optimizers import Adam
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


def build_cnn_model(dolp_shape, aolp_shape):
    """
    Build a pre-trained CNN with transfer learning.
    
    Args:
        dolp_shape: Shape of DoLP input (height, width)
        aolp_shape: Shape of AoLP input (height, width)
        
    Returns:
        Keras model for azimuth regression
    """
    print("Building pre-trained CNN model...")
    
    # Input layers for DoLP and AoLP
    dolp_input = layers.Input(shape=(*dolp_shape, 1), name='dolp_input')
    aolp_input = layers.Input(shape=(*aolp_shape, 1), name='aolp_input')
    
    # Normalize inputs
    dolp_norm = layers.Normalization()(dolp_input)
    aolp_norm = layers.Normalization()(aolp_input)
    
    # Process DoLP with small CNN (since MobileNetV2 needs 224x224)
    dolp_resized = layers.Resizing(height=128, width=128)(dolp_norm)
    dolp_repeated = layers.Concatenate()([dolp_resized, dolp_resized, dolp_resized])  # Convert to 3 channels for MobileNetV2
    
    dolp_base = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet', name='mobilenetv2_dolp')(dolp_repeated)
    dolp_pooled = layers.GlobalAveragePooling2D()(dolp_base)
    
    # Process AoLP with small CNN
    aolp_resized = layers.Resizing(height=128, width=128)(aolp_norm)
    aolp_repeated = layers.Concatenate()([aolp_resized, aolp_resized, aolp_resized])  # Convert to 3 channels
    
    aolp_base = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet', name='mobilenetv2_aolp')(aolp_repeated)
    aolp_pooled = layers.GlobalAveragePooling2D()(aolp_base)
    
    # Concatenate features
    combined = layers.Concatenate()([dolp_pooled, aolp_pooled])
    
    # Dense layers for regression
    dense1 = layers.Dense(256, activation='relu')(combined)
    dense1 = layers.Dropout(0.3)(dense1)
    
    dense2 = layers.Dense(128, activation='relu')(dense1)
    dense2 = layers.Dropout(0.3)(dense2)
    
    dense3 = layers.Dense(64, activation='relu')(dense2)
    
    # Output: azimuth angle in degrees (0-360)
    output = layers.Dense(1, activation='linear', name='azimuth_output')(dense3)
    
    # Create model
    model = models.Model(inputs=[dolp_input, aolp_input], outputs=output)
    
    print(f"✓ Model built successfully")
    print(f"  Input shapes: DoLP {dolp_shape}, AoLP {aolp_shape}")
    print(f"  Parameters: {model.count_params():,}")
    
    return model


def train_cnn_pipeline():
    """Run the complete CNN training pipeline."""
    
    if not TF_AVAILABLE:
        print("❌ TensorFlow not available. Install with: pip install tensorflow")
        return
    
    print("POLARIZATION COMPASS - CNN TRAINING")
    print("=" * 60)
    print("Testing pre-trained CNN with spatial polarization data")
    print()
    
    # ---- STEP 1: Load Data ----
    print("STEP 1: Loading & Caching Data")
    print("=" * 60)
    
    rmc_folder = Path('Bens_Data_Import/Polarization_DataLoader/rmc')
    loader = PolarizationDataLoader(rmc_folder=rmc_folder)
    
    # Get metadata
    n_samples = len(loader)
    print(f"[OK] Total samples: {n_samples}")
    
    # Load all azimuths
    all_azimuths = []
    print("Loading azimuth labels...")
    for i in range(n_samples):
        if (i + 1) % 300 == 0:
            print(f"  Loaded {i+1}/{n_samples} labels...")
        try:
            labels = loader._get_labels(i)
            all_azimuths.append(labels['azimuth'])
        except:
            all_azimuths.append(0.0)
    
    all_azimuths = np.array(all_azimuths)
    print(f"[OK] Loaded {len(all_azimuths)} azimuth labels")
    print(f"    Azimuth range: {all_azimuths.min():.1f}° → {all_azimuths.max():.1f}°")
    
    # Shuffle and split
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
    print("STEP 2: Loading Spatial Features for CNN")
    print("=" * 60)
    
    print("Loading training DoLP and AoLP...")
    dolp_train, aolp_train = load_spatial_features(loader, train_indices, verbose=True)
    print(f"[OK] Train spatial features loaded: DoLP {dolp_train.shape}, AoLP {aolp_train.shape}")
    
    print("Loading test DoLP and AoLP...")
    dolp_test, aolp_test = load_spatial_features(loader, test_indices, verbose=True)
    print(f"[OK] Test spatial features loaded: DoLP {dolp_test.shape}, AoLP {aolp_test.shape}")
    print()
    
    # ---- STEP 3: Build and Train CNN ----
    print("STEP 3: Building and Training CNN")
    print("=" * 60)
    
    # Build model
    dolp_shape = dolp_train.shape[1:]
    aolp_shape = aolp_train.shape[1:]
    
    model = build_cnn_model(dolp_shape, aolp_shape)
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    
    print("\nTraining CNN...")
    history = model.fit(
        [dolp_train, aolp_train],
        azimuth_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    print("[OK] Training completed")
    print()
    
    # ---- STEP 4: Evaluate ----
    print("STEP 4: Evaluation")
    print("=" * 60)
    
    # Train performance
    train_pred = model.predict([dolp_train, aolp_train], verbose=0).flatten()
    train_mae = np.mean(np.abs(train_pred - azimuth_train))
    train_rmse = np.sqrt(np.mean((train_pred - azimuth_train) ** 2))
    
    print(f"Train Performance:")
    print(f"  MAE: {train_mae:.3f}°")
    print(f"  RMSE: {train_rmse:.3f}°")
    print()
    
    # Test performance
    test_pred = model.predict([dolp_test, aolp_test], verbose=0).flatten()
    test_mae = np.mean(np.abs(test_pred - azimuth_test))
    test_rmse = np.sqrt(np.mean((test_pred - azimuth_test) ** 2))
    
    print(f"Test Performance:")
    print(f"  MAE: {test_mae:.3f}°")
    print(f"  RMSE: {test_rmse:.3f}°")
    print()
    
    # Comparison with L2
    print("Comparison with L2 Baseline:")
    print(f"  L2 Test MAE: 12.721°")
    print(f"  CNN Test MAE: {test_mae:.3f}°")
    print(f"  Improvement: {12.721 - test_mae:.3f}° {'✓' if test_mae < 12.721 else '✗'}")
    print()
    
    # ---- STEP 5: Save Results ----
    print("STEP 5: Saving Results")
    print("=" * 60)
    
    results_dir = Path('CNN_Models/results')
    results_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = results_dir / 'pretrained_cnn_model.h5'
    model.save(model_path)
    print(f"[OK] Model saved to {model_path}")
    
    # Save results JSON
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'MobileNetV2_Transfer_Learning',
        'architecture': 'DoLP + AoLP through MobileNetV2 + Dense layers',
        'data': {
            'total_samples': n_samples,
            'train_samples': train_size,
            'test_samples': test_size,
            'dolp_shape': dolp_train.shape,
            'aolp_shape': aolp_train.shape
        },
        'performance': {
            'train_mae': float(train_mae),
            'train_rmse': float(train_rmse),
            'test_mae': float(test_mae),
            'test_rmse': float(test_rmse)
        },
        'comparison_with_l2': {
            'l2_test_mae': 12.721,
            'cnn_test_mae': float(test_mae),
            'improvement_degrees': float(12.721 - test_mae),
            'better_than_l2': test_mae < 12.721
        },
        'training_history': {
            'epochs': len(history.history['loss']),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
    }
    
    results_path = results_dir / 'cnn_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[OK] Results saved to {results_path}")
    
    print()
    print("=" * 60)
    print("CNN TRAINING COMPLETE")
    print(f"Final Test MAE: {test_mae:.3f}°")
    print(f"Status: {'✅ Better than L2' if test_mae < 12.721 else '⚠️  Needs improvement'}")
    print()
    
    return results


if __name__ == "__main__":
    results = train_cnn_pipeline()
