"""
Data Integration Helper for Connecting Preprocessing Pipeline to ML Models

This script helps integrate your existing preprocessing pipeline with the ML training loops.
It extracts DoLP and AoLP data from your stage2/stage3 output for use in model training.
"""

import os
import sys
import numpy as np
from typing import List, Dict, Tuple, Optional

# Add preprocessing modules to path
sys.path.append('.')
sys.path.append('Preprocessing')

def extract_ml_data_from_preprocessing() -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract DoLP, AoLP, and azimuth data from your preprocessing pipeline.
    
    Returns:
        Tuple of (DoLP, AoLP, azimuth_labels)
        - DoLP: Shape (N, H, W) - Degree of Linear Polarization
        - AoLP: Shape (N, H, W) - Angle of Linear Polarization in degrees  
        - azimuth_labels: Shape (N,) - Solar azimuth labels (if available)
    """
    
    print("🔄 Extracting ML data from preprocessing pipeline...")
    
    try:
        # Import your preprocessing modules
        from data_Loader.loading_mat_data import load_mat_file, add_mosaic_to_samples
        from stage1.stage1_pipe import pseduo_four_channel_desnoising, intensity_guilded_residual_interpolation
        from stage2.stage2_pipe import polarimetric_parameters_from_stokes
        
        print("✓ Successfully imported preprocessing modules")
        
        # Run your preprocessing pipeline 
        print("Loading original data...")
        original_file = load_mat_file(noise_level="High")  # Adjust noise level as needed
        
        if not original_file:
            raise ValueError("Failed to load .mat file data")
        
        print(f"Loaded {len(original_file)} samples")
        
        # Stage 1: Mosaic and denoising
        print("Running Stage 1: Mosaic and denoising...")
        file_in_mosaic_form = add_mosaic_to_samples(original_file)
        pfcd_output = pseduo_four_channel_desnoising(file_in_mosaic_form)
        channel_images = intensity_guilded_residual_interpolation(pfcd_output)
        
        # Stage 2: Compute polarimetric parameters (DoLP, AoLP)
        print("Running Stage 2: Polarimetric parameters...")
        stage2_out = polarimetric_parameters_from_stokes(channel_images)
        
        print(f"Stage 2 complete. Processing {len(stage2_out)} samples...")
        
        # Extract DoLP and AoLP arrays
        dolp_arrays = []
        aolp_arrays = []
        
        for i, sample in enumerate(stage2_out):
            # Extract DoLP (Degree of Linear Polarization)
            dolp = sample['DoLP']  # Shape: (H, W)
            aolp_deg = sample['AoLP_deg']  # Shape: (H, W), in degrees
            
            dolp_arrays.append(dolp)
            aolp_arrays.append(aolp_deg)
            
            if i % 100 == 0:
                print(f"  Processed {i+1}/{len(stage2_out)} samples")
        
        # Stack into arrays for ML training
        dolp_data = np.stack(dolp_arrays, axis=0)  # Shape: (N, H, W)
        aolp_data = np.stack(aolp_arrays, axis=0)  # Shape: (N, H, W)
        
        print(f"✓ Extracted polarization data:")
        print(f"  DoLP shape: {dolp_data.shape}")
        print(f"  AoLP shape: {aolp_data.shape}")
        print(f"  DoLP range: [{dolp_data.min():.3f}, {dolp_data.max():.3f}]")
        print(f"  AoLP range: [{aolp_data.min():.1f}°, {aolp_data.max():.1f}°]")
        
        # TODO: Add azimuth labels 
        # You need to implement this based on your data collection setup
        print("⚠️  Azimuth labels not implemented yet - using None")
        azimuth_labels = None
        
        return dolp_data, aolp_data, azimuth_labels
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the project root directory")
        return None, None, None
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return None, None, None


def create_mock_azimuth_labels(n_samples: int, 
                              random_state: int = 42) -> np.ndarray:
    """
    Create mock azimuth labels for testing purposes.
    
    In practice, you should replace this with actual solar azimuth calculations
    based on your data collection timestamps and GPS coordinates.
    
    Args:
        n_samples: Number of samples
        random_state: Random seed
        
    Returns:
        Mock azimuth labels in degrees (0-360)
    """
    
    print(f"⚠️  Creating {n_samples} mock azimuth labels")
    print("   Replace this with real solar position calculations!")
    
    np.random.seed(random_state)
    
    # Create somewhat realistic azimuth distribution
    # Real data would have solar azimuth based on time of day and location
    azimuth_labels = np.random.uniform(0, 360, n_samples)
    
    return azimuth_labels


def save_ml_ready_data(dolp: np.ndarray, 
                      aolp: np.ndarray, 
                      azimuth: np.ndarray,
                      output_dir: str = "ml_data") -> str:
    """
    Save processed data in ML-ready format.
    
    Args:
        dolp, aolp, azimuth: Processed data arrays
        output_dir: Directory to save data
        
    Returns:
        Path to saved data directory
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save arrays
    np.save(os.path.join(output_dir, 'dolp_data.npy'), dolp)
    np.save(os.path.join(output_dir, 'aolp_data.npy'), aolp)
    np.save(os.path.join(output_dir, 'azimuth_labels.npy'), azimuth)
    
    # Save metadata
    metadata = {
        'dolp_shape': dolp.shape,
        'aolp_shape': aolp.shape,
        'azimuth_shape': azimuth.shape,
        'dolp_range': [float(dolp.min()), float(dolp.max())],
        'aolp_range': [float(aolp.min()), float(aolp.max())],
        'azimuth_range': [float(azimuth.min()), float(azimuth.max())],
        'n_samples': len(dolp)
    }
    
    import json
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ ML-ready data saved to: {output_dir}")
    return output_dir


def load_ml_ready_data(data_dir: str = "ml_data") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load previously saved ML-ready data.
    
    Args:
        data_dir: Directory containing saved data
        
    Returns:
        Tuple of (DoLP, AoLP, azimuth_labels)
    """
    
    dolp = np.load(os.path.join(data_dir, 'dolp_data.npy'))
    aolp = np.load(os.path.join(data_dir, 'aolp_data.npy'))
    azimuth = np.load(os.path.join(data_dir, 'azimuth_labels.npy'))
    
    print(f"✓ Loaded ML data from: {data_dir}")
    print(f"  DoLP: {dolp.shape}, AoLP: {aolp.shape}, Azimuth: {azimuth.shape}")
    
    return dolp, aolp, azimuth


def main():
    """
    Main function to extract and prepare ML data from preprocessing pipeline.
    """
    
    print("POLARIZATION COMPASS - ML DATA INTEGRATION")
    print("=" * 50)
    
    # Step 1: Extract data from preprocessing pipeline
    dolp, aolp, azimuth_labels = extract_ml_data_from_preprocessing()
    
    if dolp is None:
        print("❌ Failed to extract preprocessing data. Check your pipeline.")
        return
    
    # Step 2: Handle missing azimuth labels
    if azimuth_labels is None:
        print("\\nCreating mock azimuth labels...")
        azimuth_labels = create_mock_azimuth_labels(len(dolp))
        print("⚠️  Remember to replace with real solar azimuth calculations!")
    
    # Step 3: Save ML-ready data  
    print("\\nSaving ML-ready data...")
    data_dir = save_ml_ready_data(dolp, aolp, azimuth_labels)
    
    # Step 4: Test loading
    print("\\nTesting data loading...")
    dolp_test, aolp_test, azimuth_test = load_ml_ready_data(data_dir)
    
    print("\\n🎯 NEXT STEPS:")
    print("1. Implement real azimuth label calculation using:")
    print("   - Data collection timestamps") 
    print("   - GPS coordinates of collection site")
    print("   - Solar position libraries (pvlib, ephem)")
    print("\\n2. Update L2_training_loop.py to use this data:")
    print("   - Replace load_preprocessed_data() function")
    print("   - Use load_ml_ready_data() to get actual data")
    print("\\n3. Run baseline training:")
    print("   python Training_loops/L2_training_loop.py")
    
    return dolp, aolp, azimuth_labels


if __name__ == "__main__":
    main()