import numpy as np
from pathlib import Path
import sys
sys.path.append('..')
from Bens_Data_Import.Image_data_loaders.Spatial_Gradient_Loader.SpatialPolarizationLoader import SpatialPolarizationLoader

def test_spatial_loader():
    """Test the new spatial polarization loader."""
    
    # Path to Ben's PNG dataset
    data_path = Path("C:/Queens/ELEC498/Ben's Data/24-10-08-t000-forward-paradesquare/24-10-08-t000-forward-paradesquare")
    
    print("=" * 60)
    print("TESTING SPATIAL POLARIZATION LOADER")
    print("=" * 60)
    
    # Initialize the spatial loader
    loader = SpatialPolarizationLoader(
        data_path=data_path, 
        start_deg=0.0, 
        step_deg=1.0,
        target_size=(64, 64)  # Start with smaller size for testing
    )
    
    # Test loading a small subset first
    print("\n1. Testing with small sample...")
    dolp_spatial, aolp_spatial, azimuth_labels = loader.get_spatial_data(max_samples=5)
    
    print(f"\nSpatial data shapes:")
    print(f"  DoLP spatial: {dolp_spatial.shape}")
    print(f"  AoLP spatial: {aolp_spatial.shape}")
    print(f"  Azimuth labels: {azimuth_labels.shape}")
    
    # Test feature extraction methods
    print("\n2. Testing feature extraction methods...")
    
    # Method 1: Flattened features (all spatial info)
    features_flat = loader.create_feature_vectors(dolp_spatial, aolp_spatial, method='flatten')
    
    # Method 2: Statistical features (compact)
    features_stats = loader.create_feature_vectors(dolp_spatial, aolp_spatial, method='stats')
    
    print(f"\nFeature comparison:")
    print(f"  Original spatial data: {dolp_spatial.shape[1] * dolp_spatial.shape[2] * 2} potential features")
    print(f"  Flattened method: {features_flat.shape[1]} features")
    print(f"  Statistical method: {features_stats.shape[1]} features")
    
    # Check for spatial variation (not just uniform values)
    print("\n3. Checking spatial variation...")
    for i in range(min(3, len(dolp_spatial))):
        dolp_var = np.var(dolp_spatial[i])
        aolp_var = np.var(aolp_spatial[i])
        print(f"  Sample {i}: DoLP variance={dolp_var:.6f}, AoLP variance={aolp_var:.6f}")
        
        if dolp_var > 1e-6 or aolp_var > 1e-6:
            print(f"    ✓ Has spatial variation (good!)")
        else:
            print(f"    ✗ No spatial variation (uniform values)")
    
    # Test loading more data
    print("\n4. Testing larger sample...")
    dolp_all, aolp_all, azimuth_all = loader.get_spatial_data(max_samples=50)
    
    print(f"\nLarger dataset:")
    print(f"  Total samples: {len(dolp_all)}")
    print(f"  Azimuth range: {np.rad2deg(azimuth_all.min()):.1f}° to {np.rad2deg(azimuth_all.max()):.1f}°")
    print(f"  DoLP spatial range: [{dolp_all.min():.3f}, {dolp_all.max():.3f}]")
    print(f"  AoLP spatial range: [{aolp_all.min():.1f}°, {aolp_all.max():.1f}°]")
    
    print("\n" + "=" * 60)
    print("SPATIAL LOADER TEST COMPLETE")
    print("=" * 60)
    
    return dolp_all, aolp_all, azimuth_all, features_flat, features_stats

if __name__ == "__main__":
    test_spatial_loader()