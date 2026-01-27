"""Quick test of PolarizationDataLoader + SpatialStokeLoader integration"""
from Bens_Data_Import.Polarization_DataLoader.PolarizationDataLoader import PolarizationDataLoader
from pathlib import Path

print("Testing PolarizationDataLoader + SpatialStokeLoader integration...\n")

# Initialize loader
rmc_folder = Path("Bens_Data_Import/Polarization_DataLoader/rmc")
loader = PolarizationDataLoader(rmc_folder)

print(f"✓ DataLoader initialized with {len(loader)} samples\n")

# Test loading a single sample
print("Loading sample 0...")
sample = loader.get_item(0)

if sample is not None:
    print(f"✓ Sample loaded successfully!")
    print(f"  DoLP shape: {sample['features']['dolp'].shape}")
    print(f"  AoLP shape: {sample['features']['aolp'].shape}")
    print(f"  Label (azimuth): {sample['label']:.2f}°")
    print(f"  Image path: {sample['image_path']}")
    print("\n✓ Integration test PASSED - Ready to loop through all data!")
else:
    print("✗ Failed to load sample")
    print("\n✗ Integration test FAILED")
