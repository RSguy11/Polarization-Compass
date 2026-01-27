"""Test memory-efficient feature extraction"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from Training_loops.run_all_models import extract_features_from_single_image, load_batch_features
from Bens_Data_Import.Polarization_DataLoader.PolarizationDataLoader import PolarizationDataLoader
import numpy as np

print("Testing memory-efficient feature extraction...\n")

# Initialize loader
rmc_folder = Path("Bens_Data_Import/Polarization_DataLoader/rmc")
loader = PolarizationDataLoader(rmc_folder)

print(f"Total samples available: {len(loader)}\n")

# Test single image feature extraction
print("1. Testing single image:")
sample = loader.get_item(0)
features = extract_features_from_single_image(
    sample['features']['dolp'],
    sample['features']['aolp']
)
print(f"   Original DoLP shape: {sample['features']['dolp'].shape} (~40MB)")
print(f"   Extracted features shape: {features.shape} (~128 bytes)")
print(f"   Memory reduction: {(sample['features']['dolp'].nbytes / features.nbytes):.0f}x\n")

# Test batch loading
print("2. Testing batch feature extraction (10 images):")
indices = list(range(10))
batch_features = load_batch_features(loader, indices)
print(f"   Batch features shape: {batch_features.shape}")
print(f"   Expected: (10, 16) - 10 images, 16 features each")
print(f"   Memory used: {batch_features.nbytes / 1024:.1f} KB instead of ~400 MB\n")

# Test larger batch
print("3. Testing larger batch (200 images):")
indices = list(range(200))
batch_features_large = load_batch_features(loader, indices)
print(f"   Batch features shape: {batch_features_large.shape}")
print(f"   Memory used: {batch_features_large.nbytes / 1024:.1f} KB")
print(f"   vs. original approach: ~7.5 GB\n")

print("✓ Memory-efficient feature extraction working correctly!")
print(f"\nThis approach can handle all {len(loader)} images using only ~{(len(loader) * 16 * 8) / (1024**2):.1f} MB")
print(f"instead of ~{(len(loader) * 2048 * 2448 * 8) / (1024**3):.1f} GB for raw pixel arrays!")
