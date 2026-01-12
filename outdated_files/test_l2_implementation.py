"""
Memory-Efficient L2 Baseline Test

This script tests the L2 baseline with very small data to validate the implementation
without memory issues. Once validated, you can scale up with real data.
"""

import os
import sys
import numpy as np

sys.path.append('.')
from L2_Linear_reg.L2_pipeline import create_baseline_model

def create_small_test_data():
    """Create minimal test data for validation."""
    
    print("Creating small test dataset...")
    
    # Very small dataset for testing
    n_samples = 100
    h, w = 16, 16  # Very small images
    
    np.random.seed(42)
    
    # Simple mock data
    dolp = np.random.uniform(0, 1, (n_samples, h, w)).astype(np.float32)
    aolp = np.random.uniform(0, 180, (n_samples, h, w)).astype(np.float32)
    azimuth = np.random.uniform(0, 360, n_samples).astype(np.float32)
    
    print(f"✓ Test data: DoLP {dolp.shape}, AoLP {aolp.shape}, Azimuth {azimuth.shape}")
    return dolp, aolp, azimuth

def test_baseline_implementation():
    """Test the L2 baseline implementation step by step."""
    
    print("TESTING L2 BASELINE IMPLEMENTATION")
    print("=" * 40)
    
    # Create test data
    dolp, aolp, azimuth = create_small_test_data()
    
    # Test 1: Model creation
    print("\\n1. Testing model creation...")
    model = create_baseline_model(alpha=1.0)
    print("✓ Model created successfully")
    
    # Test 2: Feature extraction
    print("\\n2. Testing feature extraction...")
    try:
        features = model.extract_polarization_features(dolp, aolp)
        print(f"✓ Features extracted: {features.shape}")
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False
    
    # Test 3: Model training
    print("\\n3. Testing model training...")
    try:
        metrics = model.fit(dolp, aolp, azimuth)
        print(f"✓ Training completed - MAE: {metrics['mae']:.3f}°")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    
    # Test 4: Prediction
    print("\\n4. Testing prediction...")
    try:
        pred = model.predict(dolp[:10], aolp[:10])
        print(f"✓ Prediction successful: {pred.shape}")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return False
    
    # Test 5: Cross-validation (small)
    print("\\n5. Testing cross-validation...")
    try:
        cv_results = model.cross_validate(dolp, aolp, azimuth, cv_folds=3)
        print(f"✓ CV MAE: {cv_results['mae_mean']:.3f} ± {cv_results['mae_std']:.3f}°")
    except Exception as e:
        print(f"❌ Cross-validation failed: {e}")
        return False
    
    print("\\n✅ ALL TESTS PASSED - L2 BASELINE IMPLEMENTATION IS WORKING!")
    
    return True

def main():
    """Main test function."""
    
    success = test_baseline_implementation()
    
    if success:
        print("\\n🎯 IMPLEMENTATION STATUS:")
        print("✅ L2 Linear Regression baseline model: WORKING")
        print("✅ Feature extraction from DoLP/AoLP: WORKING") 
        print("✅ Cross-validation pipeline: WORKING")
        print("✅ Model training and prediction: WORKING")
        
        print("\\n📋 NEXT STEPS:")
        print("1. ⏳ Wait for preprocessing dependencies (bm3d) to install")
        print("2. 🔗 Connect real polarization data from preprocessing pipeline")
        print("3. 📊 Add real solar azimuth labels")
        print("4. 🚀 Scale up to full 20k dataset as per blueprint")
        print("5. 🎯 Target MAE < 5° on real data")
        
        print("\\n📂 READY FOR NEXT MODELS:")
        print("- SVR (Support Vector Regression)")
        print("- Random Forest Regression")
        
    else:
        print("\\n❌ Implementation needs fixes before proceeding")

if __name__ == "__main__":
    main()