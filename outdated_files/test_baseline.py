"""
Simple test script for L2 baseline model
"""

import sys
import os
sys.path.append('.')

from L2_Linear_reg.L2_pipeline import create_baseline_model
import numpy as np

def test_baseline():
    print('Testing L2 Baseline Model...')
    
    # Create model
    model = create_baseline_model(alpha=1.0)
    print('✓ Model created successfully')
    
    # Test with small mock data
    np.random.seed(42)
    dolp = np.random.uniform(0, 1, (100, 32, 32))
    aolp = np.random.uniform(0, 180, (100, 32, 32)) 
    azimuth = np.random.uniform(0, 360, 100)
    
    print('Training model...')
    metrics = model.fit(dolp, aolp, azimuth)
    print(f'✓ Training MAE: {metrics["mae"]:.3f}°')
    
    print('Testing prediction...')
    pred = model.predict(dolp[:10], aolp[:10])
    print(f'✓ Prediction successful, shape: {pred.shape}')
    
    print('L2 Baseline Model Test: SUCCESS!')
    return True

if __name__ == "__main__":
    test_baseline()