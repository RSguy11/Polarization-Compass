"""
Quick Start Script for L2 Baseline Model

This script provides a simple entry point to test the L2 baseline model
with mock data. Use this to verify your implementation works before
connecting to your real preprocessing pipeline.

Usage:
    python quick_start_baseline.py
"""

import os
import sys
import numpy as np

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from L2_Linear_reg.L2_pipeline import create_baseline_model
from Training_loops.L2_training_loop import L2TrainingLoop


def main():
    """Quick start demonstration of the L2 baseline model."""
    
    print("🚀 POLARIZATION COMPASS - L2 BASELINE MODEL")
    print("=" * 50)
    print()
    print("This script demonstrates the L2 baseline model with mock data.")
    print("Replace the mock data loading with your actual preprocessing pipeline.")
    print()
    
    # Option 1: Quick test with simple model
    print("OPTION 1: Quick Model Test")
    print("-" * 30)
    
    print("Creating baseline L2 model...")
    model = create_baseline_model(alpha=1.0)
    
    print("Generating mock test data...")
    n_samples = 1000
    h, w = 64, 64
    
    # Mock data (replace with your actual data)
    np.random.seed(42)
    dolp_test = np.random.uniform(0, 1, (n_samples, h, w))
    aolp_test = np.random.uniform(0, 180, (n_samples, h, w))
    azimuth_test = np.random.uniform(0, 360, n_samples)
    
    print(f"Data shapes: DoLP {dolp_test.shape}, AoLP {aolp_test.shape}, Azimuth {azimuth_test.shape}")
    
    print("Training model...")
    train_metrics = model.fit(dolp_test, aolp_test, azimuth_test)
    
    print("Performing cross-validation...")
    cv_metrics = model.cross_validate(dolp_test, aolp_test, azimuth_test, cv_folds=3)
    
    print("✓ Quick test completed successfully!")
    print()
    
    # Option 2: Full training pipeline
    print("OPTION 2: Full Training Pipeline")
    print("-" * 35)
    
    response = input("Run full training pipeline? (y/n): ").lower().strip()
    
    if response == 'y':
        print("Starting full training pipeline...")
        
        trainer = L2TrainingLoop(
            output_dir="L2_results",
            random_state=42
        )
        
        try:
            trained_model = trainer.run_complete_pipeline()
            print("✅ Full pipeline completed successfully!")
            print(f"Results saved to: L2_results/")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {str(e)}")
            print("This is expected with mock data. Connect real data for full functionality.")
    
    else:
        print("Skipping full pipeline.")
    
    print()
    print("🎯 NEXT STEPS:")
    print("-" * 15)
    print("1. Connect your preprocessing pipeline data:")
    print("   - Modify L2_training_loop.py -> load_preprocessed_data()")
    print("   - Load actual DoLP/AoLP data from your stage2/stage3 output")
    print("   - Add real azimuth labels from your data collection")
    print()
    print("2. Run with real data:")
    print("   - python Training_loops/L2_training_loop.py")
    print()
    print("3. Implement other models:")
    print("   - SVR (Support Vector Regression)")
    print("   - Random Forest Regression")
    print()
    print("4. Compare model performance against blueprint requirements:")
    print("   - MAE < 5° (target)")
    print("   - RMSE ≤ 10% (target)")
    print("   - Robustness under noise and turbidity")


if __name__ == "__main__":
    main()