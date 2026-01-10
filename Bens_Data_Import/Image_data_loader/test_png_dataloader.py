import numpy as np
from PNGDatabaseLoader import PNGDatabaseLoader
from pathlib import Path

# Path to the PNG dataset (Ben's Data folder)
BAG_PATH = Path("C:/Queens/ELEC498/Ben's Data/24-10-08-t000-forward-paradesquare/24-10-08-t000-forward-paradesquare")

def main():
    print("Initializing PNGDatabaseLoader...")
    loader = PNGDatabaseLoader(BAG_PATH, start_deg=0.0, step_deg=1.0)

    print("\nRunning sanity check (max_samples=5)...")
    X, y, t = loader.get_item(max_samples=10)

    print("\n=== SHAPES ===")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("timestamps shape:", t.shape)

    print("\n=== LABELS (degrees) ===")
    print(np.rad2deg(y))

    print("\n=== FEATURES ===")
    for i in range(len(X)):
        print(f"Sample {i}: DoLP={X[i,0]:.4f}, sin(AoLP)={X[i,1]:.4f}, cos(AoLP)={X[i,2]:.4f}")

    print("\n=== SANITY CHECKS ===")
    assert X.shape == (10, 3), f"X shape incorrect: expected (5, 3), got {X.shape}"
    assert y.shape == (10,), f"y shape incorrect: expected (5,), got {y.shape}"
    assert np.all(X[:,0] >= 0.0) and np.all(X[:,0] <= 1.0), "DoLP out of range"
    assert np.all(np.abs(X[:,1]) <= 1.0), "sin(AoLP) out of range"
    assert np.all(np.abs(X[:,2]) <= 1.0), "cos(AoLP) out of range"

    print("\n✅ PNGDatabaseLoader test PASSED")
    
    # Test loading all data
    print("\n" + "="*50)
    print("Loading ALL images...")
    X_all, y_all, t_all = loader.get_item()
    print(f"Total samples loaded: {len(X_all)}")
    print(f"Angle range: {np.rad2deg(y_all.min()):.1f}° to {np.rad2deg(y_all.max()):.1f}°")

if __name__ == "__main__":
    main()
