import numpy as np
from DatabaseLoader import DatabaseLoader
from pathlib import Path

# Update this if your relative path differs
BAG_PATH = Path(__file__).parent.parent / "rosbag2_2025_11_24-10_37_35"


def main():
    print("Initializing DatabaseLoader...")
    loader = DatabaseLoader(BAG_PATH)

    print("Running quick sanity check (max_samples=5)...")
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
    assert X.shape == (10, 3), "X shape incorrect"
    assert y.shape == (10,), "y shape incorrect"
    assert np.all(X[:,0] >= 0.0) and np.all(X[:,0] <= 1.0), "DoLP out of range"
    assert np.all(np.abs(X[:,1]) <= 1.0), "sin(AoLP) out of range"
    assert np.all(np.abs(X[:,2]) <= 1.0), "cos(AoLP) out of range"

    print("\n✅ DataLoader test PASSED")

if __name__ == "__main__":
    main()