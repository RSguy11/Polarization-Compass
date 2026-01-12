from matplotlib.colors import hsv_to_rgb
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')
from Bens_Data_Import.Image_data_loaders.Spatial_Gradient_Loader.SpatialPolarizationLoader import SpatialPolarizationLoader

def test_spatial_loader():
    """Test the new spatial polarization loader."""
    
    # Path to Ben's PNG dataset
    data_path = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/24-10-08-t000-forward-paradesquare")
    
    print("=" * 60)
    print("TESTING SPATIAL POLARIZATION LOADER")
    print("=" * 60)
    
    # Initialize the spatial loader
    loader = SpatialPolarizationLoader(
        data_path=data_path, 
        start_deg=0.0, 
        step_deg=1.0,
        target_size=(256, 256)  # Start with smaller size for testing
    )
    
    # Test loading a small subset first
    print("\n1. Testing with small sample...")
    dolp_spatial, aolp_spatial, azimuth_labels = loader.get_spatial_data(max_samples=5)

    for i in range(len(dolp_spatial)):
        visualize_dolp_aolp_inlays(
            dolp=dolp_spatial[i],
            aolp=aolp_spatial[i],
            index=i,
        )
    
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



def visualize_dolp_aolp_inlays(
    dolp: np.ndarray,
    aolp: np.ndarray,
    index: int,
    background: np.ndarray | None = None,
    arrow_step: int = None,  # Auto-calculated based on image size
    alpha_dolp: float = 0.6
):

    base_dir = "C:/Users/naesl/Polarization-Compass/Bens_Data_Import/Image_data_loaders/Spatial_Gradient_Loader/AOLP_DoLP_inlay_examples"
    dolp_dir = os.path.join(base_dir, "dolp")
    aolp_dir = os.path.join(base_dir, "aolp")
    aolp_hsv_dir = os.path.join(base_dir, "aolp_hsv")
    combined_dir = os.path.join(base_dir, "combined")
    combined_hsv_dir = os.path.join(base_dir, "combined_hsv")
    os.makedirs(base_dir, exist_ok=True)

    for d in [dolp_dir, aolp_dir, aolp_hsv_dir, combined_dir, combined_hsv_dir]:
        os.makedirs(d, exist_ok=True)

    H, W = dolp.shape
    
    # Auto-calculate arrow step based on image size for optimal density
    if arrow_step is None:
        arrow_step = max(4, H // 16)  # Keep ~16 arrows along each dimension
    
    theta = np.deg2rad(aolp)
    u = np.cos(theta)
    v = np.sin(theta)

    y, x = np.mgrid[0:H:arrow_step, 0:W:arrow_step]
    u_sub = u[::arrow_step, ::arrow_step]
    v_sub = v[::arrow_step, ::arrow_step]

    # ---------- DoLP heatmap ----------
    plt.figure(figsize=(8, 7))
    plt.imshow(dolp, cmap="inferno", interpolation='bilinear')
    plt.colorbar(label="DoLP (proxy)", fraction=0.046, pad=0.04)
    plt.title("DoLP Spatial Map", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(dolp_dir, f"dolp_{index:03d}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---------- AoLP quiver plot ----------
    plt.figure(figsize=(8, 7))
    if background is not None:
        plt.imshow(background, cmap="gray")
    else:
        plt.imshow(np.zeros_like(dolp), cmap="gray")

    plt.quiver(
        x, y, u_sub, v_sub,
        color="cyan",
        scale=25,
        width=0.004,
        headwidth=3,
        headlength=4
    )
    plt.title("AoLP Direction Field (Quiver)", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(aolp_dir, f"aolp_quiver_{index:03d}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---------- AoLP HSV color representation ----------
    # Convert angle to HSV color (common in polarimetry)
    # Hue = angle (0-180° mapped to 0-180° in HSV)
    # Saturation = 1 (full color)
    # Value = DoLP (brightness indicates polarization strength)
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Normalize angle to 0-1 for hue (0-180° -> 0-0.5 of hue range)
    hue = aolp / 180.0 * 0.5  # Use half the hue circle for 0-180°
    saturation = np.ones_like(dolp)
    # Boost brightness: set minimum brightness and scale DoLP
    value = 0.3 + 0.7 * dolp  # Brightness ranges from 0.3 to 1.0
    
    # Stack to create HSV image
    hsv = np.stack([hue, saturation, value], axis=-1)
    
    # Convert HSV to RGB for display
    rgb = hsv_to_rgb(hsv)
    
    ax.imshow(rgb, interpolation='bilinear')
    ax.set_title("AoLP as HSV Color (Hue=Angle, Brightness=DoLP)", fontsize=14, fontweight='bold')
    ax.axis("off")
    
    # Create a colorbar showing the angle mapping
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap='hsv', norm=Normalize(vmin=0, vmax=180))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('AoLP Angle (degrees)', rotation=270, labelpad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(aolp_hsv_dir, f"aolp_hsv_{index:03d}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---------- Combined overlay (quiver) ----------
    plt.figure(figsize=(8, 7))
    if background is not None:
        plt.imshow(background, cmap="gray")
    else:
        plt.imshow(np.zeros_like(dolp), cmap="gray")

    plt.imshow(dolp, cmap="inferno", alpha=alpha_dolp, interpolation='bilinear')
    plt.quiver(
        x, y, u_sub, v_sub,
        color="cyan",
        scale=25,
        width=0.004,
        headwidth=3,
        headlength=4,
        alpha=0.8
    )
    plt.title("DoLP (heat) + AoLP (arrows)", fontsize=14, fontweight='bold')
    plt.colorbar(label="DoLP", fraction=0.046, pad=0.04)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(combined_dir, f"overlay_quiver_{index:03d}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ---------- Combined HSV representation ----------
    plt.figure(figsize=(8, 7))
    plt.imshow(rgb, interpolation='bilinear')
    plt.title("Combined: AoLP (Hue) + DoLP (Brightness)", fontsize=14, fontweight='bold')
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(combined_hsv_dir, f"combined_hsv_{index:03d}.png"), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    test_spatial_loader()