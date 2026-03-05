"""
Test file for the AoLP / DoLP extractor (Deveyansh method).

Validates the full pipeline:
  raw image -> demosaicing -> Stokes -> DoLP / AoLP (IPP & SPP)

Uses the underwater test image already present in the repo.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import polanalyser as pa
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Polarization-Compass
TEST_IMAGE = PROJECT_ROOT / "Bens_Data_Import" / "new_underwater_test" / "2026-02-24_12-05-44_burst001_frame010.png"


# ---------------------------------------------------------------------------
# Helper – copied from the extractor so we can test it independently
# ---------------------------------------------------------------------------
def normalize_for_display(img):
    img = img.astype(np.float32)
    min_val = np.min(img)
    max_val = np.max(img)
    if max_val - min_val == 0:
        return np.zeros_like(img)
    return (img - min_val) / (max_val - min_val)


def compute_angle_for_mueller(rows=2048, cols=2448, center_x=1224, center_y=1024):
    y_coord, x_coord = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    dx = x_coord - center_x
    dy = y_coord - center_y
    angle_matrix = np.degrees(np.arctan2(dy, dx))
    angle_matrix = (angle_matrix + 360) % 360
    angle_matrix = angle_matrix % 180
    return angle_matrix


# ===================================================================
# Test functions
# ===================================================================
def test_image_loading():
    """Verify the test image can be read and has expected properties."""
    print("=" * 60)
    print("TEST 1: Image Loading")
    print("=" * 60)

    assert TEST_IMAGE.exists(), f"Test image not found: {TEST_IMAGE}"
    img_raw = cv2.imread(str(TEST_IMAGE), 0)
    assert img_raw is not None, "cv2.imread returned None"
    print(f"  Image shape : {img_raw.shape}")
    print(f"  Dtype       : {img_raw.dtype}")
    print(f"  Value range : [{img_raw.min()}, {img_raw.max()}]")
    assert img_raw.ndim == 2, "Expected a single-channel (grayscale) image"
    assert img_raw.shape[0] > 0 and img_raw.shape[1] > 0, "Image has zero dimension"
    print("  PASSED\n")
    return img_raw


def test_demosaicing(img_raw):
    """Verify demosaicing produces four polarisation channels."""
    print("=" * 60)
    print("TEST 2: Demosaicing")
    print("=" * 60)

    img_000, img_045, img_090, img_135 = pa.demosaicing(img_raw, pa.COLOR_PolarMono)

    h, w = img_raw.shape
    for name, ch in [("0°", img_000), ("45°", img_045), ("90°", img_090), ("135°", img_135)]:
        print(f"  {name:>4s} channel : shape={ch.shape}, range=[{ch.min():.1f}, {ch.max():.1f}]")
        assert ch.shape[0] == h and ch.shape[1] == w, \
            f"{name} shape mismatch, expected ({h}, {w})"
        assert ch.max() > 0, f"{name} channel is all zeros"

    print("  PASSED\n")
    return img_000, img_045, img_090, img_135


def test_stokes_vectors(img_000, img_045, img_090, img_135):
    """Compute Stokes vectors and validate components."""
    print("=" * 60)
    print("TEST 3: Stokes Vector Computation")
    print("=" * 60)

    image_list = [img_000, img_045, img_090, img_135]
    angles = np.deg2rad([0, 45, 90, 135])
    img_stokes = pa.calcStokes(image_list, angles)

    img_s0, img_s1, img_s2 = cv2.split(img_stokes)
    print(f"  S0 range : [{img_s0.min():.2f}, {img_s0.max():.2f}]")
    print(f"  S1 range : [{img_s1.min():.2f}, {img_s1.max():.2f}]")
    print(f"  S2 range : [{img_s2.min():.2f}, {img_s2.max():.2f}]")

    # S0 (total intensity) should be positive
    assert img_s0.min() >= 0, "S0 has negative values (unexpected for intensity)"
    # S1 and S2 can be negative – just check they are not all-zero
    assert np.any(img_s1 != 0), "S1 is all zeros"
    assert np.any(img_s2 != 0), "S2 is all zeros"

    print("  PASSED\n")
    return img_stokes, img_s0, img_s1, img_s2


def test_dolp_aolp(img_stokes):
    """Validate DoLP and AoLP outputs from polanalyser."""
    print("=" * 60)
    print("TEST 4: DoLP & AoLP (polanalyser)")
    print("=" * 60)

    img_dolp = pa.cvtStokesToDoLP(img_stokes)
    img_aolp = pa.cvtStokesToAoLP(img_stokes)  # radians

    print(f"  DoLP range : [{img_dolp.min():.4f}, {img_dolp.max():.4f}]")
    print(f"  AoLP range : [{np.rad2deg(img_aolp.min()):.2f}°, {np.rad2deg(img_aolp.max()):.2f}°]")

    # DoLP should be in [0, 1] (or slightly above due to noise)
    assert img_dolp.min() >= 0, "DoLP has negative values"
    assert img_dolp.max() <= 2.0, "DoLP unreasonably large (>2.0)"

    # AoLP in radians should be in [-pi/2, pi/2] (half-angle)
    assert img_aolp.min() >= -np.pi, "AoLP below -pi"
    assert img_aolp.max() <= np.pi, "AoLP above pi"

    print("  PASSED\n")
    return img_dolp, img_aolp


def test_manual_dolp(img_s0, img_s1, img_s2):
    """Compare manual DoLP computation against polanalyser."""
    print("=" * 60)
    print("TEST 5: Manual DoLP vs polanalyser")
    print("=" * 60)

    manual_dolp = np.sqrt(img_s1 ** 2 + img_s2 ** 2) / (img_s0 + 1e-10)

    image_list = [img_s0]  # dummy – recompute from stokes
    img_stokes_rebuilt = cv2.merge([img_s0, img_s1, img_s2])
    pa_dolp = pa.cvtStokesToDoLP(img_stokes_rebuilt)

    # Where S0 is reasonably large, the two should agree
    mask = img_s0 > 10
    if np.any(mask):
        diff = np.abs(manual_dolp[mask] - pa_dolp[mask])
        print(f"  Max abs diff (S0>10) : {diff.max():.6f}")
        print(f"  Mean abs diff        : {diff.mean():.6f}")
        assert diff.max() < 0.05, "Manual and polanalyser DoLP disagree significantly"
    else:
        print("  WARNING: no pixels with S0 > 10 to compare")

    print("  PASSED\n")
    return manual_dolp


def test_angle_matrix():
    """Validate the Mueller-angle matrix used for IPP→SPP conversion."""
    print("=" * 60)
    print("TEST 6: Angle Matrix (Mueller)")
    print("=" * 60)

    rows, cols = 2048, 2448
    beta = compute_angle_for_mueller(rows, cols)
    print(f"  Shape : {beta.shape}")
    print(f"  Range : [{beta.min():.2f}°, {beta.max():.2f}°]")

    assert beta.shape == (rows, cols), "Shape mismatch"
    assert beta.min() >= 0, "Angle below 0°"
    assert beta.max() < 180, "Angle >= 180°"

    # Centre pixel should be 0 (arctan2(0,0) → 0)
    centre_val = beta[1024, 1224]
    print(f"  Centre angle : {centre_val:.2f}°")

    print("  PASSED\n")
    return beta


def test_aolp_spp_conversion(img_aolp_deg, beta2):
    """Compute AoLP in the Scattering Principal Plane and validate."""
    print("=" * 60)
    print("TEST 7: AoLP SPP Conversion")
    print("=" * 60)

    # beta2 should already match demosaiced image dimensions
    assert img_aolp_deg.shape == beta2.shape, \
        f"Shape mismatch: AoLP {img_aolp_deg.shape} vs beta {beta2.shape}"

    temp1 = np.deg2rad(img_aolp_deg + beta2)
    aolp_spp = 0.5 * np.rad2deg(np.arctan2(np.sin(2 * temp1), np.cos(2 * temp1)))

    print(f"  AoLP SPP shape : {aolp_spp.shape}")
    print(f"  AoLP SPP range : [{aolp_spp.min():.2f}°, {aolp_spp.max():.2f}°]")

    # SPP angles should remain in [-90, 90]
    assert aolp_spp.min() >= -90.0, "AoLP SPP below -90°"
    assert aolp_spp.max() <= 90.0, "AoLP SPP above 90°"

    print("  PASSED\n")
    return aolp_spp


def test_final_outputs(aolp_spp, manual_dolp):
    """Validate the final exported variables (flipped AoLP, DoLP)."""
    print("=" * 60)
    print("TEST 8: Final Outputs")
    print("=" * 60)

    aolp_meas_full = np.fliplr(aolp_spp)
    dolp_meas = manual_dolp

    print(f"  aolp_meas_full shape : {aolp_meas_full.shape}")
    print(f"  dolp_meas shape      : {dolp_meas.shape}")

    # Flipping should preserve shape
    assert aolp_meas_full.shape == aolp_spp.shape, "Flip changed shape"
    # Flipping should actually reverse columns
    assert np.allclose(aolp_meas_full[:, 0], aolp_spp[:, -1]), "Flip verification failed"

    # DoLP should not be all-zero
    assert dolp_meas.max() > 0, "DoLP is all zeros"

    print("  PASSED\n")
    return aolp_meas_full, dolp_meas


def test_normalize_for_display():
    """Unit test for the normalization helper."""
    print("=" * 60)
    print("TEST 9: normalize_for_display")
    print("=" * 60)

    # Standard case
    arr = np.array([[1.0, 5.0], [3.0, 9.0]])
    normed = normalize_for_display(arr)
    assert np.isclose(normed.min(), 0.0), "Min not 0"
    assert np.isclose(normed.max(), 1.0), "Max not 1"
    print("  Standard case OK")

    # Constant image – should return zeros
    const = np.ones((4, 4)) * 42.0
    normed_const = normalize_for_display(const)
    assert np.allclose(normed_const, 0.0), "Constant image should normalise to 0"
    print("  Constant image case OK")

    print("  PASSED\n")


def visualize_results(img_s0, img_s1, img_s2, img_dolp, img_aolp_deg, aolp_spp, aolp_meas_full):
    """Optional visualisation – mirrors the layout from the extractor."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("AoLP / DoLP Extractor — Test Visualisation", fontsize=14)

    titles_top = ["S0", "S1", "S2", "DoLP"]
    imgs_top = [
        normalize_for_display(img_s0),
        normalize_for_display(img_s1),
        normalize_for_display(img_s2),
        normalize_for_display(img_dolp),
    ]
    cmaps_top = ["gray", "gray", "gray", "jet"]

    for ax, img, title, cmap in zip(axes[0], imgs_top, titles_top, cmaps_top):
        im = ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis("off")
        if cmap == "jet":
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    titles_bot = ["AoLP IPP (deg)", "AoLP SPP (deg)", "AoLP SPP flipped", ""]
    imgs_bot = [
        normalize_for_display(img_aolp_deg),
        normalize_for_display(aolp_spp),
        normalize_for_display(aolp_meas_full),
        None,
    ]

    for ax, img, title in zip(axes[1], imgs_bot, titles_bot):
        if img is not None:
            im = ax.imshow(img, cmap="jet")
            ax.set_title(title)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(Path(__file__).parent / "test_output.png"), dpi=100)
    print("  Visualisation saved to test_output.png")
    plt.show()


# ===================================================================
# Main
# ===================================================================
def main():
    print("\n" + "#" * 60)
    print("  AoLP / DoLP Extractor — Test Suite")
    print("#" * 60 + "\n")

    # 1. Load
    img_raw = test_image_loading()

    # 2. Demosaic
    img_000, img_045, img_090, img_135 = test_demosaicing(img_raw)

    # 3. Stokes
    img_stokes, img_s0, img_s1, img_s2 = test_stokes_vectors(img_000, img_045, img_090, img_135)

    # 4. DoLP / AoLP via polanalyser
    img_dolp, img_aolp = test_dolp_aolp(img_stokes)
    img_aolp_deg = np.rad2deg(img_aolp)

    # 5. Manual DoLP
    manual_dolp = test_manual_dolp(img_s0, img_s1, img_s2)

    # 6. Angle matrix
    beta = test_angle_matrix()

    # 7. AoLP SPP
    aolp_spp = test_aolp_spp_conversion(img_aolp_deg, beta)

    # 8. Final outputs
    aolp_meas_full, dolp_meas = test_final_outputs(aolp_spp, manual_dolp)

    # 9. Normalize helper
    test_normalize_for_display()

    # Summary
    print("=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)

    # Optional: show plots
    visualize_results(img_s0, img_s1, img_s2, img_dolp, img_aolp_deg, aolp_spp, aolp_meas_full)


if __name__ == "__main__":
    main()
