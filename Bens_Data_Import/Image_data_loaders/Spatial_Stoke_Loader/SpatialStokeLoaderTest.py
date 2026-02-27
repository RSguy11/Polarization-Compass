import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Bens_Data_Import.Image_data_loaders.Spatial_Stoke_Loader.SpatialStokeLoader import SpatialStokeDataLoader

UNDERWATER_IMAGE = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/new_underwater_test/2026-02-24_12-05-44_burst001_frame010.png")
UNDERWATER_DIR = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/new_underwater_test")


def load_dat_file(path, shape):
    data = np.loadtxt(path)
    expected_size = shape[0] * shape[1]
    if data.size != expected_size:
        raise ValueError(f"Size mismatch: got {data.size} values, expected {expected_size}")
    return data.reshape(shape)


def estimate_aolp_offset_deg(pred_aolp, gt_aolp, valid_mask):
    if np.count_nonzero(valid_mask) == 0:
        return 0.0

    delta_deg = gt_aolp[valid_mask] - pred_aolp[valid_mask]
    delta_rad = np.deg2rad(delta_deg)

    mean_sin = np.mean(np.sin(2.0 * delta_rad))
    mean_cos = np.mean(np.cos(2.0 * delta_rad))
    offset_rad = 0.5 * np.arctan2(mean_sin, mean_cos)
    return float(np.rad2deg(offset_rad))


def visualize_reference_format(s0, s1, s2, dolp, aolp_ipp, aolp_spp, sun_position, dolp_min=0.02):
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    sun_x, sun_y = sun_position

    axes[0, 0].imshow(s0, cmap="gray")
    axes[0, 0].plot(sun_x, sun_y, 'r*', markersize=15, markeredgewidth=2)  # Mark detected sun
    axes[0, 0].set_title(f"S0 (sun @ {sun_x:.0f},{sun_y:.0f})")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(s1, cmap="gray")
    axes[0, 1].set_title("S1")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(s2, cmap="gray")
    axes[0, 2].set_title("S2")
    axes[0, 2].axis("off")

    dolp_plot = axes[1, 0].imshow(dolp, cmap="jet", vmin=0.0, vmax=1.0)
    axes[1, 0].set_title("DoLP")
    axes[1, 0].axis("off")
    fig.colorbar(dolp_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)

    aolp_ipp_masked = np.ma.masked_where(dolp < dolp_min, aolp_ipp)
    axes[1, 1].imshow(aolp_ipp_masked, cmap="jet", vmin=-90, vmax=90)
    axes[1, 1].set_title("AoLP IPP")
    axes[1, 1].axis("off")

    aolp_spp_masked = np.ma.masked_where(dolp < dolp_min, aolp_spp)
    axes[1, 2].imshow(aolp_spp_masked, cmap="jet", vmin=-90, vmax=90)
    axes[1, 2].set_title("AoLP SPP")
    axes[1, 2].axis("off")

    plt.tight_layout()


def circular_smooth_aolp(aolp_deg, dolp, kernel_size=5, dolp_threshold=0.001):
    aolp_rad = np.deg2rad(aolp_deg)
    z = np.exp(2j * aolp_rad)
    weights = np.where(dolp > dolp_threshold, dolp, 0)

    z_real = (z.real * weights).astype(np.float32)
    z_imag = (z.imag * weights).astype(np.float32)
    weights = weights.astype(np.float32)

    sigma = kernel_size / 3.0
    z_real_smooth = gaussian_filter(z_real, sigma=sigma)
    z_imag_smooth = gaussian_filter(z_imag, sigma=sigma)
    weights_smooth = gaussian_filter(weights, sigma=sigma)
    weights_smooth = np.where(weights_smooth > 1e-6, weights_smooth, 1e-6)

    z_real_smooth /= weights_smooth
    z_imag_smooth /= weights_smooth

    z_smooth = z_real_smooth + 1j * z_imag_smooth
    aolp_smooth = np.angle(z_smooth) / 2.0
    return np.rad2deg(aolp_smooth)


def aggressive_circular_smooth(aolp_deg, dolp, kernel_size=21, dolp_threshold=0.01):
    aolp_rad = np.deg2rad(aolp_deg)
    z = np.exp(2j * aolp_rad)
    weights = np.where(dolp > dolp_threshold, dolp**2, 0)

    z_real = (z.real * weights).astype(np.float32)
    z_imag = (z.imag * weights).astype(np.float32)
    weights = weights.astype(np.float32)

    sigma = kernel_size / 2.5
    z_real_smooth = gaussian_filter(z_real, sigma=sigma)
    z_imag_smooth = gaussian_filter(z_imag, sigma=sigma)
    weights_smooth = gaussian_filter(weights, sigma=sigma)

    z_real_smooth = median_filter(z_real_smooth, size=5)
    z_imag_smooth = median_filter(z_imag_smooth, size=5)

    weights_smooth = np.where(weights_smooth > 1e-6, weights_smooth, 1e-6)
    z_real_smooth /= weights_smooth
    z_imag_smooth /= weights_smooth

    z_smooth = z_real_smooth + 1j * z_imag_smooth
    aolp_smooth = np.angle(z_smooth) / 2.0
    return np.rad2deg(aolp_smooth)


def bilateral_smooth_aolp(aolp_deg, dolp, d=15, sigma_color=50, sigma_space=50):
    aolp_rad = np.deg2rad(aolp_deg)
    z = np.exp(2j * aolp_rad)

    z_real = z.real.astype(np.float32)
    z_imag = z.imag.astype(np.float32)

    z_real_smooth = cv2.bilateralFilter(z_real, d, sigma_color, sigma_space)
    z_imag_smooth = cv2.bilateralFilter(z_imag, d, sigma_color, sigma_space)

    weights = np.clip(dolp, 0.001, 1.0)
    z_real_smooth *= weights
    z_imag_smooth *= weights

    z_smooth = z_real_smooth + 1j * z_imag_smooth
    aolp_smooth = np.angle(z_smooth) / 2.0
    return np.rad2deg(aolp_smooth)


def dolp_weighted_filter(aolp_deg, dolp, kernel_size=5, min_dolp=0.01):
    aolp_smooth = circular_smooth_aolp(aolp_deg, dolp, kernel_size=kernel_size)
    dolp_normalized = np.clip(dolp / min_dolp, 0, 1)

    a1 = np.deg2rad(aolp_deg)
    a2 = np.deg2rad(aolp_smooth)

    z1 = np.exp(2j * a1)
    z2 = np.exp(2j * a2)

    z_blend = dolp_normalized[:, :, np.newaxis] * z1[:, :, np.newaxis] + (1 - dolp_normalized[:, :, np.newaxis]) * z2[:, :, np.newaxis]
    z_blend = z_blend.squeeze()

    aolp_blend = np.angle(z_blend) / 2.0
    return np.rad2deg(aolp_blend)


def adaptive_dolp_smooth(dolp, kernel_size=5):
    return gaussian_filter(dolp.astype(np.float32), sigma=kernel_size / 3.0)


def apply_processing_pipeline(aolp_sensor, dolp, kernel_size=7, min_dolp=0.02):
    return {
        "spatial_smooth": circular_smooth_aolp(aolp_sensor, dolp, kernel_size=kernel_size, dolp_threshold=0.001),
        "aggressive_smooth": aggressive_circular_smooth(aolp_sensor, dolp, kernel_size=25, dolp_threshold=0.01),
        "bilateral": bilateral_smooth_aolp(aolp_sensor, dolp, d=15, sigma_color=30, sigma_space=30),
        "dolp_weighted": dolp_weighted_filter(aolp_sensor, dolp, kernel_size=kernel_size, min_dolp=min_dolp),
        "dolp_smooth": adaptive_dolp_smooth(dolp, kernel_size=kernel_size * 2),
    }


def process_image(test_image_path, test_data_dir, image_name="", apply_processing=False):
    print(f"Loading test image: {test_image_path}")
    img = cv2.imread(str(test_image_path), 0)
    if img is None:
        raise ValueError(f"Could not load image from {test_image_path}")

    print(f"Image loaded successfully, shape: {img.shape}")

    loader = SpatialStokeDataLoader(img)
    x_raw, i0, i45, i90, i135, s0 = loader.get_item()
    aolp_raw_sensor = x_raw["aolp"]
    dolp_raw = x_raw["dolp"]

    # Sun position is detected automatically by loader
    sun_x, sun_y = loader.sun_position
    print(f"Detected sun position: x={sun_x:.1f}, y={sun_y:.1f}")

    aop_gt_file = test_data_dir / "aop_global_frame.dat"
    dop_gt_file = test_data_dir / "dop.dat"

    if aop_gt_file.exists() and dop_gt_file.exists():
        aop_gt = load_dat_file(aop_gt_file, (1024, 1224))
        dop_gt = load_dat_file(dop_gt_file, (1024, 1224))
        extracted_shape = dolp_raw.shape
        if aop_gt.shape != extracted_shape:
            aop_gt = cv2.resize(aop_gt.astype(np.float32), (extracted_shape[1], extracted_shape[0]), interpolation=cv2.INTER_LINEAR)
        if dop_gt.shape != extracted_shape:
            dop_gt = cv2.resize(dop_gt.astype(np.float32), (extracted_shape[1], extracted_shape[0]), interpolation=cv2.INTER_LINEAR)

        valid_mask = (dolp_raw > 0.1) & (dop_gt > 0.1)
        if np.count_nonzero(valid_mask) > 1000:
            x_raw_global_initial, _, _, _, _, _ = loader.get_item(global_frame_offset=0.0)
            initial_global_aolp = x_raw_global_initial["aolp"]
            global_offset = estimate_aolp_offset_deg(initial_global_aolp, aop_gt, valid_mask)
            print(f"Estimated global AoLP offset (deg): {global_offset:.2f}")
        else:
            global_offset = 0.0
    else:
        global_offset = 0.0

    x_raw_global, _, _, _, _, _ = loader.get_item(global_frame_offset=global_offset)
    aolp_raw_global = x_raw_global["aolp"]

    dolp_stats = (
        float(np.min(dolp_raw)),
        float(np.percentile(dolp_raw, 5)),
        float(np.percentile(dolp_raw, 50)),
        float(np.percentile(dolp_raw, 95)),
        float(np.max(dolp_raw)),
    )
    print(f"DoLP stats [min, p5, p50, p95, max]: {dolp_stats}")

    s1 = i90 - i0
    s2 = i45 - i135

    visualize_reference_format(s0, s1, s2, dolp_raw, aolp_raw_sensor, aolp_raw_global, loader.sun_position, dolp_min=0.001)

    _ = (image_name, apply_processing)


def main():
    plt.ion()

    print("=" * 80)
    print("PROCESSING UNDERWATER TEST IMAGE")
    print("=" * 80)
    if UNDERWATER_IMAGE.exists():
        process_image(UNDERWATER_IMAGE, UNDERWATER_DIR, image_name="UNDERWATER", apply_processing=True)
    else:
        print("Underwater test image not found.")

    plt.show(block=True)


if __name__ == "__main__":
    main()

