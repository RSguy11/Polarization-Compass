from cv2 import mean, sqrt
import numpy as np
import sys
from pathlib import Path
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import cv2
import polanalyser as pa

# Add workspace root to path (3 levels up from current file)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Bens_Data_Import.Image_data_loaders.Spatial_Stoke_Loader.SpatialStokeLoader import SpatialStokeDataLoader

# Path to the test image
TEST_IMAGE_PATH = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/group48_test/2024-10-08-19-31-33_angle_0.png")
TEST_DATA_DIR = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/group48_test")

def main():
    # Load the test image
    print(f"Loading test image: {TEST_IMAGE_PATH}")
    img = cv2.imread(str(TEST_IMAGE_PATH), 0)
    
    if img is None:
        raise ValueError(f"Could not load image from {TEST_IMAGE_PATH}")
    
    print(f"Image loaded successfully, shape: {img.shape}")
    
    # Create loader with the image
    loader = SpatialStokeDataLoader(img)
    
    # --- 1. Extract raw (both sensor and global frames) ---
    # Sensor frame - no offset (instrument coordinates)
    x_raw, I0, I45, I90, I135, S0 = loader.get_item()
    aolp_raw_sensor = x_raw["aolp"]  # Sensor frame
    dolp_raw = x_raw["dolp"]
    
    # Global frame - apply offset to align with ground truth
    # You can adjust this offset based on your camera orientation
    # Set to 0° for no transformation (same as sensor frame)
    global_offset = 0.0  # Degrees - adjust this based on camera orientation relative to true North
    x_raw_global, _, _, _, _, _ = loader.get_item(global_frame_offset=global_offset)
    aolp_raw_global = x_raw_global["aolp"]  # Global frame

    print(f"Extracted features - AoLP shape: {aolp_raw_sensor.shape}, DoLP shape: {dolp_raw.shape}")

    # --- 2. Load GT and Calculate Global Frame Offset ---
    aop_gt = load_dat_file(TEST_DATA_DIR / "aop_global_frame.dat", (1024, 1224))
    dop_gt = load_dat_file(TEST_DATA_DIR / "dop.dat", (1024, 1224))
    
    # Downsample sensor frame to match ground truth resolution for offset calculation
    aolp_sensor_downsampled = aolp_raw_sensor[::2, ::2]  # Simple downsampling
    dolp_downsampled = dolp_raw[::2, ::2]
    
    # Calculate global offset from reliable pixels
    dolp_min = 0.1  # Use higher threshold for offset calculation
    valid_mask = (dolp_downsampled > dolp_min) & (dop_gt > dolp_min)
    
    if np.count_nonzero(valid_mask) > 1000:
        # For solar principal plane transformation, we just need to enable it
        # The offset calculation is not used for simple angle shifting anymore
        global_offset = 1.0  # Any non-None value enables the transformation
        print(f"\nEnabling solar principal plane transformation")
        print(f"Using {np.count_nonzero(valid_mask)} valid pixels for validation")
    else:
        global_offset = None
        print(f"\nNot enough valid pixels, disabling transformation")
    
    # Now extract with calculated offset
    x_raw_global, _, _, _, _, _ = loader.get_item(global_frame_offset=global_offset)
    aolp_raw_global = x_raw_global["aolp"]  # Global frame

    print(f"Using global frame offset: {global_offset}°")


    # --- 3. Calibrate gains (COMMENTED OUT - requires GT data) ---
    # gains, res = calibrate_gains_from_gt(
    #     I0, I45, I90, I135,
    #     aop_gt, dop_gt,
    #     dolp_min=0.02,
    #     s0_min=5.0,
    #     sample=150000,
    #     use_offsets=False
    # )
    # print("Estimated gains:", gains)
    
    # For visualization only, skip calibration
    gains = None

    # --- 4. Re-extract WITH calibration (both sensor and global frames) ---
    # Create new loader instance with same image for calibrated extraction
    loader_cal = SpatialStokeDataLoader(img)
    
    # Sensor frame - no offset
    x_cal, *_ = loader_cal.get_item(gains=gains)
    aolp_cal_sensor = x_cal["aolp"]  # Sensor frame
    dolp_cal = x_cal["dolp"]
    
    # Global frame - with offset
    x_cal_global, *_ = loader_cal.get_item(gains=gains, global_frame_offset=global_offset)
    aolp_cal_global = x_cal_global["aolp"]  # Global frame


      # --- 5. Visualizations only ---
    print("\nRAW (uncalibrated) visualization:")
    visulization_all(aolp_raw_sensor, aolp_raw_global, dolp_raw, title_prefix="RAW")

    print("\nWith global frame offset visualization:")
    visulization_all(aolp_cal_sensor, aolp_cal_global, dolp_cal, title_prefix="GLOBAL OFFSET")

    # --- 6. Error calculations (COMMENTED OUT - requires GT data) ---
    # print("RAW errors:")
    # print(error_calcs(aolp_raw_sensor, aop_gt, dolp_raw, dop_gt))
    # 
    # print("CALIBRATED errors:")
    # print(error_calcs(aolp_cal_sensor, aop_gt, dolp_cal, dop_gt))
    # 
    # # --- 7. Normalized Stokes (double-angle) alignment ---
    # dolp_min = 0.02
    # valid = (dolp_cal > dolp_min) & (dop_gt > dolp_min)
    # 
    # a = np.deg2rad(aolp_cal_sensor[valid])
    # b = np.deg2rad(aop_gt[valid])
    # 
    # score = np.mean(np.cos(2.0 * (a - b)))
    # print("Double-angle similarity (calibrated):", score)
    # 
    # a_raw = np.deg2rad(aolp_raw_sensor[valid])
    # score_raw = np.mean(np.cos(2.0 * (a_raw - b)))
    # 
    # print("Double-angle similarity (raw):", score_raw)


def aop_err_deg(a, b):
    d = a - b
    return (d + 90) % 180 - 90

def score(name, A, B):
    err = aop_err_deg(A, B)
    mae = np.mean(np.abs(err))
    corr = np.corrcoef(A.ravel(), B.ravel())[0, 1]
    return name, mae, corr


def load_dat_file(path, shape):
    data = np.loadtxt(path)  # 1D array of floats

    expected_size = shape[0] * shape[1]
    if data.size != expected_size:
        raise ValueError(
            f"Size mismatch: got {data.size} values, expected {expected_size}"
        )

    return data.reshape(shape)

def error_calcs(aop_extract, aop_truth, dop_extract, dop_truth, dolp_min = 0.02):

    dop_err = dop_extract - dop_truth

    dop_err_metrics = {
        "MAE": np.mean(np.abs(dop_err)), 
        "RMSE": np.sqrt(np.mean(dop_err**2)), 
        "Max": np.max(np.abs(dop_err))
    }

    valid = (dop_extract > dolp_min) & (dop_truth > dolp_min)

    if np.count_nonzero(valid) == 0:
        raise ValueError("No valid pixels for AoLP error after DoLP masking")

    delta = aop_extract - aop_truth
    aop_err = (delta + 90) % 180 - 90

    aop_err_valid = aop_err[valid]



    print("Valid AoLP pixels:", np.count_nonzero(valid), "/", valid.size)

    aop_err_metrics = {
    "MAE":  np.mean(np.abs(aop_err_valid)),
    "RMSE": np.sqrt(np.mean(aop_err_valid**2)),
    "Max":  np.max(np.abs(aop_err_valid)),
    }

    aop_extract_double = 2.0 * aop_extract
    aop_extract_double = (aop_extract_double + 180) % 360 - 180

    delta = aop_extract_double - aop_truth
    aop_err = (delta + 180) % 360 - 180
    print(np.mean(np.abs(aop_err)))
    print("testing half", np.mean(np.abs(aop_err)))

    return aop_err_metrics, dop_err_metrics


def visulization_all(AoLP_sensor, AoLP_global, DoLP, title_prefix="", dolp_min=0.02):
    """
    Visualize DoLP, Sensor AoLP, and Global AoLP as separate figures.
    
    Parameters:
    -----------
    AoLP_sensor : np.ndarray
        AoLP in sensor/instrument frame
    AoLP_global : np.ndarray
        AoLP in global frame
    DoLP : np.ndarray
        Degree of Linear Polarization
    title_prefix : str
        Prefix for titles (e.g., "RAW" or "CALIBRATED")
    dolp_min : float
        Minimum DoLP threshold for masking
    """
    # -----------------------------
    # DoLP visualization - Rainbow (Blue to Red)
    # -----------------------------
    plt.figure(figsize=(6, 5))
    im0 = plt.imshow(DoLP, cmap="jet", vmin=0.0, vmax=1.0)
    plt.title(f"{title_prefix} DoLP")
    plt.axis("off")
    plt.colorbar(im0, label="DoLP")

    # -----------------------------
    # Sensor AoLP visualization
    # Mask low-DoLP pixels
    # -----------------------------
    plt.figure(figsize=(6, 5))
    AoLP_sensor_masked = np.ma.masked_where(DoLP < dolp_min, AoLP_sensor)
    im1 = plt.imshow(AoLP_sensor_masked, cmap="jet_r", vmin=-90, vmax=90)
    plt.title(f"{title_prefix} AoLP (Sensor Frame)")
    plt.axis("off")
    plt.colorbar(im1, label="Angle (deg)")

    # -----------------------------
    # Global AoLP visualization
    # Mask low-DoLP pixels
    # -----------------------------
    plt.figure(figsize=(6, 5))
    AoLP_global_masked = np.ma.masked_where(DoLP < dolp_min, AoLP_global)
    im2 = plt.imshow(AoLP_global_masked, cmap="jet_r", vmin=-90, vmax=90)
    plt.title(f"{title_prefix} AoLP (Global Frame)")
    plt.axis("off")
    plt.colorbar(im2, label="Angle (deg)")

    plt.show()


def visulization(AoLP, DoLP):
    dolp_min = 0.02
    
    # -----------------------------
    # DoLP visualization - Rainbow (Blue to Red)
    # -----------------------------
    plt.figure(figsize=(6, 5))
    plt.imshow(DoLP, cmap="jet", vmin=0.0, vmax=1.0)
    plt.title("DoLP (Degree of Linear Polarization)")
    plt.axis("off")
    plt.colorbar(label="DoLP")

    # -----------------------------
    # AoLP visualization - Rainbow for angles
    # Mask low-DoLP pixels (AoLP meaningless there)
    # -----------------------------
    AoLP_masked = np.ma.masked_where(DoLP < dolp_min, AoLP)

    plt.figure(figsize=(6, 5))
    plt.imshow(AoLP_masked, cmap="jet_r", vmin=-90, vmax=90)
    plt.title("AoLP (Angle of Linear Polarization)")
    plt.axis("off")
    plt.colorbar(label="Angle (deg)")

    plt.show()


def calibrate_gains_from_gt(I0, I45, I90, I135, AoP_gt_deg, DoP_gt,
                            dolp_min=0.02, s0_min=5.0, sample=200000,
                            use_offsets=False):
    """
    Returns gains (and optionally offsets) that best match normalized Stokes (q,u)
    implied by GT AoP/DoP.

    I* are float32 arrays shape (H,W) from the mosaic tiles.
    AoP_gt_deg is GT AoLP in degrees (same shape).
    DoP_gt is GT DoLP (same shape).
    """

    # --- build GT targets in normalized Stokes space ---
    a = np.deg2rad(AoP_gt_deg)
    q_gt = DoP_gt * np.cos(2.0 * a)
    u_gt = DoP_gt * np.sin(2.0 * a)

    # --- validity mask: AoP only meaningful where DoLP & intensity are reasonable ---
    S0_raw = I0 + I90
    valid = np.isfinite(q_gt) & np.isfinite(u_gt) & (DoP_gt > dolp_min) & (S0_raw > s0_min)

    idx = np.flatnonzero(valid.ravel())
    if idx.size == 0:
        raise ValueError("No valid pixels after masking. Lower thresholds or check GT.")

    # --- subsample for speed ---
    if sample is not None and idx.size > sample:
        idx = np.random.choice(idx, size=sample, replace=False)

    # gather samples
    I0v   = I0.ravel()[idx]
    I45v  = I45.ravel()[idx]
    I90v  = I90.ravel()[idx]
    I135v = I135.ravel()[idx]
    qv    = q_gt.ravel()[idx]
    uv    = u_gt.ravel()[idx]

    eps = 1e-6

    if not use_offsets:
        # params: [g45, g90, g135], with g0 fixed = 1
        def residuals(p):
            g45, g90, g135 = p
            g0 = 1.0

            I0p   = g0   * I0v
            I45p  = g45  * I45v
            I90p  = g90  * I90v
            I135p = g135 * I135v

            S0 = I0p + I90p
            S1 = I0p - I90p
            S2 = I135p - I45p

            denom = np.clip(S0, eps, None)
            q_est = S1 / denom
            u_est = S2 / denom

            return np.concatenate([q_est - qv, u_est - uv])

        x0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        res = least_squares(residuals, x0, loss="soft_l1", f_scale=0.05)
        g45, g90, g135 = res.x
        gains = dict(g0=1.0, g45=float(g45), g90=float(g90), g135=float(g135))
        return gains, res

    else:
        # params: [g45,g90,g135, b0,b45,b90,b135], with g0 fixed=1
        def residuals(p):
            g45, g90, g135, b0, b45, b90, b135 = p
            g0 = 1.0

            I0p   = g0   * (I0v   - b0)
            I45p  = g45  * (I45v  - b45)
            I90p  = g90  * (I90v  - b90)
            I135p = g135 * (I135v - b135)

            S0 = I0p + I90p
            S1 = I0p - I90p
            S2 = I135p - I45p

            denom = np.clip(S0, eps, None)
            q_est = S1 / denom
            u_est = S2 / denom

            return np.concatenate([q_est - qv, u_est - uv])

        x0 = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        res = least_squares(residuals, x0, loss="soft_l1", f_scale=0.05)
        g45, g90, g135, b0, b45, b90, b135 = res.x
        params = dict(
            g0=1.0, g45=float(g45), g90=float(g90), g135=float(g135),
            b0=float(b0), b45=float(b45), b90=float(b90), b135=float(b135)
        )
        return params, res



if __name__ == "__main__":
    main()

