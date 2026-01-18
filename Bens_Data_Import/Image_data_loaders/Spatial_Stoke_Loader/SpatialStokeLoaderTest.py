from cv2 import mean, sqrt
import numpy as np
import sys
from pathlib import Path
from matplotlib.colors import hsv_to_rgb
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Add workspace root to path (3 levels up from current file)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Bens_Data_Import.Image_data_loaders.Spatial_Stoke_Loader.SpatialStokeLoader import SpatialStokeDataLoader

# Path to the PNG dataset (Ben's Data folder)
TEST_IMAGE = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/group48_test")

def main():
    loader = SpatialStokeDataLoader(TEST_IMAGE)
        # --- 1. Extract raw ---
    x_raw, I0, I45, I90, I135, S0 = loader.get_item()
    aolp_raw = x_raw["aolp"]
    dolp_raw = x_raw["dolp"]

    # --- 2. Load GT ---
    aop_gt = load_dat_file(TEST_IMAGE / "aop_global_frame.dat", (1024,1224))
    dop_gt = load_dat_file(TEST_IMAGE / "dop.dat", (1024,1224))


    # --- 3. Calibrate gains ---
    gains, res = calibrate_gains_from_gt(
        I0, I45, I90, I135,
        aop_gt, dop_gt,
        dolp_min=0.02,
        s0_min=5.0,
        sample=150000,
        use_offsets=False
    )

    print("Estimated gains:", gains)

    # --- 4. Re-extract WITH calibration ---
    x_cal, *_ = loader.get_item(gains=gains)
    aolp_cal = x_cal["aolp"]
    dolp_cal = x_cal["dolp"]


      # --- 5. Compare errors ---
    print("RAW errors:")
    print(error_calcs(aolp_raw, aop_gt, dolp_raw, dop_gt))

    visulization(aolp_raw, dolp_raw)

    print("CALIBRATED errors:")
    print(error_calcs(aolp_cal, aop_gt, dolp_cal, dop_gt))

    visulization(aolp_cal, dolp_cal)


    # --- 6. Normalized Stokes (double-angle) alignment ---
    dolp_min = 0.02
    valid = (dolp_cal > dolp_min) & (dop_gt > dolp_min)

    a = np.deg2rad(aolp_cal[valid])
    b = np.deg2rad(aop_gt[valid])

    score = np.mean(np.cos(2.0 * (a - b)))
    print("Double-angle similarity (calibrated):", score)

    a_raw = np.deg2rad(aolp_raw[valid])
    score_raw = np.mean(np.cos(2.0 * (a_raw - b)))

    print("Double-angle similarity (raw):", score_raw)


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


def visulization(AoLP, DoLP):
    dolp_min = 0.02
    # -----------------------------
    # DoLP visualization (scalar)
    # -----------------------------
    plt.figure(figsize=(5, 5))
    plt.imshow(DoLP, cmap="viridis", vmin=0.0, vmax=1.0)
    plt.title("DoLP")
    plt.axis("off")
    plt.colorbar(label="DoLP")

    # -----------------------------
    # AoLP visualization (angle)
    # Mask low-DoLP pixels (AoLP meaningless there)
    # -----------------------------
    AoLP_masked = np.ma.masked_where(DoLP < dolp_min, AoLP)

    plt.figure(figsize=(5, 5))
    plt.imshow(AoLP_masked, cmap="hsv", vmin=-90, vmax=90)
    plt.title("AoLP (deg)")
    plt.axis("off")
    plt.colorbar(label="deg")

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

