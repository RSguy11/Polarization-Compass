from cv2 import mean, sqrt
import numpy as np
import sys
from pathlib import Path

# Add workspace root to path (3 levels up from current file)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Bens_Data_Import.Image_data_loaders.Spatial_Stoke_Loader.SpatialStokeLoader import SpatialStokeDataLoader

# Path to the PNG dataset (Ben's Data folder)
TEST_IMAGE = Path("C:/Users/naesl/Polarization-Compass/Bens_Data_Import/group48_test")

def main():
    loader = SpatialStokeDataLoader(TEST_IMAGE)
    x = loader.get_item()

    aolp_extracts = x["aolp"]
    dolp_extracts = x["dolp"]

    print("Aolp Extracted" ,aolp_extracts.min(), aolp_extracts.max())
    print("Dolp Extracted", dolp_extracts.min(), dolp_extracts.max())

    # print("aolp", x["aolp"])
    # print("dolp", x["dolp"])

    aop_ground_truth = load_dat_file(TEST_IMAGE / "aop_sensor_frame.dat", (1024,1224))
    dop_ground_truth = load_dat_file(TEST_IMAGE / "dop.dat", (1024,1224))

    print("AoP ref:", aop_ground_truth.shape, aop_ground_truth.min(), aop_ground_truth.max())
    print("DoP ref:", dop_ground_truth.shape, dop_ground_truth.min(), dop_ground_truth.max())

    aop_errs, dop_errs = error_calcs(aolp_extracts, aop_ground_truth, dolp_extracts, dop_ground_truth)

    print("aop_errs", aop_errs)
    print("dop_errs", dop_errs)

def load_dat_file(path, shape):
    data = np.loadtxt(path)  # 1D array of floats

    expected_size = shape[0] * shape[1]
    if data.size != expected_size:
        raise ValueError(
            f"Size mismatch: got {data.size} values, expected {expected_size}"
        )

    return data.reshape(shape)

def error_calcs(aop_extract, aop_truth, dop_extract, dop_truth):

    dop_err = dop_extract - dop_truth

    delta = aop_extract - aop_truth
    aop_err = (delta + 90) % 180 - 90

    dop_err_metrics = {
        "MAE": np.mean(np.abs(dop_err)), 
        "RMSE": np.sqrt(np.mean(dop_err**2)), 
        "Max": np.max(np.abs(dop_err))
    }

    aop_err_metrics = {
    "MAE":  np.mean(np.abs(aop_err)),
    "RMSE": np.sqrt(np.mean(aop_err**2)),
    "Max":  np.max(np.abs(aop_err)),
    }

    return aop_err_metrics, dop_err_metrics

if __name__ == "__main__":
    main()

