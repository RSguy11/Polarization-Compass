import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from PIL import Image
from typing import List, Dict, Union, Literal


def _stack_polar_channels(
    I0: np.ndarray,
    I45: np.ndarray,
    I90: np.ndarray,
    I135: np.ndarray,
    file_path: str,
    noise_level: str,
) -> List[Dict]:

    #Each sample is a dict:
        #{
            #"raw":   np.ndarray,  # (4, H, W) in order [I0, I45, I90, I135]
            #"label": None,        # placeholder for azimuth labels if you add them later
            #"source": "mock",
            #"meta":  {...}
        #}
        
    samples: List[Dict] = []

    # Case 1: single frame per channel: (H, W)
    if I0.ndim == 2:
        H, W = I0.shape
        raw = np.stack([I0, I45, I90, I135], axis=0)  # (4, H, W)

        samples.append(
            {
                "raw": raw.astype(np.float32),
                "label": None,
                "source": "mock",
                "meta": {
                    "file": file_path,
                    "noise_level": noise_level,
                    "index": 0,
                },
            }
        )

    # Case 2: multiple frames per channel: (H, W, N)
    elif I0.ndim == 3:
        H, W, N = I0.shape
        assert (
            I45.shape == (H, W, N)
            and I90.shape == (H, W, N)
            and I135.shape == (H, W, N)
        ), "All polarization channels must have the same shape"

        for idx in range(N):
            raw = np.stack(
                [
                    I0[:, :, idx],
                    I45[:, :, idx],
                    I90[:, :, idx],
                    I135[:, :, idx],
                ],
                axis=0,  # (4, H, W)
            )

            samples.append(
                {
                    "raw": raw.astype(np.float32),
                    "label": None,
                    "source": "mock",
                    "meta": {
                        "file": file_path,
                        "noise_level": noise_level,
                        "index": idx,
                    },
                }
            )
    else:
        raise ValueError(
            f"Unexpected dimensionality for polarization channels: {I0.shape}"
        )

    return samples

def load_mat_file(noise_level="Low", file_path="Preprocessing/Mock_data/mosaic_test_preprocessing_database/ImageMat"):
    print("in load_mat_file function")
    data_path = os.path.join(file_path, noise_level)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"The specified file path does not exist: {file_path}")
    
    scenes:List[Dict] = []
    
    for file in os.listdir(data_path):
        if not file.endswith(".mat"):
            continue

        mat_data = loadmat(os.path.join(data_path, file))
        try:
            I0 = mat_data["Test_0"]
            I45 = mat_data["Test_45"]
            I90 = mat_data["Test_90"]
            I135 = mat_data["Test_135"]
        except KeyError as e:
            raise KeyError(
                f"Missing expected key {e} in {data_path}. "
                f"Available keys: {list(mat_data.keys())}"
            )

        samples = _stack_polar_channels(
            I0=I0,
            I45=I45,
            I90=I90,
            I135=I135,
            file_path=str(data_path),
            noise_level=noise_level,
        )

        scenes.extend(samples)

    return scenes

def main():
    mat_data = load_mat_file(noise_level="Low")
    print("Num samples:", len(mat_data))
    print("First raw shape:", mat_data[0]["raw"].shape)
    if mat_data:
        print("Successfully loaded .mat file data.")

if __name__ == "__main__":
    main()
    