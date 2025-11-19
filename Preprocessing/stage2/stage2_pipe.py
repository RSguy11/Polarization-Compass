from typing import List, Dict, Union, Literal
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm

def compute_stokes_from_raw(raw: np.ndarray) -> np.ndarray:
    I0, I45, I90, I135 = raw

    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135

    stokes = np.stack([S0, S1, S2], axis=0)

    return stokes.astype(np.float32)

def polarimetric_parameters_from_stokes(samples) -> Dict[str, np.ndarray]:
    out_samples: List[Dict] = []

    for sample in tqdm(samples, desc="Stage 2: Polarimetric Parameters"):
        raw = sample["raw"].astype(np.float32)        # (4, H, W)
        stokes = compute_stokes_from_raw(raw)         # (3, H, W)
        S0, S1, S2 = stokes

        eps = 1e-6
        denom = np.clip(S0, eps, None)

        DoLP = np.sqrt(S1**2 + S2**2) / denom         # (H, W)

        # Angle of Linear Polarization (radians, in ~(-pi/2, pi/2))
        AoLP = 0.5 * np.arctan2(S2, S1)               # (H, W)
        AoLP_deg = np.rad2deg(AoLP)

        new_sample = dict(sample)
        new_sample["stokes"] = stokes.astype(np.float32)
        new_sample["DoLP"] = DoLP.astype(np.float32)
        new_sample["AoLP"] = AoLP.astype(np.float32)
        new_sample["AoLP_deg"] = AoLP_deg.astype(np.float32)
        new_sample["meta"] = {
            **sample.get("meta", {}),
            "stage2": "stokes_DoLP_AoLP",
        }
        out_samples.append(new_sample)

    return out_samples