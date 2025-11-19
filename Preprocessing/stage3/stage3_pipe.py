from typing import List, Dict, Optional
import numpy as np


def _compute_R(I0: np.ndarray, I: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return I0 - I * (1.0 - rho) / 2.0


def _estimate_K_blockwise(
    I: np.ndarray,
    R: np.ndarray,
    block_size: int = 32,
    k_values: Optional[np.ndarray] = None,
) -> np.ndarray:

    if k_values is None:
        # Discrete candidates in [0.1, 0.9]
        k_values = np.linspace(0.1, 0.9, 17, dtype=np.float32)

    H, W = I.shape
    K_map = np.zeros((H, W), dtype=np.float32)
    eps = 1e-6

    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            ys, ye = y, min(y + block_size, H)
            xs, xe = x, min(x + block_size, W)

            I_blk = I[ys:ye, xs:xe]
            R_blk = R[ys:ye, xs:xe]

            best_K = 0.5
            best_score = -np.inf

            for Kc in k_values:
                # Eq. (21) – direct term D(x)
                D_blk = (I_blk - R_blk) / (Kc + eps)
                D_blk = np.clip(D_blk, 0.0, None)

                D_min = float(D_blk.min())
                D_max = float(D_blk.max())

                # Skip degenerate blocks
                if D_min <= 0 or D_max <= 0 or D_max <= D_min:
                    continue

                # Contrast metric inspired by Eq. (27)
                score = np.log10(D_max / D_min)

                if score > best_score:
                    best_score = score
                    best_K = float(Kc)

            K_map[ys:ye, xs:xe] = best_K

    return K_map


def _compute_B_from_K(
    I0: np.ndarray,
    I: np.ndarray,
    rho: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    eps = 1e-6
    numerator = 2.0 * I0 - I * (1.0 - rho)
    denom = 2.0 * np.clip(K, eps, None)
    return numerator / denom


def _compute_L_from_K_Binf(
    I: np.ndarray,
    I0: np.ndarray,
    rho: np.ndarray,
    K: np.ndarray,
    B_inf: float,
) -> np.ndarray:

    eps = 1e-6

    term1 = 2.0 * K * I - 2.0 * I0 + I * (1.0 - rho)
    term2 = 2.0 * K * B_inf - 2.0 * I0 + I * (1.0 - rho)

    denom = np.clip(term2, eps, None)
    L = B_inf * term1 / denom
    return L


def _simple_gamma(L: np.ndarray, gamma: float = 0.85) -> np.ndarray:
    L = np.clip(L, 0.0, None)
    L_max = float(L.max()) if L.max() > 0 else 1.0
    L_norm = L / L_max
    L_gamma = np.power(L_norm, gamma)
    return L_gamma * L_max


def stage3_underwater_restoration(
    samples: List[Dict],
    block_size: int = 32,
    k_values: Optional[np.ndarray] = None,
    gamma: float = 0.85,
) -> List[Dict]:

    out_samples: List[Dict] = []

    for sample in samples:
        I0 = sample["raw"][0].astype(np.float32)          # 0° channel
        S0, _, _ = sample["stokes"].astype(np.float32)    # total intensity
        rho = sample["DoLP"].astype(np.float32)           # DoLP

        # --- Step 1: ℜ(x) ---
        R = _compute_R(I0, S0, rho)

        # --- Step 2: K(x) blockwise optimisation ---
        K_map = _estimate_K_blockwise(S0, R, block_size=block_size, k_values=k_values)

        # --- Step 3: Backscatter B(x) ---
        B = _compute_B_from_K(I0, S0, rho, K_map)

        # --- Step 4: Global B∞ estimate ---
        # Use a high percentile of B as background backscatter approximation
        # (simplified version of Eq. (37)'s brightest-region selection).
        B_flat = B.flatten()
        B_inf = float(np.percentile(B_flat, 95))

        # --- Step 5: Unattenuated radiance L(x) ---
        L_raw = _compute_L_from_K_Binf(S0, I0, rho, K_map, B_inf)

        # --- Step 6: Gamma contrast enhancement ---
        L_gamma = _simple_gamma(L_raw, gamma=gamma)

        new_sample = dict(sample)
        new_sample["K"] = K_map.astype(np.float32)
        new_sample["B"] = B.astype(np.float32)
        new_sample["B_inf"] = B_inf
        new_sample["radiance_raw"] = L_raw.astype(np.float32)
        new_sample["radiance"] = L_gamma.astype(np.float32)
        new_sample["meta"] = {
            **sample.get("meta", {}),
            "stage3": "piom_restored",
        }

        out_samples.append(new_sample)

    return out_samples