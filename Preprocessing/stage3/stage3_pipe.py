import numpy as np
import cv2
from typing import List, Dict
from tqdm import tqdm


def _estimate_background_light(S0: np.ndarray, top_percent: float = 0.001) -> float:
    """
    Simple approximation of B∞:
    Take the mean of the brightest 'top_percent' pixels in S0.
    """
    flat = S0.reshape(-1)
    k = max(1, int(len(flat) * top_percent))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))


def _score_K(I_block: np.ndarray,
             R_block: np.ndarray,
             K: float,
             alpha: float = 1.0) -> float:
    """
    Objective L = Leme - Linf  (Eqs. (26)-(28), simplified):
      - Leme: log contrast of D(x)
      - Linf: penalty for values outside [0, 1]
    We work on normalized intensities (0..1).
    """
    eps = 1e-8
    # Eq. (24): D_i(x) = I_i(x) - R_i(x) / K_i
    D = I_block - R_block / (K + eps)

    # Contrast term (simplified: use block max/min instead of sub-blocks)
    D_max = np.max(D)
    D_min = np.min(D)

    if D_max <= eps or D_min <= eps:
        Leme = -1e9  # terrible contrast
    else:
        Leme = 20.0 * np.log10(D_max / (D_min + eps))

    # Info-loss term: squared violations outside [0, 1]
    over = np.maximum(0.0, D - 1.0)
    under = np.minimum(0.0, D)
    Linf = np.mean(over ** 2 + under ** 2)

    return float(Leme - alpha * Linf)


def _estimate_K_blockwise(I: np.ndarray,
                          R: np.ndarray,
                          block_size: int = 64,
                          k_values: np.ndarray | None = None,
                          alpha: float = 1.0) -> np.ndarray:
    """
    Approximation of APSLO + spatial fitting:
      * tile the image into blocks
      * for each block, grid-search K that maximizes L
      * upsample K map back to full resolution with bilinear interpolation
    """
    H, W = I.shape
    if k_values is None:
        # K ∈ (0,1]; avoid extremely small values
        k_values = np.linspace(0.1, 1.0, 10, dtype=np.float32)

    nH = (H + block_size - 1) // block_size
    nW = (W + block_size - 1) // block_size

    K_blocks = np.zeros((nH, nW), dtype=np.float32)

    for bi in range(nH):
        for bj in range(nW):
            y0 = bi * block_size
            x0 = bj * block_size
            y1 = min((bi + 1) * block_size, H)
            x1 = min((bj + 1) * block_size, W)

            I_block = I[y0:y1, x0:x1]
            R_block = R[y0:y1, x0:x1]

            best_score = -1e12
            best_K = 0.5

            for K in k_values:
                s = _score_K(I_block, R_block, K, alpha=alpha)
                if s > best_score:
                    best_score = s
                    best_K = K

            K_blocks[bi, bj] = best_K

    # "Spatial fitting": just interpolate the coarse K grid to full resolution
    K_full = cv2.resize(
        K_blocks,
        (W, H),
        interpolation=cv2.INTER_CUBIC
    ).astype(np.float32)

    return K_full


def stage3_restore_piom(
    stage2_samples: List[Dict],
    block_size: int = 64,
    alpha: float = 1.0,
    k_values: np.ndarray | None = None
) -> List[Dict]:
    """
    Stage 3: Implement PIOM-style restoration from Li et al. (2025),
    simplified:
      - uses Eq. (22) for ℜ(x)
      - Eq. (21) and (24) for D_i(x)
      - Eqs. (26)-(28) for the objective to estimate K locally
      - Eq. (23) for final L(x) using a simple B∞ estimator.

    Assumes each sample has:
      sample["raw"]   -> (4, H, W) [I0, I45, I90, I135]
      sample["stokes"]-> (3, H, W) [S0, S1, S2]
      sample["dolp"]  -> (H, W)    rho(x)
    """
    restored_samples: List[Dict] = []

    for sample in tqdm(stage2_samples, desc="Stage 3 (PIOM)"):
        raw = sample["raw"]      # (4, H, W)
        stokes = sample["stokes"]  # (3, H, W)
        dolp = sample["DoLP"]    # (H, W)

        I0 = raw[0].astype(np.float32)     # horizontal polarization
        S0 = stokes[0].astype(np.float32)  # total intensity I(x)
        rho = np.clip(dolp.astype(np.float32), 0.0, 0.99)

        # Normalize intensities to [0,1] for stability
        I_max = np.max(S0) + 1e-8
        I = S0 / I_max
        I0_n = I0 / I_max

        # Eq. (22): ℜ(x) = I0(x) - I(x)*(1 - ρ(x))/2
        R = I0_n - I * (1.0 - rho) * 0.5

        # Blockwise estimation of K (approximate APSLO + spatial fitting)
        K_map = _estimate_K_blockwise(
            I=I,
            R=R,
            block_size=block_size,
            k_values=k_values,
            alpha=alpha,
        )

        # Background light B∞ from S0 (use un-normalized intensity)
        B_inf = _estimate_background_light(S0)

        # We’ll compute L(x) in normalized domain, then scale back.
        B_inf_n = B_inf / I_max
        eps = 1e-8

        # Eq. (23) in normalized form:
        # L = B∞ [2K I - 2 I0 + I (1 - ρ)] / [2K B∞ - 2 I0 + I (1 - ρ)]
        num = B_inf_n * (2.0 * K_map * I - 2.0 * I0_n + I * (1.0 - rho))
        den = (2.0 * K_map * B_inf_n - 2.0 * I0_n + I * (1.0 - rho))

        L_n = num / (den + eps)

        # Undo normalization, clip to reasonable range
        L = np.clip(L_n * I_max, 0.0, I_max).astype(np.float32)

        new_sample = dict(sample)
        new_sample["stage3_L"] = L
        new_sample["stage3_K"] = K_map.astype(np.float32)
        new_sample["stage3_Binf"] = float(B_inf)

        restored_samples.append(new_sample)

    return restored_samples