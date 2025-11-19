import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_and_save_stage3(
    stage2_samples,
    stage3_samples,
    out_dir="stage3_results",
    num_samples=4,
):
    """
    Create PNG visual comparisons for Stage 3 outputs.

    Layout per sample (3 x 3 grid):

        Row 1:
            [0,0] I0 (0° channel, from Stage 2)
            [0,1] S0 (total intensity)
            [0,2] Radiance (gamma-corrected L)

        Row 2:
            [1,0] Radiance_raw L(x) (before gamma)
            [1,1] Backscatter B(x)
            [1,2] Fusion parameter K(x)

        Row 3:
            [2,0] Δ(S0 vs Radiance)  = Radiance - S0
            [2,1] Δ(I0 vs Radiance)  = Radiance - I0
            [2,2] (empty / reserved)

    This mirrors the style of your Stage 1 & 2 visualizers: 
    grayscale for intensities, viridis for K/B, seismic for diffs.
    """

    os.makedirs(out_dir, exist_ok=True)
    num_samples = min(num_samples, len(stage2_samples), len(stage3_samples))

    for idx in range(num_samples):
        s2 = stage2_samples[idx]
        s3 = stage3_samples[idx]

        # Stage 2 basics
        I0 = s2["raw"][0].astype(np.float32)          # (H, W)
        S0 = s2["stokes"][0].astype(np.float32)       # (H, W)

        # Stage 3 products
        L_raw = s3["radiance_raw"].astype(np.float32)  # (H, W)
        L     = s3["radiance"].astype(np.float32)      # (H, W)
        B     = s3["B"].astype(np.float32)             # (H, W)
        K     = s3["K"].astype(np.float32)             # (H, W)

        # Set up figure
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f"Stage 3 Restoration – Sample {idx}", fontsize=18)

        # ---------------- Row 1: original intensity vs restored radiance ----------------
        ax = axes[0, 0]
        ax.imshow(I0, cmap="gray")
        ax.set_title("I0 (0° channel)")
        ax.axis("off")

        ax = axes[0, 1]
        ax.imshow(S0, cmap="gray")
        ax.set_title("S0 (total intensity)")
        ax.axis("off")

        ax = axes[0, 2]
        ax.imshow(L, cmap="gray")
        ax.set_title("Radiance L (gamma-corrected)")
        ax.axis("off")

        # ---------------- Row 2: raw radiance, backscatter, fusion K ----------------
        ax = axes[1, 0]
        ax.imshow(L_raw, cmap="gray")
        ax.set_title("Radiance_raw L(x)")
        ax.axis("off")

        ax = axes[1, 1]
        # Normalize B for display
        B_disp = B.copy()
        B_disp = np.clip(B_disp, np.percentile(B_disp, 1), np.percentile(B_disp, 99))
        ax.imshow(B_disp, cmap="viridis")
        ax.set_title("Backscatter B(x)")
        ax.axis("off")

        ax = axes[1, 2]
        ax.imshow(K, cmap="viridis")
        ax.set_title("Fusion parameter K(x)")
        ax.axis("off")

        # ---------------- Row 3: difference maps ----------------
        # Δ(S0 vs L)
        diff_S0 = L - S0
        ax = axes[2, 0]
        dmax = np.max(np.abs(diff_S0)) + 1e-6
        ax.imshow(diff_S0, cmap="seismic", vmin=-dmax, vmax=dmax)
        ax.set_title("Δ(S0, L) = L - S0")
        ax.axis("off")

        # Δ(I0 vs L)
        diff_I0 = L - I0
        ax = axes[2, 1]
        dmax = np.max(np.abs(diff_I0)) + 1e-6
        ax.imshow(diff_I0, cmap="seismic", vmin=-dmax, vmax=dmax)
        ax.set_title("Δ(I0, L) = L - I0")
        ax.axis("off")

        # Empty slot (reserved for future metrics / masks)
        axes[2, 2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(out_dir, f"stage3_sample_{idx}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[✓] Saved {save_path}")