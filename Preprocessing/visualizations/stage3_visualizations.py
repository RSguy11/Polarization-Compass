import matplotlib.pyplot as plt
import os
import numpy as np


def visualize_and_save_stage3_piom(
    stage2_samples,
    stage3_samples,
    out_dir="stage3_piom_results",
    num_samples=4,
):
    """
    For each of the first N samples, save a PNG with:

      Row 1: I0, S0, restored L
      Row 2: K(x) map, (optional) DoLP, and difference L - S0
    """
    os.makedirs(out_dir, exist_ok=True)
    num = min(num_samples, len(stage2_samples), len(stage3_samples))

    for idx in range(num):
        s2 = stage2_samples[idx]
        s3 = stage3_samples[idx]

        raw = s2["raw"]
        stokes = s2["stokes"]
        dolp = s2["DoLP"]

        I0 = raw[0]
        S0 = stokes[0]
        L = s3["stage3_L"]
        K_map = s3["stage3_K"]

        diff = L - S0

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f"Stage 3 PIOM Restoration – Sample {idx}", fontsize=16)

        # Row 1: I0, S0, L
        ax = axes[0, 0]
        ax.imshow(I0, cmap="gray")
        ax.set_title("I0 (0°)")
        ax.axis("off")

        ax = axes[0, 1]
        ax.imshow(S0, cmap="gray")
        ax.set_title("S0 (total intensity)")
        ax.axis("off")

        ax = axes[0, 2]
        ax.imshow(L, cmap="gray")
        ax.set_title("Restored L(x)")
        ax.axis("off")

        # Row 2: K map, DoLP, diff
        ax = axes[1, 0]
        im0 = ax.imshow(K_map, cmap="viridis")
        ax.set_title("Fusion parameter K(x)")
        ax.axis("off")
        fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 1]
        im1 = ax.imshow(dolp, cmap="magma", vmin=0, vmax=1)
        ax.set_title("DoLP")
        ax.axis("off")
        fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, 2]
        diff_abs_max = np.max(np.abs(diff)) + 1e-6
        ax.imshow(diff, cmap="seismic",
                  vmin=-diff_abs_max, vmax=diff_abs_max)
        ax.set_title("Δ (L - S0)")
        ax.axis("off")

        plt.tight_layout()

        path = os.path.join(out_dir, f"stage3_sample_{idx}.png")
        plt.savefig(path, dpi=200)
        plt.close()
        print(f"[✓] Saved {path}")
