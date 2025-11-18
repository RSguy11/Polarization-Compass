import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_and_save_stage1(
    orig_samples,
    stage1_samples,
    out_dir="stage1_results",
    num_samples=4
):
    """
    Create PNG visual comparisons for the first N samples.

    Layout per PNG:
        Row 1: Original [I0, I45, I90, I135]
        Row 2: Stage 1  [I0, I45, I90, I135]
        Row 3: Difference (Stage1 - Original) per channel
    """
    os.makedirs(out_dir, exist_ok=True)

    num_samples = min(num_samples, len(orig_samples), len(stage1_samples))
    angles = ["0°", "45°", "90°", "135°"]

    for idx in range(num_samples):
        orig = orig_samples[idx]["raw"]
        proc = stage1_samples[idx]["raw"]

        assert orig.shape == proc.shape, f"Shape mismatch: {orig.shape} vs {proc.shape}"
        C, H, W = orig.shape
        assert C == 4, f"Expected 4 channels, got {C}"

        # Intensity scaling per channel based on original
        vmins = [np.min(orig[c]) for c in range(C)]
        vmaxs = [np.max(orig[c]) for c in range(C)]

        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        fig.suptitle(f"Stage 1 Comparison – Sample {idx}", fontsize=18)

        for c in range(C):
            # --- Row 1: Original ---
            ax0 = axes[0, c]
            ax0.imshow(orig[c], cmap="gray", vmin=vmins[c], vmax=vmaxs[c])
            ax0.set_title(f"Orig {angles[c]}")
            ax0.axis("off")

            # --- Row 2: Stage 1 ---
            ax1 = axes[1, c]
            ax1.imshow(proc[c], cmap="gray", vmin=vmins[c], vmax=vmaxs[c])
            ax1.set_title(f"Stage1 {angles[c]}")
            ax1.axis("off")

            # --- Row 3: Difference (Stage1 - Orig) ---
            diff = proc[c] - orig[c]
            ax2 = axes[2, c]

            # Symmetric color range around 0 for difference
            diff_abs_max = np.max(np.abs(diff)) + 1e-6
            ax2.imshow(diff, cmap="seismic", vmin=-diff_abs_max, vmax=diff_abs_max)
            ax2.set_title(f"Diff {angles[c]} (Stage1 - Orig)")
            ax2.axis("off")

        plt.tight_layout()

        save_path = os.path.join(out_dir, f"sample_{idx}.png")
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[✓] Saved {save_path}")