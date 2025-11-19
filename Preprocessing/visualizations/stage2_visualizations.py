import matplotlib.pyplot as plt
import numpy as np
import os


def visualize_and_save_stage2(
    orig_samples,
    stage2_samples,
    out_dir="stage2_results",
    num_samples=4
):
    """
    Create PNG visual comparisons for Stage 2 outputs.

    Layout per sample:
        Row 1: Stokes S0, S1, S2
        Row 2: DoLP, AoLP (deg)
        Row 3: Difference maps:
                S0_diff = S0 - (I0 + I90)
                DoLP_diff = DoLP - DoLP_from_original
    """

    os.makedirs(out_dir, exist_ok=True)
    num_samples = min(num_samples, len(orig_samples), len(stage2_samples))

    for idx in range(num_samples):

        raw_orig = orig_samples[idx]["raw"]      # (4, H, W)
        st2 = stage2_samples[idx]

        S0, S1, S2 = st2["stokes"]               # (3, H, W)
        dolp = st2["dolp"]                       # (H, W)
        aolp_deg = st2["aolp_deg"]               # (H, W)

        # compute reference stokes from original raw for difference map
        I0, I45, I90, I135 = raw_orig
        S0_ref = I0 + I90
        dolp_ref = np.sqrt((I0 - I90)**2 + (I45 - I135)**2) / (S0_ref + 1e-6)

        # --- FIGURE ---
        fig, axes = plt.subplots(3, 3, figsize=(16, 14))
        fig.suptitle(f"Stage 2 Polarization Features – Sample {idx}", fontsize=18)

        # Row 1: Stokes S0, S1, S2
        stokes = [S0, S1, S2]
        titles = ["Stokes S0", "Stokes S1", "Stokes S2"]
        for c in range(3):
            ax = axes[0, c]
            ax.imshow(stokes[c], cmap="gray")
            ax.set_title(titles[c])
            ax.axis("off")

        # Row 2: DoLP + AoLP
        ax_dolp = axes[1, 0]
        ax_dolp.imshow(dolp, cmap="gray", vmin=0, vmax=1)
        ax_dolp.set_title("DoLP (0→1)")
        ax_dolp.axis("off")

        ax_aolp = axes[1, 1]
        ax_aolp.imshow(aolp_deg, cmap="twilight", vmin=-90, vmax=90)
        ax_aolp.set_title("AoLP (degrees)")
        ax_aolp.axis("off")

        # leave 1 spot empty for symmetry
        axes[1, 2].axis("off")

        # Row 3: Difference maps
        S0_diff = S0 - S0_ref
        dolp_diff = dolp - dolp_ref

        diff_maps = [S0_diff, dolp_diff]
        diff_titles = ["ΔS0 (Stage2 - Orig)", "ΔDoLP (Stage2 - Orig)"]

        for j in range(2):
            ax = axes[2, j]
            diff = diff_maps[j]
            diff_max = np.max(np.abs(diff)) + 1e-6
            ax.imshow(diff, cmap="seismic", vmin=-diff_max, vmax=diff_max)
            ax.set_title(diff_titles[j])
            ax.axis("off")

        axes[2, 2].axis("off")  # filler

        # Save the figure
        save_path = os.path.join(out_dir, f"stage2_sample_{idx}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=200)
        plt.close()

        print(f"[✓] Saved {save_path}")