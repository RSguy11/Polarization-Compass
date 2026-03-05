"""
Validate underwater model predictions — print predictions vs ground truth
side by side.  Also performs a DATA LEAKAGE audit and trains a quick model
with a proper burst-level split for honest comparison.
"""

import sys
import gc
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Underwater_testing.UnderwaterDataLoader import UnderwaterDataLoader
from Training_loops.run_all_models import (
    extract_statistical_features_from_single_image,
    CircularRegressionWrapper,
    calculate_sample_weights,
)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def extract_features_for_indices(loader, indices, label=""):
    """Extract 468 features for a list of dataset indices."""
    features, valid = [], []
    for num, idx in enumerate(indices):
        if (num + 1) % 100 == 0:
            print(f"    {label} {num+1}/{len(indices)}...")
            sys.stdout.flush()
        try:
            sample = loader.get_item(idx)
            if sample is not None:
                f = extract_statistical_features_from_single_image(
                    sample["features"]["dolp"], sample["features"]["aolp"])
                features.append(f)
                valid.append(idx)
        except Exception:
            pass
        if (num + 1) % 50 == 0:
            gc.collect()
    return np.array(features, dtype=np.float32), valid


def print_table(loader, valid_indices, all_labels, all_bursts, pred_deg,
                overlap_bursts, n_rows=50):
    """Print side-by-side GT vs Predicted table."""
    ground_truth = all_labels[valid_indices]
    errors = np.abs(pred_deg - ground_truth)
    errors = np.minimum(errors, 360 - errors)

    print(f"\n{'#':>3}  {'Image':<68} {'GT':>8} {'Pred':>8} {'Err':>7}  {'Burst':<32} {'Leak?'}")
    print("-" * 150)
    for i, idx in enumerate(valid_indices[:n_rows]):
        img = loader.labels_df.iloc[idx]["image_path"]
        burst = all_bursts[idx]
        leaked = "YES" if burst in overlap_bursts else "-"
        print(f"{i+1:>3}  {img:<68} {ground_truth[i]:8.2f} {pred_deg[i]:8.2f} {errors[i]:7.3f}  {burst:<32} {leaked}")

    print(f"\n  MAE over {len(valid_indices)} shown samples: {errors.mean():.4f}°")
    return errors


def main():
    # ── Load dataset ─────────────────────────────────────────────────────
    loader = UnderwaterDataLoader()
    n_total = len(loader)
    print()

    all_labels = np.array([loader._get_labels(i)["azimuth"] for i in range(n_total)])
    all_bursts = np.array([loader._get_labels(i)["burst"] for i in range(n_total)])

    # ═════════════════════════════════════════════════════════════════════
    # PART A — Replicate the CURRENT (leaky) split and show predictions
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PART A: Current frame-level split (same as run_all_models)")
    print("=" * 70)

    azimuth_deg = all_labels
    n_bins = 18
    bins = np.linspace(0, 360, n_bins + 1)
    az_bin = np.clip(np.digitize(azimuth_deg, bins, right=False) - 1, 0, n_bins - 1)

    rng = np.random.default_rng(42)
    train_idx_a, test_idx_a = [], []
    for b in range(n_bins):
        bin_idxs = [i for i, ab in enumerate(az_bin) if ab == b]
        if not bin_idxs:
            continue
        n_test = max(1, int(0.2 * len(bin_idxs)))
        test_in = rng.choice(bin_idxs, size=n_test, replace=False).tolist()
        train_in = [i for i in bin_idxs if i not in test_in]
        test_idx_a.extend(test_in)
        train_idx_a.extend(train_in)
    rng.shuffle(train_idx_a)
    rng.shuffle(test_idx_a)

    # Burst overlap check
    train_bursts_a = set(all_bursts[train_idx_a])
    test_bursts_a = set(all_bursts[test_idx_a])
    overlap_a = train_bursts_a & test_bursts_a
    leaked_count = sum(1 for i in test_idx_a if all_bursts[i] in overlap_a)

    print(f"\n  DATA LEAKAGE CHECK:")
    print(f"  Train bursts: {len(train_bursts_a)},  Test bursts: {len(test_bursts_a)}")
    print(f"  Shared bursts: {len(overlap_a)}  ← {'LEAKAGE!' if overlap_a else 'Clean'}")
    print(f"  Leaked test samples: {leaked_count}/{len(test_idx_a)}")

    # Train quick RF on leaked split, predict on 50 test samples
    n_show = 50
    show_test = test_idx_a[:n_show]

    print(f"\n  Extracting features for {len(train_idx_a)} train + {n_show} test samples...")
    train_feat_a, train_valid_a = extract_features_for_indices(loader, train_idx_a, "train")
    test_feat_a, test_valid_a = extract_features_for_indices(loader, show_test, "test")

    azimuth_train_a = np.deg2rad(all_labels[train_valid_a])
    weights_a = calculate_sample_weights(all_labels[train_valid_a])

    print(f"\n  Training RF on {len(train_valid_a)} samples (frame-level split)...")
    model_a = CircularRegressionWrapper(
        RandomForestRegressor, n_estimators=300, max_depth=20,
        min_samples_split=3, min_samples_leaf=1, random_state=42, n_jobs=-1)
    model_a.fit_from_features(train_feat_a, azimuth_train_a, sample_weight=weights_a)

    pred_rad_a = model_a.predict_from_features(test_feat_a)
    pred_deg_a = np.rad2deg(pred_rad_a) % 360

    print("\n  PREDICTIONS (Frame-level split — likely leaked):")
    errors_a = print_table(loader, test_valid_a, all_labels, all_bursts,
                           pred_deg_a, overlap_a, n_rows=n_show)

    # ═════════════════════════════════════════════════════════════════════
    # PART B — Proper BURST-LEVEL split (no leakage)
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 70}")
    print("PART B: Burst-level split (NO leakage — honest evaluation)")
    print("=" * 70)

    unique_bursts = np.unique(all_bursts)
    burst_azimuths = {}
    for burst in unique_bursts:
        mask = all_bursts == burst
        burst_azimuths[burst] = all_labels[mask].mean()

    # Stratify bursts by azimuth
    burst_az = np.array([burst_azimuths[b] for b in unique_bursts])
    burst_bin = np.clip(np.digitize(burst_az, bins, right=False) - 1, 0, n_bins - 1)

    rng2 = np.random.default_rng(42)
    train_bursts_b = []
    test_bursts_b = []
    for b in range(n_bins):
        b_idxs = [i for i, bb in enumerate(burst_bin) if bb == b]
        if not b_idxs:
            continue
        n_test = max(1, int(0.2 * len(b_idxs)))
        test_in = rng2.choice(b_idxs, size=n_test, replace=False).tolist()
        train_in = [i for i in b_idxs if i not in test_in]
        test_bursts_b.extend([unique_bursts[i] for i in test_in])
        train_bursts_b.extend([unique_bursts[i] for i in train_in])

    train_burst_set = set(train_bursts_b)
    test_burst_set = set(test_bursts_b)
    overlap_b = train_burst_set & test_burst_set

    train_idx_b = [i for i in range(n_total) if all_bursts[i] in train_burst_set]
    test_idx_b = [i for i in range(n_total) if all_bursts[i] in test_burst_set]
    rng2.shuffle(train_idx_b)
    rng2.shuffle(test_idx_b)

    print(f"\n  DATA LEAKAGE CHECK:")
    print(f"  Train bursts: {len(train_burst_set)},  Test bursts: {len(test_burst_set)}")
    print(f"  Shared bursts: {len(overlap_b)}  ← {'LEAKAGE!' if overlap_b else 'Clean'}")
    print(f"  Train samples: {len(train_idx_b)},  Test samples: {len(test_idx_b)}")

    show_test_b = test_idx_b[:n_show]

    print(f"\n  Extracting features for {len(train_idx_b)} train + {n_show} test samples...")
    train_feat_b, train_valid_b = extract_features_for_indices(loader, train_idx_b, "train")
    test_feat_b, test_valid_b = extract_features_for_indices(loader, show_test_b, "test")

    azimuth_train_b = np.deg2rad(all_labels[train_valid_b])
    weights_b = calculate_sample_weights(all_labels[train_valid_b])

    print(f"\n  Training RF on {len(train_valid_b)} samples (burst-level split)...")
    model_b = CircularRegressionWrapper(
        RandomForestRegressor, n_estimators=300, max_depth=20,
        min_samples_split=3, min_samples_leaf=1, random_state=42, n_jobs=-1)
    model_b.fit_from_features(train_feat_b, azimuth_train_b, sample_weight=weights_b)

    pred_rad_b = model_b.predict_from_features(test_feat_b)
    pred_deg_b = np.rad2deg(pred_rad_b) % 360

    print("\n  PREDICTIONS (Burst-level split — honest):")
    errors_b = print_table(loader, test_valid_b, all_labels, all_bursts,
                           pred_deg_b, overlap_b, n_rows=n_show)

    # ═════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Frame-level split (leaked):   MAE = {errors_a.mean():.4f}°")
    print(f"  Burst-level split (honest):   MAE = {errors_b.mean():.4f}°")
    print(f"  Difference:                   {errors_b.mean() - errors_a.mean():.4f}°")
    print()
    if errors_b.mean() > errors_a.mean() * 2:
        print("  ⚠  The frame-level results were inflated by data leakage.")
        print("     The burst-level MAE is the honest performance metric.")
    else:
        print("  Results are consistent — minimal leakage effect.")


if __name__ == "__main__":
    main()
