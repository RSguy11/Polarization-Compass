"""
Diagnostic: Is the model actually learning polarization→azimuth,
or just memorizing temporal/lighting patterns?

Test 1: PERMUTATION TEST — shuffle labels, retrain. If MAE stays low,
        the features themselves encode time-of-day, not polarization.

Test 2: CROSS-SESSION — train on June_23, test on June_24 (different day).
        This is the only honest generalization test.

Test 3: AZIMUTH RANGE CHECK — how narrow is the label distribution?
"""

import sys
import gc
import json
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from Underwater_testing.UnderwaterDataLoader import UnderwaterDataLoader
from Training_loops.run_all_models import (
    extract_statistical_features_from_single_image,
    CircularRegressionWrapper,
    calculate_sample_weights,
)
from sklearn.ensemble import RandomForestRegressor


def extract_features(loader, indices, label=""):
    features, valid = [], []
    for num, idx in enumerate(indices):
        if (num + 1) % 200 == 0:
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
    return np.array(features, dtype=np.float32), np.array(valid)


def train_and_eval(train_feat, train_az, test_feat, test_az, label=""):
    """Train RF circular model, return test MAE in degrees."""
    weights = calculate_sample_weights(np.rad2deg(train_az))
    model = CircularRegressionWrapper(
        RandomForestRegressor, n_estimators=300, max_depth=20,
        min_samples_split=3, min_samples_leaf=1, random_state=42, n_jobs=-1)
    model.fit_from_features(train_feat, train_az, sample_weight=weights)
    pred = model.predict_from_features(test_feat)
    errors = np.abs(np.rad2deg(pred) % 360 - np.rad2deg(test_az) % 360)
    errors = np.minimum(errors, 360 - errors)
    mae = errors.mean()
    print(f"  {label}: MAE = {mae:.4f}°")
    return mae, errors, pred


class Tee:
    """Duplicate stdout to both console and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w", encoding="utf-8")
        self.stdout = sys.stdout
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()
        sys.stdout = self.stdout


def main():
    # Set up output directory and dual logging
    output_dir = Path(__file__).parent / "diagnostic_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"diagnostic_report_{timestamp}.txt"
    tee = Tee(str(report_path))
    sys.stdout = tee

    # Also prepare a dict to save structured results as JSON
    results_json = {}

    print(f"Diagnostic Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Report saved to: {report_path}")
    print()

    loader = UnderwaterDataLoader()
    n = len(loader)
    print()

    all_labels = np.array([loader._get_labels(i)["azimuth"] for i in range(n)])
    all_sessions = np.array([loader._get_labels(i)["session"] for i in range(n)])
    all_bursts = np.array([loader._get_labels(i)["burst"] for i in range(n)])

    # ═════════════════════════════════════════════════════════════════════
    # TEST 0: Azimuth distribution analysis
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 0: AZIMUTH DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"  Total range:  {all_labels.min():.1f}° – {all_labels.max():.1f}°  "
          f"(span = {all_labels.max() - all_labels.min():.1f}°)")
    print(f"  Mean:         {all_labels.mean():.1f}°")
    print(f"  Std:          {all_labels.std():.1f}°")
    mean_baseline_global = float(np.mean(np.abs(all_labels - all_labels.mean())))
    print(f"  A random guess of the mean would give MAE = {mean_baseline_global:.1f}°")
    print(f"  Full 360° range?  NO — only {all_labels.max() - all_labels.min():.0f}° of 360° covered")
    print()

    results_json["azimuth_distribution"] = {
        "min": float(all_labels.min()),
        "max": float(all_labels.max()),
        "span": float(all_labels.max() - all_labels.min()),
        "mean": float(all_labels.mean()),
        "std": float(all_labels.std()),
        "mean_baseline_mae": mean_baseline_global,
        "coverage_pct": float((all_labels.max() - all_labels.min()) / 360 * 100),
    }

    for sess in ["June_23", "June_24"]:
        mask = all_sessions == sess
        az = all_labels[mask]
        print(f"  {sess}: {az.min():.1f}° – {az.max():.1f}° "
              f"(span={az.max()-az.min():.1f}°, n={mask.sum()})")
    print()

    # ═════════════════════════════════════════════════════════════════════
    # Use a subsample for speed (every 4th sample)
    # ═════════════════════════════════════════════════════════════════════
    subsample = np.arange(0, n, 4)  # every 4th frame
    print(f"Using subsample of {len(subsample)} frames for speed\n")

    print("Extracting features for subsample...")
    sub_feat, sub_valid = extract_features(loader, subsample.tolist(), "subsample")
    sub_labels = np.deg2rad(all_labels[sub_valid])
    sub_sessions = all_sessions[sub_valid]
    sub_bursts = all_bursts[sub_valid]
    print(f"Extracted {len(sub_feat)} features\n")

    # ═════════════════════════════════════════════════════════════════════
    # TEST 1: CROSS-SESSION SPLIT (train June_23, test June_24)
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 1: CROSS-SESSION (Train June_23 → Test June_24)")
    print("=" * 70)

    train_mask = sub_sessions == "June_23"
    test_mask = sub_sessions == "June_24"

    if train_mask.sum() > 0 and test_mask.sum() > 0:
        train_feat_cs = sub_feat[train_mask]
        test_feat_cs = sub_feat[test_mask]
        train_az_cs = sub_labels[train_mask]
        test_az_cs = sub_labels[test_mask]

        print(f"  Train: {train_mask.sum()} samples (June_23)")
        print(f"  Test:  {test_mask.sum()} samples (June_24)")
        print(f"  Train azimuth: {np.rad2deg(train_az_cs).min():.1f}° – {np.rad2deg(train_az_cs).max():.1f}°")
        print(f"  Test  azimuth: {np.rad2deg(test_az_cs).min():.1f}° – {np.rad2deg(test_az_cs).max():.1f}°")

        mae_cs, _, _ = train_and_eval(
            train_feat_cs, train_az_cs, test_feat_cs, test_az_cs,
            "Cross-session RF")
    else:
        print("  Not enough data in both sessions")
        mae_cs = None

    # Also test reverse: train June_24, test June_23
    print()
    mae_cs_rev = None
    if test_mask.sum() > 0 and train_mask.sum() > 0:
        mae_cs_rev, _, _ = train_and_eval(
            sub_feat[test_mask], sub_labels[test_mask],
            sub_feat[train_mask], sub_labels[train_mask],
            "Reverse (Train June_24 → Test June_23)")
    print()

    results_json["cross_session"] = {
        "train_june23_test_june24_mae": float(mae_cs) if mae_cs is not None else None,
        "train_june24_test_june23_mae": float(mae_cs_rev) if mae_cs_rev is not None else None,
    }

    # ═════════════════════════════════════════════════════════════════════
    # TEST 2: PERMUTATION TEST (shuffle labels, retrain)
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 2: PERMUTATION TEST (shuffled labels)")
    print("=" * 70)
    print("  If the model still performs well with shuffled labels,")
    print("  the features encode time, not polarization.\n")

    # Use burst-level split for fair comparison
    unique_bursts = np.unique(sub_bursts)
    rng = np.random.default_rng(42)
    n_test_bursts = max(1, int(0.2 * len(unique_bursts)))
    test_burst_idx = rng.choice(len(unique_bursts), n_test_bursts, replace=False)
    test_burst_set = set(unique_bursts[test_burst_idx])

    train_mask_b = np.array([b not in test_burst_set for b in sub_bursts])
    test_mask_b = ~train_mask_b

    train_feat_b = sub_feat[train_mask_b]
    test_feat_b = sub_feat[test_mask_b]
    train_az_b = sub_labels[train_mask_b]
    test_az_b = sub_labels[test_mask_b]

    # Real labels
    print(f"  Burst-level split: {train_mask_b.sum()} train, {test_mask_b.sum()} test")
    mae_real, _, _ = train_and_eval(
        train_feat_b, train_az_b, test_feat_b, test_az_b,
        "Real labels")

    # Shuffled labels (permutation)
    shuffled_train_az = train_az_b.copy()
    rng.shuffle(shuffled_train_az)
    mae_shuffled, _, _ = train_and_eval(
        train_feat_b, shuffled_train_az, test_feat_b, test_az_b,
        "Shuffled labels")

    # Constant prediction (mean baseline)
    mean_pred = np.full_like(test_az_b, train_az_b.mean())
    mean_errors = np.abs(np.rad2deg(mean_pred) % 360 - np.rad2deg(test_az_b) % 360)
    mean_errors = np.minimum(mean_errors, 360 - mean_errors)
    mae_mean = mean_errors.mean()
    print(f"  Constant mean baseline: MAE = {mae_mean:.4f}°")

    results_json["permutation_test"] = {
        "burst_split_real_mae": float(mae_real),
        "shuffled_label_mae": float(mae_shuffled),
        "mean_baseline_mae": float(mae_mean),
        "train_samples": int(train_mask_b.sum()),
        "test_samples": int(test_mask_b.sum()),
    }

    print()

    # ═════════════════════════════════════════════════════════════════════
    # TEST 3: FEATURE IMPORTANCE — what's driving predictions?
    # ═════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("TEST 3: TOP FEATURE IMPORTANCES")
    print("=" * 70)

    model_diag = CircularRegressionWrapper(
        RandomForestRegressor, n_estimators=100, max_depth=15,
        random_state=42, n_jobs=-1)
    model_diag.fit_from_features(train_feat_b, train_az_b)

    # Get feature importances from the sin model (proxy)
    importances = model_diag.sin_model.feature_importances_

    # Define feature names (abbreviated)
    feat_names = (
        [f"dolp_{s}" for s in ["mean","std","median","max","min","skew","kurt"]] +
        [f"aolp_{s}" for s in ["mean","std","median","max","min","skew","kurt"]] +
        ["cos2aolp_mean","sin2aolp_mean","cos2aolp_std","sin2aolp_std"] +
        ["dolp_cos_cross","dolp_sin_cross"] +
        [f"grid_dolp_{i}" for i in range(81)] +
        [f"grid_cos_{i}" for i in range(81)] +
        [f"grid_sin_{i}" for i in range(81)] +
        [f"grid_dcross_{i}" for i in range(81)] +
        [f"grid_scross_{i}" for i in range(81)] +
        ["grad_dolp_h","grad_dolp_v","grad_aolp_h","grad_aolp_v"] +
        ["dolp_center","dolp_edge","dolp_ce_ratio",
         "aolp_center","aolp_edge","aolp_ce_diff"] +
        [f"dolp_p{p}" for p in [10,25,50,75,90]] +
        [f"aolp_p{p}" for p in [10,25,50,75,90]] +
        ["centroid_x","centroid_y"] +
        ["radial_inner","radial_mid","radial_outer"] +
        [f"row_profile_{i}" for i in range(9)] +
        [f"col_profile_{i}" for i in range(9)]
    )

    # Pad if needed
    while len(feat_names) < len(importances):
        feat_names.append(f"feat_{len(feat_names)}")

    top_idx = np.argsort(importances)[::-1][:20]
    print(f"\n  {'Rank':<5} {'Feature':<25} {'Importance':>10}")
    print("  " + "-" * 42)
    top_features_list = []
    for rank, idx in enumerate(top_idx, 1):
        name = feat_names[idx] if idx < len(feat_names) else f"feat_{idx}"
        print(f"  {rank:<5} {name:<25} {importances[idx]:10.4f}")
        top_features_list.append({"rank": rank, "feature": name, "importance": float(importances[idx])})

    results_json["feature_importances"] = top_features_list

    # ═════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 70}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Azimuth coverage:        {all_labels.max()-all_labels.min():.0f}° of 360° ({(all_labels.max()-all_labels.min())/360*100:.0f}%)")
    print(f"  Mean-baseline MAE:       {mae_mean:.2f}°")
    print(f"  Burst-split real MAE:    {mae_real:.4f}°")
    print(f"  Shuffled-label MAE:      {mae_shuffled:.4f}°")
    if mae_cs is not None:
        print(f"  Cross-session MAE:       {mae_cs:.4f}°")
    print()
    if mae_shuffled < mae_mean * 0.5:
        print("  ⚠  SHUFFLED labels still beat baseline → features encode TIME, not polarization")
    else:
        print("  ✓  Shuffled labels perform like baseline → model IS using polarization signal")
    print()
    if mae_cs is not None and mae_cs > 5.0:
        print("  ⚠  Cross-session MAE > 5° → model does NOT generalize across days")
    elif mae_cs is not None:
        print(f"  Cross-session MAE = {mae_cs:.2f}° — {'promising' if mae_cs < 5 else 'needs improvement'}")

    # Verdict
    results_json["verdict"] = {
        "shuffled_beats_baseline": bool(mae_shuffled < mae_mean * 0.5),
        "cross_session_generalizes": bool(mae_cs is not None and mae_cs < 5.0),
    }

    # Save structured JSON alongside the text report
    json_path = output_dir / f"diagnostic_results_{timestamp}.json"
    with open(json_path, "w") as jf:
        json.dump(results_json, jf, indent=2)
    print(f"\n  Structured results saved to: {json_path}")
    print(f"  Full text report saved to:   {report_path}")

    tee.close()


if __name__ == "__main__":
    main()
