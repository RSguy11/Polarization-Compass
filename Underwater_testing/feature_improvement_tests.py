"""
Feature Improvement Tests — AoLP-only features & per-run normalization.

Tests 5 feature configurations on the cross-session split (hardest test)
to see which changes actually improve generalization.

Configurations:
  A) Baseline (all 468 features, no normalization)
  B) AoLP-only features (drop all DoLP-derived features)
  C) Per-run Z-score normalization (all features)
  D) AoLP-only + per-run normalization
  E) Reduced complexity (PCA + shallow RF)

Each is tested with:
  - Cross-session: Train June_23 → Test June_24
  - Cross-session: Train June_24 → Test June_23
  - Burst-level split (both days combined, for reference)
"""

import sys
import gc
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent))

from Underwater_testing.UnderwaterDataLoader import UnderwaterDataLoader
from Training_loops.run_all_models import (
    extract_statistical_features_from_single_image,
    CircularRegressionWrapper,
    calculate_sample_weights,
)


# ═══════════════════════════════════════════════════════════════════════
# Feature index maps (468 total)
# ═══════════════════════════════════════════════════════════════════════
# Feature layout from extract_statistical_features_from_single_image:
#   [0:7]     dolp_features       — DoLP global stats (mean,std,median,max,min,skew,kurt)
#   [7:14]    aolp_features       — AoLP global stats
#   [14:18]   circular_features   — cos2aolp mean/std, sin2aolp mean/std
#   [18:20]   cross_features      — DoLP*cos, DoLP*sin (mixed)
#   [20:101]  grid_dolp           — 81 DoLP grid means
#   [101:182] grid_aolp_cos       — 81 AoLP cos grid means
#   [182:263] grid_aolp_sin       — 81 AoLP sin grid means
#   [263:344] grid_cross_cos      — 81 DoLP*cos cross grid (mixed)
#   [344:425] grid_cross_sin      — 81 DoLP*sin cross grid (mixed)
#   [425:427] gradient_dolp       — DoLP h/v gradients
#   [427:429] gradient_aolp       — AoLP h/v gradients
#   [429:432] center_edge_dolp    — DoLP center, edge, ratio
#   [432:435] center_edge_aolp    — AoLP center cos, edge cos, diff
#   [435:440] percentile_dolp     — DoLP p10,p25,p50,p75,p90
#   [440:445] percentile_aolp     — AoLP p10,p25,p50,p75,p90
#   [445:447] centroid            — DoLP-weighted centroids (mixed)
#   [447:450] radial              — DoLP radial (inner,mid,outer)
#   [450:459] row_profile         — DoLP row profiles
#   [459:468] col_profile         — DoLP col profiles

# AoLP-only indices: features derived purely from AoLP, no DoLP
AOLP_ONLY_INDICES = np.array(
    list(range(7, 14)) +       # AoLP global stats (7)
    list(range(14, 18)) +      # Circular AoLP features (4)
    list(range(101, 182)) +    # grid_aolp_cos (81)
    list(range(182, 263)) +    # grid_aolp_sin (81)
    list(range(427, 429)) +    # AoLP gradients (2)
    list(range(432, 435)) +    # AoLP center/edge (3)
    list(range(440, 445))      # AoLP percentiles (5)
)  # Total: 183 features

# AoLP + cross-product indices (cross products combine DoLP*AoLP but
# preserve directional information modulated by polarization strength)
AOLP_PLUS_CROSS_INDICES = np.array(
    list(range(7, 14)) +       # AoLP global stats (7)
    list(range(14, 20)) +      # Circular + cross features (6)
    list(range(101, 263)) +    # grid_aolp_cos + sin (162)
    list(range(263, 425)) +    # grid_cross_cos + sin (162)
    list(range(427, 429)) +    # AoLP gradients (2)
    list(range(432, 435)) +    # AoLP center/edge (3)
    list(range(440, 445))      # AoLP percentiles (5)
)  # Total: 347 features


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


def extract_all_features(loader, indices, label=""):
    """Extract raw 468-dim feature vectors + metadata."""
    features, valid_idx, runs, sessions = [], [], [], []
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
                valid_idx.append(idx)
                runs.append(sample["metadata"]["run"])
                sessions.append(sample["metadata"]["session"])
        except Exception:
            pass
        if (num + 1) % 50 == 0:
            gc.collect()
    return (np.array(features, dtype=np.float32),
            np.array(valid_idx),
            np.array(runs),
            np.array(sessions))


def normalize_per_run(features, runs, train_stats=None):
    """
    Z-score normalize features within each run.
    
    If train_stats is None, computes stats from this data (training mode).
    Returns (normalized_features, stats_dict) where stats_dict can be
    passed as train_stats for test data.
    """
    result = features.copy()
    unique_runs = np.unique(runs)
    
    if train_stats is None:
        # Training mode: compute per-run stats
        stats = {}
        for run in unique_runs:
            mask = runs == run
            run_feats = features[mask]
            mean = run_feats.mean(axis=0)
            std = run_feats.std(axis=0) + 1e-8
            stats[run] = (mean, std)
            result[mask] = (run_feats - mean) / std
        return result, stats
    else:
        # Test mode: use precomputed stats, fall back to global if run unseen
        # Compute global fallback from training stats
        all_means = np.stack([s[0] for s in train_stats.values()])
        all_stds = np.stack([s[1] for s in train_stats.values()])
        global_mean = all_means.mean(axis=0)
        global_std = all_stds.mean(axis=0)
        
        for run in unique_runs:
            mask = runs == run
            run_feats = features[mask]
            if run in train_stats:
                mean, std = train_stats[run]
            else:
                # Unseen run: use stats from this run's own data
                # (self-normalize, since we don't have training stats for it)
                mean = run_feats.mean(axis=0)
                std = run_feats.std(axis=0) + 1e-8
            result[mask] = (run_feats - mean) / std
        return result, train_stats


def train_and_eval(train_feat, train_az, test_feat, test_az,
                   n_estimators=300, max_depth=20, pca_components=None):
    """Train RF circular model, return test MAE in degrees."""
    weights = calculate_sample_weights(np.rad2deg(train_az))
    
    pca = None
    if pca_components is not None:
        n_comp = min(pca_components, train_feat.shape[1], train_feat.shape[0])
        pca = PCA(n_components=n_comp, random_state=42)
        train_feat = pca.fit_transform(train_feat)
        test_feat = pca.transform(test_feat)
    
    model = CircularRegressionWrapper(
        RandomForestRegressor, n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=3, min_samples_leaf=1, random_state=42, n_jobs=-1)
    model.fit_from_features(train_feat, train_az, sample_weight=weights)
    pred = model.predict_from_features(test_feat)
    errors = np.abs(np.rad2deg(pred) % 360 - np.rad2deg(test_az) % 360)
    errors = np.minimum(errors, 360 - errors)
    return float(errors.mean()), float(np.median(errors)), errors


def run_config(config_name, train_feat, train_az, test_feat, test_az,
               n_estimators=300, max_depth=20, pca_components=None):
    """Run a single configuration and print results."""
    mae, median_err, errors = train_and_eval(
        train_feat, train_az, test_feat, test_az,
        n_estimators=n_estimators, max_depth=max_depth,
        pca_components=pca_components)
    p90 = float(np.percentile(errors, 90))
    print(f"    MAE = {mae:.2f}°,  Median = {median_err:.2f}°,  "
          f"P90 = {p90:.2f}°,  Max = {errors.max():.2f}°")
    return {"mae": mae, "median": median_err, "p90": p90, "max": float(errors.max())}


def main():
    output_dir = Path(__file__).parent / "diagnostic_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = output_dir / f"feature_improvement_{timestamp}.txt"
    tee = Tee(str(report_path))
    sys.stdout = tee

    results_json = {}

    print(f"Feature Improvement Tests — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Report: {report_path}\n")

    # ═══════════════════════════════════════════════════════════════════
    # Load data
    # ═══════════════════════════════════════════════════════════════════
    loader = UnderwaterDataLoader()
    n = len(loader)

    all_labels = np.array([loader._get_labels(i)["azimuth"] for i in range(n)])
    all_sessions = np.array([loader._get_labels(i)["session"] for i in range(n)])

    # Subsample every 4th frame for speed
    subsample = np.arange(0, n, 4)
    print(f"Subsampling: {len(subsample)} of {n} frames\n")

    print("Extracting features...")
    sub_feat, sub_valid, sub_runs, sub_sessions = extract_all_features(
        loader, subsample.tolist(), "extract")
    sub_labels = all_labels[sub_valid]
    sub_az_rad = np.deg2rad(sub_labels)
    print(f"Extracted {len(sub_feat)} features ({sub_feat.shape[1]} dims)\n")

    # ═══════════════════════════════════════════════════════════════════
    # Define splits
    # ═══════════════════════════════════════════════════════════════════
    j23_mask = sub_sessions == "June_23"
    j24_mask = sub_sessions == "June_24"

    # Burst-level split (both days mixed, 80/20)
    sub_bursts = np.array([loader._get_labels(i)["burst"] for i in sub_valid])
    unique_bursts = np.unique(sub_bursts)
    rng = np.random.default_rng(42)
    n_test_b = max(1, int(0.2 * len(unique_bursts)))
    test_burst_set = set(unique_bursts[rng.choice(len(unique_bursts), n_test_b, replace=False)])
    burst_train = np.array([b not in test_burst_set for b in sub_bursts])
    burst_test = ~burst_train

    splits = {
        "June23→June24": (j23_mask, j24_mask),
        "June24→June23": (j24_mask, j23_mask),
        "Burst-split":   (burst_train, burst_test),
    }

    print(f"Split sizes:")
    for name, (tr, te) in splits.items():
        print(f"  {name}: train={tr.sum()}, test={te.sum()}")
    print()

    # ═══════════════════════════════════════════════════════════════════
    # Constant baseline
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("BASELINE: Constant mean prediction")
    print("=" * 70)
    baseline_results = {}
    for split_name, (tr_mask, te_mask) in splits.items():
        train_mean = sub_az_rad[tr_mask].mean()
        pred = np.full(te_mask.sum(), train_mean)
        errors = np.abs(np.rad2deg(pred) % 360 - sub_labels[te_mask] % 360)
        errors = np.minimum(errors, 360 - errors)
        mae = float(errors.mean())
        print(f"  {split_name}: MAE = {mae:.2f}°")
        baseline_results[split_name] = mae
    results_json["baseline_mean"] = baseline_results
    print()

    # ═══════════════════════════════════════════════════════════════════
    # CONFIG A: All 468 features, standard RF
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("CONFIG A: All 468 features (current baseline)")
    print("=" * 70)
    config_a = {}
    for split_name, (tr_mask, te_mask) in splits.items():
        print(f"  {split_name}:")
        config_a[split_name] = run_config(
            "A", sub_feat[tr_mask], sub_az_rad[tr_mask],
            sub_feat[te_mask], sub_az_rad[te_mask])
    results_json["A_all_features"] = config_a
    print()

    # ═══════════════════════════════════════════════════════════════════
    # CONFIG B: AoLP-only features (183 features)
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print(f"CONFIG B: AoLP-only features ({len(AOLP_ONLY_INDICES)} features)")
    print("=" * 70)
    feat_b = sub_feat[:, AOLP_ONLY_INDICES]
    config_b = {}
    for split_name, (tr_mask, te_mask) in splits.items():
        print(f"  {split_name}:")
        config_b[split_name] = run_config(
            "B", feat_b[tr_mask], sub_az_rad[tr_mask],
            feat_b[te_mask], sub_az_rad[te_mask])
    results_json["B_aolp_only"] = config_b
    print()

    # ═══════════════════════════════════════════════════════════════════
    # CONFIG C: All features + per-run normalization
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("CONFIG C: All 468 features + per-run Z-score normalization")
    print("=" * 70)
    config_c = {}
    for split_name, (tr_mask, te_mask) in splits.items():
        print(f"  {split_name}:")
        tr_norm, stats = normalize_per_run(sub_feat[tr_mask], sub_runs[tr_mask])
        te_norm, _ = normalize_per_run(sub_feat[te_mask], sub_runs[te_mask], train_stats=stats)
        config_c[split_name] = run_config(
            "C", tr_norm, sub_az_rad[tr_mask], te_norm, sub_az_rad[te_mask])
    results_json["C_perrun_norm"] = config_c
    print()

    # ═══════════════════════════════════════════════════════════════════
    # CONFIG D: AoLP-only + per-run normalization
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print(f"CONFIG D: AoLP-only ({len(AOLP_ONLY_INDICES)}) + per-run normalization")
    print("=" * 70)
    config_d = {}
    for split_name, (tr_mask, te_mask) in splits.items():
        print(f"  {split_name}:")
        tr_aolp = sub_feat[tr_mask][:, AOLP_ONLY_INDICES]
        te_aolp = sub_feat[te_mask][:, AOLP_ONLY_INDICES]
        tr_norm, stats = normalize_per_run(tr_aolp, sub_runs[tr_mask])
        te_norm, _ = normalize_per_run(te_aolp, sub_runs[te_mask], train_stats=stats)
        config_d[split_name] = run_config(
            "D", tr_norm, sub_az_rad[tr_mask], te_norm, sub_az_rad[te_mask])
    results_json["D_aolp_perrun"] = config_d
    print()

    # ═══════════════════════════════════════════════════════════════════
    # CONFIG E: All features + per-run norm + PCA(30) + shallow RF
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("CONFIG E: All features + per-run norm + PCA(30) + shallow RF (depth=8, 100 trees)")
    print("=" * 70)
    config_e = {}
    for split_name, (tr_mask, te_mask) in splits.items():
        print(f"  {split_name}:")
        tr_norm, stats = normalize_per_run(sub_feat[tr_mask], sub_runs[tr_mask])
        te_norm, _ = normalize_per_run(sub_feat[te_mask], sub_runs[te_mask], train_stats=stats)
        config_e[split_name] = run_config(
            "E", tr_norm, sub_az_rad[tr_mask], te_norm, sub_az_rad[te_mask],
            n_estimators=100, max_depth=8, pca_components=30)
    results_json["E_pca_shallow"] = config_e
    print()

    # ═══════════════════════════════════════════════════════════════════
    # CONFIG F: AoLP+Cross features + per-run norm (richer AoLP set)
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print(f"CONFIG F: AoLP + cross-product features ({len(AOLP_PLUS_CROSS_INDICES)}) + per-run norm")
    print("=" * 70)
    config_f = {}
    for split_name, (tr_mask, te_mask) in splits.items():
        print(f"  {split_name}:")
        tr_aolp_c = sub_feat[tr_mask][:, AOLP_PLUS_CROSS_INDICES]
        te_aolp_c = sub_feat[te_mask][:, AOLP_PLUS_CROSS_INDICES]
        tr_norm, stats = normalize_per_run(tr_aolp_c, sub_runs[tr_mask])
        te_norm, _ = normalize_per_run(te_aolp_c, sub_runs[te_mask], train_stats=stats)
        config_f[split_name] = run_config(
            "F", tr_norm, sub_az_rad[tr_mask], te_norm, sub_az_rad[te_mask])
    results_json["F_aolp_cross_perrun"] = config_f
    print()

    # ═══════════════════════════════════════════════════════════════════
    # COMPARISON SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("COMPARISON SUMMARY (MAE in degrees)")
    print("=" * 70)

    configs = [
        ("Mean baseline",     baseline_results),
        ("A: All 468",        {k: v["mae"] if isinstance(v, dict) else v for k, v in config_a.items()}),
        ("B: AoLP-only",      {k: v["mae"] for k, v in config_b.items()}),
        ("C: All+RunNorm",    {k: v["mae"] for k, v in config_c.items()}),
        ("D: AoLP+RunNorm",   {k: v["mae"] for k, v in config_d.items()}),
        ("E: PCA+Shallow",    {k: v["mae"] for k, v in config_e.items()}),
        ("F: AoLP+Cross+Norm",{k: v["mae"] for k, v in config_f.items()}),
    ]

    header = f"  {'Config':<22}"
    for split_name in splits:
        header += f" {split_name:>15}"
    print(header)
    print("  " + "-" * (22 + 16 * len(splits)))

    for name, vals in configs:
        row = f"  {name:<22}"
        for split_name in splits:
            v = vals.get(split_name, float('nan'))
            row += f" {v:>14.2f}°"
        print(row)
    print()

    # Find best for cross-session
    cross_key = "June23→June24"
    best_config = min(configs[1:], key=lambda c: c[1].get(cross_key, 999))
    print(f"  Best for {cross_key}: {best_config[0]} "
          f"(MAE = {best_config[1][cross_key]:.2f}°)")

    cross_key2 = "June24→June23"
    best_config2 = min(configs[1:], key=lambda c: c[1].get(cross_key2, 999))
    print(f"  Best for {cross_key2}: {best_config2[0]} "
          f"(MAE = {best_config2[1][cross_key2]:.2f}°)")

    results_json["summary"] = {
        "best_june23_to_24": best_config[0],
        "best_june24_to_23": best_config2[0],
    }

    # Save JSON
    json_path = output_dir / f"feature_improvement_{timestamp}.json"
    with open(json_path, "w") as jf:
        json.dump(results_json, jf, indent=2)
    print(f"\n  Results saved to: {json_path}")
    print(f"  Report saved to:  {report_path}")

    tee.close()


if __name__ == "__main__":
    main()
