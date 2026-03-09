"""
Underwater Polarization Compass — Complete Model Pipeline
==========================================================

Trains and evaluates the same ML models used in Training_loops/run_all_models.py
but on the Capstone underwater polarization dataset.

Pieces wired together:
  1. UnderwaterDataLoader    – reads PNGs + solar_labels.parquet
  2. SpatialStokeDataLoader  – extracts DoLP / AoLP from raw Bayer images
  3. extract_statistical_features_from_single_image()  – 468 features
  4. CircularRegressionWrapper / EnsembleCircularModel / GradientBoostingWrapper
  5. Training dashboard visualisation

Usage:
    cd <project-root>
    python -m Underwater_testing.run_all_models
"""

import os
import sys
import gc
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Project-root imports ─────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from Training_loops.run_all_models import (
    extract_statistical_features_from_single_image,
    CircularRegressionWrapper,
    EnsembleCircularModel,
    GradientBoostingWrapper,
    calculate_sample_weights,
)
from Models.SVM_classification.svm_classification_wrapper import SVMClassificationWrapper
from Training_loops.visualization import create_training_plots
from Underwater_testing.UnderwaterDataLoader import UnderwaterDataLoader

# ── Constants ────────────────────────────────────────────────────────────
OUTPUT_ROOT = Path(__file__).parent          # Underwater_testing/
SAVED_MODELS_DIR = OUTPUT_ROOT / "saved_models"
TRAINING_PLOTS_DIR = OUTPUT_ROOT / "training_plots"


# ═════════════════════════════════════════════════════════════════════════
# Feature extraction (adapted from run_all_models.load_batch_features)
# ═════════════════════════════════════════════════════════════════════════

def load_batch_features(loader: UnderwaterDataLoader, indices, verbose=False):
    """
    Load images by index, extract 468 statistical+spatial features from
    each (DoLP, AoLP) pair, and return a (N, 468) float32 matrix.
    """
    feature_list = []
    failed_count = 0

    for idx_num, idx in enumerate(indices):
        if verbose and (idx_num + 1) % 100 == 0:
            print(f"    Processing {idx_num + 1}/{len(indices)} "
                  f"(extracted: {len(feature_list)}, failed: {failed_count})...")
            sys.stdout.flush()

        try:
            sample = loader.get_item(idx)
            if sample is not None:
                features = extract_statistical_features_from_single_image(
                    sample["features"]["dolp"],
                    sample["features"]["aolp"],
                )
                feature_list.append(features)
            else:
                failed_count += 1
        except Exception as e:
            failed_count += 1

        # Release temp arrays periodically
        if (idx_num + 1) % 50 == 0:
            gc.collect()

    print(f"    Successfully extracted {len(feature_list)} features "
          f"({failed_count} failed)")

    if len(feature_list) > 0:
        return np.array(feature_list, dtype=np.float32)
    else:
        raise ValueError("No features were successfully extracted!")


# ═════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═════════════════════════════════════════════════════════════════════════

def run_complete_pipeline():
    """Train + evaluate all models on the underwater dataset."""

    from sklearn.linear_model import Ridge
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor

    print("=" * 60)
    print("UNDERWATER POLARIZATION COMPASS — MODEL PIPELINE")
    print("=" * 60)
    print("Training L2, SVR, Random Forest, Ensemble & Gradient Boosting")
    print()

    # ── STEP 1 : Load data ───────────────────────────────────────────────
    print("STEP 1: Loading Underwater Dataset")
    print("=" * 60)

    loader = UnderwaterDataLoader()  # auto-discovers Capstone_live_data
    n_total = len(loader)

    # Collect all valid indices and their azimuth labels (degrees)
    all_indices = []
    all_labels = []

    print(f"Scanning {n_total:,} samples for labels...")
    for i in range(n_total):
        try:
            labels = loader._get_labels(i)
            all_indices.append(i)
            all_labels.append(labels["azimuth"])   # degrees, 0-360
            if (i + 1) % 2000 == 0:
                print(f"  Loaded {i + 1:,}/{n_total:,} labels...")
        except Exception:
            continue

    print(f"[OK] Loaded labels for {len(all_indices):,} samples")
    print(f"    Azimuth range: {min(all_labels):.1f}° → {max(all_labels):.1f}°")

    azimuth_deg = np.array(all_labels)
    azimuth_rad = np.deg2rad(azimuth_deg)

    # ── STEP 1b : Stratified azimuth split ───────────────────────────────
    print("\nStratified azimuth splitting (test set covers all azimuths)...")
    n_bins = 18  # 20-degree bins
    bins = np.linspace(0, 360, n_bins + 1)
    az_bin = np.clip(np.digitize(azimuth_deg, bins, right=False) - 1, 0, n_bins - 1)

    rng = np.random.default_rng(42)
    train_indices = []
    test_indices = []

    for b in range(n_bins):
        bin_idxs = [i for i, ab in enumerate(az_bin) if ab == b]
        if not bin_idxs:
            continue
        n_test = max(1, int(0.2 * len(bin_idxs)))
        test_in_bin = rng.choice(bin_idxs, size=n_test, replace=False).tolist()
        train_in_bin = [i for i in bin_idxs if i not in test_in_bin]
        test_indices.extend([all_indices[i] for i in test_in_bin])
        train_indices.extend([all_indices[i] for i in train_in_bin])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    azimuth_train_deg = np.array([all_labels[i] for i in train_indices])
    azimuth_test_deg = np.array([all_labels[i] for i in test_indices])
    azimuth_train = np.deg2rad(azimuth_train_deg)
    azimuth_test = np.deg2rad(azimuth_test_deg)

    train_size = len(train_indices)
    test_size = len(test_indices)
    print(f"[OK] Train: {train_size:,} samples")
    print(f"[OK] Test:  {test_size:,} samples")
    print(f"    Train azimuth: {azimuth_train_deg.min():.1f}° → {azimuth_train_deg.max():.1f}°")
    print(f"    Test  azimuth: {azimuth_test_deg.min():.1f}° → {azimuth_test_deg.max():.1f}°")

    # ── STEP 1c : Extract features ──────────────────────────────────────
    print("\nExtracting training features (468 per sample)...")
    all_train_features = load_batch_features(loader, train_indices, verbose=True)
    print(f"[OK] Training features: {all_train_features.shape}")

    print("Extracting test features...")
    all_test_features = load_batch_features(loader, test_indices, verbose=True)
    print(f"[OK] Test features: {all_test_features.shape}\n")

    # ── STEP 2 : Define models ──────────────────────────────────────────
    print("STEP 2: Training Models")
    print("=" * 60)

    models = {
        "L2_PCA": CircularRegressionWrapper(
            Ridge,
            use_pca=True,
            n_components=50,
            alpha=0.1,
        ),
        "SVR_Circular": CircularRegressionWrapper(
            SVR,
            C=50.0,
            gamma="scale",
            epsilon=0.01,
            kernel="rbf",
        ),
        "SVM_Classification_Coarse": SVMClassificationWrapper(
            n_bins=8,
            C=10.0,
            gamma="scale",
            probability=True,
            feature_selection=150,
            class_weight="balanced"
        ),
        "SVM_Classification_Fine": SVMClassificationWrapper(
            n_bins=16,
            C=50.0,
            gamma="scale",
            probability=True,
            feature_selection=100,
            class_weight="balanced"
        ),
        "SVM_Classification_Ultra": SVMClassificationWrapper(
            n_bins=32,
            C=100.0,
            gamma="auto", 
            probability=True,
            feature_selection=80,
            class_weight="balanced"
        ),
        "RF_Enhanced": CircularRegressionWrapper(
            RandomForestRegressor,
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
        ),
        "Ensemble": EnsembleCircularModel([
            ("SVR", SVR, {"C": 50.0, "gamma": "scale", "epsilon": 0.01, "kernel": "rbf"}),
            ("RF", RandomForestRegressor, {"n_estimators": 300, "max_depth": 15,
                                           "random_state": 42, "n_jobs": -1}),
            ("Ridge", Ridge, {"alpha": 0.1}),
        ]),
        "Gradient_Boosting": GradientBoostingWrapper(
            n_estimators=1000,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.9,
            min_samples_split=3,
            min_samples_leaf=1,
        ),
    }

    # Sample weights for balanced training
    print("\nCalculating sample weights for balanced training...")
    train_weights = calculate_sample_weights(azimuth_train_deg)
    print(f"[OK] Weight range: {train_weights.min():.3f} – {train_weights.max():.3f}")

    today = datetime.now().strftime("%Y-%m-%d")
    results = {}
    training_history = {}

    # ── STEP 2b : Train each model ──────────────────────────────────────
    for model_name, model in models.items():
        print(f"\n{'─'*50}")
        print(f"Training {model_name}...")
        try:
            # ── Learning curve (skip for Ensemble) ──
            if "Ensemble" in model_name:
                print("  Skipping learning curve for ensemble model...")
                best_val_error = 0.0
                best_val_rmse = 0.0
                best_sample_size = train_size
                sample_sizes = [train_size]
                train_errors = []
                val_errors = []
            else:
                print("  Generating learning curve...")
                if train_size >= 200:
                    sample_sizes = [50, 100, 200, train_size]
                else:
                    sample_sizes = [min(50, train_size), train_size]
                sample_sizes = sorted(set(sample_sizes))

                train_errors = []
                val_errors = []
                best_val_error = float("inf")
                best_val_rmse = float("inf")
                best_sample_size = 0

                for size in sample_sizes:
                    # Create a temporary model of the right type
                    if "L2" in model_name or "PCA" in model_name:
                        model_temp = CircularRegressionWrapper(
                            Ridge, use_pca="PCA" in model_name, n_components=50, alpha=0.1)
                    elif "SVR" in model_name:
                        model_temp = CircularRegressionWrapper(
                            SVR, C=50.0, gamma="scale", epsilon=0.01, kernel="rbf")
                    elif "Gradient" in model_name:
                        model_temp = GradientBoostingWrapper(
                            n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.9)
                    else:  # RF
                        model_temp = CircularRegressionWrapper(
                            RandomForestRegressor, n_estimators=200, max_depth=15,
                            min_samples_split=3, min_samples_leaf=1,
                            random_state=42, n_jobs=-1)

                    subset_train_size = int(size * 0.8)
                    subset_val_size = size - subset_train_size
                    subset_idx = np.random.choice(
                        len(all_train_features), subset_train_size, replace=False)
                    train_feat = all_train_features[subset_idx]
                    subset_wt = train_weights[subset_idx]

                    metrics_t = model_temp.fit_from_features(
                        train_feat, azimuth_train[subset_idx], sample_weight=subset_wt)

                    if subset_val_size > 0:
                        remaining = np.setdiff1d(np.arange(len(all_train_features)), subset_idx)
                        val_idx = np.random.choice(remaining, subset_val_size, replace=False)
                        vp = model_temp.predict_from_features(all_train_features[val_idx])
                        val_error = np.rad2deg(np.mean(np.abs(vp - azimuth_train[val_idx])))
                        val_rmse = np.rad2deg(np.sqrt(np.mean((vp - azimuth_train[val_idx])**2)))
                    else:
                        val_error = metrics_t["mae"]
                        val_rmse = metrics_t["rmse"]

                    train_errors.append(metrics_t["mae"])
                    val_errors.append(val_error)

                    if val_error < best_val_error:
                        best_val_error = val_error
                        best_val_rmse = val_rmse
                        best_sample_size = size

            print(f"  Best val MAE: {best_val_error:.3f}° at {best_sample_size} samples")

            training_history[model_name] = {
                "sample_sizes": sample_sizes,
                "train_errors": train_errors,
                "val_errors": val_errors,
                "best_val_error": best_val_error,
                "best_val_rmse": best_val_rmse,
                "best_sample_size": best_sample_size,
            }

            # ── Train final model on full training set ──
            print(f"  Training final model on full set ({train_size:,} samples)...")
            train_metrics = model.fit_from_features(
                all_train_features, azimuth_train, sample_weight=train_weights)

            # ── Evaluate on held-out test set ──
            print(f"  Evaluating on test set ({test_size:,} samples)...")
            test_pred = model.predict_from_features(all_test_features)
            test_mae = np.rad2deg(np.mean(np.abs(test_pred - azimuth_test)))
            test_rmse = np.rad2deg(np.sqrt(np.mean((test_pred - azimuth_test) ** 2)))

            # ── Cross-validation ──
            cv_metrics = model.cross_validate_from_features(
                all_train_features, azimuth_train, cv_folds=5)

            cv_rmse = cv_metrics.get("rmse_mean", 0.0)
            cv_std = cv_metrics.get("mae_std", 0.0)

            results[model_name] = {
                "training_mae": float(train_metrics["mae"]),
                "cv_mae": float(cv_metrics["mae_mean"]),
                "cv_mae_std": float(cv_std),
                "cv_rmse": float(cv_rmse),
                "best_val_mae": float(best_val_error),
                "best_val_rmse": float(best_val_rmse),
                "best_val_samples": int(best_sample_size),
                "test_mae": float(test_mae),
                "test_rmse": float(test_rmse),
                "meets_requirements": bool(test_mae < 5.0),
            }

            print(f"  {model_name}")
            print(f"    Train MAE: {train_metrics['mae']:.3f}°")
            print(f"    CV MAE:    {cv_metrics['mae_mean']:.3f}°")
            print(f"    Test MAE:  {test_mae:.3f}°  "
                  f"({'PASS' if test_mae < 5.0 else 'FAIL'})")

            # ── Save model ──
            model_dir = SAVED_MODELS_DIR / today
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{model_name}_final.pkl"

            if hasattr(model, "save"):
                model.save(str(model_path))
            else:
                import joblib
                joblib.dump(model, str(model_path))
            print(f"    Saved → {model_path}")

        except Exception as e:
            print(f"  {model_name} FAILED: {e}")
            results[model_name] = {"error": str(e)}
            import traceback
            traceback.print_exc()

    # ── STEP 3 : Save results & plots ────────────────────────────────────
    results_dir = TRAINING_PLOTS_DIR / today
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "complete_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY (Underwater Dataset)")
    print(f"{'='*60}")
    for name, result in results.items():
        if "error" not in result:
            status = "PASS" if result["meets_requirements"] else "FAIL"
            print(f"  {name}:")
            print(f"    CV MAE:   {result['cv_mae']:.3f}°")
            print(f"    Test MAE: {result['test_mae']:.3f}° — {status}")

    print(f"\nGenerating training plots...")
    create_training_plots(results, training_history, str(results_dir))
    print(f"\nResults and plots saved to: {results_dir}")

    return results


if __name__ == "__main__":
    run_complete_pipeline()
