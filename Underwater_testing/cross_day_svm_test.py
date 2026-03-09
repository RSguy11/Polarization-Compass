"""
Timelapse Data Loader for Cross-Day SVM Testing
==============================================

Custom data loader for the 2-8-2026 timelapse images to test cross-day generalization.
Loads data from specific Day 1 and Day 2 directories and extracts features
for training and testing SVM models.

Structure expected:
- Day 1: C:\\Timelapse\\2-8-2026 Images\\Day 1\\
- Day 2: C:\\Timelapse\\2-8-2026 Images\\Day 2\\
- Each day: run_*/burst_*/frame*.png + burst_metadata.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import cv2
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from Models.SVM_classification.svm_classification_wrapper import (
    CircularSVMClassifier,
    SVMClassificationWrapper, 
    calculate_circular_error
)
from Training_loops.run_all_models import extract_statistical_features_from_single_image


class TimelapseDataLoader:
    """
    Data loader for timelapse polarization images with azimuth labels.
    """
    
    def __init__(self, day1_path, day2_path):
        self.day1_path = Path(day1_path)
        self.day2_path = Path(day2_path) 
        
        self.day1_samples = []
        self.day2_samples = []
        
        print("Loading Day 1 data...")
        self._load_day_data(self.day1_path, self.day1_samples)
        
        print("Loading Day 2 data...")
        self._load_day_data(self.day2_path, self.day2_samples)
        
        print(f"Loaded Day 1: {len(self.day1_samples)} samples")
        print(f"Loaded Day 2: {len(self.day2_samples)} samples")
    
    def _load_day_data(self, day_path, sample_list):
        """Load all valid samples from a day's directory."""
        if not day_path.exists():
            raise FileNotFoundError(f"Day path not found: {day_path}")
        
        run_dirs = sorted([d for d in day_path.iterdir() if d.is_dir() and d.name.startswith('run_')])
        
        for run_dir in run_dirs:
            print(f"  Processing {run_dir.name}...")
            
            # Load burst metadata if it exists
            metadata_file = run_dir / "burst_metadata.csv"
            burst_metadata = {}
            
            if metadata_file.exists():
                try:
                    df = pd.read_csv(metadata_file)
                    for _, row in df.iterrows():
                        burst_name = row.get('burst_name', '')
                        azimuth = row.get('solar_azimuth_deg', row.get('azimuth', None))
                        if azimuth is not None:
                            burst_metadata[burst_name] = float(azimuth)
                except Exception as e:
                    print(f"    Warning: Could not load metadata from {metadata_file}: {e}")
            
            # Process each burst directory
            burst_dirs = sorted([d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith('burst_')])
            
            for burst_dir in burst_dirs:
                burst_name = burst_dir.name
                
                # Get azimuth from metadata
                azimuth = burst_metadata.get(burst_name, None)
                if azimuth is None:
                    # Try to extract from csv with different column names
                    if metadata_file.exists():
                        try:
                            df = pd.read_csv(metadata_file)
                            # Look for rows matching this burst
                            burst_rows = df[df['burst_name'].str.contains(burst_name.replace('burst_', ''), na=False)]
                            if len(burst_rows) > 0:
                                azimuth = burst_rows.iloc[0].get('solar_azimuth_deg', 
                                         burst_rows.iloc[0].get('azimuth', None))
                        except:
                            pass
                
                if azimuth is None:
                    print(f"    Warning: No azimuth found for {burst_name}")
                    continue
                
                # Find image files in burst
                image_files = sorted(list(burst_dir.glob("*.png")))
                
                if len(image_files) == 0:
                    continue
                
                # Take middle frame from burst for stability
                mid_idx = len(image_files) // 2
                image_path = image_files[mid_idx]
                
                sample_list.append({
                    'image_path': str(image_path),
                    'azimuth': azimuth,
                    'burst_name': burst_name,
                    'run_name': run_dir.name,
                    'day_path': str(day_path)
                })
    
    def extract_features_from_image(self, image_path):
        """Extract polarization features from a single image."""
        try:
            # Load image
            img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Check if image is right size (Bayer pattern polarization camera)
            if img.shape != (2048, 2448):
                print(f"Warning: Unexpected image shape {img.shape} for {image_path}")
            
            # Create mock data loader for feature extraction
            class MockDataLoader:
                def extract_features(self, msg, gains=None, enhance_processing=True):
                    """Extract features using the spatial stoke method."""
                    H, W = img.shape[:2]
                    
                    # Demosaic polarization angles from Bayer-like pattern
                    A = img[0::2, 0::2].astype(np.float32)  # 0°
                    B = img[0::2, 1::2].astype(np.float32)  # 45°  
                    C = img[1::2, 0::2].astype(np.float32)  # 135°
                    D = img[1::2, 1::2].astype(np.float32)  # 90°
                    
                    I0, I45, I90, I135 = A, B, D, C
                    
                    # Calculate Stokes parameters
                    S0 = I0 + I90
                    S1 = I0 - I90  
                    S2 = I135 - I45
                    
                    eps = 1e-6
                    denom = np.clip(S0, eps, None)
                    
                    DoLP = np.sqrt(S1**2 + S2**2) / denom
                    AoLP = 0.5 * np.arctan2(S2, S1)
                    AoLP = np.rad2deg(AoLP)
                    AoLP = (AoLP + 90) % 180 - 90
                    
                    return DoLP, AoLP, I0, I45, I90, I135, S0
            
            # Create mock message
            class MockMessage:
                def __init__(self, data):
                    self.data = data.tobytes()
            
            mock_loader = MockDataLoader()
            msg = MockMessage(img)
            
            # Extract DoLP and AoLP
            dolp, aolp, _, _, _, _, _ = mock_loader.extract_features(msg)
            
            # Extract statistical features
            features = extract_statistical_features_from_single_image(dolp, aolp)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None
    
    def get_day_data(self, day_samples, max_samples=None):
        """Extract features and labels for a day's worth of data."""
        if max_samples is not None:
            day_samples = day_samples[:max_samples]
        
        features = []
        labels = []
        failed = 0
        
        print(f"Extracting features from {len(day_samples)} samples...")
        
        for i, sample in enumerate(day_samples):
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{len(day_samples)} (failed: {failed})")
            
            feat = self.extract_features_from_image(sample['image_path'])
            if feat is not None:
                features.append(feat)
                labels.append(sample['azimuth'])
            else:
                failed += 1
        
        print(f"  Successfully extracted {len(features)} features ({failed} failed)")
        
        if len(features) == 0:
            raise ValueError("No features were successfully extracted!")
        
        return np.array(features, dtype=np.float32), np.array(labels)


def run_cross_day_svm_test():
    """Test SVM classification with cross-day data (Train Day 2, Test Day 1)."""
    print("CROSS-DAY SVM CLASSIFICATION TEST")
    print("=" * 60)
    print("Training on Day 2, Testing on Day 1")
    print("=" * 60)
    
    # Data paths
    day1_path = "C:\\Timelapse\\2-8-2026 Images\\Day 1"
    day2_path = "C:\\Timelapse\\2-8-2026 Images\\Day 2"
    
    # Load data
    loader = TimelapseDataLoader(day1_path, day2_path)
    
    # Extract features (limit samples for speed)
    max_samples_per_day = 200  # Adjust based on processing time
    
    print(f"\\nExtracting Day 2 features (training)...")
    X_train, y_train = loader.get_day_data(loader.day2_samples, max_samples_per_day)
    
    print(f"\\nExtracting Day 1 features (testing)...")  
    X_test, y_test = loader.get_day_data(loader.day1_samples, max_samples_per_day)
    
    print(f"\\nDataset Summary:")
    print(f"  Training (Day 2): {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Testing (Day 1):  {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"  Train azimuth range: {y_train.min():.1f}° - {y_train.max():.1f}°")
    print(f"  Test  azimuth range: {y_test.min():.1f}° - {y_test.max():.1f}°")
    
    # Test SVM configurations
    svm_configs = [
        {"name": "SVM_8bins", "n_bins": 8, "C": 10, "feature_selection": 150},
        {"name": "SVM_16bins", "n_bins": 16, "C": 50, "feature_selection": 100},
        {"name": "SVM_24bins", "n_bins": 24, "C": 100, "feature_selection": 80},
    ]
    
    results = {}
    
    print(f"\\n" + "=" * 60)
    print("SVM TRAINING AND EVALUATION")
    print("=" * 60)
    
    for config in svm_configs:
        print(f"\\nTesting {config['name']}...")
        print(f"  Config: {config}")
        
        try:
            # Train SVM
            svm = CircularSVMClassifier(
                n_bins=config["n_bins"],
                C=config["C"], 
                gamma="scale",
                probability=True,
                feature_selection=config["feature_selection"],
                class_weight="balanced"
            )
            
            print("  Training...")
            svm.fit(X_train, y_train)
            
            print("  Predicting...")
            pred_train = svm.predict(X_train)
            pred_test = svm.predict(X_test)
            
            # Calculate errors
            train_mae, train_rmse = calculate_circular_error(y_train, pred_train)
            test_mae, test_rmse = calculate_circular_error(y_test, pred_test)
            
            results[config["name"]] = {
                "train_mae": train_mae,
                "test_mae": test_mae,
                "cross_day_gap": test_mae - train_mae,
                "config": config
            }
            
            print(f"  Train MAE: {train_mae:.2f}°")
            print(f"  Test MAE:  {test_mae:.2f}° (cross-day)")
            print(f"  Generalization gap: {test_mae - train_mae:.2f}°")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results[config["name"]] = {"error": str(e)}
    
    # Show results summary
    print(f"\\n" + "=" * 60)
    print("CROSS-DAY RESULTS SUMMARY") 
    print("=" * 60)
    
    print(f"{'SVM Config':<15} {'Train MAE':<10} {'Test MAE':<10} {'Cross-Day Gap':<15}")
    print("-" * 55)
    
    best_model = None
    best_test_mae = float('inf')
    
    for name, result in results.items():
        if "error" not in result:
            train_mae = result["train_mae"]
            test_mae = result["test_mae"] 
            gap = result["cross_day_gap"]
            
            print(f"{name:<15} {train_mae:<10.2f} {test_mae:<10.2f} {gap:<15.2f}")
            
            if test_mae < best_test_mae:
                best_test_mae = test_mae
                best_model = name
    
    # Performance analysis
    print(f"\\nPERFORMANCE ANALYSIS:")
    print("-" * 30)
    
    baseline_mae = 90  # Random baseline
    current_problem_mae = 30  # Your current cross-day problem
    
    if best_model and best_test_mae < float('inf'):
        improvement_vs_baseline = (baseline_mae - best_test_mae) / baseline_mae 
        improvement_vs_current = (current_problem_mae - best_test_mae) / current_problem_mae
        
        print(f"Best model: {best_model}")
        print(f"Cross-day test MAE: {best_test_mae:.2f}°")
        print(f"Improvement vs random baseline: {improvement_vs_baseline*100:.1f}%")
        print(f"Improvement vs current problem: {improvement_vs_current*100:.1f}%")
        
        if best_test_mae < 20:
            print("\\n✓ EXCELLENT: Significant improvement in cross-day generalization!")
        elif best_test_mae < 30:  
            print("\\n✓ GOOD: Noticeable improvement over current methods")
        else:
            print("\\n⚠ MODERATE: Some improvement but more work needed")
    
    print(f"\\nCross-day SVM evaluation complete!")
    return results


if __name__ == "__main__":
    run_cross_day_svm_test()