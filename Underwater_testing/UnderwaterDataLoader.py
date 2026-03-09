"""
Underwater Polarization Data Loader

Loads polarization images from the Capstone_live_data directory structure,
extracts DoLP/AoLP features using SpatialStokeDataLoader, and pairs them
with solar azimuth/elevation labels from solar_labels.parquet.

Directory structure expected:
    Capstone_live_data/
    ├── solar_labels.parquet
    ├── June_23/
    │   └── run_*/burst_*/*.png
    └── June_24/
        └── run_*/burst_*/*.png
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import cv2

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from Bens_Data_Import.Image_data_loaders.Spatial_Stoke_Loader.SpatialStokeLoader import SpatialStokeDataLoader


class UnderwaterDataLoader:
    """
    Data loader for the Capstone underwater polarization dataset.

    Mirrors the PolarizationDataLoader interface so it can be used
    as a drop-in replacement in the training pipeline.

    Key methods:
        __len__()              – total number of labelled frames
        get_item(index)        – returns dict with 'features', 'label', 'metadata'
        get_label_statistics() – summary stats of solar azimuth labels
    """

    def __init__(self, data_root: Optional[Path] = None, extractor_class=None,
                 max_samples: Optional[int] = None):
        """
        Parameters
        ----------
        data_root : Path, optional
            Root of the Capstone_live_data folder.  Defaults to
            ``<project>/Capstone_live_data``.
        extractor_class : class, optional
            Polarization feature extractor class (must accept an image array
            and expose ``.get_item()``).  Defaults to SpatialStokeDataLoader.
        max_samples : int, optional
            Cap the dataset size (useful for quick debugging).
        """
        if data_root is None:
            data_root = Path(__file__).parent.parent / "Capstone_live_data"
        self.data_root = Path(data_root)

        self.extractor_class = extractor_class or SpatialStokeDataLoader

        # Load the ground-truth label manifest
        parquet_path = self.data_root / "solar_labels.parquet"
        csv_path = self.data_root / "solar_labels.csv"
        
        if parquet_path.exists():
            self.labels_df = pd.read_parquet(parquet_path)
            print(f"[OK] Loaded parquet: {parquet_path}")
        elif csv_path.exists():
            self.labels_df = pd.read_csv(csv_path, parse_dates=['timestamp'])
            print(f"[OK] Loaded CSV: {csv_path}")
        else:
            raise FileNotFoundError(
                f"No manifest found at {parquet_path} or {csv_path}.\n"
                "Run  generate_day1_day2_labels.py  first."
            )

        # Validate that image files referenced in the manifest exist
        # (spot-check the first and last to avoid scanning 8k files)
        first_path = self.data_root / self.labels_df.iloc[0]["image_path"]
        if not first_path.exists():
            raise FileNotFoundError(
                f"Image referenced in manifest not found: {first_path}\n"
                "Check that the Capstone_live_data image folders are present."
            )

        if max_samples is not None:
            self.labels_df = self.labels_df.iloc[:max_samples].reset_index(drop=True)

        print(f"[OK] UnderwaterDataLoader initialized")
        print(f"  Data root:  {self.data_root}")
        print(f"  Samples:    {len(self.labels_df):,}")
        print(f"  Sessions:   {self.labels_df['session'].nunique()}")
        print(f"  Runs:       {self.labels_df['run'].nunique()}")
        print(f"  Bursts:     {self.labels_df['burst'].nunique()}")
        az = self.labels_df["solar_azimuth"]
        print(f"  Azimuth:    {az.min():.1f}° – {az.max():.1f}° (mean {az.mean():.1f}°)")

    # ------------------------------------------------------------------
    # Core interface (mirrors PolarizationDataLoader)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.labels_df)

    def _get_image_path(self, index: int) -> Path:
        rel = self.labels_df.iloc[index]["image_path"]
        return self.data_root / rel

    def _get_labels(self, index: int) -> Dict:
        row = self.labels_df.iloc[index]
        return {
            "azimuth": float(row["solar_azimuth"]),       # degrees 0-360
            "elevation": float(row["solar_elevation"]),    # degrees above horizon
            "timestamp": row["timestamp_utc"],
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "session": row["session"],
            "run": row["run"],
            "burst": row["burst"],
            "burst_number": int(row["burst_number"]),
            "frame_number": int(row["frame_number"]),
        }

    def _extract_features(self, image_path: Path,
                          gains: Optional[Dict] = None,
                          global_frame_offset: Optional[float] = None
                          ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load a raw polarization PNG and extract DoLP / AoLP via
        SpatialStokeDataLoader.

        Returns dict  {'dolp': ndarray, 'aolp': ndarray}  or None on failure.
        """
        try:
            img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARN] Failed to load image: {image_path.name}")
                return None

            extractor = self.extractor_class(img)
            result = extractor.get_item(gains=gains, global_frame_offset=global_frame_offset)

            # SpatialStokeDataLoader.get_item returns (dict, I0, I45, I90, I135, S0)
            if isinstance(result, tuple):
                features = result[0]
            else:
                features = result

            if features is None or "dolp" not in features or "aolp" not in features:
                print(f"  [WARN] Extractor returned invalid features for {image_path.name}")
                return None

            return {"dolp": features["dolp"], "aolp": features["aolp"]}

        except Exception as e:
            print(f"  [WARN] Feature extraction failed for {image_path.name}: {e}")
            return None

    def get_item(self, index: int, gains: Optional[Dict] = None,
                 global_frame_offset: Optional[float] = None) -> Optional[Dict]:
        """
        Return a single sample compatible with the training pipeline.

        Returns
        -------
        dict with keys:
            index      – integer index
            image_path – absolute path string
            features   – {'dolp': ndarray, 'aolp': ndarray}
            label      – solar azimuth in degrees (0-360)
            metadata   – dict of additional info
        or None if extraction fails.
        """
        image_path = self._get_image_path(index)
        features = self._extract_features(image_path, gains=gains,
                                          global_frame_offset=global_frame_offset)
        if features is None:
            return None

        labels = self._get_labels(index)

        return {
            "index": index,
            "image_path": str(image_path),
            "features": features,
            "label": labels["azimuth"],
            "metadata": {
                "timestamp": labels["timestamp"],
                "latitude": labels["latitude"],
                "longitude": labels["longitude"],
                "elevation": labels["elevation"],
                "session": labels["session"],
                "run": labels["run"],
                "burst": labels["burst"],
                "burst_number": labels["burst_number"],
                "frame_number": labels["frame_number"],
            },
        }

    def get_batch(self, indices: List[int]) -> List[Dict]:
        batch = []
        for idx in indices:
            sample = self.get_item(idx)
            if sample is not None:
                batch.append(sample)
        return batch

    # ------------------------------------------------------------------
    # Statistics & splitting helpers
    # ------------------------------------------------------------------

    def get_label_statistics(self) -> Dict:
        az = self.labels_df["solar_azimuth"]
        el = self.labels_df["solar_elevation"]
        return {
            "azimuth": {
                "min": float(az.min()),
                "max": float(az.max()),
                "mean": float(az.mean()),
                "std": float(az.std()),
            },
            "elevation": {
                "min": float(el.min()),
                "max": float(el.max()),
                "mean": float(el.mean()),
                "std": float(el.std()),
            },
            "n_samples": len(self.labels_df),
            "n_sessions": int(self.labels_df["session"].nunique()),
            "n_runs": int(self.labels_df["run"].nunique()),
            "n_bursts": int(self.labels_df["burst"].nunique()),
        }

    def split_train_val_test(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        random_seed: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        n = len(self)
        indices = np.arange(n)
        if shuffle:
            np.random.RandomState(random_seed).shuffle(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return (
            indices[:n_train].tolist(),
            indices[n_train : n_train + n_val].tolist(),
            indices[n_train + n_val :].tolist(),
        )


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------
if __name__ == "__main__":
    loader = UnderwaterDataLoader()
    print(f"\nDataset size: {len(loader)} samples")

    stats = loader.get_label_statistics()
    print(f"Azimuth:   [{stats['azimuth']['min']:.1f}°, {stats['azimuth']['max']:.1f}°]  "
          f"mean={stats['azimuth']['mean']:.1f}°  std={stats['azimuth']['std']:.1f}°")
    print(f"Elevation: [{stats['elevation']['min']:.1f}°, {stats['elevation']['max']:.1f}°]")

    sample = loader.get_item(0)
    if sample:
        print(f"\nSample 0:")
        print(f"  Image:  {sample['image_path']}")
        print(f"  DoLP shape: {sample['features']['dolp'].shape}")
        print(f"  AoLP shape: {sample['features']['aolp'].shape}")
        print(f"  Label (azimuth): {sample['label']:.2f}°")
        print(f"  Session: {sample['metadata']['session']}")
    else:
        print("\n[WARN] Sample 0 failed to load")

    train, val, test = loader.split_train_val_test()
    print(f"\nSplit: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    print("\nSelf-test passed")
