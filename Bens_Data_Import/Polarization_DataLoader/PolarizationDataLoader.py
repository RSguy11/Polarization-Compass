import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import sys
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class PolarizationDataLoader:
    
    def __init__(self, rmc_folder: Path, extractor_class=None, cache_labels: bool = True):
        self.rmc_folder = Path(rmc_folder)
        self.extractor_class = extractor_class
        
        # Define data paths within rmc folder
        self.image_folder = self.rmc_folder / "camera_driver_gv_vis_image_raw"
        self.gps_csv_path = self.rmc_folder / "novatel_oem7_inspva" / "novatel_oem7_inspva.csv"
        
        # Validate paths
        if not self.image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")
        if not self.gps_csv_path.exists():
            raise FileNotFoundError(f"GPS CSV not found: {self.gps_csv_path}")
        
        # Load GPS/INS labels
        print(f"Loading GPS/INS labels from {self.gps_csv_path.name}")
        self.labels_df = pd.read_csv(self.gps_csv_path)
        self.labels_df['timestamp'] = pd.to_datetime(self.labels_df['timestamp'])
        
        # Get sorted list of images
        self.image_files = sorted(self.image_folder.glob("camera_driver_gv_vis_image_raw_*.png"))
        
        # Initialize extractor (lazy initialization - only when first needed)
        self._extractor = None
        
        print(f"[OK] PolarizationDataLoader initialized")
        print(f"  Images: {len(self.image_files)}")
        print(f"  Labels: {len(self.labels_df)}")
        print(f"  Data range: {self.labels_df['timestamp'].min()} to {self.labels_df['timestamp'].max()}")
    
    @property
    def extractor(self):
        """Lazy initialization of feature extractor - not needed anymore since we process per-image"""
        # The new SpatialStokeDataLoader is initialized per-image, not per-directory
        if self._extractor is None:
            if self.extractor_class is None:
                from Bens_Data_Import.Image_data_loaders.Spatial_Stoke_Loader.SpatialStokeLoader import SpatialStokeDataLoader
                self._extractor = SpatialStokeDataLoader  # Store the class, not instance
                print(f"  Using default extractor: SpatialStokeDataLoader")
            else:
                self._extractor = self.extractor_class
                print(f"  Using custom extractor: {self.extractor_class.__name__}")
        return self._extractor
    
    def __len__(self) -> int:
        return min(len(self.image_files), len(self.labels_df))
    
    def _get_image_index_from_filename(self, image_path: Path) -> int:
        return int(image_path.stem.split('_')[-1])
    
    def _get_image_path(self, index: int) -> Path:
        if index < 0 or index >= len(self.image_files):
            raise IndexError(f"Index {index} out of range [0, {len(self.image_files)})")
        return self.image_files[index]
    
    def _apply_orientation_correction(self, azimuth: float, roll: float, pitch: float) -> float:
        # Customize this to apply roll/pitch corrections to azimuth based on camera mounting
        corrected_azimuth = azimuth
        # Example: corrected_azimuth = azimuth + roll_correction + pitch_correction
        return corrected_azimuth
    
    def _get_labels(self, index: int) -> Dict[str, float]:
        if index < 0 or index >= len(self.labels_df):
            raise IndexError(f"Index {index} out of range [0, {len(self.labels_df)})")
        
        row = self.labels_df.iloc[index]
        
        roll = float(row['roll'])
        pitch = float(row['pitch'])
        azimuth = float(row['azimuth'])
        
        corrected_azimuth = self._apply_orientation_correction(azimuth, roll, pitch)
        
        return {
            'azimuth': corrected_azimuth,
            'raw_roll': roll,
            'raw_pitch': pitch,
            'raw_azimuth': azimuth,
            'timestamp': row['timestamp'],
            'latitude': float(row['latitude']),
            'longitude': float(row['longitude']),
            'north_velocity': float(row['north_velocity']),
            'east_velocity': float(row['east_velocity']),
            'up_velocity': float(row['up_velocity'])
        }
    
    def _extract_features(self, image_path: Path, gains: Optional[Dict] = None, 
                         global_frame_offset: Optional[float] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract polarization features from an image using the updated SpatialStokeDataLoader.
        
        Parameters:
        -----------
        image_path : Path
            Path to the image file
        gains : dict, optional
            Calibration gains for each polarization angle
        global_frame_offset : float, optional
            Angle offset for solar principal plane transformation
            
        Returns:
        --------
        dict with 'aolp' and 'dolp' arrays, or None if extraction fails
        """
        try:
            # Load the image
            img = cv2.imread(str(image_path), 0)
            if img is None:
                print(f"Warning: Failed to load image {image_path.name}")
                return None
            
            # Create extractor instance for this specific image
            extractor_class = self.extractor
            extractor = extractor_class(img)
            
            # Get features with gains and global_frame_offset parameters
            # SpatialStokeDataLoader.get_item() returns (dict, I0, I45, I90, I135, S0)
            result = extractor.get_item(gains=gains, global_frame_offset=global_frame_offset)
            
            # Extract just the dict with aolp and dolp (first element of tuple)
            if isinstance(result, tuple):
                features = result[0]
            else:
                features = result
            
            if features is None or 'aolp' not in features or 'dolp' not in features:
                print(f"Warning: Extractor returned invalid features for {image_path.name}")
                return None
            
            return {
                'aolp': features['aolp'],
                'dolp': features['dolp']
            }
            
        except Exception as e:
            print(f"Warning: Feature extraction failed for {image_path.name}: {e}")
            return None
    
    def get_item(self, index: int, gains: Optional[Dict] = None, 
                 global_frame_offset: Optional[float] = None) -> Optional[Dict]:
        """
        Get a single data sample with polarization features and labels.
        
        Parameters:
        -----------
        index : int
            Index of the sample to retrieve
        gains : dict, optional
            Calibration gains for polarization channels
        global_frame_offset : float, optional
            Angle offset for solar principal plane transformation
            
        Returns:
        --------
        dict containing features, labels, and metadata, or None if failed
        """
        image_path = self._get_image_path(index)
        
        features = self._extract_features(image_path, gains=gains, 
                                         global_frame_offset=global_frame_offset)
        if features is None:
            return None
        
        labels = self._get_labels(index)
        
        sample = {
            'index': index,
            'image_path': str(image_path),
            'features': features,
            'label': labels['azimuth'],
            'metadata': {
                'timestamp': labels['timestamp'],
                'latitude': labels['latitude'],
                'longitude': labels['longitude'],
                'velocities': {
                    'north': labels['north_velocity'],
                    'east': labels['east_velocity'],
                    'up': labels['up_velocity']
                },
                'raw_roll': labels['raw_roll'],
                'raw_pitch': labels['raw_pitch'],
                'raw_azimuth': labels['raw_azimuth']
            }
        }
        
        return sample
    
    def get_batch(self, indices: List[int]) -> List[Dict]:
        batch = []
        for idx in indices:
            sample = self.get_item(idx)
            if sample is not None:
                batch.append(sample)
        return batch
    
    def get_label_statistics(self) -> Dict:
        corrected_azimuths = []
        for i in range(len(self.labels_df)):
            row = self.labels_df.iloc[i]
            corrected_az = self._apply_orientation_correction(
                float(row['azimuth']),
                float(row['roll']),
                float(row['pitch'])
            )
            corrected_azimuths.append(corrected_az)
        
        corrected_azimuths = np.array(corrected_azimuths)
        
        stats = {
            'azimuth': {
                'min': float(np.min(corrected_azimuths)),
                'max': float(np.max(corrected_azimuths)),
                'mean': float(np.mean(corrected_azimuths)),
                'std': float(np.std(corrected_azimuths))
            },
            'raw_stats': {
                'roll': {
                    'min': float(self.labels_df['roll'].min()),
                    'max': float(self.labels_df['roll'].max()),
                    'mean': float(self.labels_df['roll'].mean()),
                    'std': float(self.labels_df['roll'].std())
                },
                'pitch': {
                    'min': float(self.labels_df['pitch'].min()),
                    'max': float(self.labels_df['pitch'].max()),
                    'mean': float(self.labels_df['pitch'].mean()),
                    'std': float(self.labels_df['pitch'].std())
                },
                'azimuth': {
                    'min': float(self.labels_df['azimuth'].min()),
                    'max': float(self.labels_df['azimuth'].max()),
                    'mean': float(self.labels_df['azimuth'].mean()),
                    'std': float(self.labels_df['azimuth'].std())
                }
            }
        }
        return stats
    
    def split_train_val_test(self, 
                             train_ratio: float = 0.7,
                             val_ratio: float = 0.15,
                             test_ratio: float = 0.15,
                             shuffle: bool = True,
                             random_seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        n_samples = len(self)
        indices = np.arange(n_samples)
        
        if shuffle:
            rng = np.random.RandomState(random_seed)
            rng.shuffle(indices)
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()
        
        return train_indices, val_indices, test_indices


if __name__ == "__main__":
    print("Testing PolarizationDataLoader")
    
    rmc_folder = Path("rmc")
    loader = PolarizationDataLoader(rmc_folder)
    
    print(f"\nDataset size: {len(loader)} samples")
    
    sample = loader.get_item(100)
    
    if sample:
        print(f"\nSample {sample['index']}: {sample['image_path']}")
        print(f"  Features: AoLP {sample['features']['aolp'].shape}, DoLP {sample['features']['dolp'].shape}")
        print(f"  Label: {sample['label']:.2f}°")
        print(f"  Raw: roll={sample['metadata']['raw_roll']:.2f}°, pitch={sample['metadata']['raw_pitch']:.2f}°, azimuth={sample['metadata']['raw_azimuth']:.2f}°")
    
    stats = loader.get_label_statistics()
    print(f"\nCorrected Azimuth: [{stats['azimuth']['min']:.2f}°, {stats['azimuth']['max']:.2f}°] (mean: {stats['azimuth']['mean']:.2f}°)")
    
    train_idx, val_idx, test_idx = loader.split_train_val_test()
    print(f"\nSplit: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    print("\n✓ Tests passed")

