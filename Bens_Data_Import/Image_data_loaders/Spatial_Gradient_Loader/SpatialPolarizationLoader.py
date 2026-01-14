import numpy as np
from pathlib import Path
from PIL import Image
import re
import cv2
from typing import Tuple, Optional

class SpatialPolarizationLoader:
    # Enhanced polarization data loader that preserves spatial information
    # from full resolution images instead of reducing to scalar averages.
    
    #Target_Size = Resolution/Compressed AolP and DOLP representation
    def __init__(self, data_path: Path, start_deg=0.0, step_deg=1.0, target_size: Tuple[int, int] = (256, 256)):
        self.data_path = Path(data_path)
        self.target_size = target_size
        
        # Find all angle images
        self.image_files = sorted(
            self.data_path.glob("*_angle_*.png"),
            key=lambda x: self._extract_angle(x.name)
        )
        
        if not self.image_files:
            raise ValueError(f"No polarization images found in {self.data_path}")
        
        self.step_deg = step_deg
        self.start_deg = start_deg
        
        print(f"Found {len(self.image_files)} polarization images")
        print(f"Target processing size: {target_size}")

    
    def _extract_angle(self, filename: str) -> int:
        match = re.search(r'angle_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    def load_and_process_image(self, img_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        # Load the image
        img = Image.open(img_path)
        img_array = np.array(img, dtype=np.float32)
        
        # Handle different image formats
        if len(img_array.shape) == 3:
            # RGB image - convert to grayscale or use specific channel
            # Gives Us one intensity Value over three 
            img_array = np.mean(img_array, axis=2)
        
        H, W = img_array.shape
        
        # The rest Assumes proccessed polerization images
        
        # Normalize to 0-1 range.
        img_normalized = img_array / 255.0 if img_array.max() > 1.0 else img_array
        
        # Create spatial gdcv  radients that could represent polarization patterns Horizontal and vertical gradients
        grad_x = cv2.Sobel(img_normalized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img_normalized, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude and angle from gradients (proxy for DoLP and AoLP)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Normalize magnitude to 0-1 (DoLP-like)
        dolp_spatial = np.clip(magnitude / (magnitude.max() + 1e-8), 0, 1)
        
        # Normalize angle to 0-180 degrees (AoLP-like)
        aolp_spatial = (angle + 180) % 180
        
        # Downsample for computational efficiency while preserving spatial structure
        dolp_resized = cv2.resize(dolp_spatial, self.target_size, interpolation=cv2.INTER_AREA)
        aolp_resized = cv2.resize(aolp_spatial, self.target_size, interpolation=cv2.INTER_AREA)
        
        return dolp_resized.astype(np.float32), aolp_resized.astype(np.float32)
    
    def extract_label(self, index: int) -> float:
        #Extract azimuth label for given image index.
        azimuth_deg = self.start_deg + index * self.step_deg
        return np.deg2rad(azimuth_deg % 360)
    
    def get_spatial_data(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_files = len(self.image_files)
        if max_samples is not None:
            n_files = min(n_files, max_samples)
        
        h, w = self.target_size
        dolp_data = np.zeros((n_files, h, w), dtype=np.float32)
        aolp_data = np.zeros((n_files, h, w), dtype=np.float32)
        azimuth_labels = np.zeros(n_files, dtype=np.float32)
        
        print(f"Loading spatial polarization data from {n_files} images...")
        
        for i, img_path in enumerate(self.image_files[:n_files]):
            if i % 50 == 0:
                print(f"  Processing image {i+1}/{n_files}")
            
            try:
                dolp_spatial, aolp_spatial = self.load_and_process_image(img_path)
                dolp_data[i] = dolp_spatial
                aolp_data[i] = aolp_spatial
                azimuth_labels[i] = self.extract_label(i)
                
            except Exception as e:
                print(f"  Warning: Failed to process {img_path.name}: {e}")
                # Fill with zeros on error
                dolp_data[i] = np.zeros(self.target_size, dtype=np.float32)
                aolp_data[i] = np.zeros(self.target_size, dtype=np.float32)
                azimuth_labels[i] = self.extract_label(i)
        
        print(f"Spatial polarization dataset loaded:")
        print(f"Spatial size: {h}×{w}")

        #Ensureing all values fall within an appropriate range
        print(f"DoLP spatial range: [{dolp_data.min():.3f}, {dolp_data.max():.3f}]")
        print(f"AoLP spatial range: [{aolp_data.min():.1f}°, {aolp_data.max():.1f}°]")
        print(f"Azimuth range: [{np.rad2deg(azimuth_labels.min()):.1f}°, {np.rad2deg(azimuth_labels.max()):.1f}°]")
        
        return dolp_data, aolp_data, azimuth_labels
        #Returns:
            # Tuple of (DoLP_spatial, AoLP_spatial, azimuth_labels)
            # DoLP_spatial: (N, H, W) array of spatial DoLP patterns
            # AoLP_spatial: (N, H, W) array of spatial AoLP patterns  
            # azimuth_labels: (N,) array of azimuth angles in radians

    def create_feature_vectors(self, dolp_spatial: np.ndarray, aolp_spatial: np.ndarray, method: str = 'flatten') -> np.ndarray:
        # method: Feature extraction method ('flatten', 'stats')

        n_samples, h, w = dolp_spatial.shape
        
        if method == 'flatten':
            # Simple flattening - preserves all spatial information
            dolp_flat = dolp_spatial.reshape(n_samples, -1)
            aolp_flat = aolp_spatial.reshape(n_samples, -1)
            features = np.concatenate([dolp_flat, aolp_flat], axis=1)
            print(f"Flattened features: {features.shape} ({features.shape[1]} features per sample)")
            
        elif method == 'stats':
            # Statistical features - more compact representation
            features_list = []
            
            for i in range(n_samples):
                dolp_img = dolp_spatial[i]
                aolp_img = aolp_spatial[i]
                
                # Spatial statistics
                sample_features = [
                    np.mean(dolp_img), np.std(dolp_img), np.max(dolp_img), np.min(dolp_img),
                    np.mean(aolp_img), np.std(aolp_img), np.max(aolp_img), np.min(aolp_img),
                    # Gradients
                    np.mean(np.abs(np.gradient(dolp_img)[0])), 
                    np.mean(np.abs(np.gradient(dolp_img)[1])),
                    np.mean(np.abs(np.gradient(aolp_img)[0])), 
                    np.mean(np.abs(np.gradient(aolp_img)[1])),
                    # Center region vs edges
                    np.mean(dolp_img[h//4:3*h//4, w//4:3*w//4]),  # Center DoLP
                    np.mean(aolp_img[h//4:3*h//4, w//4:3*w//4]),  # Center AoLP
                ]
                features_list.append(sample_features)
            
            features = np.array(features_list, dtype=np.float32)
            print(f"Statistical features: {features.shape} ({features.shape[1]} features per sample)")
            
        else:
            raise ValueError(f"Unknown feature extraction method: {method}")
        
        return features