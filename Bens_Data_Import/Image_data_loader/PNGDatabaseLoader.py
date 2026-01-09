import numpy as np
from pathlib import Path
from PIL import Image
import re

class PNGDatabaseLoader():
    def __init__(self, data_path: Path, start_deg=0.0, step_deg=1.0):
        self.data_path = Path(data_path)
        
        # Find all angle images
        self.image_files = sorted(
            self.data_path.glob("*_angle_*.png"),
            key=lambda x: self._extract_angle(x.name)
        )
        
        if not self.image_files:
            raise ValueError(f"No images found in {self.data_path}")
        
        self.step_deg = step_deg
        self.start_deg = start_deg
        
        print(f"Found {len(self.image_files)} images")
    
    def _extract_angle(self, filename):
        match = re.search(r'angle_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    
    def dataReader(self):
        """Yields images in order"""
        for img_path in self.image_files:
            img = Image.open(img_path)
            # Convert to numpy array
            img_array = np.array(img, dtype=np.uint8)
            
            # Create a simple message-like object
            class ImageMsg:
                def __init__(self, data):
                    self.data = data.tobytes()
            
            # Get timestamp from filename if possible
            timestamp = self._extract_angle(img_path.name)
            
            yield ImageMsg(img_array), timestamp
    
    def extract_features(self, msg):
        H, W = 2048, 2448

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(H, W)

        I0   = img[0::2, 0::2].astype(np.float32)
        I45  = img[0::2, 1::2].astype(np.float32)
        I90  = img[1::2, 0::2].astype(np.float32)
        I135 = img[1::2, 1::2].astype(np.float32)

        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I45 - I135

        eps = 1e-6
        denom = np.clip(S0, eps, None)

        DoLP = np.sqrt(S1**2 + S2**2) / denom
        AoLP = 0.5 * np.arctan2(S2, S1)

        # Mask low-intensity pixels
        valid = S0 > 3

        if not np.any(valid):
            return 0.0, 0.0

        DoLP = DoLP[valid]
        AoLP = AoLP[valid]

        # Reduce to scalar features 
        dolp_mean = float(np.mean(DoLP))

        # Circular mean for AoLP
        aolp_mean = 0.5 * np.arctan2(
            np.mean(np.sin(2 * AoLP)),
            np.mean(np.cos(2 * AoLP))
        )

        return dolp_mean, float(aolp_mean)

    def extract_label(self, index):

        # Every image the camera was rotated by ix1  so we collect the azmuth degree 
        azimuth_deg = self.start_deg + index * self.step_deg
        return np.deg2rad(azimuth_deg)

    def get_item(self, max_samples: int | None = None):
        X = []
        y = []
        timestamps = []

        for i, (msg, timestamp) in enumerate(self.dataReader()):
            dolp, aolp = self.extract_features(msg)
            
            X.append([
                dolp,
                np.sin(aolp),
                np.cos(aolp),
            ])

            y.append(self.extract_label(i))
            timestamps.append(timestamp)

            if max_samples is not None and i + 1 >= max_samples:
                break

        return (
            np.asarray(X, dtype=np.float32),  # X = [DoLP, sin(AoLP), cos(AoLP)]
            np.asarray(y, dtype=np.float32),  # y = azimuth (rad)
            np.asarray(timestamps),           # timestamp (angle index)
        )
