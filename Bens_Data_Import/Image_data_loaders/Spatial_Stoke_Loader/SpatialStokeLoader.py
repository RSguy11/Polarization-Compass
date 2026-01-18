import numpy as np
from pathlib import Path
from PIL import Image
import re
import cv2

class SpatialStokeDataLoader():
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        
        # Find all angle images
        self.image_files = sorted(
            self.data_path.glob("*_angle_*.png"),
            key=lambda x: self._extract_angle(x.name)
        )        
        if not self.image_files:
            raise ValueError(f"No images found in {self.data_path}")
        
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
    
    def extract_features(self, msg, gains = None):
        H, W = 2048, 2448

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(H, W)

        A = img[0::2, 0::2].astype(np.float32)  # top-left
        B = img[0::2, 1::2].astype(np.float32)  # top-right
        C = img[1::2, 0::2].astype(np.float32)  # bottom-left
        D = img[1::2, 1::2].astype(np.float32)  # bottom-right

        I0, I45, I90, I135 = A, D, C, B   # Option 2

        # ---- APPLY CALIBRATION ----
        if gains is not None:
            I0   = gains["g0"]   * I0
            I45  = gains["g45"]  * I45
            I90  = gains["g90"]  * I90
            I135 = gains["g135"] * I135

        S0 = I0 + I90
        S1 = I0 - I90
        S2 = I135 - I45

        eps = 1e-6
        denom = np.clip(S0, eps, None)

        DoLP = np.sqrt(S1**2 + S2**2) / denom

        AoLP = 0.5 * np.arctan2(S2, S1)
        AoLP = np.rad2deg(AoLP) 
        AoLP = (AoLP + 90) % 180 - 90

        return DoLP, AoLP,I0,I45,I90,I135,S0

    def get_item(self, gains= None):
        return_array = {"aolp" : [], "dolp" : [] }

        for i, (msg, timestamp) in enumerate(self.dataReader()):
            dolp, aolp,I0,I45,I90,I135,S0 = self.extract_features(msg,gains)
            
            return_array["aolp"] = aolp
            return_array["dolp"] = dolp

        return return_array,I0,I45,I90,I135,S0
    
