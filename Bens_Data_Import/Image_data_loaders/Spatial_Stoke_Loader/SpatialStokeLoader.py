import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import polanalyser as pa
import re

class SpatialStokeDataLoader():
    def __init__(self, img):
        if isinstance(img, np.ndarray):
            self.img = img.astype(np.uint8)
        else:
            self.img = np.array(Image.open(img), dtype=np.uint8)
        self.sun_position = None  # Will be detected on first extract
    
    def detect_sun_position(self, S0):
        threshold = np.percentile(S0, 99.5)
        bright_mask = S0 >= threshold
        bright_y, bright_x = np.where(bright_mask)
        if len(bright_x) > 0:
            sun_x = np.mean(bright_x)
            sun_y = np.mean(bright_y)
        else:
            H, W = S0.shape
            sun_x, sun_y = W / 2, H / 2
        return sun_x, sun_y
    
    def extract_features(self, gains=None, global_frame_offset=None):

        img = self.img
        # [0°   45° ]
        # [135° 90° ]
        I0_full = img[::2, ::2].astype(np.float32)       # Top-left pixels
        I45_full = img[::2, 1::2].astype(np.float32)     # Top-right pixels  
        I135_full = img[1::2, ::2].astype(np.float32)    # Bottom-left pixels
        I90_full = img[1::2, 1::2].astype(np.float32)    # Bottom-right pixels

        I0 = I0_full
        I45 = I45_full
        I90 = I90_full
        I135 = I135_full

        if gains is not None:
            I0   = gains["g0"]   * I0
            I45  = gains["g45"]  * I45
            I90  = gains["g90"]  * I90
            I135 = gains["g135"] * I135

        image_list = [I0, I45, I90, I135]
        angles = np.deg2rad([0, 45, 90, 135])
        img_stokes = pa.calcStokes(image_list, angles)
        
        # Extract S0 for backward compatibility
        S0 = img_stokes[..., 0]
        
        # Detect and store sun position
        self.sun_position = self.detect_sun_position(S0)
        
        DoLP = pa.cvtStokesToDoLP(img_stokes)
        AoLP = pa.cvtStokesToAoLP(img_stokes)
                
        AoLP = np.rad2deg(AoLP)  # [0, 180] degrees
        AoLP = (AoLP + 90) % 180 - 90  # Convert to [-90, 90] degrees
        
        if global_frame_offset is not None:
            H, W = AoLP.shape
            y_coords, x_coords = np.mgrid[0:H, 0:W]
            
            sun_x, sun_y = self.sun_position
            
            # Image center (optical axis) - radial structure is centered here
            cy, cx = H / 2.0, W / 2.0
            
            # Angle from image center to each pixel
            pixel_angle = np.rad2deg(np.arctan2(y_coords - cy, x_coords - cx))
            
            # Angle from image center to sun
            sun_angle = np.rad2deg(np.arctan2(sun_y - cy, sun_x - cx))
            
            # Rotation: radial angle relative to the solar direction
            rotation_angle = pixel_angle - sun_angle
            
            # SPP: subtract rotation so zero-line passes through sun
            AoLP = rotation_angle - AoLP + 90
            AoLP = (AoLP + 90) % 180 - 90
            
            if np.isscalar(global_frame_offset):
                AoLP = AoLP + float(global_frame_offset)
                AoLP = (AoLP + 90) % 180 - 90

        return DoLP, AoLP, I0, I45, I90, I135, S0

    def get_item(self, gains=None, global_frame_offset=None):
        dolp, aolp, I0, I45, I90, I135, S0 = self.extract_features(gains, global_frame_offset)
        
        return_array = {"aolp": aolp, "dolp": dolp}
        return return_array, I0, I45, I90, I135, S0

