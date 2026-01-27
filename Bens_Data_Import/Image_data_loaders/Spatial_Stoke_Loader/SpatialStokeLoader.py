import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import polanalyser as pa
import re

class SpatialStokeDataLoader():
    def __init__(self, img):
        """
        Initialize with a single image.
        
        Parameters:
        -----------
        img : np.ndarray or Path
            Raw polarization image (H, W) with Bayer-like polarization pattern
            or path to image file
        """
        if isinstance(img, np.ndarray):
            self.img = img.astype(np.uint8)
        else:
            # If it's a path, load it
            self.img = np.array(Image.open(img), dtype=np.uint8)
    
    def extract_features(self, gains=None, global_frame_offset=None):
        """
        Extract polarization features from the image.
        
        Parameters:
        -----------
        gains : dict, optional
            Calibration gains for each polarization angle
        global_frame_offset : float, optional
            Angle offset in degrees to transform from instrument frame to global frame.
            This is the angle β between the camera's y-axis and true north.
            If provided, AoLP will be rotated to align with global reference (e.g., true North).
            
        Returns:
        --------
        DoLP, AoLP, I0, I45, I90, I135, S0
        """
        img = self.img
        
        # ---- DRASTICALLY DOWNSCALE IMAGE ----
        # Skip demosaicing entirely - just use binned raw values
        # This avoids massive temporary array allocations in polanalyser
        scale_factor = 0.125  # 8x reduction: 2048→256, 2448→306
        new_height = int(img.shape[0] * scale_factor)
        new_width = int(img.shape[1] * scale_factor)
        img_binned = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # ---- MANUAL DEMOSAICING (NO POLANALYSER) ----
        # Bayer/polarization mosaic: extract channels manually
        # For Polarization mosaic with 2x2 pattern (I0, I45, I90, I135):
        # Skip demosaicing library entirely - just bin the raw sensor values
        # This is equivalent to averaging over local neighborhoods
        I0 = img_binned[::2, ::2].astype(np.float32)      # Top-left pixels
        I45 = img_binned[::2, 1::2].astype(np.float32)    # Top-right pixels  
        I90 = img_binned[1::2, ::2].astype(np.float32)    # Bottom-left pixels
        I135 = img_binned[1::2, 1::2].astype(np.float32)  # Bottom-right pixels
        
        # Convert to float32 for processing
        I0 = I0.astype(np.float32)
        I45 = I45.astype(np.float32)
        I90 = I90.astype(np.float32)
        I135 = I135.astype(np.float32)

        # ---- APPLY CALIBRATION ----
        if gains is not None:
            I0   = gains["g0"]   * I0
            I45  = gains["g45"]  * I45
            I90  = gains["g90"]  * I90
            I135 = gains["g135"] * I135

        # ---- OLD MANUAL CALCULATION (COMMENTED OUT) ----
        # S0 = I0 + I90
        # S1 = I0 - I90
        # S2 = I135 - I45
        # 
        # eps = 1e-6
        # denom = np.clip(S0, eps, None)
        # 
        # DoLP = np.sqrt(S1**2 + S2**2) / denom
        # 
        # AoLP = 0.5 * np.arctan2(S2, S1)
        # AoLP = np.rad2deg(AoLP) 
        # AoLP = (AoLP + 90) % 180 - 90

        # ---- NEW POLANALYSER LIBRARY EXTRACTION ----
        # Calculate Stokes vector using polanalyser
        image_list = [I0, I45, I90, I135]
        angles = np.deg2rad([0, 45, 90, 135])
        img_stokes = pa.calcStokes(image_list, angles)
        
        # Extract S0 for backward compatibility
        S0 = img_stokes[..., 0]
        
        # Convert Stokes to DoLP and AoLP using polanalyser
        DoLP = pa.cvtStokesToDoLP(img_stokes)
        AoLP = pa.cvtStokesToAoLP(img_stokes)  # Returns radians in [0, pi]
        
        # Convert AoLP to degrees and shift to [-90, 90] range to match old convention
        AoLP = np.rad2deg(AoLP)  # [0, 180] degrees
        AoLP = (AoLP + 90) % 180 - 90  # Convert to [-90, 90] degrees
        
        # ---- TRANSFORM TO SOLAR PRINCIPAL PLANE (GLOBAL FRAME) ----
        # If global_frame_offset is provided, transform to solar principal plane
        # This is different from a simple rotation - it transforms each pixel based on its position
        if global_frame_offset is not None:
            # Create coordinate grid for each pixel
            H, W = AoLP.shape
            y_coords, x_coords = np.mgrid[0:H, 0:W]
            
            # Calculate angle α for each pixel (counterclockwise from x-axis in instrument frame)
            # Center of image is the optical axis
            center_y, center_x = H / 2, W / 2
            dy = y_coords - center_y
            dx = x_coords - center_x
            alpha = np.arctan2(dy, dx)  # Angle of each pixel from center
            
            # Transform Stokes parameters to solar principal plane (Equation 5 from paper)
            # We need to work with S1, S2 to do this properly
            # Reconstruct S1, S2 from AoLP and DoLP
            AoLP_rad = np.deg2rad(AoLP)
            S1 = DoLP * np.cos(2 * AoLP_rad) * img_stokes[..., 0]
            S2 = DoLP * np.sin(2 * AoLP_rad) * img_stokes[..., 0]
            
            # Rotation matrix transformation for solar principal plane
            # [S1_solar]   [cos(2α)   sin(2α) ] [S1_inst]
            # [S2_solar] = [-sin(2α)  cos(2α) ] [S2_inst]
            S1_solar = S1 * np.cos(2 * alpha) + S2 * np.sin(2 * alpha)
            S2_solar = -S1 * np.sin(2 * alpha) + S2 * np.cos(2 * alpha)
            
            # Calculate new AoLP in solar principal plane
            AoLP = 0.5 * np.arctan2(S2_solar, S1_solar)
            AoLP = np.rad2deg(AoLP)
            AoLP = (AoLP + 90) % 180 - 90  # Keep in [-90, 90] range

        return DoLP, AoLP, I0, I45, I90, I135, S0

    def get_item(self, gains=None, global_frame_offset=None):
        """
        Get polarization features as a dictionary.
        
        Parameters:
        -----------
        gains : dict, optional
            Calibration gains for each polarization angle
        global_frame_offset : float, optional
            Angle offset in degrees to transform from instrument frame to global frame.
            
        Returns:
        --------
        return_array : dict
            Dictionary containing 'aolp' and 'dolp' arrays
        I0, I45, I90, I135, S0 : np.ndarray
            Individual polarization channels and S0
        """
        dolp, aolp, I0, I45, I90, I135, S0 = self.extract_features(gains, global_frame_offset)
        
        return_array = {"aolp": aolp, "dolp": dolp}
        return return_array, I0, I45, I90, I135, S0

