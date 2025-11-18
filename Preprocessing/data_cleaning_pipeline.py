import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from PIL import Image
from typing import List, Dict, Union, Literal
from stage1.stage1_pipe import pseduo_four_channel_desnoising, intensity_guilded_residual_interpolation
from data_Loader.loading_mat_data import add_mosaic_to_samples, load_mat_file

def main():
    mat_data = load_mat_file(noise_level="Low")
    print("Num samples:", len(mat_data))
    print("First raw shape:", mat_data[0]["raw"].shape)
    if mat_data:
        print("Successfully loaded .mat file data.")
    
    mat_data_with_mosaic = add_mosaic_to_samples(mat_data)
    #print("Added mosaic to samples. First mosaic shape:", mat_data_with_mosaic[0]["mosaic"].shape)
    
    denoised_data = pseduo_four_channel_desnoising(mat_data_with_mosaic)
    channel_images = intensity_guilded_residual_interpolation(denoised_data)
    print("IGRI complete. Number of samples:", len(channel_images))

if __name__ == "__main__":
    main()
    