import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from PIL import Image
from typing import List, Dict, Union, Literal
from visualizations.stage1_visualizations import visualize_and_save_stage1
from stage1.stage1_pipe import pseduo_four_channel_desnoising, intensity_guilded_residual_interpolation
from data_Loader.loading_mat_data import add_mosaic_to_samples, load_mat_file

def main():
    orginal_file = load_mat_file(noise_level="High")
    print("Num samples:", len(orginal_file))
    print("First raw shape:", orginal_file[0]["raw"].shape)
    if orginal_file:
        print("Successfully loaded .mat file data.")
    
    file_in_mosaic_form = add_mosaic_to_samples(orginal_file)
    print("Added mosaic to samples. First mosaic shape:", file_in_mosaic_form[0]["mosaic"].shape)

    pfcd_output = pseduo_four_channel_desnoising(file_in_mosaic_form)
    channel_images = intensity_guilded_residual_interpolation(pfcd_output)

    visualize_and_save_stage1(
        orig_samples=orginal_file,
        stage1_samples=channel_images,
        out_dir="stage1_visuals",
        num_samples=4,
    )

    print("IGRI complete. Number of samples:", len(channel_images))

if __name__ == "__main__":
    main()
    