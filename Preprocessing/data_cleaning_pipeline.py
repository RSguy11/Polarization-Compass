import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from PIL import Image
from typing import List, Dict, Union, Literal
from Preprocessing.visualizations.stage2_visualizations import visualize_and_save_stage2
from stage2.stage2_pipe import polarimetric_parameters_from_stokes
from visualizations.stage1_visualizations import visualize_and_save_stage1
from stage1.stage1_pipe import pseduo_four_channel_desnoising, intensity_guilded_residual_interpolation
from data_Loader.loading_mat_data import add_mosaic_to_samples, load_mat_file

def main():

    ##### LOAD DATA ########

    orginal_file = load_mat_file(noise_level="High")
    print("Num samples:", len(orginal_file))
    print("First raw shape:", orginal_file[0]["raw"].shape)
    if orginal_file:
        print("Successfully loaded .mat file data.")

    ##### STAGE 1 ########

    file_in_mosaic_form = add_mosaic_to_samples(orginal_file)
    print("Added mosaic to samples. First mosaic shape:", file_in_mosaic_form[0]["mosaic"].shape)

    pfcd_output = pseduo_four_channel_desnoising(file_in_mosaic_form)
    
    channel_images = intensity_guilded_residual_interpolation(pfcd_output)
    print("IGRI complete. Number of samples:", len(channel_images))

    visualize_and_save_stage1(
        orig_samples=orginal_file,
        stage1_samples=channel_images,
        out_dir="stage1_visuals",
        num_samples=4,
    )

    ##### STAGE 2 ########

    stage2_out = polarimetric_parameters_from_stokes(channel_images)
    print("Stage 2 complete. Number of samples:", len(stage2_out))


    visualize_and_save_stage2(
        orig_samples=orginal_file, 
        stage2_samples=stage2_out,
        out_dir="stage2_results",
        num_samples=4
    )

     ##### STAGE 3 ########

if __name__ == "__main__":
    main()
    