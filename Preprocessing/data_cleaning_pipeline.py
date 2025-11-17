import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from PIL import Image
from typing import List, Dict, Union, Literal
from data_Loader.loading_mat_data import load_mat_file


def main():
    mat_data = load_mat_file(noise_level="Low")
    print("Num samples:", len(mat_data))
    print("First raw shape:", mat_data[0]["raw"].shape)
    if mat_data:
        print("Successfully loaded .mat file data.")

if __name__ == "__main__":
    main()
    