import os
from datetime import datetime
import pandas as pd
from pvlib.solarposition import get_solarposition

def calculate_solar_azimuth(time_utc, latitude, longitude):
    times = pd.DatetimeIndex([time_utc])
    solpos = get_solarposition(times, latitude, longitude)
    return solpos['azimuth'].iloc[0]

def label_image_with_azimuth(image_path, latitude, longitude, time_utc=None):
    if time_utc is None:
        time_utc = datetime.utcnow()
    azimuth = calculate_solar_azimuth(time_utc, latitude, longitude)
    image_name = os.path.basename(image_path)
    label = f"{image_name}_azimuth_{azimuth:.2f}"
    return label

if __name__ == "__main__":
    image_path = "example_image.jpg"
    latitude = 44.237768  # 460 Barrie St Kingston ON
    longitude = -76.490330
    time_utc = datetime.utcnow()

    label = label_image_with_azimuth(image_path, latitude, longitude, time_utc)
    print(f"Labeled image: {label}")
