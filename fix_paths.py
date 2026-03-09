import pandas as pd

# Load the parquet file
df = pd.read_parquet('C:/Timelapse/solar_labels.parquet')

# Fix the image paths to be relative to C:/Timelapse
def fix_path(path):
    # Remove Timelapse prefix entirely
    if 'Timelapse' in path:
        # Extract everything after "Timelapse/"
        parts = path.split('Timelapse')
        if len(parts) > 1:
            return parts[1].lstrip('\\/')  # Remove leading slashes
    return path

df['image_path'] = df['image_path'].apply(fix_path)

# Save the corrected parquet
df.to_parquet('C:/Timelapse/solar_labels.parquet', index=False)
print('Fixed image paths')
print(f'Sample paths: {df["image_path"].head(2).tolist()}')