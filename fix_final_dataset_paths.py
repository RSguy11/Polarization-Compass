import pandas as pd

# Load the parquet file
df = pd.read_parquet('C:/Queens/ELEC498/FinalDataset/solar_labels.parquet')

print(f"Original sample paths:")
print(df['image_path'].head(2).tolist())

# Fix the image paths to be relative to C:/Queens/ELEC498/FinalDataset
def fix_path_for_final_dataset(path):
    # Remove everything before "Day" 
    # Convert "Timelapse/2-8-2026 Images/Day 1/..." to "Day 1/..."
    if 'Day 1' in path:
        return path.split('Day 1')[0].replace(path.split('Day 1')[0], '') + 'Day 1' + path.split('Day 1')[1]
    elif 'Day 2' in path:
        return path.split('Day 2')[0].replace(path.split('Day 2')[0], '') + 'Day 2' + path.split('Day 2')[1]
    return path

# Simpler approach - extract just from Day onwards
def fix_path_simple(path):
    # Find "Day 1" or "Day 2" and take everything from there
    if 'Day 1' in path:
        idx = path.find('Day 1')
        return path[idx:]
    elif 'Day 2' in path:
        idx = path.find('Day 2')
        return path[idx:]
    return path

df['image_path'] = df['image_path'].apply(fix_path_simple)

print(f"Fixed sample paths:")
print(df['image_path'].head(2).tolist())

# Save the corrected parquet
df.to_parquet('C:/Queens/ELEC498/FinalDataset/solar_labels.parquet', index=False)
print('✅ Fixed image paths for FinalDataset location')