import pandas as pd
import re

# Load existing parquet
df = pd.read_parquet('C:/Timelapse/solar_labels.parquet')

# Extract run and burst from image paths  
def extract_run_burst(path):
    # Extract from paths like: 2-8-2026 Images/cntrl/burst_2026-02-12_17-39-05/2026-02-12_17-39-05_burst001_frame001.png
    run_match = re.search(r'(run_[\d-]+_[\d-]+)', str(path))
    burst_match = re.search(r'(burst_[\d-]+_[\d-]+)', str(path))
    
    # Extract frame and burst numbers
    frame_match = re.search(r'frame(\d+)', str(path))
    burst_num_match = re.search(r'burst(\d+)', str(path))
    
    run = run_match.group(1) if run_match else 'unknown_run'
    burst = burst_match.group(1) if burst_match else 'unknown_burst'
    frame_number = int(frame_match.group(1)) if frame_match else 1
    burst_number = int(burst_num_match.group(1)) if burst_num_match else 1
    
    return run, burst, frame_number, burst_number

# Apply extraction
extracted = df['image_path'].apply(extract_run_burst)
df['run'] = [e[0] for e in extracted]  
df['burst'] = [e[1] for e in extracted]
df['frame_number'] = [e[2] for e in extracted]
df['burst_number'] = [e[3] for e in extracted]

# Add missing geographical columns (use Queen's University coordinates)
df['latitude'] = 44.2253
df['longitude'] = -76.4951

# Rename timestamp to timestamp_utc for compatibility
df['timestamp_utc'] = df['timestamp']

# Save updated parquet
df.to_parquet('C:/Timelapse/solar_labels.parquet', index=False)
print('Added all required columns')
print(f'Columns: {list(df.columns)}')
print(f'Unique runs: {df["run"].nunique()}')
print(f'Unique bursts: {df["burst"].nunique()}')