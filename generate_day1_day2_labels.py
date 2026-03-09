#!/usr/bin/env python3
"""
Generate Solar Labels from Day 1 and Day 2 Folders Only
======================================================

Process only the Day 1 and Day 2 folders to create proper train/test split.
"""

import sys
import pandas as pd
import numpy as np
import cv2
import polanalyser as pa
from pathlib import Path
from datetime import datetime
import re
import os

# Import solar position calculator
sys.path.append(str(Path(__file__).parent.parent))
from solar_azimuth_generator import SolarPositionCalculator

# Queen's University coordinates  
LATITUDE = 44.2253
LONGITUDE = -76.4951
TIMEZONE = -5  # EST

def extract_timestamp_from_filename(filename):
    """Extract datetime from filename like: 2026-02-24_13-20-33_burst044_frame007.png"""
    pattern = r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})_burst\d+_frame\d+\.png'
    match = re.search(pattern, filename)
    
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime(year, month, day, hour, minute, second)
    return None

def process_polarization_image(image_path):
    """Process polarization image to extract AoLP and DoLP."""
    try:
        img_raw = cv2.imread(str(image_path), 0)
        if img_raw is None:
            return None, None
        
        # Demosaic using polanalyser
        img_000, img_045, img_090, img_135 = pa.demosaicing(img_raw, pa.COLOR_PolarMono)
        
        # Calculate Stokes vectors
        image_list = [img_000, img_045, img_090, img_135]
        angles = np.deg2rad([0, 45, 90, 135])
        img_stokes = pa.calcStokes(image_list, angles)
        
        # Extract DoLP and AoLP
        img_dolp = pa.cvtStokesToDoLP(img_stokes)
        img_aolp = pa.cvtStokesToAoLP(img_stokes)
        
        # Convert AoLP to degrees
        img_aolp = np.rad2deg(img_aolp)
        
        return img_dolp, img_aolp
        
    except Exception as e:
        print(f"  Error processing {image_path}: {e}")
        return None, None

def extract_metadata_from_path(image_path, session_name):
    """Extract run, burst, frame info from image path."""
    path_str = str(image_path)
    
    # Extract run, burst, and frame numbers
    run_match = re.search(r'(run_[\d-]+_[\d-]+)', path_str)
    burst_match = re.search(r'(burst_[\d-]+_[\d-]+)', path_str) 
    frame_match = re.search(r'frame(\d+)', path_str)
    burst_num_match = re.search(r'burst(\d+)', path_str)
    
    return {
        'run': run_match.group(1) if run_match else f'{session_name}_run',
        'burst': burst_match.group(1) if burst_match else f'{session_name}_burst', 
        'frame_number': int(frame_match.group(1)) if frame_match else 1,
        'burst_number': int(burst_num_match.group(1)) if burst_num_match else 1
    }

def process_day_folder(day_folder, session_name, max_images_per_day=50):
    """Process first image from each burst in a Day folder."""
    day_path = Path(day_folder)
    
    print(f"\\n📸 Processing {session_name} ({day_path.name})...")
    
    # Find all burst or run directories
    burst_dirs = list(day_path.glob("**/burst_*")) + list(day_path.glob("**/run_*"))
    burst_dirs.sort()  # Sort to process chronologically
    
    print(f"   Found {len(burst_dirs)} burst directories")
    
    # Get first image from each burst
    selected_images = []
    for burst_dir in burst_dirs:
        png_files = sorted(list(burst_dir.glob("*.png")))
        if png_files:
            selected_images.append(png_files[0])  # First image from this burst
    
    # Limit if requested
    if max_images_per_day and len(selected_images) > max_images_per_day:
        selected_images = selected_images[:max_images_per_day]
    
    print(f"   Selected {len(selected_images)} images (first from each burst)")
    
    # Setup solar calculator
    solar_calc = SolarPositionCalculator(LATITUDE, LONGITUDE, TIMEZONE)
    
    records = []
    processed = 0
    failed = 0
    
    for i, img_path in enumerate(selected_images):
        if (i + 1) % 5 == 0:
            print(f"   {i+1}/{len(selected_images)} (success: {processed}, failed: {failed})")
        
        # Extract timestamp from parent directory name
        parent_dir = img_path.parent.name
        if parent_dir.startswith('burst_'):
            timestamp_str = parent_dir[6:]  # Remove "burst_" prefix
        elif parent_dir.startswith('run_'):
            timestamp_str = parent_dir[4:]  # Remove "run_" prefix
        else:
            print(f"   ⚠️ Unknown directory format: {parent_dir}")
            failed += 1
            continue
            
        # Parse timestamp: 2026-02-24_11-29-43 -> datetime
        try:
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            print(f"   ⚠️ Could not parse timestamp from: {timestamp_str}")
            failed += 1
            continue
        
        # Calculate solar position
        azimuth, elevation = solar_calc.solar_position(timestamp)
        
        # Process polarization image
        dolp, aolp = process_polarization_image(img_path)
        
        if dolp is not None and aolp is not None:
            # Extract metadata
            metadata = extract_metadata_from_path(img_path, session_name)
            
            # Create record
            record = {
                'image_path': str(img_path.relative_to(Path("C:/Queens/ELEC498/FinalDataset"))),  # Relative to FinalDataset
                'timestamp': timestamp,
                'timestamp_utc': timestamp,
                'solar_azimuth': azimuth,
                'solar_elevation': elevation,
                'dolp_mean': float(np.mean(dolp)),
                'dolp_std': float(np.std(dolp)), 
                'dolp_max': float(np.max(dolp)),
                'aolp_mean': float(np.mean(aolp)),
                'aolp_std': float(np.std(aolp)),
                'image_height': dolp.shape[0],
                'image_width': dolp.shape[1],
                'session': session_name,
                'run': metadata['run'],
                'burst': metadata['burst'],
                'frame_number': metadata['frame_number'],
                'burst_number': metadata['burst_number'],
                'latitude': LATITUDE,
                'longitude': LONGITUDE,
                'hour_of_day': timestamp.hour + timestamp.minute / 60.0,
                'azimuth_cos': np.cos(np.deg2rad(azimuth)),
                'azimuth_sin': np.sin(np.deg2rad(azimuth))
            }
            
            records.append(record)
            processed += 1
        else:
            failed += 1
    
    print(f"   {session_name} complete: {processed} success, {failed} failed")
    return records

def main():
    print("FOCUSED DATASET GENERATION: Day 1 and Day 2 Only")
    print("=" * 60)
    
    # Process Day 1 and Day 2 folders specifically
    day1_folder = "C:/Queens/ELEC498/FinalDataset/Day 1"
    day2_folder = "C:/Queens/ELEC498/FinalDataset/Day 2"
    
    if not Path(day1_folder).exists():
        print(f"❌ Day 1 folder not found: {day1_folder}")
        return False
        
    if not Path(day2_folder).exists():
        print(f"❌ Day 2 folder not found: {day2_folder}")
        return False
    
    # Process both days
    all_records = []
    
    day2_records = process_day_folder(day2_folder, "Day_2", max_images_per_day=None)  # No limit, take from all bursts
    all_records.extend(day2_records)
    
    day1_records = process_day_folder(day1_folder, "Day_1", max_images_per_day=None)  # No limit, take from all bursts
    all_records.extend(day1_records)
    
    if not all_records:
        print("❌ No valid records created!")
        return False
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    print(f"\\n📊 Dataset Summary:")
    print(f"   Total records: {len(df)}")
    session_counts = df['session'].value_counts()
    for session, count in session_counts.items():
        session_data = df[df['session'] == session]
        az_min, az_max = session_data['solar_azimuth'].min(), session_data['solar_azimuth'].max()
        print(f"   {session}: {count} images, azimuth {az_min:.1f}°-{az_max:.1f}°")
    
    # Save to FinalDataset folder
    output_path = "C:/Queens/ELEC498/FinalDataset/solar_labels.parquet"
    
    # Try parquet first, fallback to CSV
    try:
        df.to_parquet(output_path, index=False)
        print(f"\n💾 Saved: {output_path}")
    except ImportError as e:
        print(f"\n⚠️ Parquet failed ({e}), saving as CSV instead...")
        csv_path = output_path.replace('.parquet', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n💾 Saved: {csv_path}")
        output_path = csv_path
    print("✅ Ready for SVM testing with proper Day_1 and Day_2 sessions!")
    
    return True

if __name__ == "__main__":
    main()