#!/usr/bin/env python3
"""
Generate Solar Labels from Timelapse Images
===========================================

Create solar_labels.parquet from your existing timelapse polarization images.
Uses the same AoLP/DoLP processing as aolp_dolp_extract.py.

Usage:
    python generate_labels_from_images.py "C:\\Timelapse\\2-8-2026 Images" ./Capstone_live_data
"""

import sys
import pandas as pd
import numpy as np
import cv2
import polanalyser as pa
from pathlib import Path
from datetime import datetime
import argparse
import json
import re
import os
from glob import glob

# Import solar position calculator
sys.path.append(str(Path(__file__).parent.parent))
from solar_azimuth_generator import SolarPositionCalculator

# Queen's University coordinates (update if needed)
DEFAULT_LATITUDE = 44.2253
DEFAULT_LONGITUDE = -76.4951
DEFAULT_TIMEZONE = -5  # EST

def extract_timestamp_from_filename(filename):
    """
    Extract datetime from filename like: 2026-02-24_13-20-33_burst044_frame007.png
    """
    # Pattern: YYYY-MM-DD_HH-MM-SS_burst###_frame###.png
    pattern = r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})_burst\d+_frame\d+\.png'
    match = re.search(pattern, filename)
    
    if match:
        year, month, day, hour, minute, second = map(int, match.groups())
        return datetime(year, month, day, hour, minute, second)
    return None

def process_polarization_image(image_path):
    """
    Process polarization image to extract AoLP and DoLP using polanalyser.
    Based on aolp_dolp_extract.py methodology.
    """
    try:
        # Read raw polarization image
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
        print(f"  ⚠️  Error processing {image_path}: {e}")
        return None, None

def find_all_images(timelapse_dir):
    """Find all polarization images in the timelapse directory."""
    timelapse_path = Path(timelapse_dir)
    
    # Recursively find all PNG files
    image_patterns = [
        "**/*.png",
        "**/*.PNG"
    ]
    
    all_images = []
    for pattern in image_patterns:
        found = list(timelapse_path.glob(pattern))
        all_images.extend(found)
    
    # Filter for files with timestamp pattern
    timestamped_images = []
    for img_path in all_images:
        timestamp = extract_timestamp_from_filename(img_path.name)
        if timestamp:
            timestamped_images.append((img_path, timestamp))
    
    # Sort by timestamp
    timestamped_images.sort(key=lambda x: x[1])
    
    return timestamped_images

def create_solar_labels_from_images(timelapse_dir, output_dir, 
                                   latitude=DEFAULT_LATITUDE, 
                                   longitude=DEFAULT_LONGITUDE,
                                   timezone_offset=DEFAULT_TIMEZONE,
                                   max_images=None):
    """
    Create solar_labels.parquet from timelapse polarization images.
    """
    
    print("🔍 SOLAR LABELS FROM TIMELAPSE IMAGES")
    print("=" * 60)
    print(f"Input directory: {timelapse_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Location: {latitude:.4f}°N, {longitude:.4f}°E")
    print(f"Timezone: UTC{timezone_offset:+d}")
    
    # Find all images
    print(f"\\n📸 Finding polarization images...")
    timestamped_images = find_all_images(timelapse_dir)
    
    if not timestamped_images:
        print("❌ No timestamped images found!")
        print("Expected filename format: YYYY-MM-DD_HH-MM-SS_burst###_frame###.png")
        return False
    
    print(f"Found {len(timestamped_images)} timestamped images")
    
    # Limit for testing if requested
    if max_images:
        timestamped_images = timestamped_images[:max_images]
        print(f"Using first {len(timestamped_images)} images for testing")
    
    # Show time range
    first_time = timestamped_images[0][1]
    last_time = timestamped_images[-1][1]
    print(f"Time range: {first_time} to {last_time}")
    print()
    
    # Create solar position calculator
    solar_calc = SolarPositionCalculator(latitude, longitude, timezone_offset)
    
    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process images and calculate solar positions
    dataset_records = []
    failed_processing = 0
    
    print("☀️  Processing images and calculating solar positions...")
    
    for i, (img_path, timestamp) in enumerate(timestamped_images):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(timestamped_images)}: {timestamp} (failed: {failed_processing})")
        
        # Calculate solar position
        azimuth, elevation = solar_calc.solar_position(timestamp)
        
        # Process polarization image
        dolp, aolp = process_polarization_image(img_path)
        
        if dolp is not None and aolp is not None:
            # Determine session from folder path (Day 1, Day 2, etc.)
            session = "Other"
            path_parts = str(img_path).split(os.sep)
            for part in path_parts:
                if part.startswith("Day "):
                    session = part.replace(" ", "_")  # "Day 1" -> "Day_1"
                    break
            
            # Store successful record
            record = {
                'image_path': str(img_path.relative_to(Path(timelapse_dir).parent)),
                'timestamp': timestamp,
                'solar_azimuth': azimuth,
                'solar_elevation': elevation,
                'dolp_mean': float(np.mean(dolp)),
                'dolp_std': float(np.std(dolp)),
                'dolp_max': float(np.max(dolp)),
                'aolp_mean': float(np.mean(aolp)),
                'aolp_std': float(np.std(aolp)),
                'image_height': dolp.shape[0],
                'image_width': dolp.shape[1],
                'session': session  # Use folder-based session
            }
            dataset_records.append(record)
        else:
            failed_processing += 1
    
    print(f"\\n✅ Processing complete:")
    print(f"   Successful: {len(dataset_records)} images")
    print(f"   Failed: {failed_processing} images")
    
    if not dataset_records:
        print("❌ No successful image processing!")
        return False
    
    # Create DataFrame
    df = pd.DataFrame(dataset_records)
    
    # Add some derived features
    df['hour_of_day'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60.0
    df['azimuth_cos'] = np.cos(np.deg2rad(df['solar_azimuth']))
    df['azimuth_sin'] = np.sin(np.deg2rad(df['solar_azimuth']))
    
    # Save as parquet
    parquet_path = output_path / "solar_labels.parquet"
    df.to_parquet(parquet_path, index=False)
    
    print(f"\\n💾 Saved dataset: {parquet_path}")
    print(f"   Records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Dataset statistics
    print(f"\\n📊 Dataset Statistics:")
    sessions = df['session'].unique()
    print(f"   Sessions: {list(sessions)}")
    for session in sessions:
        session_data = df[df['session'] == session]
        az_range = session_data['solar_azimuth']
        print(f"   {session}: {len(session_data)} images, azimuth {az_range.min():.1f}°-{az_range.max():.1f}°")
    
    print(f"   Total azimuth range: {df['solar_azimuth'].min():.1f}° - {df['solar_azimuth'].max():.1f}°")
    print(f"   Coverage: {df['solar_azimuth'].max() - df['solar_azimuth'].min():.1f}° of 360°")
    
    # Save metadata
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'source_directory': str(timelapse_dir),
        'total_images_found': len(timestamped_images),
        'successful_processing': len(dataset_records),
        'failed_processing': failed_processing,
        'time_range': [str(first_time), str(last_time)],
        'azimuth_range': [float(df['solar_azimuth'].min()), float(df['solar_azimuth'].max())],
        'elevation_range': [float(df['solar_elevation'].min()), float(df['solar_elevation'].max())],
        'location': {
            'latitude': latitude,
            'longitude': longitude,
            'timezone_offset': timezone_offset
        },
        'sessions': [str(s) for s in sessions]
    }
    
    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Saved metadata: {metadata_path}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Generate solar_labels.parquet from timelapse polarization images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_labels_from_images.py "C:\\Timelapse\\2-8-2026 Images" ./Capstone_live_data/
    python generate_labels_from_images.py "C:\\Timelapse\\2-8-2026 Images" ./output/ --max-images 100
    
Expected image filename format:
    YYYY-MM-DD_HH-MM-SS_burst###_frame###.png
    Example: 2026-02-24_13-20-33_burst044_frame007.png
        """
    )
    
    parser.add_argument("timelapse_dir", help="Directory containing timelapse images")
    parser.add_argument("output_dir", help="Output directory for dataset")
    parser.add_argument("--latitude", type=float, default=DEFAULT_LATITUDE,
                      help=f"Latitude in degrees (default: {DEFAULT_LATITUDE} - Queen's University)")
    parser.add_argument("--longitude", type=float, default=DEFAULT_LONGITUDE,
                      help=f"Longitude in degrees (default: {DEFAULT_LONGITUDE} - Queen's University)")
    parser.add_argument("--timezone", type=int, default=DEFAULT_TIMEZONE,
                      help=f"Timezone offset from UTC (default: {DEFAULT_TIMEZONE} for EST)")
    parser.add_argument("--max-images", type=int,
                      help="Limit number of images for testing (default: process all)")
    
    args = parser.parse_args()
    
    if not Path(args.timelapse_dir).exists():
        print(f"❌ Directory not found: {args.timelapse_dir}")
        sys.exit(1)
    
    success = create_solar_labels_from_images(
        timelapse_dir=args.timelapse_dir,
        output_dir=args.output_dir,
        latitude=args.latitude,
        longitude=args.longitude,
        timezone_offset=args.timezone,
        max_images=args.max_images
    )
    
    if success:
        print(f"\\n🎉 SUCCESS! Dataset created in {args.output_dir}")
        print(f"\\n🔗 Next steps:")
        print("1. Test with: python -c \"import pandas as pd; df=pd.read_parquet('solar_labels.parquet'); print(df.info())\"")
        print("2. Run SVM test: python Underwater_testing/synthetic_svm_test.py")
    else:
        print(f"\\n❌ Failed to create dataset")
        sys.exit(1)

if __name__ == "__main__":
    main()