#!/usr/bin/env python3
"""
Generate Solar Labels for Polarization Compass Dataset

This script creates the solar_labels.parquet file by combining:
1. GPS and timestamp data extracted from rosbags
2. Solar azimuth calculations using astronomical algorithms

Usage:
    python generate_solar_labels.py <rosbag_path> <output_dir>
    python generate_solar_labels.py /path/to/rosbag.bag ./Capstone_live_data/
"""

import sys
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime, timezone, timedelta
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from PIL import Image
import argparse
import json
import polanalyser as pa

# Import solar position calculator
sys.path.append(str(Path(__file__).parent.parent))
from solar_azimuth_generator import SolarPositionCalculator

# ROS topic names (same as main.py)
CAM_TOPIC = "/camera_driver_gv/vis/image_raw"
REF_TOPIC = "/novatel/oem7/inspva"  # GPS reference
IMU_TOPIC = "/xsens/imu/data"

def compute_angle_for_mueller(rows=2048, cols=2448, center_x=1224, center_y=1024):
    """Compute angle matrix for Mueller matrix calculations (from Deveyansh method)."""
    y_coord, x_coord = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    dx = x_coord - center_x
    dy = y_coord - center_y
    angle_matrix = np.degrees(np.arctan2(dy, dx))
    angle_matrix = (angle_matrix + 360) % 360
    angle_matrix = angle_matrix % 180
    return angle_matrix

def extract_deveyansh_azimuth(image_path):
    """Extract azimuth using Deveyansh polarization method."""
    try:
        # Load raw polarization image
        img_raw = cv2.imread(str(image_path), 0)
        if img_raw is None:
            return None, None, None, None
        
        # Demosaic using polanalyser
        img_000, img_045, img_090, img_135 = pa.demosaicing(img_raw, pa.COLOR_PolarMono)
        
        # Calculate Stokes vectors
        image_list = [img_000, img_045, img_090, img_135]
        angles = np.deg2rad([0, 45, 90, 135])
        img_stokes = pa.calcStokes(image_list, angles)
        
        # Decompose Stokes vector
        img_s0, img_s1, img_s2 = cv2.split(img_stokes)
        
        # Extract DoLP and AoLP
        img_dolp = pa.cvtStokesToDoLP(img_stokes)
        img_aolp = pa.cvtStokesToAoLP(img_stokes)
        img_aolp = np.rad2deg(img_aolp)  # Convert to degrees
        
        # Create angle matrix
        beta2 = compute_angle_for_mueller(img_raw.shape[0], img_raw.shape[1])
        
        # Calculate AoLP in Scattering Principal Plane (SPP)
        AolpIPP = img_aolp
        temp1 = np.deg2rad(AolpIPP + beta2)
        aolpspp2 = 0.5 * np.rad2deg(np.arctan2(np.sin(2*temp1), np.cos(2*temp1)))
        
        # Flip for aerial view (east on right)
        flipped_Aolp = np.fliplr(aolpspp2)
        
        # Calculate DOLP from Stokes components
        DOLP = np.sqrt(img_s1**2 + img_s2**2) / img_s0
        
        # Extract azimuth estimate from polarization pattern
        # Note: This is a simplified extraction - the full Deveyansh method
        # would include meridian detection and more sophisticated processing
        deveyansh_azimuth = np.mean(flipped_Aolp)  # Simplified for now
        
        return deveyansh_azimuth, img_dolp, flipped_Aolp, DOLP
        
    except Exception as e:
        print(f"  ⚠️ Deveyansh method failed for {image_path}: {str(e)[:50]}")
        return None, None, None, None

def extract_gps_and_images(bagpath, output_dir):
    """
    Extract GPS data and camera images from rosbag.
    
    Returns:
        tuple: (gps_records, image_records) where each is a list of dicts
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    gps_records = []
    image_records = []
    
    # Create typestore for ROS2
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    
    print(f"🔍 Processing rosbag: {bagpath}")
    print(f"📸 Extracting images to: {images_dir}")
    
    with AnyReader([bagpath], default_typestore=typestore) as reader:
        # Check available topics
        print(f"📋 Available topics: {list(reader.topics.keys())}")
        
        for connection, timestamp, rawdata in reader.messages():
            topic_name = connection.topic
            
            # Process GPS reference data
            if topic_name == REF_TOPIC:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                
                # Extract GPS and timestamp info
                gps_record = {
                    'timestamp': timestamp,  # ROS timestamp (nanoseconds)
                    'utc_seconds': msg.header.stamp.sec,
                    'utc_nanoseconds': msg.header.stamp.nanosec,
                    'latitude': msg.position.x if hasattr(msg, 'position') else getattr(msg, 'latitude', None),
                    'longitude': msg.position.y if hasattr(msg, 'position') else getattr(msg, 'longitude', None),
                    'altitude': msg.position.z if hasattr(msg, 'position') else getattr(msg, 'altitude', None),
                    'roll': getattr(msg, 'roll', 0.0),
                    'pitch': getattr(msg, 'pitch', 0.0), 
                    'yaw': getattr(msg, 'yaw', 0.0)
                }
                gps_records.append(gps_record)
                
                if len(gps_records) % 100 == 0:
                    print(f"  GPS records: {len(gps_records)}")
            
            # Process camera images
            elif topic_name == CAM_TOPIC:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                
                # Create timestamp-based filename
                img_timestamp = f"{msg.header.stamp.sec}_{msg.header.stamp.nanosec:09d}"
                filename = f"{img_timestamp}.png"
                filepath = images_dir / filename
                
                # Save image
                if hasattr(msg, 'data'):
                    # Convert ROS image data to PIL Image
                    width = msg.width
                    height = msg.height
                    encoding = msg.encoding
                    
                    if encoding == 'mono8':
                        # Grayscale image
                        img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width))
                        img = Image.fromarray(img_array, mode='L')
                    elif encoding == 'rgb8':
                        # Color image
                        img_array = np.frombuffer(msg.data, dtype=np.uint8).reshape((height, width, 3))
                        img = Image.fromarray(img_array, mode='RGB')
                    else:
                        print(f"⚠️  Unsupported image encoding: {encoding}")
                        continue
                    
                    img.save(filepath)
                    
                    image_record = {
                        'timestamp': timestamp,
                        'utc_seconds': msg.header.stamp.sec,
                        'utc_nanoseconds': msg.header.stamp.nanosec,
                        'image_path': str(filepath.relative_to(output_dir)),
                        'width': width,
                        'height': height,
                        'encoding': encoding
                    }
                    image_records.append(image_record)
                    
                    if len(image_records) % 50 == 0:
                        print(f"  Images extracted: {len(image_records)}")
    
    print(f"✅ Extraction complete:")
    print(f"   GPS records: {len(gps_records)}")
    print(f"   Images: {len(image_records)}")
    
    return gps_records, image_records

def calculate_solar_azimuths(gps_records, image_records, timezone_offset=-5):
    """
    Calculate solar azimuth angles for each image timestamp.
    
    Args:
        gps_records: List of GPS data records
        image_records: List of image records with timestamps
        timezone_offset: Local timezone offset from UTC (e.g., -5 for EST)
    
    Returns:
        list: Solar azimuth angles corresponding to each image
    """
    
    if not gps_records:
        raise ValueError("No GPS records found - cannot determine location")
    
    # Use average GPS location (assumes stationary or small area collection)
    valid_gps = [r for r in gps_records if r['latitude'] is not None and r['longitude'] is not None]
    if not valid_gps:
        raise ValueError("No valid GPS coordinates found")
    
    avg_lat = np.mean([r['latitude'] for r in valid_gps])
    avg_lon = np.mean([r['longitude'] for r in valid_gps])
    
    print(f"🌍 Using location: {avg_lat:.6f}°N, {avg_lon:.6f}°E")
    
    # Create solar position calculator
    solar_calc = SolarPositionCalculator(
        latitude=avg_lat, 
        longitude=avg_lon, 
        timezone_offset=timezone_offset
    )
    
    # Convert image timestamps to datetime objects
    datetimes = []
    azimuths = []
    elevations = []
    
    print(f"☀️  Calculating solar positions for {len(image_records)} images...")
    
    for i, img_rec in enumerate(image_records):
        # Convert ROS timestamp to datetime
        utc_timestamp = img_rec['utc_seconds'] + img_rec['utc_nanoseconds'] / 1e9
        dt_utc = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
        dt_local = dt_utc + timedelta(hours=timezone_offset)  # Convert to local time
        
        # Calculate solar position
        azimuth, elevation = solar_calc.solar_position(dt_local)
        
        datetimes.append(dt_local)
        azimuths.append(azimuth)
        elevations.append(elevation)
        
        if i % 100 == 0 and i > 0:
            print(f"   {i}/{len(image_records)}: {dt_local} → Az={azimuth:.1f}°, El={elevation:.1f}°")
    
    print(f"✅ Solar calculations complete:")
    print(f"   Azimuth range: [{np.min(azimuths):.1f}°, {np.max(azimuths):.1f}°]")
    print(f"   Elevation range: [{np.min(elevations):.1f}°, {np.max(elevations):.1f}°]")
    
    return datetimes, azimuths, elevations

def create_solar_labels_parquet(output_dir, gps_records, image_records, 
                               datetimes, azimuths, elevations):
    """
    Create the solar_labels.parquet file for ML training.
    Includes both solar ground truth and Deveyansh polarization compass results.
    """
    
    output_dir = Path(output_dir)
    
    # Build dataframe
    data = []
    
    print(f"🧭 Processing {len(image_records)} images with Deveyansh polarization method...")
    
    for i, img_rec in enumerate(image_records):
        # Extract azimuth using Deveyansh method
        image_full_path = output_dir / img_rec['image_path']
        deveyansh_azimuth, dolp, aolp_spp, dolp_stokes = extract_deveyansh_azimuth(image_full_path)
        
        # Calculate error if both methods succeeded
        azimuth_error = None
        if deveyansh_azimuth is not None:
            azimuth_error = abs(azimuths[i] - deveyansh_azimuth)
            if azimuth_error > 180:
                azimuth_error = 360 - azimuth_error  # Handle circular difference
        
        record = {
            'image_path': img_rec['image_path'],
            'timestamp': datetimes[i],
            'utc_seconds': img_rec['utc_seconds'],
            'utc_nanoseconds': img_rec['utc_nanoseconds'],
            'solar_azimuth': azimuths[i],
            'solar_elevation': elevations[i],
            'deveyansh_azimuth': deveyansh_azimuth,
            'azimuth_error': azimuth_error,
            'dolp_mean': float(np.mean(dolp)) if dolp is not None else None,
            'aolp_spp_mean': float(np.mean(aolp_spp)) if aolp_spp is not None else None,
            'image_width': img_rec['width'],
            'image_height': img_rec['height'],
            'encoding': img_rec['encoding']
        }
        
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(image_records)}: Solar={azimuths[i]:.1f}°, Deveyansh={deveyansh_azimuth:.1f}° (error: {azimuth_error:.1f}°)" if deveyansh_azimuth else f"  {i+1}/{len(image_records)}: Solar={azimuths[i]:.1f}°, Deveyansh=Failed")
        
        # Add GPS info if available (find closest GPS record)
        closest_gps = None
        min_time_diff = float('inf')
        
        for gps_rec in gps_records:
            time_diff = abs(gps_rec['utc_seconds'] - img_rec['utc_seconds'])
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_gps = gps_rec
        
        if closest_gps and min_time_diff < 5.0:  # Within 5 seconds
            record.update({
                'latitude': closest_gps['latitude'],
                'longitude': closest_gps['longitude'],
                'altitude': closest_gps['altitude'],
                'roll': closest_gps['roll'],
                'pitch': closest_gps['pitch'], 
                'yaw': closest_gps['yaw']
            })
        
        data.append(record)
    
    # Create dataframe
    df = pd.DataFrame(data)
    
    # Save as parquet
    parquet_path = output_dir / "solar_labels.parquet"
    df.to_parquet(parquet_path, index=False)
    
    # Calculate error statistics
    valid_errors = df['azimuth_error'].dropna()
    if len(valid_errors) > 0:
        mean_error = valid_errors.mean()
        std_error = valid_errors.std()
        max_error = valid_errors.max()
        print(f"\n📊 Deveyansh Method Error Analysis:")
        print(f"   Valid predictions: {len(valid_errors)}/{len(df)} ({100*len(valid_errors)/len(df):.1f}%)")
        print(f"   Mean error: {mean_error:.1f}° ± {std_error:.1f}°")
        print(f"   Max error: {max_error:.1f}°")
    
    print(f"\n💾 Saved solar labels: {parquet_path}")
    print(f"   Records: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # Save metadata
    # Calculate Deveyansh method statistics
    valid_deveyansh = df['deveyansh_azimuth'].dropna()
    valid_errors = df['azimuth_error'].dropna()
    
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'total_images': len(image_records),
        'total_gps_records': len(gps_records),
        'azimuth_range': [float(np.min(azimuths)), float(np.max(azimuths))],
        'elevation_range': [float(np.min(elevations)), float(np.max(elevations))],
        'time_range': [str(min(datetimes)), str(max(datetimes))],
        'deveyansh_method': {
            'valid_predictions': int(len(valid_deveyansh)),
            'success_rate': float(len(valid_deveyansh) / len(df)),
            'azimuth_range': [float(valid_deveyansh.min()), float(valid_deveyansh.max())] if len(valid_deveyansh) > 0 else None,
            'mean_error': float(valid_errors.mean()) if len(valid_errors) > 0 else None,
            'std_error': float(valid_errors.std()) if len(valid_errors) > 0 else None,
            'max_error': float(valid_errors.max()) if len(valid_errors) > 0 else None
        },
        'location': {
            'avg_latitude': float(np.mean([r['latitude'] for r in gps_records if r['latitude'] is not None])) if gps_records else None,
            'avg_longitude': float(np.mean([r['longitude'] for r in gps_records if r['longitude'] is not None])) if gps_records else None
        }
    }
    
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Saved metadata: {metadata_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Generate solar_labels.parquet from rosbag data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_solar_labels.py /path/to/rosbag.bag ./output/
    python generate_solar_labels.py /data/field_test_2024.bag ./Capstone_live_data/
    
Output structure:
    output_dir/
    ├── solar_labels.parquet     # Main dataset for ML training
    ├── dataset_metadata.json    # Dataset statistics and info
    └── images/                  # Extracted polarization images
        ├── 1640995200_123456789.png
        └── ...
        """
    )
    
    parser.add_argument("rosbag_path", help="Path to input rosbag file")
    parser.add_argument("output_dir", help="Output directory for dataset")
    parser.add_argument("--timezone", type=int, default=-5, 
                      help="Timezone offset from UTC (default: -5 for EST)")
    
    args = parser.parse_args()
    
    rosbag_path = Path(args.rosbag_path)
    output_dir = Path(args.output_dir)
    
    if not rosbag_path.exists():
        print(f"❌ Rosbag not found: {rosbag_path}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 SOLAR LABELS GENERATION")
    print("=" * 50)
    print(f"Input rosbag: {rosbag_path}")
    print(f"Output directory: {output_dir}")
    print(f"Timezone: UTC{args.timezone:+d}")
    print()
    
    try:
        # Step 1: Extract GPS and images from rosbag
        print("📦 Step 1: Extracting data from rosbag...")
        gps_records, image_records = extract_gps_and_images(rosbag_path, output_dir)
        
        if not image_records:
            print("❌ No images found in rosbag")
            sys.exit(1)
        
        # Step 2: Calculate solar positions
        print("\n☀️  Step 2: Calculating solar positions...")
        datetimes, azimuths, elevations = calculate_solar_azimuths(
            gps_records, image_records, args.timezone
        )
        
        # Step 3: Create parquet dataset
        print("\n💾 Step 3: Creating solar_labels.parquet...")
        df = create_solar_labels_parquet(
            output_dir, gps_records, image_records, 
            datetimes, azimuths, elevations
        )
        
        print(f"\n✅ SUCCESS! Solar labels dataset created in {output_dir}")
        print("\n🔗 Next steps:")
        print("1. Verify the dataset with: python -c \"import pandas as pd; print(pd.read_parquet('solar_labels.parquet').info())\"")
        print("2. Use this dataset in your SVM training with UnderwaterDataLoader")
        print("3. Run real_svm_test.py to test SVM performance")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()