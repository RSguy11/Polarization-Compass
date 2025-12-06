"""
Solar Azimuth Label Generation

This module implements solar position calculations to generate ground truth
azimuth labels for your polarization data. This is essential for supervised
learning as specified in the project blueprint.

The solar azimuth angle is the compass direction from which the sunlight is coming.
It's measured clockwise from true north (0°) and ranges from 0° to 360°.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import List, Tuple, Optional, Union
import math


class SolarPositionCalculator:
    """
    Calculate solar position (azimuth and elevation) from timestamps and location.
    
    This provides ground truth labels for training your polarization compass.
    The calculations use astronomical algorithms for high accuracy.
    """
    
    def __init__(self, latitude: float, longitude: float, timezone_offset: int = 0):
        """
        Initialize solar position calculator.
        
        Args:
            latitude: Location latitude in degrees (-90 to +90)
            longitude: Location longitude in degrees (-180 to +180) 
            timezone_offset: Timezone offset from UTC in hours
        """
        self.latitude = math.radians(latitude)
        self.longitude = math.radians(longitude)
        self.timezone_offset = timezone_offset
        
        print(f"Solar calculator initialized:")
        print(f"  Location: {math.degrees(self.latitude):.4f}°N, {math.degrees(self.longitude):.4f}°E")
        print(f"  Timezone: UTC{timezone_offset:+d}")
    
    def julian_day(self, dt: datetime) -> float:
        """Calculate Julian day number from datetime."""
        if dt.month <= 2:
            year = dt.year - 1
            month = dt.month + 12
        else:
            year = dt.year
            month = dt.month
            
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + dt.day + b - 1524.5
        jd += (dt.hour + dt.minute/60.0 + dt.second/3600.0) / 24.0
        
        return jd
    
    def solar_declination(self, julian_day: float) -> float:
        """Calculate solar declination angle."""
        n = julian_day - 2451545.0  # Days since J2000.0
        L = math.radians(280.460 + 0.9856474 * n)  # Mean longitude of sun
        g = math.radians(357.528 + 0.9856003 * n)  # Mean anomaly
        
        # Solar declination
        lambda_sun = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2*g)
        declination = math.asin(math.sin(math.radians(23.439)) * math.sin(lambda_sun))
        
        return declination
    
    def equation_of_time(self, julian_day: float) -> float:
        """Calculate equation of time (solar time correction)."""
        n = julian_day - 2451545.0
        L = math.radians(280.460 + 0.9856474 * n)
        g = math.radians(357.528 + 0.9856003 * n)
        
        lambda_sun = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2*g)
        
        # Equation of time in minutes
        eot = 4 * math.degrees(L - 0.0057183 - math.atan2(math.tan(lambda_sun), math.cos(math.radians(23.439))))
        
        return eot / 60.0  # Convert to hours
    
    def solar_position(self, dt: datetime) -> Tuple[float, float]:
        """
        Calculate solar azimuth and elevation angles.
        
        Args:
            dt: Datetime for calculation (should be in local time)
            
        Returns:
            Tuple of (azimuth, elevation) in degrees
            Azimuth: 0° = North, 90° = East, 180° = South, 270° = West
            Elevation: 0° = horizon, 90° = zenith, negative = below horizon
        """
        
        # Convert to UTC
        utc_dt = dt - timedelta(hours=self.timezone_offset)
        
        # Calculate Julian day
        jd = self.julian_day(utc_dt)
        
        # Solar declination
        declination = self.solar_declination(jd)
        
        # Equation of time
        eot = self.equation_of_time(jd)
        
        # Solar time
        solar_time = dt.hour + dt.minute/60.0 + dt.second/3600.0
        solar_time += eot + math.degrees(self.longitude) / 15.0
        
        # Hour angle
        hour_angle = math.radians(15.0 * (solar_time - 12.0))
        
        # Solar elevation
        sin_elevation = (math.sin(declination) * math.sin(self.latitude) + 
                        math.cos(declination) * math.cos(self.latitude) * math.cos(hour_angle))
        elevation = math.asin(sin_elevation)
        
        # Solar azimuth
        cos_azimuth = ((math.sin(declination) * math.cos(self.latitude) - 
                       math.cos(declination) * math.sin(self.latitude) * math.cos(hour_angle)) / 
                      math.cos(elevation))
        
        # Handle numerical precision issues
        cos_azimuth = max(-1, min(1, cos_azimuth))
        azimuth = math.acos(cos_azimuth)
        
        # Adjust azimuth based on hour angle
        if hour_angle > 0:  # Afternoon
            azimuth = 2 * math.pi - azimuth
            
        # Convert to degrees
        azimuth_deg = math.degrees(azimuth)
        elevation_deg = math.degrees(elevation)
        
        return azimuth_deg, elevation_deg
    
    def generate_azimuth_labels(self, 
                              timestamps: List[datetime],
                              verbose: bool = True) -> np.ndarray:
        """
        Generate solar azimuth labels for a list of timestamps.
        
        Args:
            timestamps: List of datetime objects when polarization images were taken
            verbose: Whether to print progress information
            
        Returns:
            Array of solar azimuth angles in degrees (0-360°)
        """
        
        if verbose:
            print(f"Generating solar azimuth labels for {len(timestamps)} timestamps...")
        
        azimuth_labels = []
        
        for i, dt in enumerate(timestamps):
            azimuth, elevation = self.solar_position(dt)
            azimuth_labels.append(azimuth)
            
            if verbose and (i % max(1, len(timestamps) // 10) == 0):
                print(f"  {i+1}/{len(timestamps)}: {dt} -> Azimuth: {azimuth:.1f}°, Elevation: {elevation:.1f}°")
        
        azimuth_array = np.array(azimuth_labels)
        
        if verbose:
            print(f"✓ Generated {len(azimuth_array)} azimuth labels")
            print(f"  Range: [{azimuth_array.min():.1f}°, {azimuth_array.max():.1f}°]")
            print(f"  Mean: {azimuth_array.mean():.1f}°, Std: {azimuth_array.std():.1f}°")
        
        return azimuth_array


def create_example_timestamps(start_time: str, 
                            duration_hours: int = 8, 
                            interval_minutes: int = 10) -> List[datetime]:
    """
    Create example timestamps for testing (simulate data collection session).
    
    Args:
        start_time: Start time as string "YYYY-MM-DD HH:MM:SS"
        duration_hours: Duration of data collection in hours
        interval_minutes: Interval between measurements in minutes
        
    Returns:
        List of datetime objects
    """
    
    start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    timestamps = []
    
    current_time = start_dt
    end_time = start_dt + timedelta(hours=duration_hours)
    
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += timedelta(minutes=interval_minutes)
    
    return timestamps


def extract_timestamps_from_image_metadata(image_paths: List[str]) -> List[datetime]:
    """
    Extract timestamps from image metadata (EXIF data).
    
    This is a placeholder function - implement based on your camera's metadata format.
    
    Args:
        image_paths: List of paths to polarization images
        
    Returns:
        List of datetime objects when images were captured
    """
    
    print("⚠️  PLACEHOLDER: extract_timestamps_from_image_metadata()")
    print("   Implement this function to extract real timestamps from your images")
    print("   Options:")
    print("   1. EXIF data from camera")
    print("   2. Filename parsing (if timestamps are in filenames)")
    print("   3. Separate log file with timestamps")
    
    # Placeholder: return example timestamps
    return create_example_timestamps("2024-06-15 08:00:00", 4, 5)


if __name__ == "__main__":
    # Example usage
    print("SOLAR AZIMUTH LABEL GENERATION - EXAMPLE")
    print("=" * 50)
    
    # Example: Queen's University, Kingston, Ontario coordinates
    # (Replace with your actual data collection location)
    latitude = 44.2253  # Queen's University approximate location
    longitude = -76.4951
    timezone_offset = -5  # EST (adjust for your timezone)
    
    print(f"Example location: Queen's University, Kingston")
    
    # Create solar position calculator
    solar_calc = SolarPositionCalculator(latitude, longitude, timezone_offset)
    
    # Example: Generate timestamps for a data collection session
    print("\\nGenerating example timestamps...")
    example_timestamps = create_example_timestamps(
        start_time="2024-06-15 08:00:00",  # Summer day
        duration_hours=6,  # 6 hours of data collection  
        interval_minutes=15  # One measurement every 15 minutes
    )
    
    print(f"Created {len(example_timestamps)} example timestamps")
    
    # Generate azimuth labels
    print("\\nCalculating solar positions...")
    azimuth_labels = solar_calc.generate_azimuth_labels(example_timestamps)
    
    # Show results
    print(f"\\n📊 SOLAR AZIMUTH RESULTS:")
    print(f"Number of labels: {len(azimuth_labels)}")
    print(f"Azimuth range: [{azimuth_labels.min():.1f}°, {azimuth_labels.max():.1f}°]")
    print(f"Mean azimuth: {azimuth_labels.mean():.1f}°")
    
    # Example of how to use with your data
    print(f"\\n🔗 INTEGRATION WITH YOUR PROJECT:")
    print("1. Replace coordinates with your actual data collection location")
    print("2. Extract timestamps from your polarization images") 
    print("3. Generate azimuth labels using SolarPositionCalculator")
    print("4. Use these labels as ground truth for training your models")
    
    print(f"\\n💾 SAVE LABELS:")
    print("np.save('azimuth_labels.npy', azimuth_labels)")
    
    # Save example labels
    np.save('example_azimuth_labels.npy', azimuth_labels)
    print("✓ Example labels saved to 'example_azimuth_labels.npy'")