import ee
import pandas as pd
import os
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    filename='data_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize Earth Engine
try:
    ee.Initialize()
    logging.info("Earth Engine initialized successfully")
except Exception as e:
    logging.error(f"Failed to initialize Earth Engine: {e}")
    raise

# Rahima Moosa Hospital coordinates
HOSPITAL_LAT = -26.175386
HOSPITAL_LON = 27.940794

# Create a point geometry for the hospital
hospital_point = ee.Geometry.Point([HOSPITAL_LON, HOSPITAL_LAT])

# Ensure output directory exists
output_dir = 'data/era5'
os.makedirs(output_dir, exist_ok=True)

def calculate_relative_humidity(t, td):
    """Calculate relative humidity using simplified formula
    t: temperature in Celsius
    td: dew point temperature in Celsius
    Returns relative humidity as a percentage
    """
    return ee.Number(100).multiply(
        ee.Number(-17.27).multiply(t.subtract(td))
        .divide(ee.Number(237.7).add(t))
        .exp()
    )

def get_era5_data_batch(start_year, end_year, batch_size=5):
    """Get ERA5-Land data in smaller batches to avoid memory issues"""
    all_data = []
    
    for year in range(start_year, end_year + 1, batch_size):
        batch_end = min(year + batch_size - 1, end_year)
        start_date = f"{year}-01-01"
        end_date = f"{batch_end}-12-31"
        
        logging.info(f"Processing batch {year}-{batch_end}")
        print(f"Processing years {year}-{batch_end}")
        
        try:
            # Load ERA5-Land daily aggregated data
            era5_daily = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')\
                .filterDate(start_date, end_date)\
                .select([
                    'temperature_2m_max', 'temperature_2m_min',
                    'dewpoint_temperature_2m', # Using instantaneous dewpoint
                    'surface_pressure',  # Using instantaneous pressure
                    'total_precipitation_sum'
                ])
            
            # Function to process daily data
            def daily_metrics(image):
                date = ee.Date(image.get('system:time_start')).format('YYYY-MM-dd')
                
                # Get values at hospital location
                values = image.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=hospital_point,
                    scale=11132,
                    maxPixels=1e9
                )
                
                # Convert temperatures from K to °C
                tmax = ee.Number(values.get('temperature_2m_max')).subtract(273.15)
                tmin = ee.Number(values.get('temperature_2m_min')).subtract(273.15)
                tdew = ee.Number(values.get('dewpoint_temperature_2m')).subtract(273.15)
                
                # Calculate relative humidity at max temperature (lowest RH)
                rh_min = calculate_relative_humidity(tmax, tdew)
                
                return ee.Feature(None, {
                    'date': date,
                    'tmax': tmax,
                    'tmin': tmin,
                    'tdew': tdew,
                    'rh': rh_min,
                    'precipitation': values.get('total_precipitation_sum'),
                    'pressure': values.get('surface_pressure')
                })
            
            # Process all images in the batch
            features = ee.FeatureCollection(era5_daily.map(daily_metrics))
            
            # Get the data directly
            batch_data = pd.DataFrame([
                feature['properties'] 
                for feature in features.getInfo()['features']
            ])
            
            all_data.append(batch_data)
            logging.info(f"Successfully processed batch {year}-{batch_end}")
            
        except Exception as e:
            logging.error(f"Error processing batch {year}-{batch_end}: {e}")
            print(f"Error processing years {year}-{batch_end}: {str(e)}")
            continue
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return None

# Process data in smaller batches
try:
    # Historical period (1980-2010)
    print("Processing historical period (1980-2010)...")
    historical_df = get_era5_data_batch(1980, 2010, batch_size=5)
    if historical_df is not None:
        historical_df.to_csv('data/era5/era5_historical_metrics.csv', index=False)
        print(f"Historical data saved: {len(historical_df)} records")
    
    # Current period (2011-2024)
    print("\nProcessing current period (2011-2024)...")
    current_df = get_era5_data_batch(2011, 2024, batch_size=5)
    if current_df is not None:
        current_df.to_csv('data/era5/era5_current_metrics.csv', index=False)
        print(f"Current data saved: {len(current_df)} records")
    
except Exception as e:
    logging.error(f"Failed to process ERA5 data: {e}")
    print(f"An error occurred. Please check data_processing.log for details.")

print("\nData processing complete!")
