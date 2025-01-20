import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Read the data
historical_df = pd.read_csv('data/era5/era5_historical_minmax.csv')
current_df = pd.read_csv('data/era5/era5_current_minmax.csv')

# Convert date columns to datetime
historical_df['date'] = pd.to_datetime(historical_df['date'])
current_df['date'] = pd.to_datetime(current_df['date'])

# Calculate DTR (Diurnal Temperature Range)
historical_df['dtr'] = historical_df['tmax'] - historical_df['tmin']
current_df['dtr'] = current_df['tmax'] - current_df['tmin']

# Function to identify heatwaves using different definitions
def identify_heatwaves(df, percentile_threshold=90, min_duration=2, dtr_threshold=12.8):
    # Calculate the percentile threshold from the data
    temp_threshold = np.percentile(df['tmax'], percentile_threshold)
    
    # Create a mask for days exceeding the threshold and DTR
    hot_days = (df['tmax'] > temp_threshold) & (df['dtr'] > dtr_threshold)
    
    # Initialize heatwave array
    heatwave = np.zeros(len(df))
    
    # Find consecutive days
    count = 0
    for i in range(len(df)):
        if hot_days.iloc[i]:
            count += 1
        else:
            if count >= min_duration:
                heatwave[i-count:i] = 1
            count = 0
    
    # Check the last sequence
    if count >= min_duration:
        heatwave[-count:] = 1
    
    return heatwave

# Split data into 10-year periods
def split_into_periods(df):
    df['year'] = df['date'].dt.year
    periods = []
    
    # Define period ranges
    ranges = [
        (1980, 1989, "1980-1989"),
        (1990, 1999, "1990-1999"),
        (2000, 2009, "2000-2009"),
        (2010, 2019, "2010-2019"),
        (2020, 2024, "2020-2024")
    ]
    
    for start, end, label in ranges:
        period_df = df[df['year'].between(start, end)].copy()
        period_df['period'] = label
        periods.append(period_df)
    
    return periods

# Combine historical and current data
all_data = pd.concat([historical_df, current_df])

# Split into periods
periods = split_into_periods(all_data)

# Analyze heatwaves for each period using different definitions
results = []
for period_df in periods:
    period_label = period_df['period'].iloc[0]
    
    # Try different heatwave definitions
    definitions = [
        {"percentile": 90, "min_duration": 2, "dtr": 12.8, "label": "Standard"},
        {"percentile": 85, "min_duration": 2, "dtr": 12.0, "label": "Moderate"},
        {"percentile": 95, "min_duration": 3, "dtr": 13.5, "label": "Severe"}
    ]
    
    for def_params in definitions:
        heatwaves = identify_heatwaves(
            period_df,
            percentile_threshold=def_params["percentile"],
            min_duration=def_params["min_duration"],
            dtr_threshold=def_params["dtr"]
        )
        
        # Calculate metrics
        total_heatwave_days = sum(heatwaves)
        total_days = len(period_df)
        avg_temp_during_heatwaves = period_df.loc[heatwaves == 1, 'tmax'].mean()
        
        results.append({
            'period': period_label,
            'definition': def_params["label"],
            'total_heatwave_days': total_heatwave_days,
            'percentage_days': (total_heatwave_days / total_days) * 100,
            'avg_temp': avg_temp_during_heatwaves
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Print results
print("\nHeatwave Analysis Results:")
print("=" * 80)
for definition in ["Standard", "Moderate", "Severe"]:
    print(f"\n{definition} Definition Results:")
    print(results_df[results_df['definition'] == definition].to_string(index=False))

# Plot results
plt.figure(figsize=(12, 6))
for definition in ["Standard", "Moderate", "Severe"]:
    def_data = results_df[results_df['definition'] == definition]
    plt.plot(def_data['period'], def_data['percentage_days'], 
             marker='o', label=definition)

plt.title('Percentage of Days with Heatwaves by Period')
plt.xlabel('Time Period')
plt.ylabel('Percentage of Days (%)')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('data/era5/heatwave_trends.png')

# Save results to CSV
results_df.to_csv('data/era5/heatwave_analysis_results.csv', index=False)
print("\nResults have been saved to 'data/era5/heatwave_analysis_results.csv'")
print("Plot has been saved to 'data/era5/heatwave_trends.png'")
