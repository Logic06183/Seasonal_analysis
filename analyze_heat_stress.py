import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Create output directories
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
fig_dir = output_dir / 'figures'
fig_dir.mkdir(exist_ok=True)

def calculate_heat_index(T, RH):
    """Calculate Heat Index using temperature (°C) and relative humidity (%)"""
    # Convert temperature to Fahrenheit for the standard equation
    T_F = T * 9/5 + 32
    
    # Simple formula for heat index
    HI = 0.5 * (T_F + 61.0 + ((T_F - 68.0) * 1.2) + (RH * 0.094))
    
    # Use more complex formula if heat index > 80°F
    mask = HI >= 80
    if np.any(mask):
        T_F_masked = T_F[mask]
        RH_masked = RH[mask]
        
        HI[mask] = -42.379 + 2.04901523*T_F_masked + 10.14333127*RH_masked \
                   - 0.22475541*T_F_masked*RH_masked - 6.83783e-3*T_F_masked**2 \
                   - 5.481717e-2*RH_masked**2 + 1.22874e-3*T_F_masked**2*RH_masked \
                   + 8.5282e-4*T_F_masked*RH_masked**2 - 1.99e-6*T_F_masked**2*RH_masked**2
    
    # Convert back to Celsius
    return (HI - 32) * 5/9

def calculate_humidex(T, RH):
    """Calculate Humidex using temperature (°C) and relative humidity (%)"""
    # Calculate dewpoint temperature
    alpha = ((17.27 * T) / (237.7 + T)) + np.log(RH/100.0)
    Td = (237.7 * alpha) / (17.27 - alpha)
    
    # Calculate vapor pressure
    e = 6.11 * np.exp(5417.7530 * ((1/273.16) - (1/(273.15 + Td))))
    
    # Calculate humidex
    humidex = T + 5/9 * (e - 10)
    return humidex

def analyze_heat_stress(historical_df, current_df):
    """Analyze heat stress trends by decade"""
    # Combine dataframes and add decade column
    historical_df['period'] = 'historical'
    current_df['period'] = 'current'
    df = pd.concat([historical_df, current_df])
    df['date'] = pd.to_datetime(df['date'])
    df['decade'] = (df['date'].dt.year // 10) * 10
    
    # Calculate heat stress indices
    df['heat_index'] = calculate_heat_index(df['tmax'], df['rh'])
    df['humidex'] = calculate_humidex(df['tmax'], df['rh'])
    
    # Define heat stress thresholds
    temp_threshold = np.percentile(df['tmax'], 80)  # 80th percentile
    hi_threshold = 32  # Heat Index > 32°C (moderate heat stress)
    hx_threshold = 35  # Humidex > 35°C (some discomfort)
    
    # Calculate heat stress days for each metric
    df['temp_stress'] = df['tmax'] > temp_threshold
    df['hi_stress'] = df['heat_index'] > hi_threshold
    df['hx_stress'] = df['humidex'] > hx_threshold
    df['any_stress'] = df['temp_stress'] | df['hi_stress'] | df['hx_stress']
    
    # Analyze by decade
    decades = []
    for decade in sorted(df['decade'].unique()):
        decade_data = df[df['decade'] == decade]
        
        # Calculate metrics
        metrics = {
            'decade': decade,
            'total_days': len(decade_data),
            'temp_stress_days': decade_data['temp_stress'].sum(),
            'hi_stress_days': decade_data['hi_stress'].sum(),
            'hx_stress_days': decade_data['hx_stress'].sum(),
            'any_stress_days': decade_data['any_stress'].sum(),
            'avg_tmax': decade_data['tmax'].mean(),
            'avg_hi': decade_data['heat_index'].mean(),
            'avg_hx': decade_data['humidex'].mean(),
            'max_tmax': decade_data['tmax'].max(),
            'max_hi': decade_data['heat_index'].max(),
            'max_hx': decade_data['humidex'].max()
        }
        
        # Add percentages
        metrics.update({
            'temp_stress_pct': metrics['temp_stress_days'] / metrics['total_days'] * 100,
            'hi_stress_pct': metrics['hi_stress_days'] / metrics['total_days'] * 100,
            'hx_stress_pct': metrics['hx_stress_days'] / metrics['total_days'] * 100,
            'any_stress_pct': metrics['any_stress_days'] / metrics['total_days'] * 100
        })
        
        decades.append(metrics)
    
    results_df = pd.DataFrame(decades)
    
    # Save results
    results_df.to_csv(output_dir / 'heat_stress_by_decade.csv', index=False)
    
    # Create visualizations
    plot_heat_stress_trends(results_df)
    plot_heat_stress_calendar(df)
    plot_indices_distribution(df)
    
    return results_df

def plot_heat_stress_trends(results_df):
    """Plot trends in heat stress metrics over decades"""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Heat stress days percentage
    plt.subplot(2, 2, 1)
    metrics = ['temp_stress_pct', 'hi_stress_pct', 'hx_stress_pct']
    labels = ['Temperature', 'Heat Index', 'Humidex']
    for metric, label in zip(metrics, labels):
        plt.plot(results_df['decade'], results_df[metric], marker='o', label=label)
    plt.title('Percentage of Heat Stress Days by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Days')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Average maximum values
    plt.subplot(2, 2, 2)
    metrics = ['avg_tmax', 'avg_hi', 'avg_hx']
    labels = ['Temperature', 'Heat Index', 'Humidex']
    for metric, label in zip(metrics, labels):
        plt.plot(results_df['decade'], results_df[metric], marker='o', label=label)
    plt.title('Average Heat Stress Indices by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Maximum values
    plt.subplot(2, 2, 3)
    metrics = ['max_tmax', 'max_hi', 'max_hx']
    labels = ['Temperature', 'Heat Index', 'Humidex']
    for metric, label in zip(metrics, labels):
        plt.plot(results_df['decade'], results_df[metric], marker='o', label=label)
    plt.title('Maximum Heat Stress Values by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Combined heat stress days
    plt.subplot(2, 2, 4)
    plt.plot(results_df['decade'], results_df['any_stress_pct'], 
             marker='o', color='red', linewidth=2)
    plt.title('Combined Heat Stress Days by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Days')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'heat_stress_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_heat_stress_calendar(df):
    """Create a calendar heatmap of heat stress days"""
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Prepare data for heatmap
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Calculate monthly heat stress days percentage
    monthly_stress = df.groupby(['year', 'month'])['any_stress'].mean() * 100
    monthly_stress = monthly_stress.unstack()
    
    # Create heatmap
    plt.imshow(monthly_stress, cmap='YlOrRd', 
                interpolation='nearest')
    plt.title('Monthly Heat Stress Days (%)')
    plt.ylabel('Year')
    plt.xticks(np.arange(len(monthly_stress.columns)), monthly_stress.columns, rotation=45)
    plt.yticks(np.arange(len(monthly_stress.index)), monthly_stress.index)
    plt.colorbar()
    plt.grid(True)
    
    plt.savefig(fig_dir / 'heat_stress_calendar.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_indices_distribution(df):
    """Plot distribution of heat stress indices"""
    plt.figure(figsize=(15, 5))
    
    # Create violin plots for each index
    plt.subplot(1, 3, 1)
    plt.violinplot(df['tmax'])
    plt.title('Temperature Distribution')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.violinplot(df['heat_index'])
    plt.title('Heat Index Distribution')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.violinplot(df['humidex'])
    plt.title('Humidex Distribution')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'heat_stress_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Load data
    historical_df = pd.read_csv('data/era5/era5_historical_metrics.csv')
    current_df = pd.read_csv('data/era5/era5_current_metrics.csv')
    
    # Run analysis
    results = analyze_heat_stress(historical_df, current_df)
    
    # Print summary
    print("\nHeat Stress Analysis Summary:")
    print("=" * 50)
    for _, row in results.iterrows():
        print(f"\nDecade {int(row['decade'])}s:")
        print(f"  Total days analyzed: {int(row['total_days'])}")
        print(f"  Days with any heat stress: {row['any_stress_pct']:.1f}%")
        print(f"  Average maximum temperature: {row['avg_tmax']:.1f}°C")
        print(f"  Average heat index: {row['avg_hi']:.1f}°C")
        print(f"  Average humidex: {row['avg_hx']:.1f}°C")
