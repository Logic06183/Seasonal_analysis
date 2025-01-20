import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from scipy import stats
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

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
    decade_stats = df.groupby('decade').agg({
        'temp_stress': 'mean',
        'hi_stress': 'mean',
        'hx_stress': 'mean',
        'any_stress': 'mean',
        'tmax': ['mean', 'max'],
        'heat_index': ['mean', 'max'],
        'humidex': ['mean', 'max']
    })
    
    # Flatten column names
    decade_stats.columns = ['_'.join(col).strip() for col in decade_stats.columns.values]
    decade_stats = decade_stats.reset_index()
    
    # Create visualizations
    plot_heat_stress_trends(decade_stats)
    plot_calendar_heatmap(df)
    seasonal_patterns = plot_seasonal_patterns(df)
    
    # Print seasonal analysis
    print("\nSeasonal Analysis of Heat Stress:")
    print("=" * 50)
    for season in seasonal_patterns.columns:
        trend = np.polyfit(range(len(seasonal_patterns.index)), 
                          seasonal_patterns[season].values, 1)[0]
        print(f"\n{season}:")
        print(f"  Average heat stress days: {seasonal_patterns[season].mean():.1f}%")
        print(f"  Trend: {trend:.2f}% change per year")
    
    return decade_stats

def plot_heat_stress_trends(results_df):
    """Plot trends in heat stress metrics over decades"""
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Heat stress days percentage
    plt.subplot(2, 2, 1)
    metrics = ['temp_stress', 'hi_stress', 'hx_stress']
    labels = ['Temperature', 'Heat Index', 'Humidex']
    for metric, label in zip(metrics, labels):
        plt.plot(results_df['decade'], results_df[f'{metric}_mean'] * 100, 
                marker='o', label=label)
    plt.title('Percentage of Heat Stress Days by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Days')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Average maximum values
    plt.subplot(2, 2, 2)
    metrics = ['tmax', 'heat_index', 'humidex']
    labels = ['Temperature', 'Heat Index', 'Humidex']
    for metric, label in zip(metrics, labels):
        plt.plot(results_df['decade'], results_df[f'{metric}_mean'], 
                marker='o', label=label)
    plt.title('Average Heat Stress Indices by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Maximum values
    plt.subplot(2, 2, 3)
    metrics = ['tmax', 'heat_index', 'humidex']
    labels = ['Temperature', 'Heat Index', 'Humidex']
    for metric, label in zip(metrics, labels):
        plt.plot(results_df['decade'], results_df[f'{metric}_max'], 
                marker='o', label=label)
    plt.title('Maximum Heat Stress Values by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Combined heat stress days
    plt.subplot(2, 2, 4)
    plt.plot(results_df['decade'], results_df['any_stress_mean'] * 100, 
             marker='o', color='red', linewidth=2)
    plt.title('Combined Heat Stress Days by Decade')
    plt.xlabel('Decade')
    plt.ylabel('Percentage of Days')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(fig_dir / 'heat_stress_trends.png', dpi=300, bbox_inches='tight')
    plt.close()

def set_economist_style():
    """Set the plotting style to match The Economist magazine with climate stripes colors"""
    # Colors based on Ed Hawkins' climate stripes
    colors = {
        'background': '#ffffff',
        'text': '#2f3030',
        'grid': '#d5d7d9',
        'highlight': '#67001f',  # Dark red from climate stripes
        'accent': '#2166ac',     # Dark blue from climate stripes
        'cold': '#053061',       # Coldest blue
        'warm': '#67001f'        # Warmest red
    }
    
    plt.rcParams.update({
        'figure.facecolor': colors['background'],
        'axes.facecolor': colors['background'],
        'axes.edgecolor': colors['text'],
        'axes.labelcolor': colors['text'],
        'text.color': colors['text'],
        'grid.color': colors['grid'],
        'grid.alpha': 0.3,
        'axes.grid': True,
        'grid.linestyle': '-',
        'xtick.color': colors['text'],
        'ytick.color': colors['text'],
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.constrained_layout.use': True
    })
    
    return colors

def set_ft_style():
    """Set Financial Times style for plots"""
    colors = {
        'background': '#FFF1E5',  # FT background cream color
        'text': '#333333',        # Dark gray for text
        'grid': '#CCC4BC',        # Light gray for grid
        'accent': '#990F3D',      # FT red
        'highlight': '#0F5499'    # FT blue
    }
    
    plt.rcParams.update({
        'figure.facecolor': colors['background'],
        'axes.facecolor': colors['background'],
        'axes.edgecolor': colors['text'],
        'axes.labelcolor': colors['text'],
        'text.color': colors['text'],
        'grid.color': colors['grid'],
        'grid.alpha': 0.2,
        'grid.linestyle': '-',
        'xtick.color': colors['text'],
        'ytick.color': colors['text'],
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.constrained_layout.use': True
    })
    
    return colors

def plot_calendar_heatmap(df):
    """Create calendar heatmap of heat stress days"""
    colors = set_ft_style()
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[4, 1], hspace=0.25)
    
    # Main heatmap subplot
    ax = fig.add_subplot(gs[0])
    
    # Prepare data
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Calculate monthly percentage of heat stress days
    monthly_stress = df.groupby(['year', 'month'])['any_stress'].mean() * 100
    monthly_stress = monthly_stress.unstack()
    
    # Create custom colormap
    colors_stress = ['#FFF1E5', '#FFE1CC', '#FFD1B2', '#FFC299', '#FFB380', 
                    '#FFA366', '#FF944D', '#FF8533', '#FF751A', '#FF6600']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors_stress)
    
    # Set color normalization
    vmin = 0
    vmax = monthly_stress.max().max()
    center = vmax / 2
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    
    # Create heatmap
    im = ax.imshow(monthly_stress, cmap=custom_cmap, norm=norm, aspect='auto', interpolation='nearest')
    
    # Customize appearance
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    bold_labels = [f'$\\mathbf{{{label}}}$' if i in [9, 10, 11, 0, 1, 2] else label 
                  for i, label in enumerate(month_labels)]
    
    # Set tick labels with increased font size
    ax.set_xticks(np.arange(len(month_labels)))
    ax.set_xticklabels(bold_labels, rotation=0, ha='center', fontsize=11)
    ax.set_yticks(np.arange(len(monthly_stress.index)))
    ax.set_yticklabels(monthly_stress.index, fontsize=11)
    
    # Add grid
    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(monthly_stress.index), 1), minor=True)
    ax.grid(which='minor', color=colors['grid'], linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Enhance colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.ax.set_ylabel('Share of days with heat stress (%)', 
                      fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=10)
    
    # Add title with improved spacing
    ax.set_title('Rising frequency of heat stress days at Rahima Moosa Hospital\n', 
                pad=30, fontsize=14, fontweight='bold', loc='left')
    
    # Add subtitle with improved spacing
    ax.text(-2, -2, 'Summer months (Oct-Mar) shown in bold', 
            fontsize=10, style='italic')
    
    # Add heat stress definition in a text box
    ax_def = fig.add_subplot(gs[1])
    ax_def.axis('off')
    definition_text = (
        'Heat stress definition:\n'
        '• Days where temperature exceeds the 80th percentile of historical records\n'
        '• Analysis based on ERA5-Land reanalysis data (1980-2024)\n'
        '• Source: Rahima Moosa Mother and Child Hospital Heat Stress Study'
    )
    ax_def.text(0.5, 0.5, definition_text, 
                ha='center', va='center', 
                fontsize=10, 
                bbox=dict(facecolor=colors['background'], 
                         edgecolor=colors['grid'],
                         alpha=0.9,
                         pad=10))
    
    # Add source information with more space from bottom
    fig.text(0.02, 0.01, 
             'Source: ERA5-Land reanalysis data (1980-2024)\n' +
             'Analysis: Rahima Moosa Mother and Child Hospital Heat Stress Study',
             fontsize=8, color=colors['text'], va='bottom')
    
    # Save figure with proper layout
    plt.savefig(fig_dir / 'heat_stress_calendar.png', 
                dpi=300, bbox_inches='tight',
                facecolor=colors['background'])
    plt.close()

def plot_seasonal_patterns(df):
    """Create enhanced visualization of seasonal patterns in heat stress"""
    # Set FT style
    colors = set_ft_style()
    
    # Create figure with two subplots (main plot and annual average)
    fig = plt.figure(figsize=(12, 8))
    
    # Create GridSpec with figure and adjust spacing
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.4)  # Increased spacing
    
    # Main subplot for seasonal patterns
    ax1 = fig.add_subplot(gs[0])
    
    # Define seasons and their properties (using FT-inspired palette)
    seasons = {
        'Summer': {'color': colors['highlight'], 'marker': 'o', 'months': [12, 1, 2], 'emphasis': True},
        'Spring': {'color': colors['accent'], 'marker': 's', 'months': [9, 10, 11], 'emphasis': True},
        'Autumn': {'color': '#70A9D6', 'marker': '^', 'months': [3, 4, 5], 'emphasis': False},
        'Winter': {'color': '#B3BE95', 'marker': 'D', 'months': [6, 7, 8], 'emphasis': False}
    }
    
    # Calculate seasonal patterns
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['season'] = pd.cut(df['month'], 
                         bins=[0, 2, 5, 8, 11, 12],
                         labels=['Summer', 'Autumn', 'Winter', 'Spring', 'Summer'],
                         ordered=False)
    
    seasonal_stress = df.groupby([df['date'].dt.year, 'season'], observed=False)['any_stress'].mean() * 100
    seasonal_stress = seasonal_stress.unstack()
    
    # Plot each season
    trend_info = []
    for season in seasons:
        data = seasonal_stress[season]
        years = data.index
        
        # Calculate trend
        X = np.arange(len(years)).reshape(-1, 1)
        y = data.values
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        
        # Calculate confidence intervals
        conf = 0.95
        n = len(years)
        mean_x = X.mean()
        se = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2) / np.sum((X - mean_x) ** 2))
        alpha = 1 - conf
        t_val = stats.t.ppf(1 - alpha / 2, n - 2)
        ci = t_val * se * np.sqrt(1 + 1/n + (X - mean_x) ** 2 / np.sum((X - mean_x) ** 2))
        ci = ci.flatten()
        
        # Plot confidence interval with increased opacity for emphasized seasons
        ci_alpha = 0.2 if seasons[season]['emphasis'] else 0.1
        ax1.fill_between(years, y_pred.flatten() - ci, y_pred.flatten() + ci, 
                        color=seasons[season]['color'], alpha=ci_alpha)
        
        # Line properties based on emphasis
        line_width = 2.5 if seasons[season]['emphasis'] else 1.5
        marker_size = 7 if seasons[season]['emphasis'] else 4
        alpha = 1.0 if seasons[season]['emphasis'] else 0.6
        
        # Plot data points and trend
        line = ax1.plot(years, data, 
                       color=seasons[season]['color'], 
                       marker=seasons[season]['marker'],
                       markersize=marker_size, 
                       label=f'{season}',
                       linewidth=line_width,
                       alpha=alpha,
                       zorder=3 if seasons[season]['emphasis'] else 2)
        
        # Plot trend line with increased width
        trend_width = 2.0 if seasons[season]['emphasis'] else 1.2
        ax1.plot(years, y_pred, 
                color=seasons[season]['color'],
                linestyle='--',
                linewidth=trend_width,
                alpha=alpha,
                zorder=2)
        
        # Store trend information
        trend_info.append({
            'season': season,
            'trend': reg.coef_[0],
            'r2': reg.score(X, y),
            'total_change': y_pred[-1] - y_pred[0]
        })
        
        # Add trend annotations with improved positioning and arrows
        if seasons[season]['emphasis']:
            mid_year = years[len(years)//2]
            mid_value = y_pred[len(years)//2]
            
            # Adjust positions for trend labels
            if season == 'Summer':
                ax1.annotate(f"{season}: {reg.coef_[0]:.2f}%/year",
                            xy=(mid_year, mid_value),
                            xytext=(-50, 30),  # Offset above the line
                            textcoords='offset points',
                            color=seasons[season]['color'],
                            bbox=dict(facecolor=colors['background'], 
                                    edgecolor=seasons[season]['color'],
                                    alpha=0.9,
                                    pad=3),
                            fontsize=9,
                            arrowprops=dict(arrowstyle='->',
                                          color=seasons[season]['color'],
                                          connectionstyle='angle3,angleA=0,angleB=90'))
            else:  # Spring
                ax1.annotate(f"{season}: {reg.coef_[0]:.2f}%/year",
                            xy=(mid_year, mid_value),
                            xytext=(50, -30),  # Offset below the line
                            textcoords='offset points',
                            color=seasons[season]['color'],
                            bbox=dict(facecolor=colors['background'], 
                                    edgecolor=seasons[season]['color'],
                                    alpha=0.9,
                                    pad=3),
                            fontsize=9,
                            arrowprops=dict(arrowstyle='->',
                                          color=seasons[season]['color'],
                                          connectionstyle='angle3,angleA=0,angleB=-90'))
        
        # Add peak annotations with improved positioning
        if season in ['Summer', 'Spring']:
            peak_year = years[np.argmax(data)]
            peak_value = data.max()
            
            # Adjust positions for peak labels
            if season == 'Summer':
                ax1.annotate(f'{season} peak ({int(peak_year)})',
                            xy=(peak_year, peak_value),
                            xytext=(-40, 15),  # Offset left and up
                            textcoords='offset points',
                            color=seasons[season]['color'],
                            bbox=dict(facecolor=colors['background'], 
                                    edgecolor=seasons[season]['color'],
                                    alpha=0.9,
                                    pad=3),
                            fontsize=9,
                            arrowprops=dict(arrowstyle='->',
                                          color=seasons[season]['color'],
                                          connectionstyle='arc3,rad=-0.2'))
            else:  # Spring
                ax1.annotate(f'{season} peak ({int(peak_year)})',
                            xy=(peak_year, peak_value),
                            xytext=(40, 15),  # Offset right and up
                            textcoords='offset points',
                            color=seasons[season]['color'],
                            bbox=dict(facecolor=colors['background'], 
                                    edgecolor=seasons[season]['color'],
                                    alpha=0.9,
                                    pad=3),
                            fontsize=9,
                            arrowprops=dict(arrowstyle='->',
                                          color=seasons[season]['color'],
                                          connectionstyle='arc3,rad=0.2'))
    
    # Move legend outside plot area
    legend_elements = []
    legend_labels = []
    for season_info in trend_info:
        season = season_info['season']
        legend_elements.append(Line2D([0], [0], color=seasons[season]['color'],
                                    marker=seasons[season]['marker'],
                                    label=season, 
                                    linewidth=2.0 if seasons[season]['emphasis'] else 1.5,
                                    markersize=6 if seasons[season]['emphasis'] else 4))
        legend_labels.append(f"{season}")
    
    leg = ax1.legend(legend_elements, legend_labels,
                    title='Seasonal Trends',
                    bbox_to_anchor=(1.15, 1.0),
                    loc='upper left',
                    fontsize=9,
                    title_fontsize=10,
                    frameon=True,
                    facecolor=colors['background'])
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor(colors['grid'])
    
    # Customize main plot
    ax1.grid(True, alpha=0.2, linestyle='-', color=colors['grid'])
    ax1.set_xlabel('Year', fontsize=10, color=colors['text'])
    ax1.set_ylabel('Share of days with heat stress (%)', fontsize=10, color=colors['text'])
    
    # Set title with improved spacing
    title_text = 'Seasonal Patterns of Heat Stress\nat Rahima Moosa Hospital (1980-2024)'
    subtitle_text = 'Heat stress defined by temperature >80th percentile'
    
    # Add title and subtitle with adjusted spacing
    fig.text(0.08, 0.99, title_text,  # Moved right, normal y-range
             fontsize=12, color=colors['text'], ha='left', va='top')
    fig.text(0.08, 0.95, subtitle_text,  # Moved right, normal y-range
             fontsize=9, style='italic', color=colors['text'], ha='left', va='top')
    
    # Annual average subplot
    ax2 = fig.add_subplot(gs[1])
    annual_avg = df.groupby(df['date'].dt.year)['any_stress'].mean() * 100
    
    # Calculate trend for annual average
    X = np.arange(len(annual_avg.index)).reshape(-1, 1)
    y = annual_avg.values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    
    # Calculate confidence intervals
    conf = 0.95
    n = len(annual_avg.index)
    mean_x = X.mean()
    se = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2) / np.sum((X - mean_x) ** 2))
    alpha = 1 - conf
    t_val = stats.t.ppf(1 - alpha / 2, n - 2)
    ci = t_val * se * np.sqrt(1 + 1/n + (X - mean_x) ** 2 / np.sum((X - mean_x) ** 2))
    
    # Plot annual average with improved legend positioning
    ax2.plot(annual_avg.index, annual_avg, color=colors['text'], 
             marker='o', markersize=4, label='Annual average',
             alpha=0.8, linewidth=1.5, zorder=3)
    ax2.plot(annual_avg.index, y_pred, color=colors['text'], 
             linestyle='--', linewidth=1.5,
             label=f'Trend (R² = {reg.score(X, y):.2f})',
             zorder=2)
    
    # Customize annual average subplot
    ax2.grid(True, alpha=0.2, linestyle='-', color=colors['grid'])
    ax2.set_xlabel('Year', fontsize=10, labelpad=10)
    ax2.set_ylabel('Annual\naverage (%)', fontsize=10, labelpad=10)
    
    # Move legend to upper left to avoid overlap
    leg2 = ax2.legend(loc='upper left', 
                     bbox_to_anchor=(0.02, 0.95),
                     fontsize=8,
                     frameon=True,
                     facecolor=colors['background'])
    leg2.get_frame().set_alpha(0.9)
    leg2.get_frame().set_edgecolor(colors['grid'])
    
    # Add source information with adjusted position
    source_text = ('Source: ERA5-Land reanalysis data (1980-2024)\n' +
                  'Analysis: Rahima Moosa Mother and Child Hospital Heat Stress Study')
    fig.text(0.02, 0.02, source_text,
             fontsize=8, color=colors['text'],
             va='bottom', ha='left')
    
    # Adjust figure margins to accommodate legend
    plt.subplots_adjust(right=0.85, bottom=0.15)
    
    # Save figure
    plt.savefig(fig_dir / 'seasonal_patterns.png', 
                dpi=300, bbox_inches='tight',
                facecolor=colors['background'])
    plt.close()

    return seasonal_stress

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
        print(f"  Days with any heat stress: {row['any_stress_mean']*100:.1f}%")
        print(f"  Average maximum temperature: {row['tmax_mean']:.1f}°C")
        print(f"  Average heat index: {row['heat_index_mean']:.1f}°C")
        print(f"  Average humidex: {row['humidex_mean']:.1f}°C")
