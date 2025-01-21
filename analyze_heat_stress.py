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
import seaborn as sns
import matplotlib.animation as animation

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
    # Set style and consistent font sizes
    colors = set_ft_style()
    TITLE_SIZE = 14
    SUBTITLE_SIZE = 11
    LABEL_SIZE = 10
    ANNOT_SIZE = 9
    LEGEND_SIZE = 8
    
    # Create figure with adjusted spacing
    fig = plt.figure(figsize=(13, 8))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.15)
    
    # Top panel - Seasonal patterns
    ax1 = fig.add_subplot(gs[0])
    
    # Calculate seasonal statistics
    seasonal_stress = calculate_seasonal_stats(df)
    
    # Set consistent x-axis limits with padding
    year_min, year_max = df['date'].dt.year.min(), df['date'].dt.year.max()
    ax1.set_xlim(year_min - 1, year_max + 1)
    
    # Plot lines for each season
    seasons = ['Summer', 'Spring', 'Autumn', 'Winter']
    season_colors = [colors['highlight'], colors['accent'], 
                    '#70A9D6', '#B3BE95']
    
    # Create legend first to get its height
    lines = []
    labels = []
    for season, color in zip(seasons, season_colors):
        line, = ax1.plot([], [], marker='o', color=color, 
                        label=season, linewidth=1.5, markersize=3)
        lines.append(line)
        labels.append(season)
    
    # Position legend in top-left with specific coordinates
    legend = ax1.legend(lines, labels, loc='upper left', frameon=True,
                       facecolor='white', framealpha=0.9,
                       fontsize=LEGEND_SIZE, borderaxespad=0.5,
                       bbox_to_anchor=(0.02, 0.85))
    
    # Now plot the actual data
    for season, color, line in zip(seasons, season_colors, lines):
        data = seasonal_stress[season]
        years = seasonal_stress.index
        
        # Update line data
        line.set_data(years, data)
        
        # Calculate and plot trend
        X = np.arange(len(years))
        reg = LinearRegression().fit(X.reshape(-1, 1), data)
        y_pred = reg.predict(X.reshape(-1, 1))
        trend_line = ax1.plot(years, y_pred, '--', 
                            color=color, alpha=0.4, linewidth=1)
        
        # Add trend rate for Summer and Spring with adjusted positions
        if season in ['Summer', 'Spring']:
            trend_rate = reg.coef_[0]  # Change per year
            if season == 'Summer':
                ax1.annotate(f'{season} trend: +{trend_rate:.2f}%/year',
                            xy=(1990, 70), xytext=(1982, 95),
                            color=color, fontsize=ANNOT_SIZE, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=color, alpha=0.6))
            else:
                ax1.annotate(f'{season} trend: +{trend_rate:.2f}%/year',
                            xy=(1990, 35), xytext=(1982, 65),
                            color=color, fontsize=ANNOT_SIZE, fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color=color, alpha=0.6))
    
    # Customize top panel with consistent padding
    ax1.set_title('Seasonal Patterns of Heat Stress in Johannesburg (1980-2024)', 
                 pad=20, fontsize=TITLE_SIZE, fontweight='bold', 
                 family='sans-serif', y=1.04)
    
    # Add peak annotations with improved positioning
    ax1.annotate('Summer peak (2015)', 
                xy=(2015, 78), xytext=(2016, 90),
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=colors['highlight'], alpha=0.9),
                color=colors['highlight'],
                fontweight='bold', fontsize=ANNOT_SIZE,
                arrowprops=dict(arrowstyle='->', color=colors['highlight'], alpha=0.6))
    
    ax1.annotate('Spring peak (2019)', 
                xy=(2019, 70), xytext=(2020, 82),
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=colors['accent'], alpha=0.9),
                color=colors['accent'],
                fontweight='bold', fontsize=ANNOT_SIZE,
                arrowprops=dict(arrowstyle='->', color=colors['accent'], alpha=0.6))
    
    # Customize axes with consistent styling
    ax1.set_ylabel('Share of days with heat stress (%)',
                  fontsize=LABEL_SIZE, labelpad=10)
    ax1.set_ylim(0, 100)  # Increased to accommodate annotations
    ax1.grid(True, alpha=0.12, linestyle='--')
    ax1.tick_params(axis='both', labelsize=ANNOT_SIZE)
    
    # Bottom panel - Annual average
    ax2 = fig.add_subplot(gs[1])
    annual_avg = df.groupby(df['date'].dt.year)['any_stress'].mean() * 100
    
    # Set consistent x-axis limits
    ax2.set_xlim(year_min - 1, year_max + 1)
    
    # Plot annual average
    ax2.plot(annual_avg.index, annual_avg, color='black', 
             label='Annual average', marker='o', 
             linewidth=1, markersize=3)
    
    # Calculate and plot trend
    X = np.arange(len(annual_avg))
    reg = LinearRegression().fit(X.reshape(-1, 1), annual_avg)
    y_pred = reg.predict(X.reshape(-1, 1))
    r2 = reg.score(X.reshape(-1, 1), annual_avg)
    
    ax2.plot(annual_avg.index, y_pred, '--', 
             color='black', alpha=0.4,
             label=f'Trend (R² = {r2:.2f})')
    
    # Customize bottom panel with consistent styling
    ax2.set_ylabel('Annual average (%)',
                  fontsize=LABEL_SIZE, labelpad=10)
    ax2.set_ylim(0, 45)
    ax2.grid(True, alpha=0.12, linestyle='--')
    ax2.tick_params(axis='both', labelsize=ANNOT_SIZE)
    
    # Position legend in empty space
    ax2.legend(loc='upper left', frameon=True, 
              facecolor='white', framealpha=0.9,
              fontsize=LEGEND_SIZE, borderaxespad=0.5)
    
    # Add definition note with improved spacing
    plt.figtext(0.02, -0.02, 
                'Heat stress defined by temperature >80th percentile\n'
                'Source: ERA5 reanalysis data (1980-2024) | Analysis: Johannesburg Heat Stress Study',
                fontsize=8, style='italic', color='#666666', 
                va='bottom')
    
    # Adjust layout with more space for annotations
    plt.subplots_adjust(right=0.95, left=0.1, bottom=0.18, top=0.95)
    
    # Save the figure
    plt.savefig(fig_dir / 'seasonal_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return seasonal_stress

def calculate_seasonal_stats(df):
    """Calculate seasonal statistics for heat stress"""
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Define seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Summer'
        elif month in [3, 4, 5]:
            return 'Autumn'
        elif month in [6, 7, 8]:
            return 'Winter'
        else:  # [9, 10, 11]
            return 'Spring'
    
    df['season'] = df['month'].map(get_season)
    
    # Calculate seasonal averages
    seasonal_stress = df.groupby(['year', 'season'])['any_stress'].mean() * 100
    seasonal_stress = seasonal_stress.unstack()
    
    # Ensure all seasons are present
    for season in ['Summer', 'Spring', 'Autumn', 'Winter']:
        if season not in seasonal_stress.columns:
            seasonal_stress[season] = 0
    
    return seasonal_stress[['Summer', 'Spring', 'Autumn', 'Winter']]

def plot_animated_calendar_heatmap(df):
    """Create animated calendar heatmap showing historical progression of heat stress"""
    colors = set_ft_style()
    
    # Create figure
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    
    # Prepare data by year
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['any_stress_pct'] = df['any_stress'] * 100
    years = sorted(df['year'].unique())
    
    def init():
        ax.clear()
        ax.set_title('Heat Stress Days at Rahima Moosa Hospital\n', 
                    pad=20, fontsize=14, fontweight='bold', loc='left')
        return ax,
    
    def animate(frame):
        ax.clear()
        
        # Get data up to current year in animation
        current_year = years[frame]
        mask = df['year'] <= current_year
        temp_df = df[mask].copy()
        
        # Create pivot table for heatmap
        pivot_data = temp_df.pivot_table(
            index='year',
            columns='month',
            values='any_stress_pct',
            aggfunc='mean'
        ).fillna(0)  # Fill NaN with 0
        
        # Ensure all months are present
        for month in range(1, 13):
            if month not in pivot_data.columns:
                pivot_data[month] = 0
        pivot_data = pivot_data.reindex(columns=range(1, 13))
        
        # Create heatmap
        sns.heatmap(pivot_data, 
                   ax=ax,
                   cmap='YlOrRd',
                   vmin=0,
                   vmax=100,
                   cbar_kws={'label': '% of days with heat stress'},
                   fmt='.0f',
                   annot=True)
        
        # Customize appearance
        ax.set_xlabel('')
        ax.set_ylabel('Year')
        
        # Format month labels
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(np.arange(12) + 0.5)
        ax.set_xticklabels(month_labels, rotation=0)
        
        # Bold spring/summer months
        for i, label in enumerate(ax.get_xticklabels()):
            if i+1 in [9, 10, 11, 12, 1, 2]:  # Spring and Summer months
                label.set_weight('bold')
        
        ax.set_title(f'Heat Stress Days at Rahima Moosa Hospital (1980-{current_year})\n', 
                    pad=20, fontsize=14, fontweight='bold', loc='left')
        
        # Add subtitle
        plt.figtext(0.02, 0.02, 'Spring and Summer months shown in bold', 
                   fontsize=10, style='italic')
        
        return ax,
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                 frames=len(years), interval=500,
                                 blit=True, repeat=True)
    
    # Save animation with higher quality settings
    anim.save(fig_dir / 'heat_stress_calendar_animated.gif',
              writer='pillow', fps=1.5, dpi=150)
    
    plt.close()

if __name__ == "__main__":
    # Load data
    historical_df = pd.read_csv('data/era5/era5_historical_metrics.csv')
    current_df = pd.read_csv('data/era5/era5_current_metrics.csv')
    
    # Existing analysis
    analyze_heat_stress(historical_df, current_df)
    
    # Create animated version
    df = pd.concat([historical_df, current_df])
    df['date'] = pd.to_datetime(df['date'])
    
    # Calculate heat stress metrics
    df['heat_index'] = calculate_heat_index(df['tmax'], df['rh'])
    df['humidex'] = calculate_humidex(df['tmax'], df['rh'])
    
    # Define heat stress thresholds
    temp_threshold = np.percentile(df['tmax'], 80)
    hi_threshold = 32  # Heat Index > 32°C (moderate heat stress)
    hx_threshold = 35  # Humidex > 35°C (some discomfort)
    
    # Calculate heat stress days
    df['temp_stress'] = df['tmax'] > temp_threshold
    df['hi_stress'] = df['heat_index'] > hi_threshold
    df['hx_stress'] = df['humidex'] > hx_threshold
    df['any_stress'] = df['temp_stress'] | df['hi_stress'] | df['hx_stress']
    
    # Create animation
    plot_animated_calendar_heatmap(df)
