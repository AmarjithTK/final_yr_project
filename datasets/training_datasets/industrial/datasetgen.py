import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================
TOTAL_DAYS = 5  # Number of days to generate data for
INTERVAL_MINUTES = 15  # Time resolution
INTERVALS_PER_DAY = 96  # 24 hours * 4 intervals per hour

# Industrial plant parameters
NOMINAL_VOLTAGE = 400  # V line-to-line (three-phase)
PEAK_CURRENT = 2000  # A during full production
BASE_CURRENT = 500  # A during night/off hours
PEAK_POWER = 1500  # kW typical peak
BASE_POWER_FACTOR = 0.88  # at full load

# ============================================================================
# MAIN GENERATION CODE
# ============================================================================

# Generate timestamp array in ISO 8601 UTC format
start_date = datetime(2025, 1, 1, 0, 0, 0)
timestamps = [
    (start_date + timedelta(minutes=INTERVAL_MINUTES * i)).strftime('%Y-%m-%dT%H:%M:%SZ')
    for i in range(TOTAL_DAYS * INTERVALS_PER_DAY)
]

# Initialize data storage
data = {
    'timestamp': timestamps,
    'voltage_r': [],
    'voltage_y': [],
    'voltage_b': [],
    'current': [],
    'power_factor': [],
    'active_power_kw': [],
    'reactive_power_kvar': [],
    'kwh_15min': []
}

# Helper function to get load profile for hour of day
def get_hourly_load_factor(hour, is_weekend):
    """Returns load factor (0-1) based on hour and day type"""
    if is_weekend:
        # Weekend - reduced operation (Saturday/Sunday)
        if 8 <= hour < 18:
            return 0.50 + np.random.uniform(-0.05, 0.05)  # 50% load during day
        else:
            return 0.25 + np.random.uniform(-0.03, 0.03)  # 25% minimal load
    else:
        # Weekday - 2-shift operation (6 AM to 10 PM)
        if hour < 5:
            # Night baseline
            return 0.30 + np.random.uniform(-0.02, 0.02)
        elif 5 <= hour < 6:
            # Early morning ramp-up
            return 0.35 + (hour - 5) * 0.30 + np.random.uniform(-0.05, 0.05)
        elif 6 <= hour < 8:
            # Morning startup with ramp
            progress = (hour - 6) / 2.0
            return 0.60 + progress * 0.30 + np.random.uniform(-0.08, 0.08)
        elif 8 <= hour < 12:
            # Full morning production
            return 0.92 + np.random.uniform(-0.05, 0.08)
        elif 12 <= hour < 13:
            # Lunch break dip
            return 0.70 + np.random.uniform(-0.05, 0.05)
        elif 13 <= hour < 18:
            # Afternoon production
            return 0.90 + np.random.uniform(-0.05, 0.08)
        elif 18 <= hour < 22:
            # Evening shift (slightly lower)
            return 0.82 + np.random.uniform(-0.05, 0.07)
        elif 22 <= hour < 24:
            # Shutdown period
            progress = (hour - 22) / 2.0
            return 0.70 - progress * 0.40 + np.random.uniform(-0.05, 0.05)
    
    return 0.30  # fallback

# Helper function to get power factor based on load level
def get_power_factor(load_factor, is_startup=False):
    """Returns power factor based on load level"""
    if is_startup:
        # Lower power factor during startup
        return np.clip(0.68 + np.random.uniform(-0.05, 0.05), 0.60, 0.75)
    
    # Power factor improves with load
    if load_factor > 0.85:
        pf = 0.88 + np.random.uniform(-0.02, 0.05)
    elif load_factor > 0.70:
        pf = 0.85 + np.random.uniform(-0.03, 0.05)
    elif load_factor > 0.50:
        pf = 0.80 + np.random.uniform(-0.03, 0.04)
    else:
        pf = 0.75 + np.random.uniform(-0.05, 0.05)
    
    return np.clip(pf, 0.60, 0.98)

# Generate data for each timestamp
print(f"Generating {TOTAL_DAYS} days of industrial load data...")

for idx, ts in enumerate(timestamps):
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(timestamps)} records ({100*idx/len(timestamps):.1f}%)")
    
    ts_dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')  # Convert string to datetime
    hour = ts_dt.hour
    minute = ts_dt.minute
    is_weekend = ts_dt.weekday() >= 5  # Saturday=5, Sunday=6
    
    # Determine if this is a startup period (morning or post-break)
    is_startup = False
    if not is_weekend:
        if (6 <= hour <= 7 and minute < 30) or (13 <= hour <= 13 and minute < 30):
            is_startup = True
    
    # Get load factor for this time
    load_factor = get_hourly_load_factor(hour, is_weekend)
    
    # Add occasional random production variations
    if np.random.random() < 0.15:  # 15% chance of variation
        load_factor *= np.random.uniform(0.92, 1.08)
    
    load_factor = np.clip(load_factor, 0.15, 1.0)
    
    # Calculate current based on load factor
    current = BASE_CURRENT + (PEAK_CURRENT - BASE_CURRENT) * load_factor
    
    # Add inrush current effect during startup
    if is_startup and np.random.random() < 0.3:  # 30% chance during startup windows
        current *= np.random.uniform(1.15, 1.35)  # Averaged inrush effect at 15-min resolution
    
    current = np.clip(current, 100, 3000)
    
    # Generate three-phase voltages with slight imbalance
    base_voltage = NOMINAL_VOLTAGE * (1 + np.random.uniform(-0.03, 0.03))
    
    # Voltage sag during high current draws
    if current > 1800:
        base_voltage *= np.random.uniform(0.95, 0.98)
    
    # Voltage rise during low load
    if load_factor < 0.35:
        base_voltage *= np.random.uniform(1.00, 1.02)
    
    # Phase imbalance (1-3% between phases)
    voltage_r = base_voltage * np.random.uniform(0.985, 1.000)
    voltage_y = base_voltage * np.random.uniform(0.995, 1.005)
    voltage_b = base_voltage * np.random.uniform(1.000, 1.015)
    
    # Ensure within bounds
    voltage_r = np.clip(voltage_r, 380, 420)
    voltage_y = np.clip(voltage_y, 380, 420)
    voltage_b = np.clip(voltage_b, 380, 420)
    
    # Average line-to-line voltage
    voltage_avg = (voltage_r + voltage_y + voltage_b) / 3
    
    # Get power factor
    pf = get_power_factor(load_factor, is_startup)
    
    # Calculate active power (three-phase)
    # P = sqrt(3) * V_LL * I * cos(phi)
    active_power = (np.sqrt(3) * voltage_avg * current * pf) / 1000  # Convert to kW
    
    # Calculate reactive power
    # Q = P * tan(arccos(pf))
    theta = np.arccos(pf)
    reactive_power = active_power * np.tan(theta)
    
    # Calculate energy consumed in 15 minutes
    kwh_15min = active_power * 0.25  # 15 min = 0.25 hours
    
    # Store data
    data['voltage_r'].append(round(voltage_r, 2))
    data['voltage_y'].append(round(voltage_y, 2))
    data['voltage_b'].append(round(voltage_b, 2))
    data['current'].append(round(current, 2))
    data['power_factor'].append(round(pf, 4))
    data['active_power_kw'].append(round(active_power, 2))
    data['reactive_power_kvar'].append(round(reactive_power, 2))
    data['kwh_15min'].append(round(kwh_15min, 2))

# Create DataFrame
df = pd.DataFrame(data)

# Ensure timestamp column is string in ISO 8601 UTC format
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Save to CSV
output_filename = f'industrial_load_data_{TOTAL_DAYS}days.csv'
df.to_csv(output_filename, index=False)

print(f"\n{'='*70}")
print(f"Dataset generated successfully!")
print(f"{'='*70}")
print(f"Filename: {output_filename}")
print(f"Total records: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"\nDataset preview:")
print(df.head(10))
print(f"\nStatistical Summary:")
print(df.describe())

# Calculate and display daily energy consumption
df['date'] = pd.to_datetime(df['timestamp']).dt.date
daily_energy = df.groupby('date')['kwh_15min'].sum()
print(f"\nDaily energy consumption (kWh/day):")
print(daily_energy.head(10))
print(f"\nAverage daily consumption: {daily_energy.mean():.2f} kWh")
print(f"Weekday average: {daily_energy[daily_energy.index.map(lambda x: x.weekday() < 5)].mean():.2f} kWh")
print(f"Weekend average: {daily_energy[daily_energy.index.map(lambda x: x.weekday() >= 5)].mean():.2f} kWh")

# Calculate load factor
total_energy = daily_energy.sum()
peak_power = df['active_power_kw'].max()
hours = TOTAL_DAYS * 24
load_factor = total_energy / (peak_power * hours)
print(f"\nLoad Factor: {load_factor:.3f}")
print(f"Peak Power: {peak_power:.2f} kW")
print(f"Average Power: {df['active_power_kw'].mean():.2f} kW")
