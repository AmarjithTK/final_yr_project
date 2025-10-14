import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Industrial data Config
TOTAL_DAYS = 5
INTERVAL_MINUTES = 15
INTERVALS_PER_DAY = 96

NOMINAL_VOLTAGE = 400
PEAK_CURRENT = 2000
BASE_CURRENT = 500
PEAK_POWER = 1500
BASE_POWER_FACTOR = 0.88

# Generate timestamps (UTC ISO8601)
start_date = datetime(2025, 1, 1)
timestamps = [
    (start_date + timedelta(minutes=INTERVAL_MINUTES * i)).strftime('%Y-%m-%dT%H:%M:%SZ')
    for i in range(TOTAL_DAYS * INTERVALS_PER_DAY)
]

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

def get_hourly_load_factor(hour, is_weekend):
    if is_weekend:
        if 8 <= hour < 18:
            return 0.50 + np.random.uniform(-0.05, 0.05)
        else:
            return 0.25 + np.random.uniform(-0.03, 0.03)
    else:
        if hour < 5:
            return 0.30 + np.random.uniform(-0.02, 0.02)
        elif 5 <= hour < 6:
            return 0.35 + (hour - 5) * 0.30 + np.random.uniform(-0.05, 0.05)
        elif 6 <= hour < 8:
            progress = (hour - 6) / 2.0
            return 0.60 + progress * 0.30 + np.random.uniform(-0.08, 0.08)
        elif 8 <= hour < 12:
            return 0.92 + np.random.uniform(-0.05, 0.08)
        elif 12 <= hour < 13:
            return 0.70 + np.random.uniform(-0.05, 0.05)
        elif 13 <= hour < 18:
            return 0.90 + np.random.uniform(-0.05, 0.08)
        elif 18 <= hour < 22:
            return 0.82 + np.random.uniform(-0.05, 0.07)
        elif 22 <= hour < 24:
            progress = (hour - 22) / 2.0
            return 0.70 - progress * 0.40 + np.random.uniform(-0.05, 0.05)
    return 0.30

def get_power_factor(load_factor, is_startup=False):
    if is_startup:
        return np.clip(0.68 + np.random.uniform(-0.05, 0.05), 0.60, 0.75)
    if load_factor > 0.85:
        pf = 0.88 + np.random.uniform(-0.02, 0.05)
    elif load_factor > 0.70:
        pf = 0.85 + np.random.uniform(-0.03, 0.05)
    elif load_factor > 0.50:
        pf = 0.80 + np.random.uniform(-0.03, 0.04)
    else:
        pf = 0.75 + np.random.uniform(-0.05, 0.05)
    return np.clip(pf, 0.60, 0.98)

print(f"Generating {TOTAL_DAYS} days of industrial load data...")

for idx, ts in enumerate(timestamps):
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(timestamps)} records ({100*idx/len(timestamps):.1f}%)")
    ts_dt = datetime.strptime(ts, '%Y-%m-%dT%H:%M:%SZ')
    hour = ts_dt.hour
    minute = ts_dt.minute
    is_weekend = ts_dt.weekday() >= 5

    is_startup = False
    if not is_weekend:
        if (6 <= hour <= 7 and minute < 30) or (13 <= hour <= 13 and minute < 30):
            is_startup = True

    load_factor = get_hourly_load_factor(hour, is_weekend)
    if np.random.random() < 0.15:
        load_factor *= np.random.uniform(0.92, 1.08)
    load_factor = np.clip(load_factor, 0.15, 1.0)

    current = BASE_CURRENT + (PEAK_CURRENT - BASE_CURRENT) * load_factor
    if is_startup and np.random.random() < 0.3:
        current *= np.random.uniform(1.15, 1.35)
    current = np.clip(current, 100, 3000)

    base_voltage = NOMINAL_VOLTAGE * (1 + np.random.uniform(-0.03, 0.03))
    if current > 1800:
        base_voltage *= np.random.uniform(0.95, 0.98)
    if load_factor < 0.35:
        base_voltage *= np.random.uniform(1.00, 1.02)

    voltage_r = base_voltage * np.random.uniform(0.985, 1.000)
    voltage_y = base_voltage * np.random.uniform(0.995, 1.005)
    voltage_b = base_voltage * np.random.uniform(1.000, 1.015)
    voltage_r = np.clip(voltage_r, 380, 420)
    voltage_y = np.clip(voltage_y, 380, 420)
    voltage_b = np.clip(voltage_b, 380, 420)
    voltage_avg = (voltage_r + voltage_y + voltage_b) / 3

    pf = get_power_factor(load_factor, is_startup)
    active_power = (np.sqrt(3) * voltage_avg * current * pf) / 1000
    theta = np.arccos(pf)
    reactive_power = active_power * np.tan(theta)
    kwh_15min = active_power * 0.25

    data['voltage_r'].append(round(voltage_r, 2))
    data['voltage_y'].append(round(voltage_y, 2))
    data['voltage_b'].append(round(voltage_b, 2))
    data['current'].append(round(current, 2))
    data['power_factor'].append(round(pf, 4))
    data['active_power_kw'].append(round(active_power, 2))
    data['reactive_power_kvar'].append(round(reactive_power, 2))
    data['kwh_15min'].append(round(kwh_15min, 2))

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')
output_filename = f'industrial_load_data_{TOTAL_DAYS}days.csv'
df.to_csv(output_filename, index=False)

print("="*70)
print("Dataset generated successfully!")
print("="*70)
print(f"Filename: {output_filename}")
print(f"Total records: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print("\nDataset preview:")
print(df.head(10))
print("\nStatistical Summary:")
print(df.describe())

df['date'] = pd.to_datetime(df['timestamp']).dt.date
daily_energy = df.groupby('date')['kwh_15min'].sum()
print("\nDaily energy consumption (kWh/day):")
print(daily_energy.head(10))
print(f"\nAverage daily consumption: {daily_energy.mean():.2f} kWh")
print(f"Weekday average: {daily_energy[daily_energy.index.map(lambda x: x.weekday() < 5)].mean():.2f} kWh")
print(f"Weekend average: {daily_energy[daily_energy.index.map(lambda x: x.weekday() >= 5)].mean():.2f} kWh")

total_energy = daily_energy.sum()
peak_power = df['active_power_kw'].max()
hours = TOTAL_DAYS * 24
load_factor = total_energy / (peak_power * hours)
print(f"\nLoad Factor: {load_factor:.3f}")
print(f"Peak Power: {peak_power:.2f} kW")
print(f"Average Power: {df['active_power_kw'].mean():.2f} kW")
