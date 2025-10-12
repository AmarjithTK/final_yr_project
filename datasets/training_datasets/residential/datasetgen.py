import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================
START_DATE = datetime(2025, 1, 1, 0, 0, 0, tzinfo=pytz.UTC)  # Start date (UTC)
DAYS_TO_GEN = 90  # Number of days to generate data for
INTERVAL_MINUTES = 15  # Time resolution
INTERVALS_PER_DAY = 96  # 24 hours * 4 intervals per hour

# Output timestamp format compatible with InfluxDB / Grafana (RFC3339 / Zulu)
OUTPUT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Residential household parameters (Single-phase)
HOUSEHOLD_TYPE = "typical_3person"
NOMINAL_VOLTAGE = 230  # V (single-phase)
PEAK_POWER = 3.5  # kW (typical household peak)
BASE_POWER_NIGHT = 0.15  # kW (baseline - refrigerator, standby devices)
SEASON = "summer"  # Options: "summer", "winter", "moderate"

# Timezone for Grafana/InfluxDB (use UTC or local timezone)
TIMEZONE = pytz.UTC  # or pytz.timezone('Asia/Kolkata') for IST

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_residential_load_factor(hour, minute, is_weekend, season='moderate'):
    """
    Returns normalized load factor (0-1) for residential consumption
    Based on time of day, weekend status, and season
    """
    time_decimal = hour + minute / 60.0
    
    if is_weekend:
        return get_weekend_load_factor(time_decimal, season)
    
    # Weekday pattern with three peaks
    
    # Night baseline (12 AM - 5 AM)
    if time_decimal < 5.0:
        base = 0.05 + 0.03 * np.random.rand()  # 5-8% of peak
        if season == 'summer':
            base *= 2.0  # AC running through night
        return base
    
    # Early morning wake-up (5 AM - 6 AM)
    elif 5.0 <= time_decimal < 6.0:
        progress = (time_decimal - 5.0) / 1.0
        load = 0.08 + progress * 0.20  # 8% → 28%
        load += np.random.uniform(-0.05, 0.05)
        return np.clip(load, 0.05, 0.35)
    
    # Morning ramp-up (6 AM - 7 AM)
    elif 6.0 <= time_decimal < 7.0:
        progress = (time_decimal - 6.0) / 1.0
        load = 0.28 + progress * 0.30  # 28% → 58%
        load += np.random.uniform(-0.08, 0.08)
        return np.clip(load, 0.20, 0.65)
    
    # Morning peak (7 AM - 9 AM) - water heater, cooking, getting ready
    elif 7.0 <= time_decimal < 9.0:
        base = 0.57  # 57% of peak (2.0 kW for 3.5 kW peak)
        load = base + np.random.uniform(-0.10, 0.15)
        # Water heater cycling
        if np.random.rand() < 0.30:  # 30% chance water heater is on
            load += 0.15
        return np.clip(load, 0.45, 0.72)
    
    # Post-morning decline (9 AM - 11 AM)
    elif 9.0 <= time_decimal < 11.0:
        progress = (time_decimal - 9.0) / 2.0
        load = 0.57 - progress * 0.45  # 57% → 12%
        load += np.random.uniform(-0.05, 0.05)
        return np.clip(load, 0.08, 0.60)
    
    # Midday trough (11 AM - 4 PM) - house mostly empty
    elif 11.0 <= time_decimal < 16.0:
        base = 0.11  # 11% of peak (0.4 kW baseline)
        load = base + np.random.uniform(-0.03, 0.06)
        # Occasional midday cooking/laundry
        if np.random.rand() < 0.15:  # 15% chance
            load += 0.15
        return np.clip(load, 0.06, 0.25)
    
    # Afternoon return (4 PM - 6 PM)
    elif 16.0 <= time_decimal < 18.0:
        progress = (time_decimal - 16.0) / 2.0
        load = 0.11 + progress * 0.30  # 11% → 41%
        load += np.random.uniform(-0.05, 0.10)
        return np.clip(load, 0.10, 0.50)
    
    # Evening ramp-up (6 PM - 7 PM)
    elif 18.0 <= time_decimal < 19.0:
        progress = (time_decimal - 18.0) / 1.0
        load = 0.41 + progress * 0.35  # 41% → 76%
        load += np.random.uniform(-0.08, 0.12)
        return np.clip(load, 0.35, 0.85)
    
    # Evening peak (7 PM - 10 PM) - cooking, lighting, TV, AC
    elif 19.0 <= time_decimal < 22.0:
        base = 0.91  # 91% of peak (3.2 kW for 3.5 kW peak)
        load = base + np.random.uniform(-0.10, 0.09)
        # Peak load can occasionally hit 100%
        if np.random.rand() < 0.10:  # 10% chance
            load = 1.0
        if season == 'summer':
            load = min(1.0, load * 1.2)
        return np.clip(load, 0.75, 1.0)
    
    # Late evening decline (10 PM - 12 AM)
    elif 22.0 <= time_decimal < 24.0:
        progress = (time_decimal - 22.0) / 2.0
        load = 0.91 - progress * 0.80  # 91% → 11%
        load += np.random.uniform(-0.08, 0.05)
        return np.clip(load, 0.08, 0.90)
    
    return 0.10


def get_weekend_load_factor(time_decimal, season):
    """
    Weekend load pattern - more gradual, flatter, delayed morning peak
    """
    # Night (12 AM - 7 AM) - sleeping in
    if time_decimal < 7.0:
        base = 0.06 + 0.04 * np.random.rand()
        if season == 'summer':
            base *= 2.0
        return base
    
    # Late morning (7 AM - 10 AM) - gradual wake-up
    elif 7.0 <= time_decimal < 10.0:
        progress = (time_decimal - 7.0) / 3.0
        load = 0.10 + progress * 0.45  # Slower ramp
        load += np.random.uniform(-0.08, 0.08)
        return np.clip(load, 0.08, 0.60)
    
    # Morning peak (10 AM - 12 PM) - delayed, leisurely breakfast
    elif 10.0 <= time_decimal < 12.0:
        base = 0.55
        load = base + np.random.uniform(-0.10, 0.12)
        return np.clip(load, 0.45, 0.70)
    
    # Afternoon (12 PM - 5 PM) - higher than weekday (people at home)
    elif 12.0 <= time_decimal < 17.0:
        base = 0.35  # Much higher than weekday midday
        load = base + np.random.uniform(-0.08, 0.12)
        # Weekend activities - cooking, laundry, etc.
        if np.random.rand() < 0.30:
            load += 0.15
        return np.clip(load, 0.25, 0.55)
    
    # Evening (5 PM - 10 PM) - similar to weekday
    elif 17.0 <= time_decimal < 22.0:
        base = 0.85
        load = base + np.random.uniform(-0.10, 0.15)
        if season == 'summer':
            load = min(1.0, load * 1.15)
        return np.clip(load, 0.70, 1.0)
    
    # Night decline (10 PM - 12 AM)
    else:
        progress = (time_decimal - 22.0) / 2.0
        load = 0.85 - progress * 0.75
        return np.clip(load, 0.08, 0.85)


def apply_seasonal_adjustment(load_factor, hour, season):
    """
    Apply seasonal multipliers based on HVAC usage patterns
    """
    if season == 'summer':
        # Summer: AC usage increases load
        if 12 <= hour <= 16:  # Hottest part of day
            return min(1.0, load_factor * 1.4)
        elif 0 <= hour <= 5:  # AC through night
            return min(1.0, load_factor * 2.2)
        elif 18 <= hour <= 23:  # Evening AC
            return min(1.0, load_factor * 1.3)
        else:
            return min(1.0, load_factor * 1.15)
    
    elif season == 'winter':
        # Winter: Heating and water heater increase morning/evening loads
        if 6 <= hour <= 9:  # Morning heating + hot water
            return min(1.0, load_factor * 1.35)
        elif 18 <= hour <= 23:  # Evening heating
            return min(1.0, load_factor * 1.25)
        else:
            return min(1.0, load_factor * 1.1)
    
    else:  # Spring/Fall - moderate, lowest consumption
        return load_factor * 0.85


def generate_residential_voltage(load_kw, base_voltage=230):
    """
    Generate single-phase voltage with realistic variations
    """
    # Base voltage with ±2% normal variation
    voltage = base_voltage * (1 + np.random.uniform(-0.02, 0.02))
    
    # Voltage sag during high loads (>3 kW)
    if load_kw > 3.0:
        sag_percent = min(0.04, (load_kw - 3.0) * 0.01)
        voltage *= (1 - sag_percent)
    
    # Voltage rise during very low loads (<0.3 kW)
    elif load_kw < 0.3:
        voltage *= (1 + np.random.uniform(0.005, 0.015))
    
    # Ensure within bounds
    return np.clip(voltage, 220, 240)


def add_voltage_sag_events(voltage, interval_count):
    """
    Occasionally add severe voltage sag events for realism
    Drops to 100-200V range in ~0.5-2% of intervals
    """
    if np.random.rand() < 0.015:  # 1.5% probability
        sag_type = np.random.rand()
        
        if sag_type < 0.60:  # 60% - Moderate sag
            sag_factor = np.random.uniform(0.70, 0.87)  # 161-200V
        elif sag_type < 0.90:  # 30% - Deep sag
            sag_factor = np.random.uniform(0.52, 0.70)  # 120-161V
        else:  # 10% - Severe sag
            sag_factor = np.random.uniform(0.43, 0.52)  # 100-120V
        
        voltage *= sag_factor
        voltage = max(voltage, 80)  # Absolute minimum
    
    return voltage


def get_residential_power_factor(load_kw, time_hour):
    """
    Power factor varies based on load level and appliance mix
    """
    # High load (>2.5 kW) - AC/water heater/motors running
    if load_kw > 2.5:
        pf = 0.96 + np.random.uniform(-0.02, 0.02)
        # Motor startup events
        if np.random.rand() < 0.05:  # 5% chance
            pf = 0.92 + np.random.uniform(0, 0.03)
    
    # Medium load (0.8-2.5 kW) - mixed appliances
    elif 0.8 <= load_kw <= 2.5:
        pf = 0.97 + np.random.uniform(-0.02, 0.02)
    
    # Light load (<0.8 kW) - mostly electronics
    else:
        pf = 0.98 + np.random.uniform(-0.01, 0.02)
    
    return np.clip(pf, 0.95, 1.0)


def add_appliance_events(base_load_kw, hour, minute):
    """
    Add random appliance startup events for realism
    """
    load = base_load_kw
    
    # Water heater cycles (3-4 kW, runs 15-30 min)
    if 6 <= hour <= 9 or 18 <= hour <= 21:
        if np.random.rand() < 0.20:  # 20% chance
            load += np.random.uniform(2.5, 4.0)
    
    # Washing machine/dryer (1.5-2.5 kW)
    if 9 <= hour <= 21:
        if np.random.rand() < 0.08:  # 8% chance
            load += np.random.uniform(1.5, 2.5)
    
    # Microwave/oven (1-2 kW, brief)
    if (7 <= hour <= 9) or (12 <= hour <= 14) or (18 <= hour <= 21):
        if np.random.rand() < 0.15:  # 15% chance during meal times
            load += np.random.uniform(0.8, 2.0)
    
    return load


# ============================================================================
# MAIN DATA GENERATION
# ============================================================================

def main():
    print(f"Generating residential load data for {DAYS_TO_GEN} days...")
    print(f"Household Type: {HOUSEHOLD_TYPE.replace('_', ' ').title()}")
    print(f"Peak Power: {PEAK_POWER} kW")
    print(f"Season: {SEASON.title()}")
    print(f"Baseline Power: {BASE_POWER_NIGHT} kW")
    print(f"Start Date: {START_DATE.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Timezone: {TIMEZONE}")
    
    start_time = datetime.now()
    
    # Generate timestamp array with timezone-aware UTC datetimes
    timestamps = []
    for i in range(DAYS_TO_GEN * INTERVALS_PER_DAY):
        ts = START_DATE + timedelta(minutes=INTERVAL_MINUTES * i)  # tz-aware UTC
        timestamps.append(ts)
    
    # Initialize data storage
    data = {
        'timestamp': [],
        'voltage': [],
        'current': [],
        'power_factor': [],
        'active_power_kw': [],
        'reactive_power_kvar': [],
        'kwh_15min': []
    }
    
    # Generate data for each timestamp
    for idx, ts in enumerate(timestamps):
        hour = ts.hour
        minute = ts.minute
        day_of_week = ts.weekday()
        is_weekend = day_of_week >= 5  # 5=Saturday, 6=Sunday
        
        # Get base load factor
        load_factor = get_residential_load_factor(hour, minute, is_weekend, SEASON)
        
        # Apply seasonal adjustment
        load_factor = apply_seasonal_adjustment(load_factor, hour, SEASON)
        
        # Convert to actual power (kW)
        base_power = PEAK_POWER * load_factor
        
        # Add appliance events
        power_kw = add_appliance_events(base_power, hour, minute)
        
        # Add random noise (±10%)
        power_kw *= (1 + np.random.uniform(-0.10, 0.10))
        
        # Ensure within bounds
        power_kw = np.clip(power_kw, 0.05, 5.0)
        
        # Generate voltage
        voltage = generate_residential_voltage(power_kw)
        
        # Add occasional voltage sag events
        voltage = add_voltage_sag_events(voltage, idx)
        
        # Get power factor
        pf = get_residential_power_factor(power_kw, hour)
        
        # Calculate current (single-phase)
        current = power_kw * 1000 / (voltage * pf)
        current = np.clip(current, 0.5, 22)
        
        # Calculate reactive power
        theta = np.arccos(pf)
        reactive_power = power_kw * np.tan(theta)
        
        # Calculate 15-minute energy
        kwh_15min = power_kw * 0.25
        
        # Store timestamp in RFC3339 'Z' format (Influx/Grafana friendly, UTC)
        ts_utc = ts.astimezone(pytz.UTC)
        data['timestamp'].append(ts_utc.strftime(OUTPUT_TIME_FORMAT))
        data['voltage'].append(round(voltage, 2))
        data['current'].append(round(current, 2))
        data['power_factor'].append(round(pf, 4))
        data['active_power_kw'].append(round(power_kw, 2))
        data['reactive_power_kvar'].append(round(reactive_power, 2))
        data['kwh_15min'].append(round(kwh_15min, 2))
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"Progress: {idx + 1}/{len(timestamps)} records ({100*(idx+1)/len(timestamps):.1f}%)")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_filename = f'residential_load_data_{HOUSEHOLD_TYPE}_{PEAK_POWER}kW_{DAYS_TO_GEN}days_{SEASON}.csv'
    df.to_csv(output_filename, index=False)
    
    elapsed = datetime.now() - start_time
    
    print(f"\n{'='*70}")
    print(f"Dataset generated successfully!")
    print(f"{'='*70}")
    print(f"Filename: {output_filename}")
    print(f"Total records: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Generation time: {elapsed.total_seconds():.2f} seconds")
    
    print(f"\nDataset preview:")
    print(df.head(10))
    
    print(f"\nStatistical Summary:")
    print(df[['voltage', 'current', 'power_factor', 'active_power_kw', 
             'reactive_power_kvar', 'kwh_15min']].describe())
    
    # Calculate daily energy consumption
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    daily_energy = df.groupby('date')['kwh_15min'].sum()
    
    print(f"\nDaily energy consumption (first 7 days):")
    print(daily_energy.head(7))
    
    # Separate weekday and weekend analysis
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    weekday_energy = daily_energy[df.groupby('date')['day_of_week'].first() < 5]
    weekend_energy = daily_energy[df.groupby('date')['day_of_week'].first() >= 5]
    
    print(f"\nEnergy Analysis:")
    print(f"Average weekday consumption: {weekday_energy.mean():.2f} kWh/day")
    print(f"Average weekend consumption: {weekend_energy.mean():.2f} kWh/day")
    print(f"Weekend/Weekday ratio: {weekend_energy.mean()/weekday_energy.mean():.2%}")
    
    # Calculate load factor
    peak_power = df['active_power_kw'].max()
    avg_power = df['active_power_kw'].mean()
    load_factor = avg_power / peak_power
    
    print(f"\nLoad Factor Analysis:")
    print(f"Peak Power: {peak_power:.2f} kW")
    print(f"Average Power: {avg_power:.2f} kW")
    print(f"Load Factor: {load_factor:.3f}")
    
    # Power factor statistics
    print(f"\nPower Factor Statistics:")
    print(f"Average: {df['power_factor'].mean():.4f}")
    print(f"Minimum: {df['power_factor'].min():.4f}")
    print(f"Maximum: {df['power_factor'].max():.4f}")
    
    # Voltage statistics including sag events
    print(f"\nVoltage Statistics:")
    print(f"Average: {df['voltage'].mean():.2f} V")
    print(f"Minimum: {df['voltage'].min():.2f} V")
    print(f"Maximum: {df['voltage'].max():.2f} V")
    voltage_sag_events = len(df[df['voltage'] < 200])
    print(f"Severe voltage sag events (<200V): {voltage_sag_events} ({100*voltage_sag_events/len(df):.2f}%)")
    
    print(f"\n{'='*70}")
    print("Generation complete!")
    print(f"Timestamp format: RFC3339/ISO8601 (InfluxDB/Grafana compatible)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
