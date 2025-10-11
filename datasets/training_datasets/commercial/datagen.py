import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================
DAYS_TO_GEN = 90  # Number of days to generate data for
INTERVAL_MINUTES = 15  # Time resolution
INTERVALS_PER_DAY = 96  # 24 hours * 4 intervals per hour

# Start date (declare at top, timezone-aware UTC) — change this as needed
START_DATE = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

# Output timestamp format compatible with InfluxDB / Grafana (RFC3339 / Zulu)
OUTPUT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Commercial building parameters (Medium Office - 150 kW peak)
BUILDING_TYPE = "medium_office"
NOMINAL_VOLTAGE = 400  # V line-to-line (three-phase)
PEAK_POWER = 150  # kW (adjust for building size: 50, 150, 400)
BASE_POWER_NIGHT = 18  # kW (baseline when unoccupied)
SEASON = "summer"  # Options: "summer", "winter", "moderate"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_commercial_load_factor(hour, minute, is_weekend, season='moderate'):
    """
    Returns normalized load factor (0-1) for commercial office building
    Based on time of day, weekend status, and season
    """
    time_decimal = hour + minute / 60.0
    
    if is_weekend:
        # Weekend - minimal load, mostly baseline
        if 9.0 <= time_decimal < 14.0:  # Brief Saturday activity
            base = 0.25 if is_weekend == 5 else 0.12  # Saturday higher than Sunday
            return base + np.random.uniform(-0.03, 0.05)
        else:
            return 0.12 + np.random.uniform(-0.02, 0.02)
    
    # Weekday pattern - strong occupancy correlation
    
    # Night baseline (12 AM - 6 AM)
    if time_decimal < 6.0:
        return 0.12 + np.random.uniform(-0.02, 0.02)
    
    # Pre-occupancy HVAC startup (6 AM - 8 AM)
    elif 6.0 <= time_decimal < 8.0:
        progress = (time_decimal - 6.0) / 2.0
        load = 0.12 + progress * 0.28  # 12% → 40%
        load += np.random.uniform(-0.05, 0.05)
        return np.clip(load, 0.10, 0.45)
    
    # Morning arrival ramp-up (8 AM - 9 AM)
    elif 8.0 <= time_decimal < 9.0:
        progress = (time_decimal - 8.0) / 1.0
        load = 0.40 + progress * 0.35  # 40% → 75%
        load += np.random.uniform(-0.08, 0.08)
        return np.clip(load, 0.35, 0.82)
    
    # Morning full occupancy (9 AM - 12 PM)
    elif 9.0 <= time_decimal < 12.0:
        base = 0.95  # 95% of peak
        load = base + np.random.uniform(-0.08, 0.05)
        return np.clip(load, 0.85, 1.0)
    
    # Lunch period slight dip (12 PM - 1 PM)
    elif 12.0 <= time_decimal < 13.0:
        base = 0.88  # 88% of peak (minor dip)
        load = base + np.random.uniform(-0.08, 0.08)
        return np.clip(load, 0.80, 0.95)
    
    # Afternoon peak (1 PM - 5 PM) - highest consumption
    elif 13.0 <= time_decimal < 17.0:
        base = 1.0  # 100% peak
        load = base + np.random.uniform(-0.05, 0.0)
        # Afternoon peak in summer due to cooling
        if season == 'summer' and 14.0 <= time_decimal < 16.0:
            load = min(1.05, load + 0.05)
        return np.clip(load, 0.90, 1.05)
    
    # Evening departure decline (5 PM - 7 PM)
    elif 17.0 <= time_decimal < 19.0:
        progress = (time_decimal - 17.0) / 2.0
        load = 1.0 - progress * 0.60  # 100% → 40%
        load += np.random.uniform(-0.08, 0.05)
        return np.clip(load, 0.35, 0.95)
    
    # Evening setback (7 PM - 10 PM)
    elif 19.0 <= time_decimal < 22.0:
        progress = (time_decimal - 19.0) / 3.0
        load = 0.40 - progress * 0.25  # 40% → 15%
        load += np.random.uniform(-0.03, 0.03)
        return np.clip(load, 0.12, 0.42)
    
    # Late night baseline (10 PM - 12 AM)
    else:
        return 0.12 + np.random.uniform(-0.02, 0.02)


def apply_seasonal_adjustment(load_factor, hour, season):
    """
    Apply seasonal multipliers based on HVAC usage patterns
    """
    if season == 'summer':
        # Summer: High cooling loads especially afternoon
        if 12 <= hour <= 18:  # Peak cooling period
            return min(1.05, load_factor * 1.30)
        elif 8 <= hour <= 12 or 18 <= hour <= 20:
            return min(1.05, load_factor * 1.20)
        else:
            return load_factor * 1.10
    
    elif season == 'winter':
        # Winter: Morning heating startup more pronounced
        if 6 <= hour <= 10:
            return min(1.05, load_factor * 1.25)
        elif 10 <= hour <= 18:
            return min(1.05, load_factor * 1.10)
        else:
            return load_factor * 1.05
    
    else:  # Spring/Fall - moderate, minimal HVAC
        return load_factor * 0.88


def generate_three_phase_voltage(base_voltage=400, load_factor=0.5):
    """
    Generate three-phase voltages (R, Y, B) with realistic imbalance
    """
    # Base voltage with ±2-3% normal variation
    voltage_avg = base_voltage * (1 + np.random.uniform(-0.025, 0.025))
    
    # Voltage sag during high loads
    if load_factor > 0.85:
        sag_percent = min(0.05, (load_factor - 0.85) * 0.15)
        voltage_avg *= (1 - sag_percent)
    
    # Voltage rise during very low loads
    elif load_factor < 0.20:
        voltage_avg *= (1 + np.random.uniform(0.005, 0.015))
    
    # Phase imbalance (1-3% difference between phases)
    voltage_r = voltage_avg * np.random.uniform(0.985, 1.000)
    voltage_y = voltage_avg * np.random.uniform(0.995, 1.005)
    voltage_b = voltage_avg * np.random.uniform(1.000, 1.015)
    
    # Ensure within bounds
    voltage_r = np.clip(voltage_r, 380, 420)
    voltage_y = np.clip(voltage_y, 380, 420)
    voltage_b = np.clip(voltage_b, 380, 420)
    
    return voltage_r, voltage_y, voltage_b


def get_commercial_power_factor(load_factor, season):
    """
    Power factor varies based on load level and season
    Commercial buildings typically have good PF due to correction
    """
    # High load - HVAC dominated
    if load_factor > 0.80:
        if season == 'summer':
            pf = 0.92 + np.random.uniform(-0.02, 0.03)  # Lower with heavy AC
        else:
            pf = 0.94 + np.random.uniform(-0.02, 0.03)
    
    # Medium load - mixed
    elif load_factor > 0.40:
        pf = 0.95 + np.random.uniform(-0.02, 0.03)
    
    # Light load - baseload equipment
    else:
        pf = 0.91 + np.random.uniform(-0.02, 0.03)
    
    return np.clip(pf, 0.90, 0.98)


def add_voltage_sag_events(voltage_r, voltage_y, voltage_b, interval_count):
    """
    Occasionally add severe voltage sag events for realism
    Occurs in ~0.5% of intervals
    """
    if np.random.rand() < 0.005:  # 0.5% probability
        sag_type = np.random.rand()
        
        if sag_type < 0.65:  # 65% - Moderate sag
            sag_factor = np.random.uniform(0.70, 0.87)  # 70-87% retained
        elif sag_type < 0.90:  # 25% - Deep sag
            sag_factor = np.random.uniform(0.52, 0.70)  # 52-70% retained
        else:  # 10% - Severe sag
            sag_factor = np.random.uniform(0.43, 0.52)  # 43-52% retained
        
        voltage_r *= sag_factor
        voltage_y *= sag_factor
        voltage_b *= sag_factor
        
        # Ensure minimum bounds
        voltage_r = max(voltage_r, 100)
        voltage_y = max(voltage_y, 100)
        voltage_b = max(voltage_b, 100)
    
    return voltage_r, voltage_y, voltage_b


# ============================================================================
# MAIN DATA GENERATION
# ============================================================================

def main():
    print(f"Generating commercial load data for {DAYS_TO_GEN} days...")
    print(f"Building Type: {BUILDING_TYPE.replace('_', ' ').title()}")
    print(f"Peak Power: {PEAK_POWER} kW")
    print(f"Season: {SEASON.title()}")
    print(f"Baseline Power: {BASE_POWER_NIGHT} kW")
    
    start_time = datetime.now()
    
    # Generate timezone-aware datetime array (used for calculations)
    start_date = START_DATE
    timestamps_dt = [start_date + timedelta(minutes=INTERVAL_MINUTES * i) 
                     for i in range(DAYS_TO_GEN * INTERVALS_PER_DAY)]
    
    # Initialize data storage
    data = {
        'timestamp': [],
        'voltage_r': [],
        'voltage_y': [],
        'voltage_b': [],
        'current': [],
        'power_factor': [],
        'active_power_kw': [],
        'reactive_power_kvar': [],
        'kwh_15min': []
    }
    
    # Generate data for each timestamp
    for idx, ts in enumerate(timestamps_dt):
        hour = ts.hour
        minute = ts.minute
        day_of_week = ts.weekday()
        is_weekend = day_of_week >= 5  # 5=Saturday, 6=Sunday
        
        # Get base load factor
        load_factor = get_commercial_load_factor(hour, minute, is_weekend, SEASON)
        
        # Apply seasonal adjustment
        load_factor = apply_seasonal_adjustment(load_factor, hour, SEASON)
        
        # Add occasional random variations (equipment, special events)
        if np.random.rand() < 0.10:  # 10% chance
            load_factor *= np.random.uniform(0.92, 1.08)
        
        load_factor = np.clip(load_factor, 0.10, 1.05)
        
        # Calculate active power
        active_power = PEAK_POWER * load_factor
        
        # Add Gaussian noise (±8%)
        active_power *= (1 + np.random.uniform(-0.08, 0.08))
        active_power = np.clip(active_power, BASE_POWER_NIGHT * 0.8, PEAK_POWER * 1.05)
        
        # Generate three-phase voltages
        voltage_r, voltage_y, voltage_b = generate_three_phase_voltage(
            NOMINAL_VOLTAGE, load_factor
        )
        
        # Add occasional voltage sag events
        voltage_r, voltage_y, voltage_b = add_voltage_sag_events(
            voltage_r, voltage_y, voltage_b, idx
        )
        
        # Average line-to-line voltage for calculations
        voltage_avg = (voltage_r + voltage_y + voltage_b) / 3
        
        # Get power factor
        pf = get_commercial_power_factor(load_factor, SEASON)
        
        # Calculate current (three-phase)
        # I = P / (sqrt(3) * V_LL * cos(phi))
        current = (active_power * 1000) / (np.sqrt(3) * voltage_avg * pf)
        current = np.clip(current, 10, 800)
        
        # Calculate reactive power
        # Q = P * tan(arccos(pf))
        theta = np.arccos(pf)
        reactive_power = active_power * np.tan(theta)
        
        # Calculate energy consumed in 15 minutes
        kwh_15min = active_power * 0.25  # 15 min = 0.25 hours
        
        # Store data
        data['timestamp'].append(ts.strftime(OUTPUT_TIME_FORMAT))
        data['voltage_r'].append(round(voltage_r, 2))
        data['voltage_y'].append(round(voltage_y, 2))
        data['voltage_b'].append(round(voltage_b, 2))
        data['current'].append(round(current, 2))
        data['power_factor'].append(round(pf, 4))
        data['active_power_kw'].append(round(active_power, 2))
        data['reactive_power_kvar'].append(round(reactive_power, 2))
        data['kwh_15min'].append(round(kwh_15min, 2))
        
        # Progress indicator
        if (idx + 1) % 500 == 0:
            print(f"Progress: {idx + 1}/{len(timestamps_dt)} records ({100*(idx+1)/len(timestamps_dt):.1f}%)")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_filename = f'commercial_load_data_{BUILDING_TYPE}_{PEAK_POWER}kW_{DAYS_TO_GEN}days_{SEASON}.csv'
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
    print(df.describe())
    
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
    
    print(f"\n{'='*70}")
    print("Generation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
