"""
Kerala Distribution Network Load Generator (K-DNLG)
Version: 2.0 (Deep Research Edition)
Target: 2024-2025 Operational Years
Author: Expert System (Energy Analytics Persona)

Description:
This script generates high-fidelity, minute-resolution electrical load data 
specifically calibrated for the Kerala, India context. It integrates:
1. A Hybrid Calendar Engine (Gregorian/Malayalam/Hijri events).
2. A Tropical Weather Model (Heat Index driven cooling loads).
3. Detailed Consumer Profiling (Res/Com/Ind).
4. Electrical Parameter Derivation (P, Q, V, pf).
"""

import pandas as pd
import numpy as np
import datetime
import math
import random

# ==============================================================================
# SECTION A: CONFIGURATION & CONSTANTS
# ==============================================================================

# Simulation Parameters
START_DATE = datetime.date(2024, 1, 1)
END_DATE = datetime.date(2025, 12, 31)
FREQUENCY = '15T'  # 15-minute resolution
FEEDER_ID = "KL-Tvm-North-F04"
NOMINAL_VOLTAGE = 11000 # 11 kV Feeder Level Simulation
BASE_MVA = 5.0 # 5 MVA Feeder Capacity

# Consumer Demographics (The "Rurban" Mix)
NUM_CONSUMERS = 1200
MIX = {
    'Residential': 0.78, # 78% Domestic 
    'Commercial': 0.18,  # Shops, Offices
    'Industrial': 0.04   # Small scale mills
}

# Seasonal Config (Köppen Am Climate)
SEASONS = {
    'Winter':  {'months': [1, 2],      'base_temp': 24, 'temp_var': 5, 'rh_base': 60, 'rh_var': 15},
    'Summer':  {'months': [3, 4, 5],   'base_temp': 29, 'temp_var': 7, 'rh_base': 72, 'rh_var': 15},
    'Monsoon': {'months': [6, 7, 8, 9] ,'base_temp': 25, 'temp_var': 4, 'rh_base': 88, 'rh_var': 10},
    'PostMon': {'months': [10, 11, 12] ,'base_temp': 26, 'temp_var': 5, 'rh_base': 75, 'rh_var': 12}
}

# ==============================================================================
# SECTION B: KERALA CALENDAR ENGINE
# ==============================================================================

class KeralaCalendar:
    """
    Handles the complex interaction of Fixed and Variable holidays.
    """
    def __init__(self):
        # Fixed Gregorian Holidays
        self.fixed = {
            (1, 26): "Republic Day", (5, 1): "May Day",
            (8, 15): "Independence Day", (10, 2): "Gandhi Jayanthi",
            (12, 25): "Christmas"
        }
        
        # Variable Holidays [10, 13]
        # Dictionary format: Year -> {Date: Name}
        self.variable = {
            2024: {
                datetime.date(2024, 3, 29): "Good Friday",
                datetime.date(2024, 4, 10): "Id-ul-Fitr",
                datetime.date(2024, 4, 14): "Vishu",
                datetime.date(2024, 6, 17): "Bakrid",
                datetime.date(2024, 9, 15): "Thiruvonam", # Onam Main Day
                datetime.date(2024, 9, 16): "Third Onam",
                datetime.date(2024, 10, 12): "Mahanavami"
            },
            2025: {
                datetime.date(2025, 3, 31): "Id-ul-Fitr",
                datetime.date(2025, 4, 14): "Vishu", # Fixed Solar Date
                datetime.date(2025, 4, 18): "Good Friday",
                datetime.date(2025, 6, 6): "Bakrid",
                datetime.date(2025, 9, 4): "First Onam",
                datetime.date(2025, 9, 5): "Thiruvonam", # Onam Main Day
                datetime.date(2025, 10, 20): "Deepavali"
            }
        }

    def get_day_type(self, date_obj):
        """
        Returns:
        0: Workday
        1: Weekend (Sunday)
        2: Holiday (General)
        3: Festival (High Load - Onam/Vishu/Xmas)
        """
        # 1. Check Festival (High Priority)
        # Check variable festivals
        if date_obj in self.variable.get(date_obj.year, {}):
            name = self.variable[date_obj.year][date_obj]
            if name in ["Onam", "Vishu", "Christmas", "Id-ul-Fitr", "Bakrid", "Deepavali", "Thiruvonam", "First Onam", "Third Onam"]:
                return 3, name
            return 2, name
            
        # Check fixed festivals
        if (date_obj.month, date_obj.day) in self.fixed:
            name = self.fixed[(date_obj.month, date_obj.day)]
            if name == "Christmas":
                return 3, name
            return 2, name

        # 2. Check Weekend
        if date_obj.weekday() == 6: # Sunday
            return 1, "Sunday"
            
        # 3. Default Workday
        return 0, "Workday"

# ==============================================================================
# SECTION C: TROPICAL WEATHER MODEL
# ==============================================================================

class TropicalWeather:
    """
    Simulates Am climate and calculates Rothfusz Heat Index.
    """
    def __init__(self, seed=42):
        np.random.seed(seed)
        
    def _rothfusz_heat_index(self, t_c, rh):
        """
        Calculates Heat Index in Celsius given Temp (C) and RH (%).
        """
        T = (t_c * 9/5) + 32 # Convert to F
        R = rh
        
        # Simple equation for low temps
        hi_f = 0.5 * (T + 61.0 + ((T-68.0)*1.2) + (R*0.094))
        
        if hi_f > 80:
            # Full Regression
            hi_f = -42.379 + 2.04901523*T + 10.14333127*R - 0.22475541*T*R - \
                   6.83783e-3*T**2 - 5.481717e-2*R**2 + 1.22874e-3*T**2*R + \
                   8.5282e-4*T*R**2 - 1.99e-6*T**2*R**2
            # Adjustments
            if R < 13 and 80 < T < 112:
                hi_f -= ((13-R)/4) * math.sqrt((17-abs(T-95.))/17)
            if R > 85 and 80 < T < 87:
                hi_f += ((R-85)/10) * ((87-T)/5)
                
        return (hi_f - 32) * 5/9 # Back to C

    def generate_day_weather(self, date_obj):
        """
        Generates 24-hour profile for T and RH.
        """
        # Determine season
        season_data = SEASONS['PostMon'] # Default
        for s_name, s_data in SEASONS.items():
            if date_obj.month in s_data['months']:
                season_data = s_data
                break
        
        # Daily stochastic shift (Heatwave or Cool day)
        day_t_offset = np.random.normal(0, 1.5)
        day_rh_offset = np.random.normal(0, 5)
        
        temps = []
        rhs = []
        
        for h in range(24):
            # Diurnal Cycle: T peaks ~15:00, RH peaks ~05:00
            # Normalize hour to 0-2pi
            # T_curve: min at 4, max at 15 -> Shifted Cosine
            norm_t = -math.cos((h - 4) * 2 * math.pi / 24)
            
            t_val = season_data['base_temp'] + day_t_offset + (season_data['temp_var'] * norm_t)
            # Add hourly noise
            t_val += np.random.normal(0, 0.3)
            
            # RH is roughly inverse to T
            rh_val = season_data['rh_base'] + day_rh_offset - (season_data['rh_var'] * norm_t)
            rh_val = min(100, max(40, rh_val + np.random.normal(0, 2)))
            
            temps.append(t_val)
            rhs.append(rh_val)
            
        return temps, rhs

# ==============================================================================
# SECTION D: CONSUMER LOAD PROFILER
# ==============================================================================

class LoadProfiler:
    """
    Provides normalized load curves and environmental multipliers.
    """
    def __init__(self):
        # 24-point normalized curves (0 to 1)
        # Based on Research [16, 17]
        
        self.profiles = {
            'Residential': np.array([
                0.35, 0.30, 0.30, 0.30, 0.35, 0.50, 0.75, 0.85, 0.60, 0.45, # 00-09
                0.40, 0.40, 0.45, 0.45, 0.50, 0.55, 0.65, 0.85, 1.00, 0.95, # 10-19 (Peak 18-19)
                0.90, 0.85, 0.70, 0.50                                      # 20-23
            ]),
            'Commercial': np.array([
                0.10, 0.10, 0.10, 0.10, 0.15, 0.20, 0.40, 0.70, 0.90, 1.00, # 00-09
                1.00, 1.00, 0.95, 0.95, 1.00, 1.00, 1.00, 0.90, 0.80, 0.60, # 10-19
                0.40, 0.20, 0.15, 0.10                                      # 20-23
            ]),
            'Industrial': np.array([
                0.40, 0.40, 0.40, 0.40, 0.40, 0.50, 0.70, 0.90, 1.00, 1.00, # 00-09
                1.00, 1.00, 0.70, 1.00, 1.00, 1.00, 0.95, 0.90, 0.80, 0.60, # 10-19 (Lunch dip)
                0.50, 0.50, 0.45, 0.40                                      # 20-23
            ])
        }

    def get_load(self, c_type, hour, day_type, heat_index):
        """
        Returns scaler for specific hour.
        """
        base = self.profiles[c_type][hour]
        
        # 1. Calendar Modifiers
        if c_type == 'Residential':
            if day_type == 1: # Sunday
                if 9 <= hour <= 16: base *= 1.25 # People home
            elif day_type == 3: # Festival
                if hour <= 9: base *= 1.4 # Cooking
                if hour >= 18: base *= 1.3 # Lights
                
        elif c_type in ['Commercial', 'Industrial']:
            if day_type >= 1: base *= 0.3 # Closed/Low Ops
            
        # 2. Weather Modifiers (The Heat Index Effect)
        # Critical Logic: AC loads kick in when HI > 29C
        if heat_index > 29:
            excess_heat = heat_index - 29
            if c_type == 'Residential':
                # Night cooling sensitivity is high
                if hour >= 21 or hour <= 6:
                    base += (excess_heat * 0.04) # 4% load increase per degree C
                else:
                    base += (excess_heat * 0.02)
            elif c_type == 'Commercial':
                if 9 <= hour <= 20: # Operating hours
                    base += (excess_heat * 0.03)
                    
        return max(0.05, base)

# ==============================================================================
# SECTION E: MASTER GENERATOR
# ==============================================================================

def generate_kerala_dataset():
    print(f"Initializing Kerala Data Generation for {START_DATE} to {END_DATE}...")
    
    cal = KeralaCalendar()
    weather_eng = TropicalWeather()
    profiler = LoadProfiler()
    
    # Time Index
    dates = pd.date_range(START_DATE, END_DATE, freq=FREQUENCY)
    results = []
    
    # Cache daily weather to optimize
    daily_weather = {}
    curr = START_DATE
    while curr <= END_DATE:
        daily_weather[curr] = weather_eng.generate_day_weather(curr)
        curr += datetime.timedelta(days=1)
        
    for ts in dates:
        d = ts.date()
        h = ts.hour
        m = ts.minute
        
        # 1. Context
        day_type_code, day_name = cal.get_day_type(d)
        temps, rhs = daily_weather[d]
        
        # Interpolate Weather (Linear)
        t_curr = temps[h]
        rh_curr = rhs[h]
        hi_curr = weather_eng._rothfusz_heat_index(t_curr, rh_curr)
        
        # 2. Load Aggregation
        total_kw = 0
        
        # Residential
        n_res = int(NUM_CONSUMERS * MIX['Residential'])
        # Peak avg demand per home ~1.5kW (Diversified)
        p_res = profiler.get_load('Residential', h, day_type_code, hi_curr)
        load_res = n_res * 1.5 * p_res * np.random.normal(1, 0.05)
        
        # Commercial
        n_com = int(NUM_CONSUMERS * MIX['Commercial'])
        # Peak avg demand ~3kW
        p_com = profiler.get_load('Commercial', h, day_type_code, hi_curr)
        load_com = n_com * 3.0 * p_com * np.random.normal(1, 0.05)
        
        # Industrial
        n_ind = int(NUM_CONSUMERS * MIX['Industrial'])
        # Peak avg demand ~15kW
        p_ind = profiler.get_load('Industrial', h, day_type_code, hi_curr)
        load_ind = n_ind * 15.0 * p_ind * np.random.normal(1, 0.02)
        
        total_kw = load_res + load_com + load_ind
        
        # 3. Electrical Derivatives
        # Power Factor Logic
        # Res is inductive (0.85) at night, Com is compensated (0.95)
        w_res = load_res / total_kw if total_kw > 0 else 0
        pf_avg = (w_res * 0.88) + ((1-w_res) * 0.94)
        pf_avg += np.random.normal(0, 0.01)
        pf_avg = min(0.99, max(0.8, pf_avg))
        
        total_kvar = total_kw * math.tan(math.acos(pf_avg))
        total_kva = math.sqrt(total_kw**2 + total_kvar**2)
        
        # Voltage Estimation (Radial Drop)
        # Assuming 5% drop at full capacity (Base MVA)
        # V_pu = 1.0 - (Load_kVA / Base_kVA * Impedance)
        v_drop = (total_kva / (BASE_MVA * 1000)) * 0.05 
        v_pu = 1.0 - v_drop + np.random.normal(0, 0.002)
        
        # Calculate Current (I) in Amps
        # I = S / (sqrt(3) * V_line)
        # S = total_kva * 1000 (to get VA)
        # V_line = v_pu * NOMINAL_VOLTAGE
        
        v_actual = v_pu * NOMINAL_VOLTAGE
        current_amps = (total_kva * 1000) / (math.sqrt(3) * v_actual) if v_actual > 0 else 0

        results.append({
            'timestamp': ts,
            'date': d,
            'day_name': day_name,
            'is_holiday': day_type_code >= 2,
            'temp_c': round(t_curr, 2),
            'humidity_pct': round(rh_curr, 1),
            'heat_index_c': round(hi_curr, 2),
            'P': round(total_kw * 1000, 2),
            'Q': round(total_kvar * 1000, 2),
            'V': round(v_actual, 2),
            'I': round(current_amps, 2),
            'PF': round(pf_avg, 3)
        })
        
    df = pd.DataFrame(results)
    # Ensure column order
    df = df[['timestamp', 'date', 'day_name', 'is_holiday', 'temp_c', 'humidity_pct', 'heat_index_c', 'P', 'Q', 'V', 'I', 'PF']]
    return df

if __name__ == "__main__":
    kerala_data = generate_kerala_dataset()
    print("Generation Complete.")
    print(kerala_data.head())
    print(f"Generated {len(kerala_data)} records.")
    kerala_data.to_csv("kerala_datasetv2.csv", index=False)