from datetime import datetime
import math
import random

def solar_power(max_capacity, timestamp=None, randomness=True):
    """
    Generate solar power based on time of day.
    
    Parameters:
    - max_capacity: float, peak solar power at midday (kW)
    - timestamp: datetime object, optional; if None, uses current time
    - randomness: bool, if True adds a small random fluctuation
    
    Returns:
    - float: solar power output at the given time
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    hour = timestamp.hour + timestamp.minute / 60
    
    # Solar generation only between 6 AM and 6 PM
    if hour < 6 or hour > 18:
        return 0.0
    
    # Peak at 12 PM, use a sine curve to simulate smooth rise/fall
    # Scale factor between 0 (sunrise/sunset) and 1 (noon)
    solar_factor = math.sin(math.pi * (hour - 6) / 12)
    
    power = max_capacity * solar_factor
    
    # Optional: add small random variation ±5%
    if randomness:
        variation = random.uniform(-0.05, 0.05) * max_capacity
        power += variation
        power = max(0, power)  # prevent negative power
    
    return round(power, 2)

# Example usage
max_solar = 10  # kW
for h in range(0, 24):
    for m in range(0, 60, 15):
        t = datetime(2025, 8, 24, h, m)
        print(f"{t.hour:02d}:{t.minute:02d} -> Solar Power: {solar_power(max_solar, t)} kW")
