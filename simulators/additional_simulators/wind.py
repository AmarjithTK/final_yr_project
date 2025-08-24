import math
import random
from datetime import datetime

def wind_power(max_capacity, timestamp=None):
    """
    Simulate wind turbine power output.
    
    Parameters:
    - max_capacity: rated turbine power (kW)
    - timestamp: optional datetime object
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    hour = timestamp.hour + timestamp.minute/60
    
    # Optional smooth daily trend (0.4 - 0.7 of max)
    base = max_capacity * (0.5 + 0.2 * math.sin(math.pi * (hour - 6)/12))
    
    # Small random fluctuation ±30% of max_capacity
    fluctuation = random.uniform(-0.3, 0.3) * max_capacity
    
    power = base + fluctuation
    
    # Clamp to physical limits
    power = max(0, min(power, max_capacity))
    
    return round(power, 2)

# Example usage
for h in range(0, 24, 2):
    t = datetime(2025, 8, 24, h, 0)
    print(f"{t.hour}:00 -> Wind Power: {wind_power(400, t)} kW")
