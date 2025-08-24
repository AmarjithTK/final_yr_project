import math
from datetime import datetime

def get_season(month):
    if month in [6, 7, 8]:
        return 'Southwest Monsoon'
    elif month in [10, 11]:
        return 'Northeast Monsoon'
    elif month in [3, 4, 5]:
        return 'Summer'
    elif month in [12, 1, 2]:
        return 'Winter'

def simulate_weather(latitude, longitude, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now()

    month = timestamp.month
    season = get_season(month)

    # Define base temperature and humidity ranges for each season
    season_data = {
        'Summer': {'temp_range': (28, 35), 'humidity_range': (60, 80)},
        'Southwest Monsoon': {'temp_range': (24, 30), 'humidity_range': (80, 95)},
        'Northeast Monsoon': {'temp_range': (25, 32), 'humidity_range': (75, 90)},
        'Winter': {'temp_range': (20, 28), 'humidity_range': (50, 70)},
    }

    temp_min, temp_max = season_data[season]['temp_range']
    humidity_min, humidity_max = season_data[season]['humidity_range']

    # Simulate temperature and humidity
    temperature = temp_min + (temp_max - temp_min) * math.sin(math.pi * (timestamp.hour - 6) / 12)
    humidity = humidity_min + (humidity_max - humidity_min) * math.sin(math.pi * (timestamp.hour - 6) / 12)

    return round(temperature, 2), round(humidity, 2)

# Example usage
latitude = 10.8505  # Example: Kochi
longitude = 76.2711
current_time = datetime(2025, 8, 24, 14, 0)  # August 24, 2025, 2:00 PM
temperature, humidity = simulate_weather(latitude, longitude, current_time)
print(f"At {current_time}, the simulated temperature is {temperature}°C and humidity is {humidity}%.")
