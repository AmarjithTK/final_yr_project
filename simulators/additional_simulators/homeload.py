import math
import random
from datetime import datetime, timedelta
from weather import simulate_weather  # import your weather simulator

class HomeLoad:
    def __init__(self, max_home_load, latitude, longitude):
        """
        Initialize a HomeLoad simulator.

        Parameters:
        - max_home_load: float, maximum instantaneous load in kW
        - latitude, longitude: for weather simulation
        """
        self.max_home_load = max_home_load
        self.latitude = latitude
        self.longitude = longitude

    def activity_factor(self, timestamp):
        """
        Estimate load factor based on time of day and typical household activity.
        Returns a scaling factor between 0 and 1.
        """
        hour = timestamp.hour + timestamp.minute / 60

        if 6 <= hour < 9:   # Morning: breakfast, cooking
            return 0.6 + random.uniform(-0.1, 0.1)
        elif 9 <= hour < 12:  # Late morning: low activity
            return 0.3 + random.uniform(-0.05, 0.05)
        elif 12 <= hour < 15:  # Afternoon: lunch + AC/comfort devices
            return 0.5 + random.uniform(-0.1, 0.1)
        elif 15 <= hour < 18:  # Evening: snacks, prep dinner
            return 0.6 + random.uniform(-0.1, 0.1)
        elif 18 <= hour < 23:  # Night: dinner, lighting, appliances (extended to 23:00)
            return 0.8 + random.uniform(-0.1, 0.1)
        else:  # Late night & early morning (after 23:00 and before 6:00)
            return 0.2 + random.uniform(-0.05, 0.05)

    def weather_factor(self, temperature, humidity):
        """
        Adjust load based on weather (e.g., AC/heating usage):
        - Higher temperature → more cooling → increase load
        - Higher humidity → more dehumidifier/AC → increase load
        Returns a scaling factor around 1 (e.g., 0.8-1.2)
        """
        factor = 1.0
        # Simple example: AC/heating kicks in if temp outside 24-28°C
        if temperature > 28:
            factor += 0.3  # hotter → more cooling
        elif temperature < 22:
            factor += 0.2  # colder → more heating

        # Humidity impact
        if humidity > 75:
            factor += 0.1  # more humid → slight extra load

        return factor

    def generate_daily_load(self, date, interval_minutes=15):
        """
        Generate 15-min interval load profile for the whole day.

        Returns: list of dicts with timestamp, load (kW), temperature, humidity
        """
        load_profile = []
        start_time = datetime(date.year, date.month, date.day, 0, 0)
        end_time = start_time + timedelta(days=1)
        current_time = start_time

        while current_time < end_time:
            # Get weather at this timestamp
            temp, hum = simulate_weather(self.latitude, self.longitude, current_time)

            # Compute load factors
            activity = self.activity_factor(current_time)
            weather = self.weather_factor(temp, hum)

            # Calculate instantaneous load
            load = self.max_home_load * activity * weather
            # Optional: add random noise ±5%
            load += random.uniform(-0.05, 0.05) * self.max_home_load
            load = max(0, load)  # prevent negative load

            # Save data
            load_profile.append({
                'timestamp': current_time,
                'load_kW': round(load, 2),
                'temperature_C': temp,
                'humidity_%': hum
            })

            current_time += timedelta(minutes=interval_minutes)

        return load_profile


# Example usage
if __name__ == "__main__":
    home = HomeLoad(max_home_load=8, latitude=10.8505, longitude=76.2711)
    profile = home.generate_daily_load(datetime(2025, 8, 24))
    for entry in profile:  # print first 10 intervals
        t = entry['timestamp'].strftime("%H:%M")
        load = entry['load_kW']
        print(f"{t} - Load: {load} kW")
