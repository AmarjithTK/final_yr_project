from datetime import datetime
from homeload import HomeLoad

class ResidentialNetwork:
    def __init__(self, home_specs):
        """
        Initialize a residential network.

        Parameters:
        - home_specs: list of dicts, each dict contains:
            - max_home_load: float
            - latitude: float
            - longitude: float
        Example:
        home_specs = [
            {'max_home_load': 8, 'latitude': 10.85, 'longitude': 76.27},
            {'max_home_load': 5, 'latitude': 10.85, 'longitude': 76.28},
            ...
        ]
        """
        self.homes = [HomeLoad(**spec) for spec in home_specs]

    def generate_network_load(self, date, interval_minutes=15):
        """
        Generate aggregated load profile for all homes in the network.
        
        Returns:
        - list of dicts: each dict has timestamp, total_load, and per-home load + weather
        """
        network_profile = []

        # Assume all homes generate 15-min interval data independently
        home_profiles = [home.generate_daily_load(date, interval_minutes) for home in self.homes]

        num_intervals = len(home_profiles[0])
        for i in range(num_intervals):
            timestamp = home_profiles[0][i]['timestamp']

            total_load = 0
            per_home_load = []

            for idx, profile in enumerate(home_profiles):
                load = profile[i]['load_kW']
                total_load += load
                per_home_load.append({
                    f'home_{idx+1}_load_kW': load,
                    f'home_{idx+1}_temperature_C': profile[i]['temperature_C'],
                    f'home_{idx+1}_humidity_%': profile[i]['humidity_%']
                })

            network_profile.append({
                'timestamp': timestamp,
                'total_load_kW': round(total_load, 2),
                'per_home': per_home_load
            })

        return network_profile


# Example usage
if __name__ == "__main__":
    # Define 3 homes for demo
    home_specs = [
        {'max_home_load': 8, 'latitude': 10.8505, 'longitude': 76.2711},
        {'max_home_load': 6, 'latitude': 10.8560, 'longitude': 76.2750},
        {'max_home_load': 5, 'latitude': 10.8450, 'longitude': 76.2800}
    ]

    network = ResidentialNetwork(home_specs)
    profile = network.generate_network_load(datetime(2025, 8, 24))

    # Print first 5 intervals
    for entry in profile[:5]:
        print(entry)
