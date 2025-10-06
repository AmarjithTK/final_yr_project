import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class ResidentialNode:
    def __init__(self, node_id, device_type='residential', voltage_nominal=230, pf=0.9):
        self.node_id = node_id
        self.device_type = device_type
        self.voltage_nominal = voltage_nominal
        self.pf = pf

    def generate_load_profile(self, timestamps):
        """Generate active power profile with morning/evening peaks"""
        base_power = np.zeros(len(timestamps))
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute/60
            # Morning peak 6-9 AM
            if 6 <= hour < 9:
                base_power[i] = np.random.uniform(0.8, 1.2)
            # Evening peak 6-10 PM
            elif 18 <= hour < 22:
                base_power[i] = np.random.uniform(0.9, 1.5)
            else:
                base_power[i] = np.random.uniform(0.2, 0.5)
        return base_power

    def generate_voltage(self, timestamps):
        return np.random.uniform(self.voltage_nominal*0.95, self.voltage_nominal*1.05, len(timestamps))

    def generate_current(self, power, voltage):
        # I = P/V
        return power / voltage

    def generate_reactive_power(self, power):
        # Q = P * sqrt(1/PF^2 - 1)
        noise = np.random.normal(0, 0.02, len(power))
        return power * np.sqrt(1/self.pf**2 - 1) + noise

    def generate_kwh(self, power, interval_minutes=15):
        return power * (interval_minutes/60)

class ResidentialLoadGenerator:
    def __init__(self, start_date, end_date, nodes, interval_minutes=15):
        self.start_date = start_date
        self.end_date = end_date
        self.interval_minutes = interval_minutes
        self.nodes = [ResidentialNode(node_id) for node_id in nodes]

    def generate_timestamps(self):
        timestamps = pd.date_range(start=self.start_date, end=self.end_date, freq=f'{self.interval_minutes}T')
        return timestamps

    def generate(self):
        timestamps = self.generate_timestamps()
        data_rows = []
        for node in self.nodes:
            P = node.generate_load_profile(timestamps)
            V = node.generate_voltage(timestamps)
            I = node.generate_current(P, V)
            Q = node.generate_reactive_power(P)
            kWh = node.generate_kwh(P, self.interval_minutes)

            for ts, v, i, p, q, k in zip(timestamps, V, I, P, Q, kWh):
                data_rows.append({
                    'node': node.node_id,
                    'device_type': node.device_type,
                    'timestamp': ts,
                    'V': round(v, 2),
                    'I': round(i, 3),
                    'P': round(p, 2),
                    'Q': round(q, 2),
                    'kwh': round(k, 4)
                })
        df = pd.DataFrame(data_rows)
        return df

if __name__ == '__main__':
    # Define generator
    nodes = ['node1', 'node2', 'node3']
    start_date = '2025-01-01 00:00'
    end_date = '2025-03-31 23:45'

    generator = ResidentialLoadGenerator(start_date, end_date, nodes)
    df = generator.generate()

    # Export to CSV
    df.to_csv('residential_3months.csv', index=False)
    print('CSV file generated: residential_3months.csv')
