import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 10000

cpu_usage = np.random.uniform(10, 95, n_samples)
memory_usage = np.random.uniform(2, 64, n_samples)
requests_per_sec = np.random.uniform(100, 10000, n_samples)
network_io = np.random.uniform(50, 5000, n_samples)
pod_count = np.random.randint(1, 20, n_samples)

cloud_cost = (
    0.3 * cpu_usage +
    0.5 * memory_usage +
    0.01 * requests_per_sec +
    0.2 * network_io +
    2 * pod_count +
    np.random.normal(0, 20, n_samples)
)

df = pd.DataFrame({
    "cpu_usage": cpu_usage,
    "memory_usage": memory_usage,
    "requests_per_sec": requests_per_sec,
    "network_io": network_io,
    "pod_count": pod_count,
    "cloud_cost": cloud_cost
})

df.to_csv("data/telemetry_data.csv", index=False)

print("Dataset generated successfully!")
print(df.head())