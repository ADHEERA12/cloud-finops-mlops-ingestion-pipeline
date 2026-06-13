import pandas as pd

df = pd.read_csv("data/telemetry_data.csv")
df["cpu_per_pod"] = df["cpu_usage"] / df["pod_count"]
df["memory_per_request"] = (
    df["memory_usage"] /
    df["requests_per_sec"]
)
df["network_per_request"] = (
    df["network_io"] /
    df["requests_per_sec"]
)
df["resource_intensity"] = (
    df["cpu_usage"] +
    df["memory_usage"] +
    df["network_io"]
)
df["cost_efficiency"] = (
    df["requests_per_sec"] /
    (df["cpu_usage"] + 1)
)
print(df.head())
df.to_csv(
    "data/processed/feature_engineered.csv",
    index=False
)