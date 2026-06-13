from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create FastAPI app

app = FastAPI()

# Load trained model
model = joblib.load(
    "models/cloud_cost_model.pkl"
)

# Home endpoint

@app.get("/")
def home():
    return {"message": "Cloud Cost Prediction API Running"}

# Input schema

class PredictionInput(BaseModel):
    cpu_usage: float
    memory_usage: float
    requests_per_sec: float
    network_io: float
    pod_count: int

# Prediction endpoint

@app.post("/predict")
def predict(data: PredictionInput):
    cpu_per_pod = data.cpu_usage / data.pod_count

    memory_per_request = data.memory_usage / data.requests_per_sec

    
    network_per_request = data.network_io / data.requests_per_sec

    resource_intensity = data.cpu_usage + data.memory_usage + data.network_io

    cost_efficiency = data.requests_per_sec / (data.cpu_usage + 1)

    features = [[
        data.cpu_usage,
        data.memory_usage,
        data.requests_per_sec,
        data.network_io,
        data.pod_count,
        cpu_per_pod,
        memory_per_request,
      
        network_per_request,
        resource_intensity,
        cost_efficiency
    ]]

    prediction = model.predict(features)

    return {"predicted_cost": float(prediction[0])}
