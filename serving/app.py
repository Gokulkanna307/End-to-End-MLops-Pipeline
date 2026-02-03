import mlflow.pyfunc
from fastapi import FastAPI
import numpy as np
import os

app = FastAPI()

# Path INSIDE container (because Dockerfile copies mlruns → /app/mlruns)
MODELS_DIR = "/app/mlruns/1/models"

# Pick latest model directory
model_dirs = sorted(
    [os.path.join(MODELS_DIR, d) for d in os.listdir(MODELS_DIR)],
    key=os.path.getmtime,
    reverse=True
)

MODEL_PATH = os.path.join(model_dirs[0], "artifacts")

print(f"Loading model from: {MODEL_PATH}")

model = mlflow.pyfunc.load_model(MODEL_PATH)

@app.post("/predict")
def predict(data: dict):
    features = np.array(data["features"]).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}

