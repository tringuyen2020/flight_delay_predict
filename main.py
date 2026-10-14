import uvicorn
import os
import numpy as np
import joblib
from fastapi import FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from keras.models import load_model
from contextlib import asynccontextmanager

ml_models = {}
templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load the ML model when the application starts
        ml_models["flight_delay"] = load_model(os.path.join("models", "flight_delay_model.keras"))
        ml_models["scaler"] = joblib.load(os.path.join("models", "scaler.joblib"))
    except Exception as e:
        print(f"Error loading ML model: {e}")
    yield
    # Clean up resources when the application shuts down
    del ml_models["flight_delay"]
    del ml_models["scaler"]

app = FastAPI(lifespan=lifespan)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.post("/predict", response_class=HTMLResponse)
async def predict_delay(request: Request, air_time: float = Form(...), distance: float = Form(...)):
    model = ml_models.get("flight_delay")
    scaler = ml_models.get("scaler")

    if not model:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        # Perform prediction using the loaded ML model
        user_inputs = np.array([[air_time, distance]], dtype=np.float32)
        
        # Scale the user inputs using the loaded scaler
        user_inputs = scaler.transform(user_inputs)

        predictions = model.predict(user_inputs)
        print(f"Predictions: {predictions}")

        if predictions[0][1] > 0.5:
            # Handle the case where the prediction indicates a delay
            prediction = f"The flight is likely to be delayed by {predictions[0][1]:.2f} minutes."
        else:
            prediction = "The flight is not likely to be delayed."

        return templates.TemplateResponse(request, "index.html", {"prediction": prediction})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {e}")
    

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
