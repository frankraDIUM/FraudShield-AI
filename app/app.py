from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Fraud Detection API")

# Load artifacts
model = joblib.load('fraud_model_final.pkl')
# Define names exactly as the model expects them
FEATURE_NAMES = [f'V{i}' for i in range(1, 29)] + ['Hour', 'scaled_amount', 'scaled_time']

class Transaction(BaseModel):
    features: list # Must be length 31

@app.post("/predict")
def predict_fraud(tx: Transaction):
    # Check feature length
    if len(tx.features) != 31:
        raise HTTPException(
            status_code=400,
            detail=f"Model expects 31 features, but received {len(tx.features)}. Please check the feature engineering."
        )
    try:
        # Convert to DataFrame to avoid "Feature Names Mismatch" warning
        input_data = pd.DataFrame([tx.features], columns=FEATURE_NAMES)
        # Get Probability
        prob = float(model.predict_proba(input_data)[0][1])

        # Apply the sliding threshold logic
        amount = tx.features[29]
        threshold = 0.25 if amount > 5.0 else 0.55
        decision = "BLOCK" if prob >= threshold else "ALLOW"

        return {
        "probability": prob,
        "threshold_used": threshold,
        "decision": decision
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
