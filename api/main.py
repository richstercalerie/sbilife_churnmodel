from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="SBI Life - Churn Prediction API")

# Absolute path to the model (adjust if needed)
MODEL_PATH = r"C:\Users\Anurag mishra\sbilife_churnmodel\model\sbilife_churn_model.pkl"

# Load model
try:
    print("Loading model from:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    raise RuntimeError(f"Could not load model: {e}")

# Define expected schema
class CustomerData(BaseModel):
    Age: int
    Gender: int
    Region_Code: int
    Previously_Insured: int
    Vehicle_Damage: int
    Annual_Premium: float
    Policy_Sales_Channel: int
    Vintage: int

@app.post("/predict_churn")
def predict_churn(data: CustomerData):
    try:
        # Convert input to DataFrame
        input_dict = data.dict()
        print("Input received:", input_dict)
        df = pd.DataFrame([input_dict])
        # Ensure correct column order
        df = df[['Age', 'Gender', 'Region_Code', 'Previously_Insured',
                 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']]
        print("Prepared DataFrame:", df)

        # Prediction
        prob = model.predict_proba(df)[0][1]
        prob = float(prob)
        print("Predicted probability:", prob)

        # Use optimized threshold instead of 0.5 (set manually based on your evaluation)
        THRESHOLD = 0.3  
        prediction = int(prob > THRESHOLD)
        return {
            "churn_probability": round(prob * 100, 2),
            "prediction": "Churn Likely" if prediction else "Retention Likely"
        }

    except Exception as e:
        print("Error during prediction:", e)
        return {"error": str(e)}
