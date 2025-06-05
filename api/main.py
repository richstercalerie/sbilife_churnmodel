from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import numpy as np
import os
import csv
from io import StringIO

app = FastAPI(title="SBI Life - Churn Prediction & SHAP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = "model/sbilife_churn_model.pkl"
TRAIN_CSV = "data/train.csv"

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

# === SCHEMAS ===

class CustomerData(BaseModel):
    Age: int
    Gender: int
    Region_Code: int
    Previously_Insured: int
    Vehicle_Damage: int
    Annual_Premium: float
    Policy_Sales_Channel: int
    Vintage: int

class DataRow(BaseModel):
    id: str = ""
    Gender: str
    Age: float
    Driving_License: int
    Region_Code: float
    Previously_Insured: int
    Vehicle_Age: str
    Vehicle_Damage: str
    Annual_Premium: float
    Policy_Sales_Channel: float
    Vintage: int
    Response: int

# === PREDICTION ENDPOINT ===

@app.post("/predict_churn")
def predict_churn(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])
        prob = model.predict_proba(df)[0][1]
        prediction = int(prob > 0.3)
        return {
            "churn_probability": round(prob * 100, 2),
            "prediction": "Churn Likely" if prediction else "Retention Likely"
        }
    except Exception as e:
        return {"error": str(e)}

# === SHAP SUMMARY ENDPOINT (FIXED) ===

@app.get("/shap_summary")
def shap_summary():
    try:
        df = pd.read_csv(TRAIN_CSV)
        features = [
            'Age', 'Gender', 'Region_Code', 'Previously_Insured',
            'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'
        ]
        df = df[features]
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(inplace=True)

        if len(df) > 1000:
            df = df.sample(n=1000, random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_vals_full = explainer.shap_values(df)

        if isinstance(shap_vals_full, list):
            shap_vals = shap_vals_full[1]
        else:
            shap_vals = shap_vals_full

        avg_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        percent_shap = (avg_abs_shap / np.sum(avg_abs_shap)) * 100

        expected_value = explainer.expected_value
        # For binary classifier, expected_value can be list, pick index 1
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

        return {
            "expected_value": float(expected_value),
            "shap_summary": {k: float(v) for k, v in zip(features, percent_shap.tolist())}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing SHAP summary: {e}")

# === CSV UPLOAD (append new data) ===

@app.post("/update_data")
async def update_data(request: Request):
    expected_fields = [
        'id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
        'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response'
    ]
    try:
        if os.path.exists(TRAIN_CSV):
            df = pd.read_csv(TRAIN_CSV)
            last_id = int(df['id'].iloc[-1]) if 'id' in df.columns else 0
        else:
            last_id = 0

        content_type = request.headers.get("Content-Type", "").lower()
        new_rows = []

        if "application/json" in content_type:
            data = await request.json()
            if isinstance(data, dict): data = [data]
            for entry in data:
                row = DataRow(**entry).dict()
                if row["id"] == "":
                    last_id += 1
                    row["id"] = str(last_id)
                new_rows.append(",".join(str(row[f]) for f in expected_fields))
        else:
            text = (await request.body()).decode()
            reader = csv.reader(StringIO(text))
            for fields in reader:
                if len(fields) != len(expected_fields):
                    raise HTTPException(status_code=400, detail="Incorrect number of fields.")
                if fields[0] == "":
                    last_id += 1
                    fields[0] = str(last_id)
                new_rows.append(",".join(fields))

        mode = "a" if os.path.exists(TRAIN_CSV) else "w"
        with open(TRAIN_CSV, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(",".join(expected_fields) + "\n")
            for line in new_rows:
                f.write("\n" + line)
        return {"detail": f"{len(new_rows)} rows added."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating CSV: {e}")

# === CSV UPDATE (update row by ID) ===

@app.post("/update_data_by_id")
async def update_by_id(data: DataRow):
    try:
        if not data.id:
            raise HTTPException(status_code=400, detail="Missing ID.")
        df = pd.read_csv(TRAIN_CSV)
        if data.id not in df["id"].astype(str).values:
            raise HTTPException(status_code=404, detail=f"No record with id {data.id}")
        for k, v in data.dict().items():
            if k != "id":
                df.loc[df["id"].astype(str) == data.id, k] = v
        df.to_csv(TRAIN_CSV, index=False)
        return df[df["id"].astype(str) == data.id].to_dict(orient="records")[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update failed: {e}")

# === ROOT ===

@app.get("/")
def root():
    return {"message": "Welcome to the SBILife Churn Prediction & SHAP API"}
