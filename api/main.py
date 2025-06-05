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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SBI Life - Churn Prediction & SHAP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
MODEL_PATH = os.getenv("MODEL_PATH", "model/sbilife_churn_model.pkl")
TRAIN_CSV = os.getenv("TRAIN_CSV", "data/train.csv")

# Expected columns for SHAP
expected_columns = [
    'Age', 'Gender', 'Region_Code', 'Previously_Insured',
    'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'
]

# Expected CSV fields
expected_csv_fields = [
    'id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
    'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response'
]

# Load model and initialize SHAP explainer
try:
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    logger.info("Model and SHAP explainer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model or initialize explainer: {e}")
    raise RuntimeError(f"Could not load model or initialize explainer: {e}")

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
        prediction = int(prob > 0.5)  # Standard threshold
        logger.info(f"Prediction made: probability={prob*100:.2f}%, prediction={'Churn Likely' if prediction else 'Retention Likely'}")
        return {
            "churn_probability": round(prob * 100, 2),
            "prediction": "Churn Likely" if prediction else "Retention Likely"
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error predicting churn: {str(e)}")

# === SHAP SUMMARY ENDPOINT ===

def load_train_data():
    try:
        if not os.path.exists(TRAIN_CSV):
            logger.error(f"Training CSV file not found at {TRAIN_CSV}")
            raise HTTPException(status_code=500, detail=f"Training CSV file not found at {TRAIN_CSV}")
        df = pd.read_csv(TRAIN_CSV)
    except Exception as e:
        logger.error(f"Could not load training data: {e}")
        raise HTTPException(status_code=500, detail=f"Could not load training data: {e}")

    if not all(col in df.columns for col in expected_columns):
        missing_cols = [col for col in expected_columns if col not in df.columns]
        logger.error(f"Missing required columns: {missing_cols}")
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

    # Preprocess string columns
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0, '0': 0, '1': 1}).fillna(0).astype(float)
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0, '0': 0, '1': 1}).fillna(0).astype(float)
    df = df[expected_columns]
    for col in expected_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    if df.empty:
        logger.error("DataFrame is empty after preprocessing")
        raise HTTPException(status_code=400, detail="No valid data after preprocessing")
    logger.info(f"Loaded {len(df)} rows from training data")
    return df

def compute_shap_summary():
    df = load_train_data()
    sample_size = 1000
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        logger.info(f"Sampled DataFrame to {len(df)} rows")
    chunk_size = 100
    n = len(df)
    num_chunks = 0
    aggregated_shap = np.zeros(len(expected_columns))
    for i in range(0, n, chunk_size):
        chunk = df.iloc[i:i + chunk_size]
        if chunk.empty:
            logger.warning(f"Empty chunk at index {i}")
            continue
        shap_values_chunk = explainer.shap_values(chunk)
        logger.info(f"SHAP values shape for chunk {i//chunk_size + 1}: {np.shape(shap_values_chunk)}")
        # Handle SHAP values
        if isinstance(shap_values_chunk, list):
            shap_values_chunk = shap_values_chunk[1]  # Positive class
        elif isinstance(shap_values_chunk, np.ndarray):
            if shap_values_chunk.ndim == 1:
                shap_values_chunk = shap_values_chunk.reshape(1, -1)  # Single-sample case
        else:
            logger.error(f"Unexpected SHAP values type: {type(shap_values_chunk)}")
            raise HTTPException(status_code=500, detail=f"Unexpected SHAP values type")
        if shap_values_chunk.shape[1] != len(expected_columns):
            logger.error(f"Invalid SHAP values shape: {shap_values_chunk.shape}")
            raise HTTPException(status_code=500, detail=f"Invalid SHAP values shape")
        mean_abs_shap_chunk = np.mean(np.abs(shap_values_chunk), axis=0)
        aggregated_shap += mean_abs_shap_chunk
        num_chunks += 1
    if num_chunks == 0:
        logger.error("No valid chunks processed")
        raise HTTPException(status_code=500, detail="No valid data chunks processed")
    avg_shap = aggregated_shap / num_chunks
    total_shap = np.sum(avg_shap)
    percentage_importance = (avg_shap / total_shap) * 100 if total_shap > 0 else np.zeros(len(expected_columns))
    summary_dict = {k: float(v) for k, v in zip(expected_columns, percentage_importance.tolist())}
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = expected_value[1]
    expected_value_percentage = float(expected_value) * 100
    logger.info("SHAP summary computed successfully")
    return {
        "expected_value": round(expected_value_percentage, 2),
        "shap_summary": summary_dict
    }

@app.get("/shap_summary")
def get_shap_summary():
    try:
        result = compute_shap_summary()
        return result
    except Exception as e:
        logger.error(f"Error computing SHAP summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error computing SHAP summary: {str(e)}")

# === CSV UPLOAD (append new data) ===

@app.post("/update_data")
async def update_data(request: Request):
    try:
        # Get last ID
        if os.path.exists(TRAIN_CSV) and os.path.getsize(TRAIN_CSV) > 0:
            df_existing = pd.read_csv(TRAIN_CSV)
            last_id = df_existing.iloc[-1]["id"] if "id" in df_existing.columns else 0
            try:
                last_id = int(last_id)
            except:
                last_id = 0
        else:
            last_id = 0

        # Determine content type
        content_type = request.headers.get("Content-Type", "").lower()
        new_rows = []

        if "application/json" in content_type:
            # Handle JSON input
            data = await request.json()
            if isinstance(data, dict):
                data = [data]
            elif not isinstance(data, list):
                raise HTTPException(status_code=400, detail="JSON input must be an object or list of objects")
            for item in data:
                row = DataRow(**item).dict()
                if row["id"] == "":
                    last_id += 1
                    row["id"] = str(last_id)
                csv_row = [str(row[field]) for field in expected_csv_fields]
                new_rows.append(",".join(csv_row))
        else:
            # Handle CSV text input
            body = await request.body()
            text = body.decode("utf-8").replace('\r\n', '\n').replace('\r', '\n').strip()
            if not text:
                raise HTTPException(status_code=400, detail="Empty request body")
            reader = csv.reader(StringIO(text))
            for fields in reader:
                fields = [field.strip() for field in fields]
                if len(fields) != len(expected_csv_fields):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Expected {len(expected_csv_fields)} fields but got {len(fields)}"
                    )
                if fields[0] == "":
                    last_id += 1
                    fields[0] = str(last_id)
                new_rows.append(",".join(fields))

        # Append to CSV
        mode = "a" if os.path.exists(TRAIN_CSV) and os.path.getsize(TRAIN_CSV) > 0 else "w"
        newline_prefix = "\n" if mode == "a" and os.path.getsize(TRAIN_CSV) > 0 else ""
        with open(TRAIN_CSV, mode, encoding="utf-8") as f:
            if mode == "w":
                f.write(",".join(expected_csv_fields) + "\n")
            for i, row_str in enumerate(new_rows):
                f.write((newline_prefix if i == 0 else "\n") + row_str)
        logger.info(f"Added {len(new_rows)} rows to {TRAIN_CSV}")
        return {"detail": f"{len(new_rows)} rows added successfully"}
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating data: {str(e)}")

# === CSV UPDATE (update row by ID) ===

@app.post("/update_data_by_id")
async def update_data_by_id(data: DataRow):
    try:
        if not data.id:
            raise HTTPException(status_code=400, detail="Missing required field 'id'")
        update_id = data.id
        if not os.path.exists(TRAIN_CSV):
            raise HTTPException(status_code=500, detail=f"Training CSV file not found at {TRAIN_CSV}")
        df = pd.read_csv(TRAIN_CSV)
        if update_id not in df['id'].astype(str).values:
            raise HTTPException(status_code=404, detail=f"No row found with id {update_id}")
        for key, value in data.dict().items():
            if key != 'id' and key in df.columns:
                df.loc[df['id'].astype(str) == update_id, key] = value
        df.to_csv(TRAIN_CSV, index=False)
        updated_row = df[df['id'].astype(str) == update_id].to_dict(orient='records')[0]
        logger.info(f"Updated record with id {update_id}")
        return updated_row
    except Exception as e:
        logger.error(f"Error updating data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating data: {str(e)}")

# === ROOT ===

@app.get("/")
def root():
    return {"message": "SBI Life Churn Prediction & SHAP API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8888)
