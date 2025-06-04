from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import shap
import numpy as np
import os
import csv
from io import StringIO
from pydantic import BaseModel
from typing import List, Union

app = FastAPI(title="SHAP Summary API")

# Allow all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths to model and training CSV
MODEL_PATH = r"C:\Users\Anurag mishra\sbilife_churnmodel\model\sbilife_churn_model.pkl"
TRAIN_CSV = r"C:\Users\Anurag mishra\sbilife_churnmodel\data\train.csv"

# Expected columns (for SHAP summary computation)
expected_columns = [
    'Age', 'Gender', 'Region_Code', 'Previously_Insured',
    'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage'
]

# Expected CSV fields (12 fields including id and others)
expected_csv_fields = [
    'id', 'Gender', 'Age', 'Driving_License', 'Region_Code', 'Previously_Insured',
    'Vehicle_Age', 'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response'
]

# Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Could not load model: {e}")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

# Pydantic model for JSON input validation
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

def load_train_data():
    try:
        df = pd.read_csv(TRAIN_CSV)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not load training data: {e}")
    if not all(col in df.columns for col in expected_columns):
        raise HTTPException(status_code=400, detail=f"Training data missing required columns: {expected_columns}")
    df = df[expected_columns]
    for col in expected_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def compute_shap_summary():
    df = load_train_data()
    sample_size = 1000
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    chunk_size = 100
    n = len(df)
    num_chunks = 0
    aggregated_shap = np.zeros(len(expected_columns))
    for i in range(0, n, chunk_size):
        chunk = df.iloc[i: i + chunk_size]
        shap_values_chunk = explainer.shap_values(chunk)
        mean_abs_shap_chunk = np.mean(np.abs(shap_values_chunk), axis=0)
        aggregated_shap += mean_abs_shap_chunk
        num_chunks += 1
    avg_shap = aggregated_shap / num_chunks
    total_shap = np.sum(avg_shap)
    percentage_importance = (avg_shap / total_shap) * 100
    summary_dict = dict(zip(expected_columns, percentage_importance.tolist()))
    expected_value = (explainer.expected_value.tolist()
                     if hasattr(explainer.expected_value, "tolist")
                     else explainer.expected_value)
    expected_value_percentage = float(expected_value) * 100
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
        raise HTTPException(status_code=500, detail=f"Error computing SHAP summary: {e}")

@app.post("/update_data")
async def update_data(request: Request):
    """
    Accepts CSV text (single or multiple rows) or JSON (single object or list).
    For CSV, expects 12 comma-separated fields per row.
    For JSON, expects object(s) matching DataRow schema.
    If id is empty, auto-generates a new id.
    """
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
                data = [data]  # Convert single object to list
            elif not isinstance(data, list):
                raise HTTPException(status_code=400, detail="JSON input must be an object or list of objects.")

            for item in data:
                row = DataRow(**item).dict()
                if row["id"] == "":
                    last_id += 1
                    row["id"] = str(last_id)
                # Convert to CSV row
                csv_row = [str(row[field]) for field in expected_csv_fields]
                new_rows.append(",".join(csv_row))
        else:
            # Handle CSV text input
            body = await request.body()
            text = body.decode("utf-8").replace('\r\n', '\n').replace('\r', '\n').strip()
            if not text:
                raise HTTPException(status_code=400, detail="Empty request body.")
            
            reader = csv.reader(StringIO(text))
            for fields in reader:
                fields = [field.strip() for field in fields]
                if len(fields) != len(expected_csv_fields):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Expected {len(expected_csv_fields)} fields but got {len(fields)} in row: {fields}"
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
        return {"detail": f"{len(new_rows)} rows added successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating data: {str(e)}")

@app.post("/update_data_by_id")
async def update_data_by_id(data: DataRow):
    """
    Updates an existing record in train.csv by ID using JSON input.
    """
    try:
        if not data.id:
            raise HTTPException(status_code=400, detail='Missing required field "id".')
        update_id = data.id
        df = pd.read_csv(TRAIN_CSV)
        if update_id not in df['id'].values:
            raise HTTPException(status_code=404, detail=f"No row found with id {update_id}.")
        for key, value in data.dict().items():
            if key != 'id' and key in df.columns:
                df.loc[df['id'] == update_id, key] = value
        df.to_csv(TRAIN_CSV, index=False)
        updated_row = df[df['id'] == update_id].to_dict(orient='records')[0]
        return updated_row
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating data: {str(e)}")

@app.get("/")
def root():
    return {"message": "SHAP Summary API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8888)