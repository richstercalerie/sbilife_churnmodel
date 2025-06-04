# Use official Python image
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code
COPY api/ ./api/
COPY model/ ./model/
COPY data/ ./data/

# Expose port
EXPOSE 8888

# Start FastAPI with uvicorn
CMD ["uvicorn", "api.shap_api:app", "--host", "0.0.0.0", "--port", "8888"]