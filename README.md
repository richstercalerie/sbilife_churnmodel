# SBILife Churn Model & SHAP API

This project implements an **XGBoost-based churn prediction model** along with a **FastAPI service** to expose **SHAP (SHapley Additive exPlanations)** summary statistics for model interpretability. The API is designed to integrate with a **React dashboard** for visualizing key factors behind customer churn.

---

## 📑 Table of Contents

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Project Structure](#project-structure)
* [Setup Instructions](#setup-instructions)
* [Running the API](#running-the-api)
* [Testing the Endpoint](#testing-the-endpoint)
* [Integration with React](#integration-with-react)


---

## 📌 Overview

* **Model**: XGBoost classifier trained on SBILife customer data to predict churn.
* **Explainability**: SHAP values provide transparency behind predictions.
* **API**: FastAPI endpoints deliver predictions and SHAP summaries.
* **Visualization**: API integrates easily with a React dashboard for analytics.

---

## 🧰 Prerequisites

Make sure the following are installed:

* Python ≥ 3.8
* `pip` (latest recommended)
* `venv` or `virtualenv`
* [Microsoft C++ Build Tools (Windows)](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
* [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
* [Docker](https://www.docker.com/)

---

## 🗂 Project Structure

```
sbilife_churnmodel/
├── api/
│   ├── batch_prediction.py
│   ├── main.py
│   ├── shap_api.py
│   ├── Dockerfile
│   ├── requirements.txt
├── data/
│   ├── train.csv
│   ├── test.csv
├── model/
│   ├── sbilife_churn_model.pkl
│   ├── shap_values.pkl
├── notebook/
│   ├── sbilife_model_train.ipynb
│   ├── test.py
├── batch_prediction.csv
├── README.md
├── .gitignore
├── requirements.txt
├── venv/ (ignored)
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository_url>
cd sbilife_churnmodel
```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train or Verify the Model

* Open `notebook/sbilife_model_train.ipynb`
* Ensure the model is saved as: `model/sbilife_churn_model.pkl`
* SHAP values should be generated and saved as: `model/shap_values.pkl`

---

## 🚀 Running the API

```bash
python api/shap_api.py
```

* Runs on `http://localhost:8888`
* Uvicorn is used under the hood

---

## ✅ Testing the Endpoint

Navigate to:

```bash
http://localhost:8888/shap_summary
```

Expected JSON response:

```json
{
  "expected_value": 0.2843,
  "shap_summary": {
    "Age": 0.213,
    "Vintage": 0.192,
    ...
  },
  "sample_data": [...]
}
```

---

## ⚛️ Integration with React

In your React app, use `fetch` or `axios` to hit:

```js
fetch('http://localhost:8888/shap_summary')
```

* Use the returned SHAP data to render visualizations (e.g., bar charts).

---



## 🧪 Sample PowerShell Call (Multiple Inputs)

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8888/update_data" `
  -Method Post `
  -Headers @{ "accept" = "application/json"; "Content-Type" = "text/plain" } `
  -Body @'
6,Male,30,1,28.0,0,1-2 Year,No,35000.0,26.0,100,1
7,Female,45,1,15.0,1,> 2 Years,Yes,90000.0,50.0,200,1
8,Female,55,1,99.0,1,> 2 Years,Yes,150000.0,77.0,300,1
'@#
```

---

## 📌 Notes

* 🔁 **Performance**: Adjust sampling in `shap_api.py` for larger datasets.
* 🛡 **Security**: Secure the GCP endpoint with IAM roles.
* 💸 **Cost Monitoring**: Keep track of GCP usage to avoid unexpected billing.
* 🔄 **Model Updates**: Re-run notebook and redeploy if retraining the model.

---

## 👨‍💻 Maintainers

For questions or support, feel free to open an issue on anurag17sw@gmail.com

