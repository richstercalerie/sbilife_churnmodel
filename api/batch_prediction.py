import pandas as pd
import joblib

MODEL_PATH = r"C:\Users\Anurag mishra\sbilife_churnmodel\model\sbilife_churn_model.pkl"

# Load model
try:
    print("Loading model from:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Failed to load model:", e)
    raise RuntimeError(f"Could not load model: {e}")

# Load test data
TEST_FILE = r"c:\Users\Anurag mishra\sbilife_churnmodel\data\test.csv"
try:
    df = pd.read_csv(TEST_FILE)
    print("Test data loaded successfully, shape:", df.shape)
except Exception as e:
    print("Failed to load test file:", e)
    exit(1)

# Ensure column order matches the training schema
expected_columns = ['Age', 'Gender', 'Region_Code', 'Previously_Insured',
                    'Vehicle_Damage', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']

if not all(col in df.columns for col in expected_columns):
    print("Test data is missing one or more required columns:", expected_columns)
    exit(1)

df = df[expected_columns]

# Convert categorical columns to numeric
if df['Gender'].dtype == object:
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
if df['Vehicle_Damage'].dtype == object:
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

for col in ['Age', 'Gender', 'Region_Code', 'Previously_Insured', 'Vehicle_Damage', 'Policy_Sales_Channel', 'Vintage']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['Annual_Premium'] = pd.to_numeric(df['Annual_Premium'], errors='coerce')

# Run predictions
print("Running batch predictions...")
THRESHOLD = 0.3  # Use threshold determined by your validation
pred_probs = model.predict_proba(df)
results = []
for idx, row in enumerate(df.iterrows()):
    prob = float(pred_probs[idx][1])
    prediction_label = "Churn Likely" if prob > THRESHOLD else "Retention Likely"
    results.append({
         "index": idx,
         "churn_probability": round(prob * 100, 2),
         "prediction": prediction_label
     })

results_df = pd.DataFrame(results)
print("Batch predictions:")
print(results_df)

# Optionally, save the results to a CSV file
results_df.to_csv(r"c:\Users\Anurag mishra\sbilife_churnmodel\batch_predictions.csv", index=False)
print("Predictions saved to batch_predictions.csv")