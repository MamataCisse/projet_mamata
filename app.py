from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load dataset
df_path = "Loan_Data.csv"
df = pd.read_csv(df_path)

# Load models
with open("logistic_regression_model.pkl", "rb") as f:
    logreg_model = pickle.load(f)

with open("random_forest_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    # Get JSON request data
    data = request.get_json()

    # Check if customer_id is provided
    customer_id = data.get("customer_id")
    if not customer_id:
        return jsonify({"error": "No customer_id provided"}), 400

    # Find customer data in the dataset
    customer_data = df[df["customer_id"] == customer_id]

    if customer_data.empty:
        return jsonify({"error": "Customer ID not found"}), 404

    # Extract features (excluding 'customer_id' and 'default')
    features = customer_data[['credit_lines_outstanding', 'loan_amt_outstanding', 
                              'total_debt_outstanding', 'income', 
                              'years_employed', 'fico_score']].values

    # Scale the features if necessary (e.g., use the scaler applied during training)
    # Here, assuming features are already scaled if models are trained on scaled data

    # Logistic Regression prediction
    logreg_pred = logreg_model.predict(features)
    logreg_proba = logreg_model.predict_proba(features)[:, 1]

    # Random Forest prediction
    rf_pred = rf_model.predict(features)
    rf_proba = rf_model.predict_proba(features)[:, 1]

    # Respond with predictions
    return jsonify({
        "customer_id": customer_id,
        "logreg_prediction": int(logreg_pred[0]),
        "logreg_probability": float(logreg_proba[0]),
        "rf_prediction": int(rf_pred[0]),
        "rf_probability": float(rf_proba[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
