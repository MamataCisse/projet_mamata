import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests

# Load the dataset
df_path = "Loan_Data.csv"
df = pd.read_csv(df_path)

# Title
st.title("Loan Default Prediction and Analysis")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Data Visualization", "Make Predictions"])

# Data Visualization Section
if section == "Data Visualization":
    st.header("Data Visualization")

    # Plot 1: Bar plot for default distribution
    st.subheader("Distribution of the 'Default' Class")

    # Count occurrences of each class in 'default'
    default_counts = df['default'].value_counts()

    # Bar plot
    fig, ax = plt.subplots()
    sns.barplot(x=default_counts.index, y=default_counts.values, ax=ax)
    ax.set_title('Distribution of the Default Class')
    ax.set_xlabel('Default (0 = Non, 1 = Oui)')
    ax.set_ylabel('Number of Customers')
    st.pyplot(fig)

    # Plot 2: Pie chart for default proportion
    st.subheader("Proportion of Default vs Non-Default")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(default_counts, labels=['Non-Default', 'Default'], autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
    ax.set_title('Proportion of Default vs Non-Default')
    st.pyplot(fig)

# Prediction Section
elif section == "Make Predictions":
    st.header("Predict Loan Default for a Customer")

    # Dropdown to select customer ID
    customer_ids = df['customer_id'].unique()
    selected_id = st.selectbox("Select Customer ID", customer_ids)

    # Button to trigger prediction
    if st.button("Predict"):
        # Call the Flask API
        try:
            response = requests.post("http://127.0.0.1:5000/predict", json={"customer_id": selected_id})
            if response.status_code == 200:
                prediction = response.json()

                # Display the prediction
                st.subheader(f"Prediction for Customer ID: {selected_id}")
                st.write(f"Logistic Regression Prediction: {prediction['logreg_prediction']}")
                st.write(f"Logistic Regression Probability: {prediction['logreg_probability']:.2f}")
                st.write(f"Random Forest Prediction: {prediction['rf_prediction']}")
                st.write(f"Random Forest Probability: {prediction['rf_probability']:.2f}")
            else:
                st.error("Error in prediction: " + response.text)

        except Exception as e:
            st.error(f"Error connecting to the Flask app: {e}")
