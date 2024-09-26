# Loan Default Prediction Using Flask API and Streamlit

This project is focused on predicting loan defaults using machine learning models. It combines Flask for serving the trained models via an API and Streamlit for data visualization and user-friendly predictions. The goal is to use customer data, such as outstanding loans, credit lines, and FICO scores, to predict whether a customer is likely to default on a loan.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
  - [Logistic Regression](#logistic-regression)
  - [Random Forest](#random-forest)
- [MLflow for Experiment Tracking](#mlflow-for-experiment-tracking)
- [Application Setup](#application-setup)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Dataset Download](#dataset-download)
- [Ways to Run the Project](#ways-to-run-the-project)
  - [1. Running the Flask API (`app.py`)](#1-running-the-flask-api-apppy)
  - [2. Running the Streamlit App (`streamlit_app.py`)](#2-running-the-streamlit-app-streamlit_apppy)
  - [3. Running Both Flask and Streamlit Together](#3-running-both-flask-and-streamlit-together)
- [Endpoints](#endpoints)
  - [POST /predict](#post-predict)
- [Streamlit Visualizations](#streamlit-visualizations)
  - [Bar Plot: Default Class Distribution](#bar-plot-default-class-distribution)
  - [Pie Chart: Default vs Non-Default](#pie-chart-default-vs-non-default)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project showcases how machine learning can be integrated into web applications for real-world usage. Two models — Logistic Regression and Random Forest — have been trained on customer loan data to predict loan defaults. These models are then deployed via a Flask API, which can be interacted with using tools like Postman or cURL. A Streamlit app serves as the front-end for easy data exploration and model interaction.

Both models are tracked using **MLflow**, an open-source platform for managing the machine learning lifecycle, including experimentation, reproducibility, and deployment.

## Project Structure

The structure of the project is as follows:

```plaintext
.
├── app.py                      # Flask API for model predictions
├── streamlit_app.py             # Streamlit app for visualizations and predictions
├── Loan_Data.csv                # The dataset used for training and prediction
├── logistic_regression_model.pkl  # Trained Logistic Regression model
├── random_forest_model.pkl        # Trained Random Forest model
├── README.md                    # Project documentation
├── requirements.txt             # List of required packages
├── mlruns/                      # MLflow tracking data
├── static/                      # Static files (if any)
└── templates/                   # HTML templates for Flask (optional, if extended)
