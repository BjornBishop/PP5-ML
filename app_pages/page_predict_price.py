import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from app_pages.pipeline_definitions import final_pipeline

def page_predict_sale_price():
    # Load the pipeline
    pipeline = joblib.load('/workspace/PP5-ML/outputs/ml_pipeline/predict_SalePrice/v3/best_regressor_pipeline.pkl')

    # Define the features used in the pipeline
    used_features = ['1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'GarageArea', 'GrLivArea', 'LotArea', 
                     'OverallCond', 'OverallQual', 'TotalBsmtSF', 'YearBuilt']

    # Load training columns to ensure consistency
    X_train = pd.read_csv('/workspace/PP5-ML/outputs/ml_pipeline/predict_SalePrice/v3/br_X_train.csv')
    expected_columns = X_train[used_features].columns

    # Streamlit dashboard
    st.title("House Price Prediction")

    # Information about the page
    st.info("This page allows you to quickly predict the sale price of a property on the fly. The model uses various features of the property to estimate the price based on a trained machine learning pipeline.")

    # User input for house details with hints
    st.write("### Enter Details of Your House")

    input_data = {}

    feature_descriptions = {
        '1stFlrSF': 'Measurement: Square Feet',
        '2ndFlrSF': 'Measurement: Square Feet',
        'BsmtFinSF1': 'Measurement: Square Feet',
        'GarageArea': 'Measurement: Square Feet',
        'GrLivArea': 'Measurement: Square Feet',
        'LotArea': 'Measurement: Square Feet',
        'OverallCond': 'Scale 1 - 10 : Poor - Good',
        'OverallQual': 'Scale 1 - 10 : Low - High',
        'TotalBsmtSF': 'Measurement: Square Feet',
        'YearBuilt': 'Year'
    }

    for feature in used_features:
        hint = feature_descriptions.get(feature, "")
        input_data[feature] = st.number_input(f"{feature} ({hint})", min_value=0, step=1)

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Ensure input_df has the same columns as expected by the pipeline
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)

    # Make prediction
    if st.button("Predict Sale Price"):
        try:
            prediction = pipeline.predict(input_df)
            st.write(f"### Predicted Sale Price: ${prediction[0]:,.2f}")

            # Load and clean inherited houses data
            Other_houses = pd.read_csv('/workspace/PP5-ML/outputs/datasets/cleaned/CleanedDataset.csv')
            Cleaned_data = Other_houses[used_features].reindex(columns=expected_columns, fill_value=0)

            # Get predictions for the cleaned dataset
            predictions = pipeline.predict(Cleaned_data)

            # Combine user's prediction with the cleaned dataset predictions
            comparison_df = pd.DataFrame({
                'Property': ['Your Property'] + Other_houses.index.tolist(),
                'Sale Price': [prediction[0]] + predictions.tolist()
            })

            # Scatter plot with regression line
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.scatterplot(x=Other_houses['YearBuilt'], y=predictions, label='Other Houses', ax=ax)
            sns.regplot(x=Other_houses['YearBuilt'], y=predictions, scatter=False, ax=ax, color='blue')
            sns.scatterplot(x=[input_df['YearBuilt'][0]], y=[prediction[0]], color='red', s=100, label='Your Property', ax=ax)
            plt.title('Predicted Sale Prices by Year Built')
            plt.xlabel('Year Built')
            plt.ylabel('Predicted Sale Price (USD)')
            st.pyplot(fig)

            # Histogram
            fig, ax = plt.subplots(figsize=(14, 8))
            sns.histplot(predictions, bins=30, kde=True, ax=ax)
            plt.axvline(x=prediction[0], color='red', linestyle='--', label='Your Property')
            plt.title('Distribution of Predicted Sale Prices')
            plt.xlabel('Predicted Sale Price (USD)')
            plt.ylabel('Frequency')
            plt.legend()
            st.pyplot(fig)

        except ValueError as e:
            st.error(f"Error: {e}")
