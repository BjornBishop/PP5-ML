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


def page_predict_price_body():
    # Load the pipeline
    pipeline = joblib.load('/workspace/PP5-ML/outputs/ml_pipeline/predict_SalePrice/v3/best_regressor_pipeline.pkl')

    # Define numerical features
    num_features = ['OverallCond', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'LotArea']

    # Preprocessor for the pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features)
        ]
    )

    # Example imputer (if not already included in the pipeline)
    imputer_num = SimpleImputer(strategy='mean')

    # Streamlit dashboard
    st.title("House Price Prediction")

    # User input for house details
    st.write("### Enter Details of Your House")

    input_data = {}

    for feature in num_features:
        input_data[feature] = st.number_input(feature, min_value=0.0)

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # Impute and scale the input data (if required)
    input_df[num_features] = imputer_num.fit_transform(input_df[num_features])
    input_df = pd.DataFrame(preprocessor.fit_transform(input_df), columns=num_features)

    # Make prediction
    if st.button("Predict Sale Price"):
        prediction = pipeline.predict(input_df)
        st.write(f"### Predicted Sale Price: {prediction[0]:.2f}")

        # Plot predictions (Optional)
        st.write("### Predicted Sale Prices for Inherited Houses")
        st.write("The plot below shows the predicted sale prices for the inherited houses:")
        
        # Assuming you have actual predictions for inherited houses
        Inherited_houses = pd.read_csv('/workspace/PP5-ML/inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/inherited_houses.csv')
        plt.figure(figsize=(14, 8))
        sns.barplot(x=Inherited_houses.index, y=prediction)
        plt.title('Predicted Sale Prices for Inherited Houses')
        plt.xlabel('Property')
        plt.ylabel('Predicted Sale Price')
        plt.xticks(rotation=90)
        st.pyplot(plt)
