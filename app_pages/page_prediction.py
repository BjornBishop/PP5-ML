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
from sklearn.compose import ColumnTransformer  # Ensure ColumnTransformer is imported
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from app_pages.pipeline_definitions import final_pipeline  # Adjusted import path



# Ensure preprocessor is defined
num_features = ['OverallCond', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'LotArea']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features)
    ]
)

# Load the pipeline
def load_pkl_file(file_path):
    return joblib.load(file_path)

# Regression performance functions
def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    st.write("### Model Evaluation \n")
    
    st.write("#### Train Set")
    regression_evaluation(X_train, y_train, pipeline)
    
    st.write("#### Test Set")
    regression_evaluation(X_test, y_test, pipeline)

def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)
    st.write('R2 Score:', r2_score(y, prediction).round(3))
    st.write('Mean Absolute Error:', mean_absolute_error(y, prediction).round(3))
    st.write('Mean Squared Error:', mean_squared_error(y, prediction).round(3))
    st.write('Root Mean Squared Error:', np.sqrt(mean_squared_error(y, prediction)).round(3))
    st.write("\n")

def page_prediction_body():
    version = 'v3'
    # Load needed files
    v1_pipeline = load_pkl_file(f'outputs/ml_pipeline/predict_SalePrice/v1.1/pipeline1.pkl')
    v1_important_features = plt.imread(f'outputs/ml_pipeline/predict_SalePrice/v1.1/features_importance.png')
    v2_pipeline = load_pkl_file(f'outputs/ml_pipeline/predict_SalePrice/v2/clf_pipeline.pkl')
    v2_important_features = plt.imread(f'outputs/ml_pipeline/predict_SalePrice/v2/features_importance.png')
    v3_pipeline = load_pkl_file(f'outputs/ml_pipeline/predict_SalePrice/v3/best_regressor_pipeline.pkl')
    v3_important_features = plt.imread(f'outputs/ml_pipeline/predict_SalePrice/v3/features_importance.png')
    
    X_train = pd.read_csv(f'outputs/ml_pipeline/predict_SalePrice/v3/br_X_train.csv')
    y_train = pd.read_csv(f'outputs/ml_pipeline/predict_SalePrice/v3/br_y_train.csv')
    X_test = pd.read_csv(f'outputs/ml_pipeline/predict_SalePrice/v3/br_X_test.csv')
    y_test = pd.read_csv(f'outputs/ml_pipeline/predict_SalePrice/v3/br_y_test.csv')

    st.title("House Price Prediction Model Performance Dashboard")

    st.write("### Feature Importance")

    st.write("#### V1 Pipeline Feature Importance")
    st.image(v1_important_features)

    st.write("#### V2 Pipeline Feature Importance")
    st.image(v2_important_features)

    st.write("#### V3 Pipeline Feature Importance")
    st.image(v3_important_features)

    if st.checkbox("Show Regression Metrics"):
        st.write("### Model Evaluation Metrics")
        st.write("#### Train Set Metrics")
        regression_performance(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=v3_pipeline)

        st.write("#### Test Set Metrics")
        regression_performance(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=v3_pipeline)

    if st.checkbox("Show Box Plot of Predictions"):
        predictions = v3_pipeline.predict(X_test)
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=predictions)
        plt.title('Box Plot of Predicted Sale Prices')
        plt.ylabel('Predicted Sale Price')
        st.pyplot(plt)
