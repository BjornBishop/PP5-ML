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

def load_pkl_file(file_path):
    if os.path.exists(file_path):  
        return joblib.load(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")

def regression_performance(X_train, y_train, X_test, y_test, pipeline):
    st.write("### Model Evaluation \n")
    st.write("#### Train Set")
    regression_evaluation(X_train, y_train, pipeline)
    st.write("#### Test Set")
    regression_evaluation(X_test, y_test, pipeline)

def regression_evaluation(X, y, pipeline):
    prediction = pipeline.predict(X)
    st.write('**R2 Score:**', r2_score(y, prediction).round(3))
    st.write('**Mean Absolute Error:**', mean_absolute_error(y, prediction).round(3))
    st.write('**Mean Squared Error:**', mean_squared_error(y, prediction).round(3))
    st.write('**Root Mean Squared Error:**', np.sqrt(mean_squared_error(y, prediction)).round(3))
    st.write("\n")

def regression_evaluation_plots(X_train, y_train, X_test, y_test, pipeline, alpha_scatter=0.5):
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    # Ensure y_train and y_test are 1-dimensional
    if y_train.ndim > 1:
        y_train = y_train.ravel()
    if y_test.ndim > 1:
        y_test = y_test.ravel()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=y_train, y=y_train, color='red', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predictions")
    axes[0].set_title("Train Set")

    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_test, y=y_test, color='red', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("Test Set")

    st.pyplot(fig)


import os

def page_sale_price_prediction():
    version = 'v3'
    base_path = 'outputs/ml_pipeline/predict_SalePrice/'  
    version_path = os.path.join(base_path, version)

    # Debugging: Log directory and file contents
    print("Working Directory:", os.getcwd())
    if os.path.exists(version_path):
        print("Files in 'outputs/ml_pipeline/predict_SalePrice/v3':", os.listdir(version_path))
    else:
        print(f"Directory not found: {version_path}")
        st.error(f"Directory not found: {version_path}")
        return

    # Load needed files with error handling
    try:
        v3_pipeline = load_pkl_file(os.path.join(version_path, 'best_regressor_pipeline.pkl'))
        v3_feat_importance = plt.imread(os.path.join(version_path, 'features_importance.png'))
        X_train = pd.read_csv(os.path.join(version_path, 'br_X_train.csv'))
        X_test = pd.read_csv(os.path.join(version_path, 'br_x_test.csv'))
        y_train = pd.read_csv(os.path.join(version_path, 'br_y_train.csv')).values
        y_test = pd.read_csv(os.path.join(version_path, 'br_y_test.csv')).values
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # Flatten y_train and y_test
    if y_train.ndim > 1:
        y_train = y_train.ravel()
    if y_test.ndim > 1:
        y_test = y_test.ravel()

    # Section: Introduction
    st.write("### ML Pipeline: Predict Sales Price")
    st.info(
        f"The pipeline was trained to predict house sales prices based on a variety of features. "
        f"The model achieves solid performance, exceeding the client's R2 score target of 0.75."
    )

    # Section: Model Details and Feature Importance
    st.write("---")
    st.write("#### Model Overview")
    st.write("This is the regression pipeline used for prediction:")
    st.write(v3_pipeline)

    st.write("---")
    st.write("#### Key Features and Importance")
    st.image(v3_feat_importance)
    st.info(
        f"The model identified the following features as the most influential:\n"
        f"- Above Ground Living Area\n"
        f"- Total Basement Size\n"
        f"- Year Built"
    )

    # Section: Performance Metrics
    st.write("---")
    st.write("### Model Performance Metrics")

    st.write("#### Train Set Performance:")
    regression_evaluation(X_train, y_train, v3_pipeline)

    st.write("#### Test Set Performance:")
    regression_evaluation(X_test, y_test, v3_pipeline)

    st.info(
        f"The test set R2 score of **0.797** demonstrates the model's ability to explain nearly 80% of the variance "
        f"in house sale prices based on the features provided."
    )

    # Section: Actual vs Predicted Plots
    st.write("---")
    st.write("### Actual vs Predicted Plots")
    if st.checkbox("Show Plots"):
        regression_evaluation_plots(X_train, y_train, X_test, y_test, v3_pipeline)
        st.info(
            f"The scatter plots below show how well the model's predictions align with actual sales prices:\n"
            f"- **Train Set**: Predicted values follow the ideal trend closely.\n"
            f"- **Test Set**: Minor deviations for high-priced houses (outliers) are observed."
        )

