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
    return joblib.load(file_path)

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
    base_path = '/workspace/PP5-ML/outputs/ml_pipeline/predict_SalePrice/'
    version_path = os.path.join(base_path, version)

    # Load needed files
    v3_pipeline = load_pkl_file(os.path.join(version_path, 'best_regressor_pipeline.pkl'))
    metrics = load_pkl_file(os.path.join(version_path, 'metrics.pkl'))
    v3_feat_importance = plt.imread(os.path.join(version_path, 'features_importance.png'))
    X_train = pd.read_csv(os.path.join(version_path, 'br_X_train.csv'))
    X_test = pd.read_csv(os.path.join(version_path, 'br_x_test.csv'))
    y_train = pd.read_csv(os.path.join(version_path, 'br_y_train.csv')).values
    y_test = pd.read_csv(os.path.join(version_path, 'br_y_test.csv')).values

    st.write("### ML Pipeline: Predict Sales Price")

    st.info(
        f"* The pipeline was tuned aiming at least 0.75 R2 score on predicting sales price "
        f"since we are interested in creating predictions on potential sales values. \n"
        f"* The pipeline performance on train and test set is 0.87 and 0.79, respectively."
    )

    st.write("---")
    st.write("#### This is a lama. Oh, I mean a lamna.")
    st.write("* The second is for feature scaling and modelling.")
    st.write(v3_pipeline)

    st.write("---")
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(v3_feat_importance)

    st.write("---")
    st.write("### Pipeline Performance")
    regression_performance(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=v3_pipeline)

    st.write("---")
    if st.checkbox("Show Regression Evaluation Plots"):
        regression_evaluation_plots(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=v3_pipeline)

