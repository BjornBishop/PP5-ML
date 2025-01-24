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
    st.write("* This is the model that predicted most accurately.")
    st.write(v3_pipeline)

    st.write("---")
    st.write("* The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(v3_feat_importance)
    st.info(
        f"* We can note here than from the original 25 features"
        f" The model that predicted the most accurately was only"
        f" utilising 3 features: Above Ground Living Area, Total Basement size"
        f" and the year the building was built."
    )

    st.write("---")
    st.write("### Pipeline Performance")
    regression_performance(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=v3_pipeline)
    st.info(
        f"* As we can see here, the model performed quite well."
        f"* Train Set R2 Score: 0.861"
        f"* Test set R2 Score: 0.79"
        f"* This surpassed the required prediction score requested by the client"
    )

    st.write("---")
    if st.checkbox("Show Regression Evaluation Plots"):
        regression_evaluation_plots(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=v3_pipeline)
        st.info(
            f"* The blue dots represent the actual and predicted value provided by the ML.  "
            f" The red line indicated where the predicted value is. \n"
            f"* As it should, the blue dots follow the red line to a pretty accurate degree."
            f"This occured both for test and training sets of data."
            f"*  We note that there are few datapoints above 400,000+ so these values are often "
            f" harder to predict than the others. "
    )

