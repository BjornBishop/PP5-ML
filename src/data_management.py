import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_housing_data():
    df_raw = pd.read_csv("inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/house_prices_records.csv")
    return df_raw

def load_prediction_pipeline():
    pipeline_path = 'outputs/ml_pipeline/predict_SalePrice/v3/best_regressor_pipeline.pkl'
    return joblib.load(pipeline_path)

def load_pkl_file(file_path):
    return joblib.load(filename=file_path)

def load_housing_data_transformed():
    df_trans = pd.read_csv("outputs/datasets/collection/housing_prices_transformed.csv")
    return df_trans

def final_pipeline():
    ppl_final = joblib.load("outputs/ml_pipeline/predict_SalePrice/v1.1/final_pipeline.pkl")
    return ppl_final

def CleanedDataset():
    ppl_cleaned = joblib.load("outputs/datasets/cleaned/CleanedDataset.csv")
    return ppl_cleaned