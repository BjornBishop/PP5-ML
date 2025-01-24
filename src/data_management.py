import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_housing_data():
    df_raw = pd.read_csv("inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/house_prices_records.csv")
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)

def load_housing_data_transformed():
    df_trans = pd.read_csv("/workspace/PP5-ML/outputs/datasets/collection/Housing_prices_transformed.csv")
    return df_trans

def final_pipeline():
    ppl_final = joblib.load("/workspace/PP5-ML/outputs/ml_pipeline/predict_SalePrice/v1.1/final_pipeline.pkl")
    return ppl_final

def CleanedDataset():
    ppl_cleaned = joblib.load("/workspace/PP5-ML/outputs/datasets/cleaned/CleanedDataset.csv.pkl")
    return ppl_cleaned