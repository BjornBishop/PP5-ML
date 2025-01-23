import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_housing_data():
    df = pd.read_csv("inputs/datasets/raw/house-price-20211124T154130Z-001/house-price/house_prices_records.csv")
    return df


def load_pkl_file(file_path):
    return joblib.load(filename=file_path)