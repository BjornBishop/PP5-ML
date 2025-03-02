import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_housing_data_transformed
import pandas as pd

def page_variable_importance_study_body():
    # Load data
    df = load_housing_data_transformed()

    vars_to_study = ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageArea', 'KitchenQual', 'YearBuilt', '1stFlrSF']

    st.write("### House Sales Data")
    st.info(
        f"* The client is interested in understanding the patterns from the historical house sales data "
        f"so that the client can learn the most relevant variables correlated "
        f"to high house sale prices."
    )

    # Inspect data
    if st.checkbox("Inspect database"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns. "
            f"Find below the first 10 rows."
        )
        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to sales price. \n"
        f"The most correlated variables are: **{vars_to_study}**"
    )

    # Text based on correlation study results
    st.info(
        f"The correlation indications and plots below converge. "
        f"It is indicated that: \n"
        f"* The year the house was built affects the sales price. \n"
        f"* The overall quality of the building affects the sales price. \n"
        f"* The total basement square footage affects the sale price. \n"
        f"* The garage size affects the sales price. \n"
        f"* The 1st floor size is more effective for sales price than the 2nd floor. \n"
    )

    df_eda = df.filter(vars_to_study + ['SalePrice'])

    # Interactive Scatter Plot
    if st.checkbox("Interactive Scatter Plot"):
        st.write("* Explore the relationship between Sale Price and key features dynamically.")
        feature = st.selectbox("Choose a feature for scatter plot:", df_eda.columns[:-1])  # Exclude SalePrice
        fig = px.scatter(
            df_eda, x=feature, y='SalePrice',
            title=f"{feature} vs SalePrice",
            labels={feature: feature, 'SalePrice': 'Sale Price'},
            color='SalePrice',
            color_continuous_scale='Viridis',
            hover_data=df_eda.columns
        )
        st.plotly_chart(fig)

    st.write("---")

    # Interactive Heatmap for Correlations
    if st.checkbox("Show Correlation Heatmap"):
        st.write(
            "* An interactive heatmap to visualize the correlation between selected house features."
        )
        selected_columns = st.multiselect(
            'Select features for heatmap', 
            df_eda.columns.tolist(), 
            default=['GarageArea', 'GrLivArea', 'OverallQual', '1stFlrSF', 'TotalBsmtSF', 'YearBuilt', 'SalePrice']
        )
        if selected_columns:
            correlation_matrix = df_eda[selected_columns].corr()
            fig = px.imshow(
                correlation_matrix, 
                text_auto=True,
                color_continuous_scale='RdBu_r',
                title="Correlation Heatmap"
            )
            st.plotly_chart(fig)

    # Box plot for cleaned data
    if st.checkbox("Interactive Sale Price Boxplot"):
        st.write(
            "* Shows a box plot of the sales prices to highlight average price ranges and outliers."
        )
        fig = px.box(df_eda, y='SalePrice', title='Sale Price Distribution')
        st.plotly_chart(fig)

    st.write("---")

    # Individual scatter plots per variable
    if st.checkbox("Sales Price Scatter Plot by Variable"):
        st.write(
            "* Shows scatter plots of sales price against selected key features."
        )
        feature = st.selectbox(
            "Select a variable for comparison with Sale Price:", 
            vars_to_study
        )
        fig = px.scatter(
            df_eda, x=feature, y='SalePrice',
            title=f"{feature} vs Sale Price",
            labels={feature: feature, 'SalePrice': 'Sale Price'},
            color='SalePrice',
            trendline='ols',
            hover_data=df_eda.columns
        )
        st.plotly_chart(fig)
