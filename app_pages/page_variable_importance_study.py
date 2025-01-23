import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_telco_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_churned_customer_study_body():

    # load data
    df = load_housing_data()

    vars_to_study = ['GrLivArea', 'OverallQual',
                     'TotalBsmtSF', 'GarageArea', 'KitchenQual', 'YearBuilt', '1stFlrSF']

    st.write("### Churned Customer Study")
    st.info(
        f"* The client is interested in understanding the patterns from the historical house sales data "
        f"so that the client can learn the most relevant variables correlated "
        f"to a high house sales prices.")

    # inspect data
    if st.checkbox("Inspect Customer Base"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to sales price. \n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    # Text based on things ive made up.
    st.info(
        f"The correlation indications and plots below interpretation converge. "
        f"It is indicated that: \n"
        f"* The year the house was built effects the sales price. \n"
        f"* The overall Quality of the building effects the sales price \n"
        f"* The total basement square footage effects the sale price. \n"
        f"* The garage size effects the sales price. \n"
        f"* THe 1st floor size is more effective for sales price than 2nd floor.. \n"
    )

    df_eda = df.filter(vars_to_study + ['SalePrice'])

    # Streamlit interface
st.title("House Feature Analysis Dashboard")

# Box plot for cleaned data
if st.checkbox("Sale Price Boxplot"):
    st.write(
        "* Shows a box plot of the sales prices to show average price range and outliers"
    )
    sale_price_block_plot(df_eda)

# Heatmap for correlations
if st.checkbox("Show Correlation Heatmap"):
    st.write(
        "* Shows a heatmap of the correlation between selected house features"
    )
    selected_columns = st.multiselect('Select features for heatmap', df_eda.columns.tolist(), default=[
        'GarageArea', 'GrLivArea', 'KitchenQual_encoded', 'OverallQual', '1stFlrSF', 'TotalBsmtSF', 'YearBuilt', 'SalePrice'])
    if selected_columns:
        plot_heatmap(df_eda, selected_columns)

# Individual plots per variable
if st.checkbox("Sales Price per Variable"):
    st.write(
        "* Shows scatter plots of sales price against key features"
    )
    sales_price_per_variable(df_eda)


# function created using "02 - Churned Customer Study" notebook code - "Variables Distribution by Churn" section
def sales_price_per_variable(df_eda):
    # Create subplots for top relationships
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    fig.suptitle('Key Features vs Sale Price', fontsize=16)

# Plot 1
    sns.scatterplot(data=df_eda, x='GrLivArea', y='SalePrice', hue='SalePrice', palette='magma', alpha=0.6, ax=axes[0,0])
    axes[0,0].set_title('Living Area vs Price')

# Plot 2
    sns.scatterplot(data=df_eda, x='OverallQual', y='SalePrice', hue='SalePrice', palette='magma', alpha=0.6, ax=axes[0,1])
    axes[0,1].set_title('Overall Quality vs Price')

# Plot 3
    sns.scatterplot(data=df_eda, x='TotalBsmtSF', y='SalePrice', hue='SalePrice', palette='magma', alpha=0.6, ax=axes[1,0])
    axes[1,0].set_title('Total Basement Area vs Price')

# Plot 4
    sns.scatterplot(data=df_eda, x='GarageArea', y='SalePrice', hue='SalePrice', palette='magma', alpha=0.6, ax=axes[1,1])
    axes[1,1].set_title('Garage Area vs Price')

    plt.tight_layout()
    plt.show()


# code copied from "02 - Churned Customer Study" notebook - "Variables Distribution by Churn" section
def plot_heatmap(df, cols):
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[cols].corr()  # Use provided DataFrame and columns
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of House Features')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()




