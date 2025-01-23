import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **customer** is a person who consumes your service or product.\n"
        f"* A **Sale Price** is a potential house sale value.\n"

        f"**Project Dataset**\n"
        f"* The dataset represents a **housing sale price data from a specific state** "
        f"containing house features "
        f"(like kitchen quality, basement quality, overall condition and overall quality. etc), "
        f"measurements (like lot size, basement size, garage size, above ground living space) "
        f"and unique qualities (like year house was built and year it was last renovated).")

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/BjornBishop/PP5-ML/blob/main/README.md).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* Business Requirement 1: Data Visualization and Correlation study"  
        f"We will inspect the data related to the customer base."
        f"We will conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to Churn."
        f"We will plot the main variables against SalePrice to visualize insights.\n"

        f"* Business Requirement 2: Regression Analysis"
        f"We want to predict with reasonable accuracy the approximate sales price for the customer that has inherited properties in a state where she is unaware of the prices."
        f"We want to predict the sale price for a house based off a variety of features of other houses that have been sold in the same state."
        )

        