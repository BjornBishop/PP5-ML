import streamlit as st


def page_hypothesis_validation_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.success(
        f"* We suspect there are less than 5 features that can determine the sales value of a house: Correct. "
        f"The correlation study at Variable_importance_study supports that. \n\n"

        f"* Analysis shows above ground living space, year built and overall condition are valued by buyers "
        f"A valuable house typically has good overall quality building, large living area and is built in a specific year range. "
        f"This insight will be used by the survey team for further discussions and investigations."
    )
