import streamlit as st


def page_hypothesis_validation_body():

    st.write("### Project Hypotheses and Validation")

    # Hypothesis 1
    st.info(
        f"**Hypothesis 1:**\n"
        f"We suspect there are less than 5 features that significantly determine the sales value of a house."
    )
    st.success(
        f"**Validation:**\n"
        f"The correlation study in the 'Variable Importance Study' confirmed this hypothesis. "
        f"Key variables such as **Above Ground Living Area**, **Year Built**, and **Overall Condition** "
        f"show strong correlations with Sale Price."
    )
    st.write("---")

    # Hypothesis 2
    st.info(
        f"**Hypothesis 2:**\n"
        f"Analysis shows that buyers value houses with large living areas, good overall quality, and construction "
        f"within certain year ranges."
    )
    st.success(
        f"**Validation:**\n"
        f"The analysis supports this hypothesis. Houses with good **Overall Quality**, **Large Living Areas**, "
        f"and built in favorable year ranges were found to have significantly higher Sale Prices. "
        f"These insights align with buyer behavior and will guide further discussions and surveys by the team."
    )
    st.write("---")

    # Correlation Study Insights
    st.write("### Supporting Insights")
    st.markdown(
        f"""
        - The top 3 features influencing sales price are:
            - **Above Ground Living Area**: Strong correlation with sale price.
            - **Overall Quality**: Positively impacts buyer perception and price.
            - **Year Built**: Indicates the desirability of newer constructions.
        - These results are consistent with the correlation study findings in the 'Variable Importance Study' section.
        """
    )
