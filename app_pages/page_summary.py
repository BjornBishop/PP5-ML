import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # Define terms and dataset information
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **customer** is a person who consumes your service or product.\n"
        f"* A **Sale Price** is a potential house sale value.\n\n"

        f"**Project Dataset**\n"
        f"* The dataset represents **housing sale price data from a specific state**, "
        f"containing house features "
        f"(like kitchen quality, basement quality, overall condition and overall quality, etc.), "
        f"measurements (like lot size, basement size, garage size, above ground living space), "
        f"and unique qualities (like the year the house was built and the year it was last renovated)."
    )

    # Link to README file
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/BjornBishop/PP5-ML/blob/main/README.md)."
    )

    # Business requirements
    st.success(
        f"The project has two business requirements:\n"
        f"* **Business Requirement 1**: Data Visualization and Correlation Study\n"  
        f"   - Conduct a correlation study (using Pearson and Spearman methods) to understand how features interact with sale price.\n"
        f"   - Provide visualizations of the most relevant features affecting house sale prices to gain actionable insights.\n\n"

        f"* **Business Requirement 2**: Regression Analysis\n"
        f"   - Develop a model to predict, with reasonable accuracy, the approximate sales price of houses based on historical data.\n"
        f"   - The model should empower the client to price inherited properties effectively."
    )

    # Objective conclusion (addressing criterion 4.2)
    st.info(
        f"### Conclusion:\n"
        f"The regression model developed for this project successfully addresses Business Requirement 2. "
        f"It demonstrates strong predictive performance in estimating property sale prices based on housing features. \n\n"

        f"#### Model Evaluation:\n"
        f"* **Train Set Performance:**\n"
        f"  - **R2 Score**: 0.861\n"
        f"  - **Mean Absolute Error (MAE)**: $20,676\n"
        f"  - **Root Mean Squared Error (RMSE)**: $29,300\n\n"

        f"* **Test Set Performance:**\n"
        f"  - **R2 Score**: 0.797\n"
        f"  - **Mean Absolute Error (MAE)**: $23,790\n"
        f"  - **Root Mean Squared Error (RMSE)**: $37,484\n\n"

        f"These metrics demonstrate the model's ability to predict sales prices with reasonable accuracy. "
        f"The R2 score of 0.797 on the test set indicates that the model captures about 80% of the variance in the sale price data. "
        f"This makes it a valuable tool for assisting clients in pricing inherited properties effectively."
)

