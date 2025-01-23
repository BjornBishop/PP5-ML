import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_churned_customer_study import page_variable_importance_study_body
from app_pages.page_prospect import page_prediction_body
from app_pages.page_project_hypothesis import page_predict_price_body
from app_pages.page_predict_churn import page_hypothesis_validation_body

app = MultiPage(app_name= "Price Predictor") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Variables Important Study", page_variable_importance_study_body)
app.add_page("Sale Price Prediction", page_prediction_body)
app.add_page("Predict Sale Price", page_predict_price_body)
app.add_page("Hypothesis and Validation", page_hypothesis_validation_body)


app.run() # Run the  app