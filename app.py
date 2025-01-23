import sys
import os
sys.path.append(os.path.abspath('.'))

from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body
from app_pages.page_variable_importance_study import page_variable_importance_study_body
from app_pages.page_prediction import page_prediction_body
from app_pages.page_predict_price import page_predict_price_body
from app_pages.page_hypothesis_validation import page_hypothesis_validation_body
from app_pages.pipeline_definitions import final_pipeline  # Adjusted import path

app = MultiPage(app_name="Price Predictor")

app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Variables Important Study", page_variable_importance_study_body)
app.add_page("Sale Price Prediction", page_prediction_body)
app.add_page("Predict Sale Price", page_predict_price_body)
app.add_page("Hypothesis and Validation", page_hypothesis_validation_body)

app.run()
