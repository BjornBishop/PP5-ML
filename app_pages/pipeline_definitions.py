from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer  # Ensure ColumnTransformer is imported

num_features = ['OverallCond', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', 'OverallQual', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'LotArea']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features)
    ]
)

def final_pipeline(lasso_params, model):
    the_pipeline_base = Pipeline([
        ('preprocessor', preprocessor),
        ('feature_selection', SelectFromModel(Lasso(**lasso_params, random_state=0))),
        ('model', model)
    ])
    return the_pipeline_base
