# Welcome to the ReadME file. 

## How to use this repo

1. Fork this repo and copy the https URL of your forked churnometer repo

1. Log into the cloud IDE with your GitHub account.

1. On your Dashboard, click on the New Workspace button

1. Paste in the URL you copied from GitHub earlier

1. Click Create

1. Wait for the workspace to open. This can take a few minutes.

1. Open a new terminal and `pip3 install -r requirements.txt`

1. Click the kernel button and choose Python Environments.

1. Choose the kernel Python 3.12.2 as it inherits from the workspace, so it will be Python-3.12.2 as installed by our template. To confirm this, you can use `! python --version` in a notebook code cell.

Your workspace is now ready to use. When you want to return to this project, you can find it in your Cloud IDE Dashboard. You should only create 1 workspace per project.

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We created then a fictitious user story where predictive analytics can be applied in a real project in the workplace.
Each row represents a customer, each column contains a customer attribute. The data set includes information about:
- Housing Sales data, such as basement size, above ground living space, 2nd floor living space, lot size, garage size, 1st floor SF, basement finished area and more.
- More catagorical data as well such as overall quality, overall condition, basement exposure, Basement finish type and kitchen quality

----1stFlrSF----2ndFlrSF----BedroomAbvGr----BsmtExposure----BsmtFinSF1----BsmtFinType1----BsmtUnfSF----EnclosedPorch

0----856---------854.0-----------3.0----------------No----------------706------------GLQ----------------150--------------0.0

1----1262---------0.0-------------3.0----------------Gd----------------978------------ALQ----------------284--------------NaN	

2----920---------866.0-------------3.0----------------Mn----------------486------------GLQ----------------434--------------0.0


## Project Terms & Jargon
	- A client is the one contracting for the job.
    - The key to the initial descriptions: 
      - 1stFlrSF: First Floor square feet
        - 334 - 4692
    - 2ndFlrSF: Second floor square feet
      - 0 - 2065
    - BedroomAbvGr: Bedrooms above grade (does NOT include basement bedrooms)
      - 0 - 8
    - BsmtExposure: Refers to walkout or garden level walls
      - Gd: Good Exposure;
      - Av: Average Exposure;
      - Mn: Mimimum Exposure;
      - No: No Exposure;
      - None: No Basement
    - BsmtFinType1: Rating of basement finished area
      - GLQ: Good Living Quarters;
      - ALQ: Average Living Quarters;
      - BLQ: Below Average Living Quarters;
      - Rec: Average Rec Room;
      - LwQ: Low Quality;
      - Unf: Unfinshed;
      - None: No Basement
    - BsmtFinSF1: Type 1 finished square feet
      - 0 - 5644
    - BsmtUnfSF: Unfinished square feet of basement area
      - 0 - 2336
    - TotalBsmtSF: Total square feet of basement area
      - 0 - 6110
    - GarageArea: Size of garage in square feet
      -0 - 1418
    - GarageFinish: Interior finish of the garage
      - Fin: Finished;
      - RFn: Rough Finished;
      - Unf: Unfinished;
      - None: No Garage
    - GarageYrBlt: Year garage was built
      - 1900 - 2010
    - GrLivArea: Above grade (ground) living area square feet
      - 334 - 5642
    - KitchenQual: Kitchen quality
      - Ex: Excellent;
      - Gd: Good;
      - TA: Typical/Average;
      - Fa: Fair;
      - Po: Poor
    - LotArea: Lot size in square feet
      - 1300 - 215245
    - LotFrontage: Linear feet of street connected to property
      - 21 - 313
    - MasVnrArea: Masonry veneer area in square feet
      - 0 - 1600
    - EnclosedPorch: Enclosed porch area in square feet
      - 0 - 286
    - OpenPorchSF: Open porch area in square feet
      - 0 - 547
    - OverallCond: Rates the overall condition of the house
      - 10: Very Excellent;
      - 9: Excellent;
      - 8: Very Good;
      - 7: Good;
      - 6: Above Average;
      - 5: Average;
      - 4: Below Average;
      - 3: Fair;
      - 2: Poor;
      - 1: Very Poor
    - OverallQual: Rates the overall material and finish of the house
      - 10: Very Excellent;
      - 9: Excellent;
      - 8: Very Good;
      - 7: Good;
      - 6: Above Average;
      - 5: Average;
      - 4: Below Average;
      - 3: Fair;
      - 2: Poor;
      - 1: Very Poor
    - WoodDeckSF: Wood deck area in square feet
      - 0 - 736
    - YearBuilt: Original construction date
      - 1872 - 2010
    - YearRemodAdd: Remodel date (same as construction date if no remodeling or additions)
      - 1950 - 2010
    - SalePrice: Sale Price
      - 34900 - 755000|

## Hypothesis and how to validate?
- 1 - We suspect customers are churning with low tenure levels.
	- A Correlation study can help in this investigation
- 2 - A customer survey showed our customers appreciate Fibre Optic.
	- A Correlation study can help in this investigation


## The rationale to map the business requirements to the Data Visualizations and ML tasks
- **Business Requirement 1:** Data Visualization and Correlation study
	- We will inspect the data related to the customer base.
	- We will conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to Churn.
	- We will plot the main variables against SalePrice to visualize insights.

- **Business Requirement 2:** Regression Analysis
	- We want to predict with reasonable accuracy the approximate sales price for the customer that has inherited properties in a state where she is unaware of the prices.
	- We want to predict the sale price for a house based off a variety of features of other houses that have been sold in the same state.

  ## ML Business Case

### Predict Property Sale Price
#### Regression Model
- We want an ML model to predict sale price of a house based on historical data from the same state, which include the sales price as these are houses that have already been sold. The target variable is numerical and contains a wide range of prices and other features. We consider a **regression model**. which is supervised and uni-dimensional, regression model output: approximate sale price based on user input data. 
- Our ideal outcome is to provide our customers with the ability to input their housing data and reasonably predict the sale value of their property.
- The model success metrics are
	- at least 80% Recall for Churn, on train and test set 
	- The ML model is considered a failure if:
		- it has less than 0.75 R2 score leading to untrustworthy sale price valuations.
		- Precision for sale price is too conservative leading to lower than average sale price. We dont want to lose our cut of the big sales right?
- The model output is defined as an approximation of the sales price of a given property based on 4 key features. If a property has been sold, the input features are part of the sales evualuation process via a form. Our salespeople can use this data to gather the key features and reasonably predict the sales value to prospective clients on the fly. (not in batches)
- Heuristics: Currently, there is no approach to predict sales price for potential clients.
- The training data to fit the model comes from the housing-prices-data. This dataset contains about 1500 customer records.
	- Train data 
    - target: Sale Price; 
    - features dropped: 'EnclosedPorch', 'WoodDeckSF', 'LotFrontage', 'GarageFinish', 'BsmtFinType1', 'BedroomAbvGr', 'GarageYrBlt', 'MasVnrArea', 'BsmtExposure', 'YearRemodAdd', 'OpenPorchSF', 'BsmtUnfSF', 'KitchenQual'
    - features kept: '1stFlrSF'	'2ndFlrSF'	'BsmtFinSF1'	'GarageArea'	'GrLivArea'	'LotArea'	'OverallCond'	'OverallQual'	'TotalBsmtSF'	'YearBuilt'

## The wrong turns: 

Throughout building pipelines, I ended up making a wide array of models in an attempt to get the best scoring model possible. I tried calassification models. I tried RGBXetc models, I tried decisionTreeRegressor models. I even threw in some Lasso from time to time in an attempt to change the outcome. 

## Back to the regression model

The regression model with some standard scaling on the data turned out to be the best way forward once certain features had been removed. Its currently 3am. I have pulled 4 all nighters 10am until 8am kind of all nighters. So forgive me for not wanting to fetch those results right now. 

# House Price Prediction Project
## Dashboard Design (Streamlit App User Interface)
### Page 1: Quick Project Summary

- Project Terms & Jargon
- Describe Project Dataset
- State Business Requirements
- Business Case Assessment: Summarize the business requirements, the expected project outcomes, and the criteria for success.

### Page 2: House Sale Price Analysis
Objective: To answer business requirement 1.

Contents:

State business requirement 1
- Checkbox: Data inspection on house dataset (number of rows and columns, first ten rows of the data)
- Correlation Study: Display the most correlated variables to sale price and summarize the insights.
- Checkbox: Individual plots showing the sale price levels for each correlated variable
- Checkbox: Parallel plot using Sale Price and correlated variables

### Page 3: House Price Predictor
Objective: To address business requirement 2.

Contents:

Widget inputs related to house features for predicting the sale price.
- "Run predictive analysis" button to serve the house data to our ML pipelines and predict the sale price.
- Compare the predicted price to other houses built in the same year using visualizations like scatter plots and histograms.
- Interactive Prediction: Display the summed predicted price for all 4 inherited houses and allow users to input real-time house data to predict sale prices.

### Page 4: Project Hypothesis and Validation
Objective: To describe and validate project hypotheses.

Contents:

- Hypothesis 1: Older houses have lower sale prices. (Validated)

- Hypothesis 2: Houses with larger living areas have higher sale prices. (Validated)

- Describe each hypothesis, the conclusions, and how validation was performed.

### Page 5: Predict House Sale Price
Contents:

Considerations and conclusions after the pipeline is trained

- Present ML pipeline steps

- Feature importance

- Pipeline performance

### Page 6: Predict Tenure
Contents:

- Considerations and conclusions after the pipeline is trained

- Present ML pipeline steps

- Feature importance

- Pipeline performance

#### Project Considerations
- Data Suggestion: The data suggests using a regressor where the target is the sale price.
- Model Inputs and Outputs: Inputs are house attribute information; output is the predicted sale price.
- Performance Goal: R2 score of at least 0.75 on both the train set and the test set.
- Client Benefits: Maximize the sales price for the inherited properties.
- Ethical or Privacy Concerns: None, as a public dataset is used.

### Business Requirements
- Business Requirement 1: Perform a correlation and/or PPS study to investigate the most relevant variables correlated to the sale price, and visualize these variables against the sale price to summarize the insights.

- Business Requirement 2: Deliver an ML system capable of reliably predicting the summed sales price of the 4 inherited houses and any other house in Ames, Iowa.

- Use either conventional ML or Neural Networks to map the relationships between features and the target.

- Conduct extensive hyperparameter optimization for the chosen algorithm.

- Display model performance and pipeline steps in the dashboard.

### Dashboard Expectations
Your dashboard should contain:

- Project Summary Page: Show the project dataset summary and the client's requirements.

- Analysis Page: Findings related to which features have the strongest correlation to the house sale price.

- Prediction Page: Display the 4 houses' attributes and their respective predicted sale prices, along with the summed predicted price for all 4 inherited houses. Add interactive input widgets for real-time house data to predict sale prices.

- Hypothesis Page: Indicate project hypotheses and their validation across the project.

- Technical Page: Display model performance and pipeline steps.


