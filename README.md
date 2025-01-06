# Predicting Price Evolutions of the Purchases of Raw Materials (Python 3.10.12)

## Description - WIP

Accurate and effective raw material price prediction is critical for manufacturing enterprises to optimize the timing of
raw material purchases. Company A is an international manufacturer of hygiene products, and has traditionally relied on
local experts from production sites to provide price forecasts, which can be time-consuming and susceptible to bias and
subjectivity. To address these challenges, this project aims to employ data-driven modelling techniques to provide
accurate and efficient predictions for the unit price of raw materials at Company A.

LASSO regression models that incorporates both univariate information (historical prices of raw materials) and exogenous
variables (price indices of identified price drivers) are constructed for different raw materials across various
forecasting horizons. These models are compared against baseline persistence forecast and LASSO regression models using
only univariate information to assess whether this modelling approach can improve the accuracy of raw material price
prediction and whether the inclusion of exogenous variables enhances prediction accuracy.

Additionally, a feature important analysis is conducted to investigate the most predictive features for raw material
prices.

The project contains several major sections.

- `data`
  - The `raw/` contains target variable data and electricity data.
- `results`
  - The `coefficient_export/` contains exported feature coefficients of final Lasso regression models to help identify whether a specific price driver has critical impact on target variable or not.
  - The `exploration/` contains the visualisation of historical rm codes prices.
  - The `generated_features/` contains the outputs of ETL pipelines for further modelling.
  - The `modelling/` contains trained models.
- `src/lambda`
  - The `testing` folder contains mock data for unit testing.
  - The `lambda_extract_and_clean` contains functions to extract data from API and S3 bucket.
  - The `lambda_transform` contains functions for necessary data transformation.
  - The `lambda_feature_engineer` contains functions to perform feature engineering for further modelling.
- `src`
  - The `forecastor` contains functions to train models.
  - The `naiveforecastor` contains functions to calculate naive forecasting as baseline models.
  - The `visualiser` contains functions to draw graphs.

## Project layout

```text
Raw_Material_Price_Prediction/
    ├── data/
        ├── raw/
        └── generated_feature/
    ├── results/
        ├── coefficient_export/
        ├── exploration/
        └── modelling/
    ├── src/
        ├── lambda/
            ├── testing/
            ├── lambda_extract_and_clean.py
            ├── lambda_feature_engineer.py
            └── lambda_transform.py
        ├── __init__.py
        ├── forecastor.py
        ├── format_handle.py
        ├── naiveforecastor.py
        ├── print_all_rm_codes.py
        └── visualiser.py
    ├── README.md
    └── requirements.txt
```

## Approaches - WIP

```text
## Python 3.10
### Naive Forecast - Baseline Model
[] Import and extract raw material prices data
[] Create all combinations of Year Month and Key RM Codes
[] Map the combinations with the imported raw material prices to ensure having all RM Codes for each months
[] Impute monthly data of all RM codes by forward fill approach
[] Filter data with years and monthes of actual procurement
[] To calculate monthly average prices per Key RM code and perform naive forecast

### Lasso Regression
[] Import monthly average prices of external price drivers
[] Create rows and encoding with Key RM Codes, ex: Alkalis_RM02_0001, Alkalis_RM02_0002
[] To calculate the monthly average prices of the target variables
[] Create 12*N features, price changes from {start} months before prices to {end} months before prices, features including historical prices of external price drivers and autoregressive prices of y
[] Combine features with target variables
[] train_test_split() - do calculation and scaling only based on train data set to prevent data leakage
[] Detect outliers - skip
[] Check data distribution
[] Data scaling - log transformation and standardlisation
[] check multicollinearity(to run one regression using each features, and find corr of all feature, filtering those with higher performance and least corr for our last model) - skip
[] Lasso regression - fit and transform train data set
[] Cross validation and Hyperparameter tuning using RandomizedSearchCV
[] Lasso regression - transform test data set
[] Lasso regression - transform new data and match the predicted values with real values
[] Visualisation - all Key RM codes
[] Visualisation - individual Key RM codes
[] Compare Lasso with Naive forecast

### Onto AWS Cloud
[] Split preprocessor into extract_and_clean, transform, and feature_engineer
[] Add def lambda_handler()
    [] Modify def input/output as json formats 
[] Import Numpy, Pandas, and fredapi onto AWS Lambda
[] Enable extract_and_clean to get raw data from S3 buckets
[] Have Lambda extract_and_clean to trigger Lambda transform, and Lambda transform to trigger Lambda feature_engineer
[] Add each materials as individual test events
```

| Group Description     | Price Driver                               | Size |
|-----------------------|--------------------------------------------|------|
| Acid                  | Gas/Electricity/Ammonia/Wheat              | 3255 |
| Alkalis               | Gas/Electricity                            | 6181 |
| Anionic Surfactants   | Gas/Electricity/Palm Oil/Ethylene oxide    | 1158 |
| Bleaching Agents      | Gas/Electricity                            | 4153 |
| Builder               | Beets/Corn/Ethylene/Phosphate              | 1771 |
| Fatty Acid            | Palm Oil/Transport                         | 689  |
| Non-ionic surfactants | Gas/Electricity/Crude Oil/Palm Oil/Ammonia | 3220 |
| Solvents              | Glycerol/Natural Oil                       | 710  |

## External factor sources

[Electricity](https://my.elexys.be/MarketInformation/IceEndexAverage.aspx)
[Fred](https://fred.stlouisfed.org/)

## Initialisation Tutorial for first-time user

## 1. Clone repository

1. Set up git accounts
    - $ git config --global user.name "John Doe"
    - $ git config --global user.email <johndoe@example.com>
2. Clone repository from github

## 2. Install dependencies

### 2.1. Virtual Environment

1. Install homebrew
2. Install pyenv to create virtual machine
    - $ brew install pyenv
    - $ brew install pyenv-virtualenv
3. To install Python version 3.10.12 and activate
    - $ pyenv install 3.10.12
    - $ pyenv virtualenv 3.10.12 <given_name>
    - $ pyenv local <given_name>
4. Check Python interpreter version
    1. Pycharm > Python Interpreter > Add new Interpreter > Add local interpreter > Virtualenv Environment > Location: ~
       /Raw_Material_Price_Prediction/.venv
    2. Check interpreter name as : "Python 3.10 given_name"

### 2.2. Install requirements

1. To install requirements from requirements.txt
    - pip install -r /path_to/requirements.txt