import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler


def persistence_Naive_MAPE(raw_df: pd.DataFrame, code: str, lag: int, test_periods):
    """
    Calculate MAPE based on persistent Naive approach.
    
    Calculation approach:
    - lag = 1: 'Average_price' VS 'AR_1'
    - lag = 3: 'Average_price' VS 'AR_3'
    - lag = 6: 'Average_price' VS 'AR_6'
    """
    assert not len(raw_df) == 0, "df is empty!"
    assert not len(code) == 0, "RM_codes are missed."
    assert isinstance(lag, int), "Time lag is missed."
    assert lag > 0, "Time lag should be a positive int."
    assert len(test_periods) == 2, "There should only be one test_start and one test_end."

    df = raw_df[raw_df[code] == True]
    test_start, test_end = test_periods

    test_df = df[df.Time.between(test_start, test_end, inclusive="left")]

    if lag == 1:
        mape = mean_absolute_percentage_error(test_df['Average_price'], test_df['AR_1'])
        assert mape > 0, "mape is negative!"
        return round(mape * 100, 3)
    elif lag > 1:
        mape = mean_absolute_percentage_error(test_df['Average_price'], test_df[f'AR_{lag}'])
        assert mape > 0, "mape is negative!"
        return round(mape * 100, 3)


def train_model_AR(raw_df: pd.DataFrame, code: str, lag: int, test_periods, alpha_bottom=0.01, return_coef=False):
    """
    Train Lasso models based on individual RM code with only autoregression features and given test periods
    
    ## X_train headers
    - lag = 1: ['AR_1', 'AR_2', 'AR_3', 'AR_4', 'AR_5', 'AR_6', 'AR_7', 'AR_8', 'AR_9', 'AR_10', 'AR_11', 'AR_12']
    - lag = 3: ['AR_3', 'AR_4', 'AR_5', 'AR_6', 'AR_7', 'AR_8', 'AR_9', 'AR_10', 'AR_11', 'AR_12']
    - lag = 6: ['AR_6', 'AR_7', 'AR_8', 'AR_9', 'AR_10', 'AR_11', 'AR_12']
    
    ## Model parameters:
    - param_grid = {'alpha': np.linspace(alpha_bottom, 1, 3000)}
    - random_search = RandomizedSearchCV(estimator=lasso,
                                         param_distributions=param_grid,
                                         n_iter=300,
                                         cv=5,
                                         random_state=42)
                                         
    ## If you want to print out coef of all models, return_coef == True:                                         
    coef_list = list()
    for code in RM_codes:

        for lag in lags:
            test_df = pd.DataFrame()
            count = 0

            for period in test_periods:
                count += 1
                mape, coef_sum = train_model_AR(feature_df,code,lag,period,alpha_bottom, True)
                temp = pd.DataFrame.from_dict(coef_sum, 
                                              orient='index',
                                              columns=[f"{code}_lag-{lag}_test-{count}"])
                coef_list.append(temp)


    merged_df = pd.concat(coef_list, axis=1)
    result = pd.DataFrame.transpose(merged_df)
    print(result)
   
    """
    assert not len(raw_df) == 0, "df is empty!"
    assert not len(code) == 0, "RM_codes are missed."
    assert isinstance(lag, int), "Time lag is missed."
    assert lag > 0, "Time lag should be a positive int."
    assert len(test_periods) == 2, "There should only be one test_start and one test_end."

    df = raw_df[raw_df[code] == True]

    test_start, test_end = test_periods

    # Split data into train and test sets based on given test periods
    train_df = df[df.Time < test_start]
    test_df = df[df.Time.between(test_start, test_end, inclusive="left")]

    X_train = train_df.filter(regex='^AR_')
    X_test = test_df.filter(regex='^AR_')

    # Handle time lag parameter
    if lag > 1:
        conditions = tuple((f"_{i}" for i in range(1, lag)))
        assert_con = tuple((f"_{i}$" for i in range(1, lag)))
        X_train = X_train.loc[:, ~X_train.columns.str.endswith(conditions)]
        X_test = X_test.loc[:, ~X_test.columns.str.endswith(conditions)]
        assert not X_train.filter(regex='|'.join(assert_con)).any(axis=1).any(), "X_train not filtered correctly"
        assert not X_test.filter(regex='|'.join(assert_con)).any(axis=1).any(), "X_test not filtered correctly"

    y_train = train_df['Average_price'].values
    y_test = test_df['Average_price'].values

    # Standardlisation
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    # Define the parameter grid
    param_grid = {'alpha': np.linspace(alpha_bottom, 1, 3000)}
    # Create a Lasso regression model
    lasso = Lasso()
    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=lasso,
                                       param_distributions=param_grid,
                                       n_iter=300,
                                       cv=5,
                                       random_state=42)
    # Fit the data to perform a grid search
    random_search.fit(X_train_scaled, y_train_scaled)
    assert random_search.n_features_in_ == len(X_train.columns)

    # Get the best Lasso model from RandomizedSearchCV
    best_lasso_model = random_search.best_estimator_
    # Predict on the test data
    y_pred_test = best_lasso_model.predict(X_test_scaled)
    y_pred_test_inverse = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
    mape = mean_absolute_percentage_error(y_test, y_pred_test_inverse)
    assert mape > 0, "mape is negative!"

    ## To return coef of each models if necessary
    if return_coef == True:
        coef_sum = dict()
        assert random_search.n_features_in_ == len(X_test.columns), "n_features !=len(X_test.columns)"
        for feature, coefficient in zip(X_test.columns, random_search.best_estimator_.coef_):
            coef_sum[feature] = round(coefficient, 5)
        return mape * 100, coef_sum
    else:
        return mape * 100


def train_model_all_features(raw_df: pd.DataFrame, code: str, lag: int, test_periods, alpha_bottom=0.01,
                             return_coef=False):
    """
    Train Lasso models based on individual RM code with autoregression features and external price drivers and given test periods
    
    ## X_train headers
    - lag = 1: ['Electricity_1', 'Electricity_2', ... 'Electricity_12', 'AR_1', 'AR_2' ... 'AR_12']
    - lag = 3: ['Electricity_3', 'Electricity_4', ... 'Electricity_12', 'AR_3', 'AR_4' ... 'AR_12']
    - lag = 6: ['Electricity_6', 'Electricity_7', ... 'Electricity_12', 'AR_6', 'AR_7' ... 'AR_12']
    
    ## Model parameters:
    - param_grid = {'alpha': np.linspace(alpha_bottom, 1, 3000)}
    - random_search = RandomizedSearchCV(estimator=lasso,
                                         param_distributions=param_grid,
                                         n_iter=300,
                                         cv=5,
                                         random_state=42)
                                         
    ## If you want to print out coef of all models, return_coef == True:                                         
    coef_list = list()
    for code in RM_codes:

        for lag in lags:
            test_df = pd.DataFrame()
            count = 0

            for period in test_periods:
                count += 1
                mape, coef_sum = train_model_all_features(feature_df,code,lag,period,alpha_bottom, True)
                temp = pd.DataFrame.from_dict(coef_sum, 
                                              orient='index',
                                              columns=[f"{code}_lag-{lag}_test-{count}"])
                coef_list.append(temp)


    merged_df = pd.concat(coef_list, axis=1)
    result = pd.DataFrame.transpose(merged_df)
    print(result)
   
    """
    assert not len(raw_df) == 0, "df is empty!"
    assert not len(code) == 0, "RM_codes are missed."
    assert isinstance(lag, int), "Time lag is missed."
    assert lag > 0, "Time lag should be a positive int."
    assert len(test_periods) == 2, "There should only be one test_start and one test_end."

    df = raw_df[raw_df[code] == True]
    df = df.loc[:, ~df.columns.str.startswith("RM")]

    test_start, test_end = test_periods

    # Split data into train and test sets based on given test periods
    train_df = df[df.Time < test_start]
    test_df = df[df.Time.between(test_start, test_end, inclusive="left")]

    X_train = train_df.drop(['Time', 'Group Description', 'Year', 'Month', 'Average_price'], axis=1)
    X_test = test_df.drop(['Time', 'Group Description', 'Year', 'Month', 'Average_price'], axis=1)

    # Handle time lag parameter
    if lag > 1:
        conditions = tuple((f"_{i}" for i in range(1, lag)))
        assert_con = tuple((f"_{i}$" for i in range(1, lag)))
        X_train = X_train.loc[:, ~X_train.columns.str.endswith(conditions)]
        X_test = X_test.loc[:, ~X_test.columns.str.endswith(conditions)]
        assert not X_train.filter(regex='|'.join(assert_con)).any(axis=1).any(), "X_train not filtered correctly"
        assert not X_test.filter(regex='|'.join(assert_con)).any(axis=1).any(), "X_test not filtered correctly"

    y_train = train_df['Average_price'].values
    y_test = test_df['Average_price'].values

    # Standardlisation
    scaler_x = StandardScaler()
    X_train_scaled = scaler_x.fit_transform(X_train)
    X_test_scaled = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

    # Define the parameter grid
    param_grid = {'alpha': np.linspace(alpha_bottom, 1, 3000)}
    # Create a Lasso regression model
    lasso = Lasso()
    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=lasso,
                                       param_distributions=param_grid,
                                       n_iter=300,
                                       cv=5,
                                       random_state=42)
    # Fit the data to perform a grid search
    random_search.fit(X_train_scaled, y_train_scaled)
    assert random_search.n_features_in_ == len(X_train.columns)

    # Get the best Lasso model from RandomizedSearchCV
    best_lasso_model = random_search.best_estimator_
    # Predict on the test data
    y_pred_test = best_lasso_model.predict(X_test_scaled)
    y_pred_test_inverse = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
    mape = mean_absolute_percentage_error(y_test, y_pred_test_inverse)
    assert mape > 0, "mape is negative!"

    ## To return coef of each models if necessary
    if return_coef == True:
        coef_sum = dict()
        assert random_search.n_features_in_ == len(X_test.columns), "n_features !=len(X_test.columns)"
        for feature, coefficient in zip(X_test.columns, random_search.best_estimator_.coef_):
            coef_sum[feature] = round(coefficient, 5)
        return mape * 100, coef_sum
    else:
        return mape * 100
