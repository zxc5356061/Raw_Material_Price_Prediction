#%% md
# # Acid_random_split
# 
# ```
# RM01/0006     503
# RM01/0007     573
# RM01/0004     672
# RM01/0001    1507
# 
# Earliest data of Amonia: 2014-12-01
# ```
# 
# ## Table of contents
# 0. Data preparations
# * Import data
# * Impute data
# * Slice data for naive forecast
# * Feature engineering
# * Split data into train and test sets
# * Create X, y
# * Log transformation
# * Standardlisation
# * Drop external price drivers to create X_train_ar and X_test_ar
# 1. 1-month predictions
# * Naive forecast with test_df and visualisation
# * Lasso with only autoregressions, visualise testing set
# * Lasso with autoregressions and external price drivers, visualise testing set
# 
# 2. 3-month predictions
# * Slice data for 3-month lag
# * Naive forecast with test_df and visualisation
# * Lasso with only autoregressions, visualise testing set
# * Lasso with autoregressions and external price drivers, visualise testing set
# 
# 3. 6-month predictions
# * Slice data for 6-month lag
# * Naive forecast with test_df and visualisation
# * Lasso with only autoregressions, visualise testing set
# * Lasso with autoregressions and external price drivers, visualise testing set

#%% md
# # 0. Data preparation
#%%
import preprocessor as pre
import naiveforecastor as nf
import visualiser as visual
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np
from matplotlib import pyplot as plt

# Import data
gas_df = pre.get_Fred_data('PNGASEUUSDM',2014,2024)
wheat_df = pre.get_Fred_data('PWHEAMTUSDM',2014,2024)
ammonia_df = pre.get_Fred_data('WPU0652013A',2014,2024)
elec_df = pre.clean_elec_csv('Data_flat_files/ELECTRICITY_03_2024.csv',2014,2024)

df = pre.clean_pred_price_evo_csv("Data_flat_files/Dataset_Future_Predicting_Price_Evolutions_202403.csv",2014,2023)

target = 'acid'.lower()

RM_codes = ['RM01/0001','RM01/0004','RM01/0006','RM01/0007']

external_drivers = {
    "PNGASEUUSDM": gas_df,
    "PWHEAMTUSDM": wheat_df,
    "WPU0652013A": ammonia_df,
    "Electricity": elec_df
}

slicing_columns = ["PNGASEUUSDM","PWHEAMTUSDM","WPU0652013A","Electricity","AR"]


#%%
# Impute raw data of target variables 
imputed_df, missing = pre.impute_pred_price_evo_csv(df)

# Slice data for naive forecast
naive_df = imputed_df[imputed_df.Year == 2023]

# Feature engineering
dummy_df = pre.get_dummies_and_average_price(imputed_df,target,*RM_codes)
feature_df = pre.generate_features(1,12,dummy_df,*RM_codes, **external_drivers)

# feature_df.to_csv("feature_df.csv")
# Create X, y
feature_list = feature_df.drop(['Time', 'Group Description', 'Year','Month','Average_price'],axis=1)
X = feature_list
y = feature_df['Average_price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 30% as test set

# Log transformation and standardlisation
# y_train_log = np.log(y_train)
# y_test_log = np.log(y_test)

scaler_x = StandardScaler()
X_train_scaled = scaler_x.fit_transform(X_train)
X_test_scaled = scaler_x.transform(X_test)

scaler_y = StandardScaler()
# y_train_scaled = scaler_y.fit_transform(y_train_log.reshape(-1,1))
# y_test_scaled = scaler_y.transform(y_test_log.reshape(-1,1))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1))
y_test_scaled = scaler_y.transform(y_test.reshape(-1,1))

#%%
# Drop external price drivers to create X_train_ar and X_test_ar
X_train_ar = X_train.copy()
X_test_ar = X_test.copy()

try:
    for driver in external_drivers.keys():
        X_train_ar = X_train_ar.drop([f"{driver}_{i}" for i in range(1,13)], axis=1, errors="ignore")
        X_test_ar = X_test_ar.drop([f"{driver}_{i}" for i in range(1,13)], axis=1, errors="ignore")
    
    assert not any(col.startswith(tuple(key for key in external_drivers.keys())) for col in X_train_ar.columns), "df not sliced correctly"
    assert not any(col.startswith(tuple(key for key in external_drivers.keys())) for col in X_test_ar.columns), "df not sliced correctly"
    print(X_train_ar.columns)
    print(X_test_ar.columns)
    
except AssertionError:
    print("Unable to slice DataFrame")
    
scaler_ar = StandardScaler()
X_train_ar_scaled = scaler_ar.fit_transform(X_train_ar)
X_test_ar_scaled = scaler_ar.transform(X_test_ar)
#%% md
# # 1-month predictions
#%% md
# ## Naive forecast with test_df and visualisation
#%%
nf.naive_forest(naive_df,target,1,missing)
#%% md
# ## Lasso with only autoregressions, visualise testing set
#%%
## Lasso regression - fit and transform train data set
## Cross validation and Hyperparameter tuning using RandomizedSearchCV

# Define the parameter grid
param_grid = {'alpha': np.linspace(0.0000001, 1, 3000)}

# Create a Lasso regression model
lasso = Lasso()

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=lasso, 
                                   param_distributions=param_grid, 
                                   n_iter=300, 
                                   cv=5, 
                                   random_state=42)

# Fit the data to perform a grid search
random_search.fit(X_train_ar_scaled, y_train_scaled)

# Best alpha parameter
print("Best alpha parameter:", random_search.best_params_['alpha'])

# Best R-squared score
print("Best R-squared score:", round(random_search.best_score_, 3))

# Coefficients of the best Lasso model
print("Coefficients of the selected features in the best Lasso model:")
for feature, coefficient in zip(X_train_ar.columns, random_search.best_estimator_.coef_):
    print(f"{feature}: {round(coefficient,3)}")
    
# Get the best Lasso model from RandomizedSearchCV
best_lasso_model = random_search.best_estimator_
    
# Calculate MAPE and MSE of training set
y_pred_train = best_lasso_model.predict(X_train_ar_scaled)
# y_pred_train_inverse = np.exp(scaler_y.inverse_transform(y_pred_train.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y
y_pred_train_inverse = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))

print("MAPE, 1-month, AR only, train: ", round(mean_absolute_percentage_error(y_train,y_pred_train_inverse), 3))
#%%
## Lasso regression - transform test data set
# Predict on the test data
y_pred_test = best_lasso_model.predict(X_test_ar_scaled)
# y_pred_test_inverse = np.exp(scaler_y.inverse_transform(y_pred_test.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y
y_pred_test_inverse = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))
# Evaluate the model performance on the test data
test_score = best_lasso_model.score(X_test_ar_scaled, y_test_scaled)
print("Best Model:", best_lasso_model)
print("Test Set R-squared score:", round(test_score, 3))

# Calculate MAPE
print("MAPE, 1-month, AR only, test: ", round(mean_absolute_percentage_error(y_test,y_pred_test_inverse), 3))
#%% md
# ## Lasso with autoregressions and external price drivers, visualise testing set
#%%
## Lasso regression - fit and transform train data set
## Cross validation and Hyperparameter tuning using RandomizedSearchCV

# Define the parameter grid
param_grid = {'alpha': np.linspace(0.0000001, 1, 3000)}

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



# Best alpha parameter
print("Best alpha parameter:", random_search.best_params_['alpha'])

# Best R-squared score
print("Best R-squared score:", round(random_search.best_score_, 3))

# Coefficients of the best Lasso model
assert random_search.n_features_in_ == len(X_train.columns)
print("Coefficients of the selected features in the best Lasso model:")
for feature, coefficient in zip(X_train.columns, random_search.best_estimator_.coef_):
    print(f"{feature}: {round(coefficient,3)}")
    
# Get the best Lasso model from RandomizedSearchCV
best_lasso_model = random_search.best_estimator_
    
# Calculate MAPE and MSE of training set
y_pred_train = best_lasso_model.predict(X_train_scaled)
# y_pred_train_inverse = np.exp(scaler_y.inverse_transform(y_pred_train.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y
y_pred_train_inverse = scaler_y.inverse_transform(y_pred_train.reshape(-1,1))
print("MAPE, 1-month, all features, train: ", round(mean_absolute_percentage_error(y_train,y_pred_train_inverse), 3))
#%%
## Lasso regression - transform test data set
# Predict on the test data
y_pred_test = best_lasso_model.predict(X_test_scaled)
# y_pred_test_inverse = np.exp(scaler_y.inverse_transform(y_pred_test.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y
y_pred_test_inverse = scaler_y.inverse_transform(y_pred_test.reshape(-1,1))

# Evaluate the model performance on the test data
test_score = best_lasso_model.score(X_test_scaled, y_test_scaled)
print("Best Model:", best_lasso_model)
print("Test Set R-squared score:", round(test_score, 3))

# Calculate MAPE
print("MAPE, 1-month, all features, test: ", round(mean_absolute_percentage_error(y_test,y_pred_test_inverse), 3))
#%% md
# # 3-month predictions
#%%
nf.naive_forest(naive_df,target,3,missing)
#%% md
# ## Lasso with only autoregressions visualise testing set
#%%
## Lasso regression - fit and transform train data set
## Cross validation and Hyperparameter tuning using RandomizedSearchCV

# Slice data for 3-month lag
try:
    X_train_ar=X_train_ar.drop([f"AR_{i}" for i in range(1,3)], axis=1, errors='ignore')
    assert not any(col.endswith(("_1", "_2" )) for col in X_train_ar.columns) , "df not sliced correctly"
    X_test_ar=X_test_ar.drop([f"AR_{i}" for i in range(1,3)], axis=1, errors='ignore')
    assert not any(col.endswith(("_1", "_2" )) for col in X_test_ar.columns) , "df not sliced correctly"
    print(X_train_ar.columns)
    print(X_test_ar.columns)
except AssertionError:
    print("Unable to slice DataFrame")


scaler_ar_3 = StandardScaler()
X_train_ar_scaled = scaler_ar_3.fit_transform(X_train_ar)
X_test_ar_scaled = scaler_ar_3.transform(X_test_ar)

# Define the parameter grid
param_grid = {'alpha': np.linspace(0.0000001, 1, 3000)}

# Create a Lasso regression model
lasso = Lasso()

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=lasso, 
                                   param_distributions=param_grid, 
                                   n_iter=300, 
                                   cv=5, 
                                   random_state=42)

# Fit the data to perform a grid search
random_search.fit(X_train_ar_scaled, y_train_scaled)

# Best alpha parameter
print("Best alpha parameter:", random_search.best_params_['alpha'])

# Best R-squared score
print("Best R-squared score:", round(random_search.best_score_, 3))

# Coefficients of the best Lasso model
print("Coefficients of the selected features in the best Lasso model:")
for feature, coefficient in zip(X_train_ar.columns, random_search.best_estimator_.coef_):
    print(f"{feature}: {round(coefficient,3)}")
    
# Get the best Lasso model from RandomizedSearchCV
best_lasso_model = random_search.best_estimator_
    
# Calculate MAPE and MSE of training set
y_pred_train = best_lasso_model.predict(X_train_ar_scaled)
y_pred_train_inverse = np.exp(scaler_y.inverse_transform(y_pred_train.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y

print("MAPE, 3-month, AR only, train: ", round(mean_absolute_percentage_error(y_train,y_pred_train_inverse), 3))
#%%
## Lasso regression - transform test data set
# Predict on the test data
y_pred_test = best_lasso_model.predict(X_test_ar_scaled)
y_pred_test_inverse = np.exp(scaler_y.inverse_transform(y_pred_test.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y

# Evaluate the model performance on the test data
test_score = best_lasso_model.score(X_test_ar_scaled, y_test_scaled)
print("Best Model:", best_lasso_model)
print("Test Set R-squared score:", round(test_score, 3))

# Calculate MAPE
print("MAPE, 3-month, AR only, test: ", round(mean_absolute_percentage_error(y_test,y_pred_test_inverse), 3))
#%% md
# ## Lasso with autoregressions and external price drivers, visualise testing set
#%%
## train_test_split()
## Check data distribution
## Data scaling - log transformation and standardlisation

# Slice data for 3-month lag
try:
    for feature in slicing_columns:
        X_train=X_train.drop([f"{feature}_{i}" for i in range(1,3)], axis=1, errors='ignore')
        X_test=X_test.drop([f"{feature}_{i}" for i in range(1,3)], axis=1, errors='ignore')
    assert not any(col.endswith(("_1", "_2" )) for col in X_train.columns) , "df not sliced correctly"
    assert not any(col.endswith(("_1", "_2" )) for col in X_test.columns) , "df not sliced correctly"
    print(X_train.columns)
    print(X_test.columns)
except AssertionError:
    print("Unable to slice DataFrame")

scaler_3 = StandardScaler()
X_train_scaled = scaler_3.fit_transform(X_train)
X_test_scaled = scaler_3.transform(X_test)

## Lasso regression - fit and transform train data set
## Cross validation and Hyperparameter tuning using RandomizedSearchCV

# Define the parameter grid
param_grid = {'alpha': np.linspace(0.0000001, 1, 3000)}

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



# Best alpha parameter
print("Best alpha parameter:", random_search.best_params_['alpha'])

# Best R-squared score
print("Best R-squared score:", round(random_search.best_score_, 3))

# Coefficients of the best Lasso model
assert random_search.n_features_in_ == len(X_train.columns)
print("Coefficients of the selected features in the best Lasso model:")
for feature, coefficient in zip(X_train.columns, random_search.best_estimator_.coef_):
    print(f"{feature}: {round(coefficient,3)}")
    
# Get the best Lasso model from RandomizedSearchCV
best_lasso_model = random_search.best_estimator_
    
# Calculate MAPE and MSE of training set
y_pred_train = best_lasso_model.predict(X_train_scaled)
y_pred_train_inverse = np.exp(scaler_y.inverse_transform(y_pred_train.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y

print("MAPE, 3-month, all features, train: ", round(mean_absolute_percentage_error(y_train,y_pred_train_inverse), 3))
#%%
## Lasso regression - transform test data set
# Predict on the test data
y_pred_test = best_lasso_model.predict(X_test_scaled)
y_pred_test_inverse = np.exp(scaler_y.inverse_transform(y_pred_test.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y

# Evaluate the model performance on the test data
test_score = best_lasso_model.score(X_test_scaled, y_test_scaled)
print("Best Model:", best_lasso_model)
print("Test Set R-squared score:", round(test_score, 3))

# Calculate MAPE
print("MAPE, 3-month, all features, test: ", round(mean_absolute_percentage_error(y_test,y_pred_test_inverse), 3))
#%% md
# # 6-month predictions
#%% md
# ## Naive forecast with test_df and visualisation
#%%
nf.naive_forest(naive_df,target,6,missing)
#%% md
# ## Lasso with only autoregressions, visualise testing set
#%%
## Lasso regression - fit and transform train data set
## Cross validation and Hyperparameter tuning using RandomizedSearchCV

# Slice data for 6-month lag
try:
    X_train_ar=X_train_ar.drop([f"AR_{i}" for i in range(1,6)], axis=1, errors='ignore')
    assert not any(col.endswith(("_1", "_2", "_3", "_4", "_5")) for col in X_train_ar.columns) , "df not sliced correctly"
    X_test_ar=X_test_ar.drop([f"AR_{i}" for i in range(1,6)], axis=1, errors='ignore')
    assert not any(col.endswith(("_1", "_2", "_3", "_4", "_5")) for col in X_test_ar.columns) , "df not sliced correctly"
    print(X_train_ar.columns)
    print(X_test_ar.columns)
except AssertionError:
    print("Unable to slice DataFrame")


scaler_ar_6 = StandardScaler()
X_train_ar_scaled = scaler_ar_6.fit_transform(X_train_ar)
X_test_ar_scaled = scaler_ar_6.transform(X_test_ar)

# Define the parameter grid
param_grid = {'alpha': np.linspace(0.0000001, 1, 3000)}

# Create a Lasso regression model
lasso = Lasso()

# Create RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=lasso, 
                                   param_distributions=param_grid, 
                                   n_iter=300, 
                                   cv=5, 
                                   random_state=42)

# Fit the data to perform a grid search
random_search.fit(X_train_ar_scaled, y_train_scaled)

# Best alpha parameter
print("Best alpha parameter:", random_search.best_params_['alpha'])

# Best R-squared score
print("Best R-squared score:", round(random_search.best_score_, 3))

# Coefficients of the best Lasso model
print("Coefficients of the selected features in the best Lasso model:")
for feature, coefficient in zip(X_train_ar.columns, random_search.best_estimator_.coef_):
    print(f"{feature}: {round(coefficient,3)}")
    
# Get the best Lasso model from RandomizedSearchCV
best_lasso_model = random_search.best_estimator_
    
# Calculate MAPE and MSE of training set
y_pred_train = best_lasso_model.predict(X_train_ar_scaled)
y_pred_train_inverse = np.exp(scaler_y.inverse_transform(y_pred_train.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y

print("MAPE, 6-month, AR only, train: ", round(mean_absolute_percentage_error(y_train,y_pred_train_inverse), 3))
#%%
## Lasso regression - transform test data set
# Predict on the test data
y_pred_test = best_lasso_model.predict(X_test_ar_scaled)
y_pred_test_inverse = np.exp(scaler_y.inverse_transform(y_pred_test.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y

# Evaluate the model performance on the test data
test_score = best_lasso_model.score(X_test_ar_scaled, y_test_scaled)
print("Best Model:", best_lasso_model)
print("Test Set R-squared score:", round(test_score, 3))

# Calculate MAPE
print("MAPE, 6-month, AR only, test: ", round(mean_absolute_percentage_error(y_test,y_pred_test_inverse), 3))
#%% md
# ## Lasso with autoregressions and external price drivers, visualise testing set
#%%
## train_test_split()
## Check data distribution
## Data scaling - log transformation and standardlisation

# Slice data for 3-month lag
try:
    for feature in slicing_columns:
        X_train=X_train.drop([f"{feature}_{i}" for i in range(1,6)], axis=1, errors='ignore')
        X_test=X_test.drop([f"{feature}_{i}" for i in range(1,6)], axis=1, errors='ignore')
    assert not any(col.endswith(("_1", "_2", "_3", "_4", "_5")) for col in X_train.columns) , "df not sliced correctly"
    assert not any(col.endswith(("_1", "_2", "_3", "_4", "_5")) for col in X_test.columns) , "df not sliced correctly"
    print(X_train.columns)
    print(X_test.columns)
except AssertionError:
    print("Unable to slice DataFrame")

scaler_6 = StandardScaler()
X_train_scaled = scaler_6.fit_transform(X_train)
X_test_scaled = scaler_6.transform(X_test)

## Lasso regression - fit and transform train data set
## Cross validation and Hyperparameter tuning using RandomizedSearchCV

# Define the parameter grid
param_grid = {'alpha': np.linspace(0.0000001, 1, 3000)}

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



# Best alpha parameter
print("Best alpha parameter:", random_search.best_params_['alpha'])

# Best R-squared score
print("Best R-squared score:", round(random_search.best_score_, 3))

# Coefficients of the best Lasso model
assert random_search.n_features_in_ == len(X_train.columns)
print("Coefficients of the selected features in the best Lasso model:")
for feature, coefficient in zip(X_train.columns, random_search.best_estimator_.coef_):
    print(f"{feature}: {round(coefficient,3)}")
    
# Get the best Lasso model from RandomizedSearchCV
best_lasso_model = random_search.best_estimator_
    
# Calculate MAPE and MSE of training set
y_pred_train = best_lasso_model.predict(X_train_scaled)
y_pred_train_inverse = np.exp(scaler_y.inverse_transform(y_pred_train.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y

print("MAPE, 6-month, all features, train: ", round(mean_absolute_percentage_error(y_train,y_pred_train_inverse), 3))
#%%
## Lasso regression - transform test data set
# Predict on the test data
y_pred_test = best_lasso_model.predict(X_test_scaled)
y_pred_test_inverse = np.exp(scaler_y.inverse_transform(y_pred_test.reshape(-1,1))).reshape(-1) # the model was trained with log-transformed and standardlised y

# Evaluate the model performance on the test data
test_score = best_lasso_model.score(X_test_scaled, y_test_scaled)
print("Best Model:", best_lasso_model)
print("Test Set R-squared score:", round(test_score, 3))

# Calculate MAPE
print("MAPE, 6-month, all features, test: ", round(mean_absolute_percentage_error(y_test,y_pred_test_inverse), 3))