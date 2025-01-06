#%% md
# # Reference: Predicting Price Evolutions_2022
#%%
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import date
import numpy as np
from datetime import timedelta
import warnings
warnings.simplefilter(action="ignore")
#%% md
# ### Create Timeseries DataFrame
#%%
today = date.today()
df_calendar = pd.DataFrame({'date':pd.date_range(start='2018-01-01', end=today)})
df_calendar['Day'] = df_calendar['date'].dt.day
df_calendar['Day of Week'] = df_calendar['date'].dt.dayofweek
df_calendar['Day of Week'] = df_calendar['Day of Week'].map({
    0: 'Monday',
    1: 'Tuesday',
    2: 'Wednesday',
    3: 'Thursday',
    4: 'Friday',
    5: 'Saturday',
    6: 'Sunday'})
df_calendar['Month'] = df_calendar['date'].dt.month
df_calendar['Year'] = df_calendar['date'].dt.year
df_calendar['ISO Week'] = df_calendar['date'].dt.isocalendar().week
df_calendar['ISO Year'] = df_calendar['date'].dt.isocalendar().year
df_calendar.tail(10)
#%% md
# ### Load Dataset containing Top Raw Materials
#%%
path = os.getcwd()
file = 'Dataset_Predicting_Price_Evolutions.csv'
print("current path: ", path)
#%%
RM = pd.read_csv(path + "\\" + file)
RM['POSTING DATE'] = pd.to_datetime(RM['POSTING DATE'], format='%Y-%m-%d')
RM['Year'] = RM['POSTING DATE'].dt.year
RM.head()
#%% md
# ### Data Analysis
#%%
# total weight per year and per Key RM code
total_volume = RM.groupby(by=['Key RM code', 'Year'], as_index=False)['WEIGHT (kg)'].sum()
total_volume = total_volume[(total_volume['Year'] >= 2022)].reset_index(drop=True)
# pivot table
total_volume = total_volume.pivot(index='Key RM code', columns='Year', values='WEIGHT (kg)').fillna(0).reset_index()
total_volume = total_volume.rename_axis(None).rename_axis(None, axis=1)
# sort table by total volume 2023
total_volume = total_volume.sort_values(by=[2023], ascending=False).reset_index(drop=True)

total_volume.head()
#%%
# Pie Plots
fig, axs = plt.subplots(3, 1)
fig.set_size_inches(12, 12, forward=True)

# plot weight per Group Description
RM_GroupDescription = RM.groupby(by=['Group Description'], as_index=False)['WEIGHT (kg)'].sum()
RM_GroupDescription = RM_GroupDescription.sort_values(by=['WEIGHT (kg)'], ascending=False).reset_index(drop=True)
values = RM_GroupDescription['WEIGHT (kg)'] 
label = RM_GroupDescription['Group Description']
axs[0].pie(values, autopct='%1.1f%%')
axs[0].legend(labels=label, bbox_to_anchor=(1.1,1.025), loc="upper left")

# plot weight per Production Site
RM_Site = RM.groupby(by=['SITE'], as_index=False)['WEIGHT (kg)'].sum()
RM_Site = RM_Site.sort_values(by=['WEIGHT (kg)'], ascending=False).reset_index(drop=True)
values = RM_Site['WEIGHT (kg)'] 
label = RM_Site['SITE']
axs[1].pie(values, autopct='%1.1f%%')
axs[1].legend(labels=label, bbox_to_anchor=(1.1,1.025), loc="upper left")

# plot weight per Key RM
RM_KeyRM = RM.groupby(by=['Key RM code'], as_index=False)['WEIGHT (kg)'].sum()
RM_KeyRM = RM_KeyRM.sort_values(by=['WEIGHT (kg)'], ascending=False).reset_index(drop=True)
values = RM_KeyRM['WEIGHT (kg)'] 
label = RM_KeyRM['Key RM code']
axs[2].pie(values, autopct='%1.1f%%')
axs[2].legend(labels=label[:10], bbox_to_anchor=(1.1,1.025), loc="upper left")

plt.show()
#%%
# count uniques per class 
count_unique = pd.DataFrame(RM.groupby('Group Description', as_index=False)['Key RM code'].nunique())
max_value = pd.DataFrame(count_unique.max())

# Number of datapoints per Group Description
plt.figure(figsize=(14,4))
sns.barplot(x=count_unique['Group Description'], y=count_unique['Key RM code'])
plt.xticks(rotation=45)
plt.ylabel('RM Count')
plt.title('Number of RM per Group Description')
plt.show()
#%% md
# ### Proof of Concept: Correlation between Key RM and External Indices
#%%
# Get Commodity Indices
from fredapi import Fred
api = '6f6e8a770c28c70d77668ab2f5654960'
fred = Fred(api_key=api)

# Natural Gas prices in Europe per month
TTF_GAS = pd.DataFrame(fred.get_series('PNGASEUUSDM'), columns=['PNGASEUUSDM']).reset_index() 
TTF_GAS['index'] = pd.to_datetime(TTF_GAS['index'], format='%Y-%m-%d')
TTF_GAS['Year'] = TTF_GAS['index'].dt.year
TTF_GAS['Month'] = TTF_GAS['index'].dt.month
TTF_GAS = TTF_GAS.drop(['index'], axis=1)

# Crude Oil prices on a daily level
CRUDE_OIL = pd.DataFrame(fred.get_series('DCOILBRENTEU'), columns=['DCOILBRENTEU']).reset_index()
CRUDE_OIL['index'] = pd.to_datetime(CRUDE_OIL['index'], format='%Y-%m-%d')
CRUDE_OIL = CRUDE_OIL.rename(columns={"index": "date"})

# Wheat prices on a monthly level
WHEAT = pd.DataFrame(fred.get_series('PWHEAMTUSDM'), columns=['PWHEAMTUSDM']).reset_index() 
WHEAT['index'] = pd.to_datetime(WHEAT['index'], format='%Y-%m-%d')
WHEAT['Year'] = WHEAT['index'].dt.year
WHEAT['Month'] = WHEAT['index'].dt.month
WHEAT = WHEAT.drop(['index'], axis=1)

# Palm Oil Prices on a monthly level
PALM_OIL = pd.DataFrame(fred.get_series('PPOILUSDM'), columns=['PPOILUSDM']).reset_index()
PALM_OIL['index'] = pd.to_datetime(PALM_OIL['index'], format='%Y-%m-%d')
PALM_OIL['Year'] = PALM_OIL['index'].dt.year
PALM_OIL['Month'] = PALM_OIL['index'].dt.month
PALM_OIL = PALM_OIL.drop(['index'], axis=1)

# Electricity -> different source
file_electricity = 'ELECTRICITY.csv'
ELECTRICITY = pd.read_csv(path + "\\Commodity Indices\\" + file_electricity)
ELECTRICITY = ELECTRICITY[['Year', 'Month', 'Electricity']]

# Combine all together in the calendar DataFrame
df_indices = df_calendar.merge(CRUDE_OIL, how='left', on='date')
df_indices = df_indices.merge(TTF_GAS, how='left', on=['Year', 'Month'])
df_indices = df_indices.merge(WHEAT, how='left', on=['Year', 'Month'])
df_indices = df_indices.merge(PALM_OIL, how='left', on=['Year', 'Month'])
df_indices = df_indices.merge(ELECTRICITY, how='left', on=['Year', 'Month'])

# preview
df_indices.head()
#%%
Key_RM = "RM02/0001"

df_to_use = RM[(RM['Key RM code'] == Key_RM) & (RM['POSTING DATE'] > '2020-01-01')]
df_to_use = df_to_use.groupby(by=['POSTING DATE'], as_index=False).agg(WEIGHT=('WEIGHT (kg)', 'sum'),
                                                                       PRICE=('PRICE (EUR/kg)', 'mean'))

df_to_use = df_to_use.rename(columns={"POSTING DATE": "date"})

df_to_use = df_indices.merge(df_to_use, how='left', on='date')
df_to_use = df_to_use.groupby(by=['Year', 'Month'], as_index=False).agg(WEIGHT=('WEIGHT', 'sum'),
                                                                       PRICE=('PRICE', 'mean'),
                                                                       PNGASEUUSDM=('PNGASEUUSDM', 'mean'),
                                                                       DCOILBRENTEU=('DCOILBRENTEU', 'mean'), 
                                                                       PWHEAMTUSDM=('PWHEAMTUSDM', 'mean'), 
                                                                       PPOILUSDM=('PPOILUSDM', 'mean'), 
                                                                       Electricity=('Electricity', 'mean'),
                                                                       date=('date', 'first'))

# Visualize RM prices versus external features
fig, ax = plt.subplots(5, 1)
fig.set_size_inches(12, 12, forward=True)

# Gas Prices
ax[0].plot(df_to_use['date'], df_to_use['PRICE'], c='g', marker="s", label=Key_RM)
ax0 = ax[0].twinx()
ax0.plot(df_to_use['date'], df_to_use['PNGASEUUSDM'], c='b', marker="s", label='TTF GAS')

ax[0].set_ylabel(Key_RM, color='g')
ax0.set_ylabel('TTF GAS')
ax0.legend(loc='upper left')

# Crude Oil
ax[1].plot(df_to_use['date'], df_to_use['PRICE'], c='g', marker="s", label=Key_RM)
ax1 = ax[1].twinx()
ax1.plot(df_to_use['date'], df_to_use['DCOILBRENTEU'], c='r', marker="s", label='Crude Oil')

ax[1].set_ylabel(Key_RM, color='g')
ax1.set_ylabel('Crude Oil')
ax1.legend(loc='upper left')

# Wheat Prices
ax[2].plot(df_to_use['date'], df_to_use['PRICE'], c='g', marker="s", label=Key_RM)
ax2 = ax[2].twinx()
ax2.plot(df_to_use['date'], df_to_use['PWHEAMTUSDM'], c='y', marker="s", label='Wheat Price')

ax[2].set_ylabel(Key_RM, color='g')
ax2.set_ylabel('Wheat Price')
ax2.legend(loc='upper left')

# Palm Oil Prices
ax[3].plot(df_to_use['date'], df_to_use['PRICE'], c='g', marker="s", label=Key_RM)
ax3 = ax[3].twinx()
ax3.plot(df_to_use['date'], df_to_use['PPOILUSDM'], c='m', marker="s", label='Palm Oil')

ax[3].set_ylabel(Key_RM, color='g')
ax3.set_ylabel('Palm Oil')
ax3.legend(loc='upper left')

# Electicity Prices
ax[4].plot(df_to_use['date'], df_to_use['PRICE'], c='g', marker="s", label=Key_RM)
ax4 = ax[4].twinx()
ax4.plot(df_to_use['date'], df_to_use['Electricity'], c='c', marker="s", label='Electricity')

ax[4].set_ylabel(Key_RM, color='g')
ax4.set_ylabel('Electricity')
ax4.legend(loc='upper left')

plt.show()
#%%
from scipy import stats
# List of commodity index columns
index_columns = ['PNGASEUUSDM', 'DCOILBRENTEU', 'PWHEAMTUSDM', 'PPOILUSDM', 'Electricity']

# Dictionary to store correlation coefficients for each index
correlation_coefficients = {}

# Calculate correlation coefficient for each index
for index_column in index_columns:
    correlation_coefficient, _ = stats.pearsonr(df_to_use[index_column].fillna(0), df_to_use['PRICE'].fillna(0))
    correlation_coefficients[index_column] = correlation_coefficient

# Find the index with the highest correlation
most_interesting_index = max(correlation_coefficients, key=correlation_coefficients.get)
max_correlation_coefficient = correlation_coefficients[most_interesting_index]

print(f"The most interesting index is {most_interesting_index} with a correlation coefficient of {max_correlation_coefficient}")

#%%
# Visualize the correlations
fig, ax1 = plt.subplots(figsize=(10, 6)) # initializes figure and plots
plt.bar(correlation_coefficients.keys(), correlation_coefficients.values())
plt.title('Correlation with Raw Material Price')
plt.xlabel('Commodity Indices')
plt.ylabel('Correlation Coefficient')
plt.show()
#%%
plt.scatter(df_to_use['Electricity'], df_to_use['PRICE'])
plt.title('Electricity vs Raw Material Price')
plt.xlabel('Commodity Index')
plt.ylabel('Raw Material Price')
plt.show()
#%% md
# For RM02/0001, we expect TTF Gas to be our price driver
#%%
def date_feature(df, TYPE): 
    options = df[TYPE].unique()
    features = []
    for opt in options:
        new_col = np.where(df[TYPE]==opt,1,0)
        col_name = TYPE+"_"+str(opt)
        df[col_name]=new_col
        features.append(col_name)
    return df, features
#%%
# max gas price
max_gas = df_to_use[df_to_use['PNGASEUUSDM'] == df_to_use['PNGASEUUSDM'].max()]
print("max gas price: ", df_to_use['PNGASEUUSDM'].max(), " on ", max_gas['date'].max())

# max raw material price
max_price = df_to_use[(df_to_use['PRICE'] == df_to_use['PRICE'].max())]
print("max price RM: ", max_price['PRICE'].max(), " on ", max_price['date'].max())

# calculate time shift in reaching highest peak: 
days_difference = max_price['date'].max() - max_gas['date'].max()
print("Days difference: ", days_difference.days)
#%%
# Create Features incl date shift
features_shift = []

RM_ = RM[(RM['Key RM code'] == Key_RM) & (RM['POSTING DATE'] > '2020-01-01')]
RM_ = RM_.groupby(by=['POSTING DATE', 'SITE', 'Key RM code'], as_index=False).agg(WEIGHT=('WEIGHT (kg)', 'sum'),
                                                                                  PRICE=('PRICE (EUR/kg)', 'mean'))

# 1) SITE
RM_['SITE_'] = 1
RM_['CHR SITE'] = RM_['SITE']
df_ = RM_.copy()
df_.rename(columns={"POSTING DATE": "DATE"}, inplace=True)
df_pivot = df_.pivot(index=['DATE', 'CHR SITE', 'Key RM code', 'PRICE', 'WEIGHT'], columns='SITE', values=['SITE_']).reset_index()
df_pivot.columns = [''.join(col) for col in df_pivot.columns.values]
df_pivot.fillna(0, inplace=True)
for col in df_pivot: 
    if "SITE_" in col: 
        features_shift.append(col)

        
# 2) DATES
min_date = '2020-01-01'
max_date = today
df_date = pd.DataFrame(pd.date_range(start=min_date, end=max_date), columns=['DATE'])

# Year
df_date['YEAR'] = df_date['DATE'].dt.year
df_date, year_features = date_feature(df_date, "YEAR")
# Quarter
df_date['QUARTER'] = df_date['DATE'].dt.quarter
df_date, quarter_features = date_feature(df_date, "QUARTER")
# Month
df_date['MONTH'] = df_date['DATE'].dt.month
df_date, month_features = date_feature(df_date, "MONTH")
# WEEK
df_date['WEEK'] = df_date['DATE'].dt.week
df_date, week_features = date_feature(df_date, "WEEK")
# DAY
df_date['DAY'] = df_date['DATE'].dt.day
df_date, day_features = date_feature(df_date, "DAY")
# COVID
df_date['COVID'] = ((df_date.DATE >='2020-02-23') & (df_date.DATE <='2022-03-07')).astype(int)

features_shift = features_shift + year_features + quarter_features + month_features + week_features + day_features + ['COVID']

# 3) External Features
from fredapi import Fred
api = '6f6e8a770c28c70d77668ab2f5654960'
fred = Fred(api_key=api)

fred_data = fred.get_series('PNGASEUUSDM')
TTF_GAS = pd.DataFrame(fred_data, columns=['TTF GAS'])
TTF_GAS = TTF_GAS.reset_index().rename(columns={"index":"DATE"})
TTF_GAS['DATE'] = pd.to_datetime(TTF_GAS.DATE)
TTF_GAS['Month'] = TTF_GAS['DATE'].dt.month
TTF_GAS['Year'] = TTF_GAS['DATE'].dt.year
TTF_GAS = df_calendar.merge(TTF_GAS, how='left' , on=['Year', 'Month'])
TTF_GAS = TTF_GAS[['date', 'TTF GAS']]
TTF_GAS.columns = ['DATE', 'TTF GAS']
TTF_GAS["X_DATE"] = TTF_GAS["DATE"] + timedelta(days=days_difference.days)
TTF_GAS = TTF_GAS[TTF_GAS['X_DATE'] >= '2020-01-01']
TTF_GAS = TTF_GAS.drop(['DATE'], axis=1)
TTF_GAS.rename(columns={"X_DATE": "DATE"}, inplace=True)

fred_data = fred.get_series('DCOILBRENTEU')
CRUDE_OIL = pd.DataFrame(fred_data, columns=['DCOILBRENTEU'])
CRUDE_OIL = CRUDE_OIL.reset_index().rename(columns={"index":"DATE"})
CRUDE_OIL['DATE'] = pd.to_datetime(CRUDE_OIL.DATE)
CRUDE_OIL["X_DATE"] = CRUDE_OIL["DATE"] + timedelta(days=days_difference.days)
CRUDE_OIL = CRUDE_OIL[CRUDE_OIL['X_DATE'] >= '2020-01-01']
CRUDE_OIL = CRUDE_OIL.drop(['DATE'], axis=1)
CRUDE_OIL.rename(columns={"X_DATE": "DATE"}, inplace=True)

features_shift = features_shift + ['TTF GAS', 'DCOILBRENTEU']

# 4) combine together
df_featurized_shift = df_pivot.merge(df_date, how='left', on='DATE')
df_featurized_shift = df_featurized_shift.merge(CRUDE_OIL, how='left', on='DATE')
df_featurized_shift = df_featurized_shift.merge(TTF_GAS, how='left', on='DATE')

df_featurized_shift['TTF GAS'].fillna(method='ffill', inplace=True)
df_featurized_shift['DCOILBRENTEU'].fillna(method='ffill', inplace=True)

# 5) Visualize
df_featurized_shift = df_featurized_shift.sort_values('DATE', ascending=False).reset_index(drop=True)
df_featurized_shift.head()

#%%
# ANALYZING TIMESHIFT IN EXTERNAL FEATURES
site = "CHBE"
Key_RM = 'RM02/0001'
df_to_plot = df_featurized_shift[(df_featurized_shift['CHR SITE'] == site) & (df_featurized_shift['Key RM code'] == Key_RM)]
fig, ax1 = plt.subplots(figsize=(10, 6)) # initializes figure and plots
ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 

sns.set(style="ticks")
sns.scatterplot(x="DATE", y="PRICE", data=df_to_plot, ax=ax1)
sns.lineplot(x='DATE', y='TTF GAS', data=df_to_plot, color="b", ax=ax2)
sns.lineplot(x='DATE', y='DCOILBRENTEU', data=df_to_plot, color="r", ax=ax2)

ax1.set_xlabel('DATE')
ax1.set_ylabel('Price')
ax1.yaxis.grid(True) # Show the horizontal gridlines
ax1.xaxis.grid(True) # Show the vertical gridlines

ax2.set_ylabel('TTF GAS (blue) - DCOILBRENTEU (red)')

plt.title(Key_RM + " - " + site)
plt.show()
#%% md
# ### Train Model
#%%
# Packages for MLP
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
import pickle
#%%
df_featurized_shift = df_featurized_shift[df_featurized_shift['Key RM code'] == Key_RM].reset_index(drop=True)
df_featurized_shift.head()
#%%
# Create dataframes
df_features = df_featurized_shift[features_shift]
df_target = df_featurized_shift['PRICE']
df_info = df_featurized_shift[['DATE', 'CHR SITE', 'Key RM code', 'WEIGHT', 'TTF GAS', 'DCOILBRENTEU']]
#%%
# Define the hyperparameters and their possible values
param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 25), (100, 50, 25)],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'batch_size': [16, 32, 64],
    'max_iter': [200, 500, 1000]
}
#%%
# Split the data into training and testing sets 
features_train_, features_test_, target_train, target_test = train_test_split(df_features, df_target, random_state=1)
#%%
# Normalize the features
scaler = preprocessing.RobustScaler()
features_train = scaler.fit_transform(features_train_)
features_test = scaler.transform(features_test_)
print("Train set:  {'X_Train': ", features_train.shape, " 'Y_Train': ", target_train.shape, "}")
print("Test set:  {'X_Test': ", features_test.shape, " 'Y_Test': ", target_test.shape, "}")
#%%
# Create an MLP regressor object
mlp = MLPRegressor(random_state=1)
#%%
# Create a GridSearchCV object to search for the best hyperparameters
grid_search = GridSearchCV(mlp, param_grid, cv=5, n_jobs=-1)
#%%
# Fit the GridSearchCV object to the training data
grid_search.fit(features_train, target_train)
#%%
# Print the best hyperparameters found by GridSearchCV
print("Best hyperparameters:", grid_search.best_params_)
#%%
# Make predictions on the test set using the best model found by GridSearchCV
best_model = grid_search.best_estimator_
predictions = best_model.predict(features_test)
#%%
# Evaluate the performance of the best model on the test set
score = best_model.score(features_test, target_test)
print("Test set score:", score)
#%%
# save best model
filename = Key_RM.replace("/", "_") + '_MLP_' + str(today).replace("-", "") + '.sav'
model_path = './Saved models/' + filename

pickle.dump(best_model, open(model_path, 'wb'))
print(filename, " is saved!")
#%%
results = pd.DataFrame(target_test)
results['Prediction'] = predictions
results['mae'] = abs(results['Prediction'] - results['PRICE'])

results = results.merge(features_test_, left_index=True, right_index=True)
results = results[['PRICE', 'Prediction', 'mae']]
results.sort_values('mae', ascending=False).head()
#%% md
# ### Evaluate model on complete Dataset
#%%
# open saved model
filename = Key_RM.replace("/", "_") + '_MLP_' + str(today).replace("-", "") + '.sav'
model_path = './Saved models/' + filename

loaded_model = pickle.load(open(model_path,'rb'))

print(loaded_model)
#%%
# predict full dataset
features_all = scaler.fit_transform(df_features)

predictions_all = loaded_model.predict(features_all)
#%%
df_info['predictions'] = predictions_all
df_info['PRICE'] = df_target
df_info['mae'] = abs(df_info['predictions'] - df_info['PRICE'])

df_info1_ = df_info.drop(['predictions'], axis=1)
df_info1_['TYPE'] = "ground_truth"
df_info1_.rename(columns={"PRICE": "value"}, inplace=True)

df_info2_ = df_info.drop(['PRICE'], axis=1)
df_info2_['TYPE'] = "prediction"
df_info2_.rename(columns={"predictions": "value"}, inplace=True)

df_results_all_predictions = df_info1_.append(df_info2_)
#%%
df_info.head()
#%%
site = "CHBE"
df_to_plot = df_results_all_predictions[df_results_all_predictions['CHR SITE'] == site]

fig, ax1 = plt.subplots(figsize=(10, 6)) # initializes figure and plots
ax2 = ax1.twinx() # applies twinx to ax2, which is the second y axis. 

sns.set(style="ticks")
sns.scatterplot(x="DATE", y="value", hue='TYPE', data=df_to_plot, ax=ax1)
sns.lineplot(x='DATE', y='TTF GAS', data=df_to_plot, color="b", ax=ax2)
sns.lineplot(x='DATE', y='DCOILBRENTEU', data=df_to_plot, color="r", ax=ax2)

ax1.set_xlabel('DATE')
ax1.set_ylabel('Price')
ax1.yaxis.grid(True) # Show the horizontal gridlines
ax1.xaxis.grid(True) # Show the vertical gridlines

ax2.set_ylabel('TTF GAS (blue) - DCOILBRENTEU (red)')

plt.title(Key_RM + " - " + site +" - datetime shift")
plt.show()