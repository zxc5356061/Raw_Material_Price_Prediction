from fredapi import Fred
import pandas as pd
from datetime import datetime


def get_Fred_data(target:str,\
                  start_year:int,\
                  end_year:int,\
                  apiKey: str ='29219060bc68b2802af8584e0f328b52') -> pd.DataFrame:
    '''To extract data from Fred database: https://fred.stlouisfed.org/ 
    apiKey = '29219060bc68b2802af8584e0f328b52'
    PWHEAMTUSDM - wheat https://fred.stlouisfed.org/series/PWHEAMTUSDM
    WPU0652013A - Ammonia https://fred.stlouisfed.org/series/WPU0652013A
    PNGASEUUSDM - TTG_Gas https://fred.stlouisfed.org/series/PNGASEUUSDM
    ''' # get_Fred_data.__doc__
    assert start_year <= end_year, 'start_year can not exceed end_year'
    assert end_year <= datetime.now().year, 'end_year can not include future date'
    
    target = target.upper()
    # Import monthly data
    _key = apiKey
    fred = Fred(api_key=_key)
    df = pd.DataFrame(fred.get_series(target),
                       columns=[target]).reset_index()
    df['index'] = pd.to_datetime(df['index'], format='%Y-%m-%d')
    df['Year'] = df['index'].dt.year
    df['Month'] = df['index'].dt.month
    df = df[df['Year'].between(start_year,end_year)]
    df.rename(columns = {'index':'Time'}, inplace = True)
    
    assert df.isnull().values.any() == False, "Imported data contains NaN."
    
    return df


def clean_elec_csv(file: str, start_year:int, end_year:int) -> pd.DataFrame:
    '''
    To clean ELECTRICITY.csv correctly for following pre-processing steps
    '''
    assert start_year <= end_year, 'start_year can not exceed end_year'
    assert end_year <= datetime.now().year, 'end_year can not include future date'
    
    # Import monthly electrcity data
    df = pd.read_csv(file).iloc[:,1:]
    df['Time'] = pd.to_datetime(df['Year'].astype(str) + df['Month'].astype(str), format='%Y%m')
    df = df[df['Year'].between(start_year,end_year)].reset_index().drop('index',axis=1)
    
    assert df.isnull().values.any() == False, "Imported/Returned data contains NaN."
    return df
    
    
def clean_pred_price_evo_csv(file: str, start_year:int, end_year:int) -> pd.DataFrame:
    '''
    To clean Dataset_Predicting_Price_Evolutions.csv correctly for following pre-processing steps
    '''
    assert start_year <= end_year, 'start_year can not exceed end_year'
    assert end_year <= datetime.now().year, 'end_year can not include future date'
    
    # Import price evaluatioin data
    df = pd.read_csv(file).iloc[:,1:]
    df['POSTING DATE'] = pd.to_datetime(df['POSTING DATE'], format='%Y-%m-%d')
    df['Year'] = df['POSTING DATE'].dt.year
    df['Month'] = df['POSTING DATE'].dt.month
    df = df.sort_values(['Year','Month'],ascending=True)
    df = df[df['Year'].between(start_year,end_year)].reset_index().drop(['index'], axis=1)
    df.rename(columns = {'POSTING DATE':'Time'}, inplace = True)
    
    # Drop unnecessary columns
    df = df.drop(['SITE', 'SUPPLIER NUMBER', 'PURCHASE NUMBER', 'WEIGHT (kg)'], axis=1)
    
    assert df.isnull().values.any() == False, "Imported/Returned data contains NaN."
    return df


def get_dummies_and_average_price(raw_df: pd.DataFrame, target: str, *args: str) -> pd.DataFrame:
    '''
    *args: str -> to input key rm codes
    # auto filter df based on given target_name and key rm codes
    # to auto calculate the monthly average prices of the target variable
    # auto dummy variables and concat
    # output columns: 'Time', 'Group Description', 'Year', 'Month', 'RM02/0002',
       'Average_price'
    # To aggregate all observations with year, month, Key RM Code -> Not implemented yet
    '''
    # To ensure inputted Key RM Codes belong to corresponding Group Description
    valid_codes = raw_df['Key RM code']
    assert raw_df.loc[valid_codes.isin(args), 'Group Description'].unique() == target, "RM codes don't align with the group description."
    for i in args:
        if i not in valid_codes.unique():
            raise Exception(f"{i} is not a valid RM code.")
            
            
    # Filter raw_df by target_variable_name, ex: acid
    filtered_df = raw_df[raw_df['Group Description'] == target]\
                    .sort_values(['Year','Month'],ascending=True)
    filtered_df = filtered_df.reset_index().drop('index',axis=1)
    
    # Create n-1 dummy variables
    dummies = pd.get_dummies(filtered_df['Key RM code'], drop_first=True)
    dummies = pd.concat([filtered_df, dummies], axis=1)
    filtered_df_dummies = dummies.drop('Key RM code', axis=1)
    
    
    # Calculate the average raw material price
    conditions = ['Year','Month'] + list(args[1:]) # args -> Key RM Coded, drop_first=True
    average_price = filtered_df_dummies.groupby(conditions)['PRICE (EUR/kg)']\
                                        .mean()\
                                        .reset_index()
    
    ## To be discussed - To aggregate all observations with year, month, Key RM Code
    filtered_df_dummies['Time_index']=filtered_df_dummies['Time']
    filtered_df_dummies.set_index('Time_index', inplace=True)
    # Group by 'RM02/0002' and resample to monthly frequency while keeping the last value
    filtered_df_dummies = filtered_df_dummies.groupby(list(args[1:]))\
                                             .resample('M')\
                                             .last()\
                                             .drop(list(args[1:]),axis=1)\
                                             .reset_index()\
                                             .drop('Time_index',axis=1)\
                                             .dropna()
    
    # Merge the average monthly price with the original dataframe
    filtered_df_dummies = pd.merge(filtered_df_dummies, average_price,\
                                   on = conditions,\
                                   suffixes=('', '_avg'))
    
    # Rename the new column to 'Average_price'
    filtered_df_dummies.rename(columns={'PRICE (EUR/kg)_avg': 'Average_price'},\
                               inplace=True)
    filtered_df_dummies = filtered_df_dummies.drop('PRICE (EUR/kg)', axis=1)
    
    assert filtered_df_dummies.isnull().values.any() == False, "Imported/Returned data contains NaN."
    return filtered_df_dummies


def generate_features(start:int, end:int, y_df:pd.DataFrame, **kwargs: pd.DataFrame) -> pd.DataFrame:
    '''
    # idea: to support importing n df inputs, 1+ is the target variables df, with other n external factors df -> *args
    # y_df -> the df contains target variable data
    ## Create 12*N features, external factor prices from one-month before to 12-month before
    ## Combine features with target variables
    '''
    first_key = next(iter(kwargs))
    feature_df = kwargs[first_key]
    # enumerate(kwargs.items()): This part of the expression iterates over the items of the kwargs dictionary using the items() method, yielding pairs of (index, (key, value)) tuples where index is the index of the item in the enumeration, and (key, value) is the key-value pair from the dictionary.
    for key, df in {k: v for i, (k, v) in enumerate(kwargs.items()) if i != 0}.items():
        feature_df = pd.merge(feature_df, df, how='left', on = (['Year', 'Month', 'Time']))

    # Add time label on feature df  
    feature_df['Time_label'] = feature_df['Time'].dt.strftime('%Y-%m')
    feature_df = feature_df.drop(['Year','Month', 'Time'], axis=1) # to prevent duplicate columns when merging

    # Combine historical feature prices with target variables:
    ## Step 1. create time labels based on the 'Time' of target df
    ## Step 2. merge time labels with feature df to get historical prices
    ## Step 3. drop time labels
    ## Step 4. merge target df with historical prices
    # step 1
    label_dfs=[]    # To store time labels
                # ref: 'https://pandas.pydata.org/docs/user_guide/merging.html'
    for i in range(start,end+1):
        label = y_df[['Time']]
        label.rename(columns = {'Time':f'Time_label{i}'}, inplace = True)
        label = (label[f'Time_label{i}'] - pd.DateOffset(months=i)).dt.strftime('%Y-%m')
        label_dfs.append(label)
        
    result = pd.concat(label_dfs, axis=1)

    # step 2
    for i in range(start,end+1):
        result = result.merge(feature_df, how='left',\
                              left_on=[f'Time_label{i}'],\
                              right_on=['Time_label'])
        [result.rename(columns={key: f'{key}_{i}'}, inplace=True) for key, value in kwargs.items()]
        # step3
        result = result.drop(['Time_label',f'Time_label{i}'], axis=1)
    # step 4    
    y_df = pd.concat([y_df,result],axis=1)
    assert y_df.isnull().values.any() == False, "Imported/Returned data contains NaN."
    return y_df


# def monthly_mean_to_daily(df_monthly: pd.core.frame.DataFrame ) -> pd.core.frame.DataFrame:
#     """
#     Convert Monthly data into Daily data and impute with monthly mean prices
#     """
#     df_monthly['Date'] = pd.to_datetime(df_monthly[['Year', 'Month']].assign(DAY=1))
#     df = df_monthly.explode('Date') # The explode() method converts each element of the specified column(s) into a row.

#     # Generate a complete range of daily dates for the year for imputation
#     start_date = df['Date'].min() # represents the starting point of your data
#     end_date = df['Date'].max() + pd.offsets.MonthEnd(1)  # finds the maximum (or latest) date and include the last month fully
#     full_date_range = pd.date_range(start=start_date, end=end_date, freq='D') # generates a fixed-frequency DatetimeIndex

#     # Merge the full date range with the monthly averages to fill in all days
#     df_full_date_range = pd.DataFrame(full_date_range, columns=['Date'])
#     df = pd.merge(df_full_date_range, df_monthly, on='Date', how='left')
#     df_daily = df.ffill(axis=0) # to fill the missing value based on last valid observation following index sequence
#     return df_daily

