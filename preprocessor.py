import numpy as np
import pandas as pd
from datetime import datetime
from fredapi import Fred


def get_Fred_data(target: str, \
                  start_year: int, \
                  end_year: int, \
                  apiKey: str = '29219060bc68b2802af8584e0f328b52') -> pd.DataFrame:
    '''To extract data from Fred database: https://fred.stlouisfed.org/ 
    apiKey = '29219060bc68b2802af8584e0f328b52'
    PWHEAMTUSDM - wheat https://fred.stlouisfed.org/series/PWHEAMTUSDM
    WPU0652013A - Ammonia https://fred.stlouisfed.org/series/WPU0652013A
    PNGASEUUSDM - TTG_Gas https://fred.stlouisfed.org/series/PNGASEUUSDM
    '''  # get_Fred_data.__doc__
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
    df = df[df['Year'].between(start_year, end_year)]
    df.rename(columns={'index': 'Time'}, inplace=True)

    df[target] = df[target].ffill()

    assert df.isnull().values.any() == False, "Imported data contains NaN."

    return df


def clean_elec_csv(file: str, start_year: int, end_year: int) -> pd.DataFrame:
    '''
    To clean ELECTRICITY.csv correctly for following pre-processing steps
    '''
    assert start_year <= end_year, 'start_year can not exceed end_year'
    assert end_year <= datetime.now().year, 'end_year can not include future date'

    # Import monthly electrcity data
    df = pd.read_csv(file).iloc[:, 1:]
    df['Time'] = pd.to_datetime(df['Year'].astype(str) + df['Month'].astype(str), format='%Y%m')
    df = df[df['Year'].between(start_year, end_year)].reset_index().drop('index', axis=1)

    assert df.isnull().values.any() == False, "Imported/Returned data contains NaN."
    return df


def clean_pred_price_evo_csv(file: str, start_year: int, end_year: int) -> pd.DataFrame:
    '''
    To clean Dataset_Predicting_Price_Evolutions.csv correctly for following pre-processing steps
    '''
    assert start_year <= end_year, 'start_year can not exceed end_year'
    assert end_year <= datetime.now().year, 'end_year can not include future date'

    # Import price evaluatioin data
    df = pd.read_csv(file).iloc[:, 1:]
    df['Group Description'] = df['Group Description'].str.lower()
    df['POSTING DATE'] = pd.to_datetime(df['POSTING DATE'], format='ISO8601')
    # df['POSTING DATE'] = pd.to_datetime(df['POSTING DATE'], format='%Y-%m-%d')
    df['Year'] = df['POSTING DATE'].dt.year
    df['Month'] = df['POSTING DATE'].dt.month
    df = df.sort_values(['Year', 'Month'], ascending=True)
    df = df[df['Year'].between(start_year, end_year)].reset_index().drop(['index'], axis=1)
    df.rename(columns={'POSTING DATE': 'Time'}, inplace=True)

    # Drop unnecessary columns
    df = df.drop(['SITE', 'SUPPLIER NUMBER', 'PURCHASE NUMBER', 'WEIGHT (kg)'], axis=1)

    assert df.isnull().values.any() == False, "Imported/Returned data contains NaN."
    return df


def impute_pred_price_evo_csv(old_df: pd.DataFrame) -> pd.DataFrame:
    '''
    1. Create all combinations of Year Month and Key RM Codes
    2. Map the combinations with the imported raw material prices to ensure having all RM Codes for each months
    3. Impute Year, Month, Prices (Forward Fill)
    
    Return two Dataframes: df_not_null, missing
    -> df_not_null: Complete dataframe imputed by forward fill method
    -> missing: Rows needed to be imputed
    '''
    RM_list = old_df['Key RM code'].unique()

    # Create combinations of ['Year','Month','Key RM code']
    combinations = pd.DataFrame(
        [(year, month, rm_code) for year in old_df['Year'].unique() \
         for month in old_df['Month'].unique() \
         for rm_code in RM_list], \
        columns=['Year', 'Month', 'Key RM code']
    )

    # Merge combinations with the original DataFrame to identify missing combinations, because we intend to have all RM codes appear in each Years+Months.
    df = pd.merge(combinations, old_df, on=['Year', 'Month', 'Key RM code'], how='left')

    ## Filter out and impute missing values
    missing = df[df['PRICE (EUR/kg)'].isnull()].drop(["Time", "Group Description", "PRICE (EUR/kg)"], axis=1)
    missing_codes = missing['Key RM code'].unique()

    df_notna = df.dropna(how="any", axis=0)

    for code in missing_codes:
        # Impute group description
        df.loc[(df['Key RM code'] == code) & (df['Group Description'].isnull()), 'Group Description'] = \
        df_notna.loc[(df_notna['Key RM code'] == code), 'Group Description'].unique()[0]

        # Impute Time
        df.loc[df['Time'].isnull(), 'Time'] = pd.to_datetime(
            {'year': missing['Year'], 'month': missing['Month'], 'day': 15})
        df = df.sort_values(by=['Time']).reset_index().drop('index', axis=1)

        # Impute Price
        df['PRICE (EUR/kg)'] = df.groupby('Key RM code')['PRICE (EUR/kg)'].ffill()
        df_cleaned = df.dropna(how="any", axis=0)
        assert not df_cleaned.isnull().values.any(), print(df[df['PRICE (EUR/kg)'].isna()])

    return df_cleaned, missing


def get_dummies_and_average_price(raw_df: pd.DataFrame, target: str, *args: str) -> pd.DataFrame:
    '''
    *args: str -> Key RM Codes corresponding to the target variable, create N dummy variables
    # auto filter df based on given target_name and key rm codes
    # to auto calculate the monthly average prices of the target variable
    # auto dummy variables and concat
    # output columns: 'Time', 'Group Description', 'Year', 'Month', 'RM02/0002',
       'Average_price'
    # To aggregate all observations with year, month, Key RM Code -> Not implemented yet
    '''
    # To ensure inputted Key RM Codes belong to corresponding Group Description
    valid_codes = raw_df['Key RM code']
    raw_df['Group Description'] = raw_df['Group Description'].str.lower()
    assert raw_df.loc[valid_codes.isin(
        args), 'Group Description'].unique().all() == target, "RM codes don't align with the group description."
    for i in args:
        if i not in valid_codes.unique():
            raise Exception(f"{i} is not a valid RM code.")

    # Filter raw_df by target_variable_name, ex: acid
    filtered_df = raw_df[raw_df['Group Description'] == target] \
        .sort_values(['Year', 'Month'], ascending=True)
    filtered_df = filtered_df.reset_index().drop('index', axis=1)

    dummies = pd.get_dummies(filtered_df['Key RM code'], drop_first=False)
    # if len(args) == 1:
    #   dummies = pd.get_dummies(filtered_df['Key RM code'], drop_first=False)
    # else:
    #  	dummies = pd.get_dummies(filtered_df['Key RM code'], drop_first=True)

    dummies = pd.concat([filtered_df, dummies], axis=1)
    filtered_df_dummies = dummies.drop('Key RM code', axis=1)

    # Verify if columns in args exist in filtered_df_dummies
    missing_columns = set(args) - set(filtered_df_dummies.columns)
    if missing_columns == True:
        raise Exception(f"Columns {', '.join(missing_columns)} are missing in filtered_df_dummies.")

    ## Calculate the average raw material price
    conditions = ['Year', 'Month'] + list(args)
    mask = list(args)
    # if len(args) == 1:
    #     conditions = ['Year','Month'] + list(args)
    #     mask = list(args)
    # else:
    #     conditions = ['Year','Month'] + list(args[1:]) # args -> Key RM Coded, drop_first=True
    #     mask = list(args[1:])

    average_price = filtered_df_dummies.groupby(conditions)['PRICE (EUR/kg)'] \
        .mean() \
        .reset_index()

    ## To aggregate all observations with year, month, Key RM Code
    filtered_df_dummies['Time_index'] = filtered_df_dummies['Time']
    filtered_df_dummies.set_index('Time_index', inplace=True)
    # Group by 'RM02/0002' and resample to monthly frequency while keeping the last value
    filtered_df_dummies = filtered_df_dummies.groupby(mask) \
        .resample('M') \
        .last() \
        .drop(mask, axis=1) \
        .reset_index() \
        .drop('Time_index', axis=1) \
        .dropna()

    # Merge the average monthly price with the original dataframe
    filtered_df_dummies = pd.merge(filtered_df_dummies, average_price, \
                                   on=conditions, \
                                   suffixes=('', '_avg'))

    # Rename the new column to 'Average_price'
    filtered_df_dummies.rename(columns={'PRICE (EUR/kg)_avg': 'Average_price'}, \
                               inplace=True)
    filtered_df_dummies = filtered_df_dummies.drop('PRICE (EUR/kg)', axis=1)

    assert filtered_df_dummies.isnull().values.any() == False, "Imported/Returned data contains NaN."
    return filtered_df_dummies


def exclude_imputed_data_from_y(y_df: pd.DataFrame, missing: pd.DataFrame, RM_codes: list) -> pd.DataFrame:
    """
    This function will exclude imputed records without real purchasing data from y, i.e. if there were purchases in Jan-2022 and Mar-2022, then we only predict y based on the data of Jan-2022 and Mar-2022, and exclude the imputed values in Feb-2022 to be one of the target variables(y).
    
    - missing -> second return of preprocessor.impute_pred_price_evo_csv(), the list of imputed rows, ["Year", "Month", "Key RM code"]
    """
    assert len(y_df) != 0, "Warning: Empty dataframe!"
    assert len(missing) != 0, "Warning: impute_list is empty."
    assert len(RM_codes) != 0, "Warning: RM codes are empty."

    ## To slide imputed list by RM codes with the same group, and create dummy variables
    missing_ = missing[missing['Key RM code'].isin(RM_codes)]
    dummies_ = pd.get_dummies(missing_['Key RM code'], drop_first=False)
    dummies_df = pd.concat([missing_, dummies_], axis=1).drop('Key RM code', axis=1)

    # Unit testing
    to_test = set(dummies_df.drop(['Year', 'Month'], axis=1).columns)
    assert to_test.issubset(set(RM_codes)), "filtered_df_dummies includes incorrect RM code."

    ## Anti join between feature_df and imputed list
    outer = y_df.merge(dummies_df, how='outer', indicator=True)  # indicator True creates a column "_merge"
    anti_join = outer[(outer._merge == 'left_only')].drop('_merge', axis=1)

    # Unit testing
    for code in RM_codes:
        # Ensure anti_join and missing do not have matching "Year", "Month", rm code
        missing_filtered = missing_.loc[missing_['Key RM code'] == code, ['Year', 'Month']]
        anti_join_filtered = anti_join.loc[anti_join[code] == True, ['Year', 'Month']]
        missing_set = set([tuple(x) for x in missing_filtered.values])
        anti_join_set = set([tuple(x) for x in anti_join_filtered.values])

        # Assert that there are no common elements
        assert missing_set.isdisjoint(
            anti_join_set), f"There are matching 'Year', 'Month', {code} between anti_join and missing."

    return anti_join


def generate_features(start: int, end: int, y_df: pd.DataFrame, impute_list: pd.DataFrame, *args: str,
                      **kwargs: pd.DataFrame) -> pd.DataFrame:
    '''
    To generate features and combine y_df with historical prices of external price drivers and autoregressive prices of y.
    Supports N feature inputs and creates 12*N features, price changes from {start} months before prices to {end} months before prices
    This function only take imputed records without real purchasing for autoregression, but will exclude them from y, i.e. if there were purchases in Jan-2022 and Mar-2022, then we only predict y based on the data of Jan-2022 and Mar-2022, and exclude the imputed values in Feb-2022 to be one of the target variables(y).
    
    Inputs:
    - start -> starting point of time lag duration
    - end -> ending point of time lag duration
    - y_df -> df containing target variables and dummies
    - impute_list -> second return of preprocessor.impute_pred_price_evo_csv(), the list of imputed rows, ["Year", "Month", "Key RM code"]
    - *args -> all Key RM codes correspinding to the group description of y
    - **kwargs -> {Name of external price drivers: df of external price driver data}
    - Note: Unable to check the correctness of inputted Key RM codes
    
    Return:
    - y_df_non_na -> y_df with 1*N features, rows containing NaN will be excluded
    
    Usage example:
    acid_df = generate_features(1,2,dummy_df,missing,\
                                 'RM01/0001',\
                                 'RM01/0004',\
                                 'RM01/0006',\
                                 'RM01/0007',\
                                PNGASEUUSDM=gas_df,\
                                PWHEAMTUSDM=wheat_df,\
                                WPU0652013A=ammonia_df,\
                                Electricity=elec_df
                               )

    Return df.columns example:
    ['RM01/0004', 'RM01/0006', 'RM01/0007', 'Year', 'Month', 'Time',
       'Group Description', 'Average_price', 'PNGASEUUSDM_1', 'PWHEAMTUSDM_1',
       'WPU0652013A_1', 'Electricity_1', 'PNGASEUUSDM_2', 'PWHEAMTUSDM_2',
       'WPU0652013A_2', 'Electricity_2', 'AR_1', 'AR_2']
    '''
    assert start <= end, 'Ending point should be later than starting point.'
    assert len(y_df) != 0, "Warning: Empty dataframe!"
    assert len(impute_list) != 0, "Warning: impute_list is empty."
    assert len(args) != 0, "Warning: RM codes are empty."
    assert len(kwargs) != 0, "Warning: External price drivers are empty."

    ## To create time-labelled feature lists of external price drivers
    first_key = next(iter(kwargs))
    feature_df = kwargs[first_key]  # to caputer the first given df for following merging

    # enumerate(kwargs.items()): This part of the expression iterates over the items of the kwargs dictionary using the items() method, yielding pairs of (index, (key, value)) tuples where index is the index of the item in the enumeration, and (key, value) is the key-value pair from the dictionary.
    for key, df in {k: v for i, (k, v) in enumerate(kwargs.items()) if i != 0}.items():
        feature_df = pd.merge(feature_df, df, how='left', on=(['Year', 'Month', 'Time']))

    # Add time label on feature df  
    feature_df['Time_label'] = feature_df['Time'].dt.strftime('%Y-%m')
    feature_df = feature_df.drop(['Year', 'Month', 'Time'], axis=1)  # to prevent duplicate columns when merging

    ## To create autoregression parts
    # to filter RM codes with dummies only
    RM_dummy = [arg for arg in args if arg in y_df.columns]

    ar_df = pd.DataFrame()
    ar_df = y_df[["Time", "Average_price", *RM_dummy]]
    ar_df.rename(columns={"Average_price": "AR"}, inplace=True)
    ar_df['Time_label'] = ar_df['Time'].dt.strftime('%Y-%m')
    ar_df = ar_df.drop(["Time"], axis=1)  # to prevent duplicate columns when merging

    ## To combine historical feature prices and autoregression with target variables:
    ## Step 1. create time labels based on y_df['Time']
    ## Step 2.1 merge labels with feature_df to get historical prices of external factors
    ## Step 2.2 drop time labels
    ## Step 3.1 merge labels with ar_df to get autoregressive prices
    ## Step 3.2 drop time labels
    ## Step 4. merge y_df with historical prices of external factors and AR prices
    # step 1
    label_dfs = []  # To store time labels
    # ref: 'https://pandas.pydata.org/docs/user_guide/merging.html'
    for i in range(start, end + 1):
        label = y_df[['Time']]
        label.rename(columns={'Time': f'Time_label{i}'}, inplace=True)
        label = (label[f'Time_label{i}'] - pd.DateOffset(months=i)).dt.strftime('%Y-%m')
        label_dfs.append(label)

    df_1 = pd.concat(label_dfs, axis=1)  # transform labels into a df
    df_1['Time'] = y_df['Time']  # add current time as merging keys
    df_2 = pd.concat(label_dfs, axis=1)  # transform labels into a df
    df_2['Time'] = y_df['Time']  # add current time as merging keys
    df_2[[*RM_dummy]] = ar_df[[*RM_dummy]]  # add dummy variables as merging keys

    # step 2_1
    for i in range(start, end + 1):
        df_1 = df_1.merge(feature_df, how='left', \
                          left_on=[f'Time_label{i}'], \
                          right_on=['Time_label'])
        [df_1.rename(columns={key: f'{key}_{i}'}, inplace=True) for key, value in kwargs.items()]
        # step2_2
        df_1 = df_1.drop(['Time_label', f'Time_label{i}'], axis=1)

    # step 3_1   
    for i in range(start, end + 1):
        df_2 = df_2.merge(ar_df, how='left', \
                          left_on=[f'Time_label{i}', *RM_dummy], \
                          right_on=["Time_label", *RM_dummy])
        df_2.rename(columns={"AR": f"AR_{i}"}, inplace=True)
        # step3_2
        df_2 = df_2.drop(['Time_label', f'Time_label{i}'], axis=1)

    # step 4    
    y_df = pd.merge(y_df, df_1, how='left', on=['Time'])
    y_df = pd.merge(y_df, df_2, how='left', on=["Time", *RM_dummy])
    non_na_df = y_df.dropna(axis=0, how='any').drop_duplicates(subset=None)

    ## To exclude records without real purchasing
    # i.e. if there were purchases in Jan-2022 and Mar-2022, then we only consider the performances of models based on the predictions of Jan-2022 and Mar-2022, and exclude the imputed value in Feb-2022.
    y_df_non_na = exclude_imputed_data_from_y(non_na_df, impute_list, args)

    # Unit testing
    assert not y_df_non_na.isnull().values.any(), "Returned data contains NaN."
    assert y_df_non_na.shape[0] > 0, "The returned DataFrame is empty."

    if len(args) == 1:
        assert y_df_non_na.shape[1] == ((len(kwargs) + 1) * (end - start + 1) + len(
            args) + 5), "The number of columns in the returned DataFrame is incorrect."
    else:
        # assert y_df_non_na.shape[1] == ((len(kwargs)+1)*(end-start+1)+len(args)-1+5), "The number of columns in the returned DataFrame is incorrect."
        assert y_df_non_na.shape[1] == ((len(kwargs) + 1) * (end - start + 1) + len(
            args) + 5), "The number of columns in the returned DataFrame is incorrect."
    for key, value in kwargs.items():
        for i in range(start, end + 1):
            assert y_df_non_na.dtypes[
                       f'{key}_{i}'] == np.float64, f"The data type of column {key}_{i} is not np.float64."

    assert y_df_non_na.shape[0] <= y_df.shape[0], "Returned df has more rows than inputted y_df."

    return y_df_non_na


def get_interaction_terms(raw_df: pd.DataFrame) -> pd.DataFrame:
    assert raw_df.shape[0] > 0, "Given DataFrame is empty!"

    df = raw_df.copy()

    rm_lists = [col for col in raw_df.columns if col.startswith("RM")]
    ar_lists = [col for col in raw_df.columns if col.startswith("AR")]

    for rm in rm_lists:
        for ar in ar_lists:
            new_column = f"{rm}_{ar}"
            raw_df[new_column] = raw_df[rm] * raw_df[ar]

    assert not raw_df.isnull().values.any(), "Returned data contains NaN."
    assert raw_df.shape[0] == df.shape[0], "The No. of rows changed!"
    assert raw_df.shape[1] == (len(df.columns) + len(rm_lists) * len(ar_lists)), "Incorrect shape"

    return raw_df
