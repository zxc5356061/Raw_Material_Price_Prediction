import pandas as pd


def lambda_handler(event):
    pass


def impute_pred_price_evo_csv(old_df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Create all combinations of Year Month and Key RM Codes
    2. Map the combinations with the imported raw material prices to ensure having all RM Codes for each months
    3. Impute Year, Month, Prices (Forward Fill)

    Return two Dataframes: df_not_null, missing
    -> df_not_null: Complete dataframe imputed by forward fill method
    -> missing: Rows needed to be imputed
    """
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
    """
    *args: str -> Key RM Codes corresponding to the target variable, create N dummy variables
    # auto filter df based on given target_name and key rm codes
    # to auto calculate the monthly average prices of the target variable
    # auto dummy variables and concat
    # output columns: 'Time', 'Group Description', 'Year', 'Month', 'RM02/0002',
       'Average_price'
    # To aggregate all observations with year, month, Key RM Code -> Not implemented yet
    """
    # To ensure inputted Key RM Codes belong to corresponding Group Description
    valid_codes = raw_df['Key RM code']
    raw_df['Group Description'] = raw_df['Group Description'].str.lower()

    try:
        check = raw_df.loc[valid_codes.isin(args), 'Group Description'].drop_duplicates().item()
    except ValueError:
        raise Exception("Please check input RM codes")

    assert check == target, f"RM codes don't align with the group description: {check}."
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
