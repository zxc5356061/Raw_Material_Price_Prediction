from datetime import datetime

import pandas as pd
from fredapi import Fred


def lambda_handler(event, context):
    pass


def get_fred_data(target: str,
                  start_year: int,
                  end_year: int,
                  apikey: str = '29219060bc68b2802af8584e0f328b52') -> dict:
    """To extract data from Fred database: https://fred.stlouisfed.org/ 
    apiKey = '29219060bc68b2802af8584e0f328b52'
    PWHEAMTUSDM - wheat https://fred.stlouisfed.org/series/PWHEAMTUSDM
    WPU0652013A - Ammonia https://fred.stlouisfed.org/series/WPU0652013A
    PNGASEUUSDM - TTG_Gas https://fred.stlouisfed.org/series/PNGASEUUSDM
    
    Returns: dict
        "statusCode": 200
        "body": df.to_json(orient='records', date_format='iso')
    """  # get_Fred_data.__doc__
    assert start_year <= end_year, 'start_year can not exceed end_year'
    assert end_year <= datetime.now().year, 'end_year can not include future date'

    target = target.upper()

    # Import monthly data
    _key = apikey
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
    return {
        "statusCode": 200,
        "body": df.to_json(orient='records', date_format='iso')
    }


def clean_elec_csv(file: str, start_year: int, end_year: int) -> dict:
    """
    To clean ELECTRICITY.csv correctly for following pre-processing steps

    Returns: dict
        "statusCode": 200
        "body": df.to_json(orient='records', date_format='iso')
    """
    assert start_year <= end_year, 'start_year can not exceed end_year'
    assert end_year <= datetime.now().year, 'end_year can not include future date'

    # Import monthly electricity data
    df = pd.read_csv(file).iloc[:, 1:]
    df['Time'] = pd.to_datetime(df['Year'].astype(str) + df['Month'].astype(str), format='%Y%m')
    df = df[df['Year'].between(start_year, end_year)].reset_index().drop('index', axis=1)

    assert df.isnull().values.any() == False, "Imported/Returned data contains NaN."
    return {
        "statusCode": 200,
        "body": df.to_json(orient='records', date_format='iso')
    }


def clean_pred_price_evo_csv(file: str, start_year: int, end_year: int) -> dict:
    """
    To clean Dataset_Predicting_Price_Evolutions.csv correctly for following pre-processing steps

    Returns: dict
        "statusCode": 200
        "body": df.to_json(orient='records', date_format='iso')
    """
    assert start_year <= end_year, 'start_year can not exceed end_year'
    assert end_year <= datetime.now().year, 'end_year can not include future date'

    # Import price evaluation data
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
    return {
        "statusCode": 200,
        "body": df.to_json(orient='records', date_format='iso')
    }
