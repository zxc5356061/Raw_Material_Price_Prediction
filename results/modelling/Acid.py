import numpy as np
import pandas as pd

from src import (forecastor as fc)

# Import processed data
target = 'acid'.lower()

rm_codes = ['RM01/0001', 'RM01/0004', 'RM01/0006', 'RM01/0007']

feature_df = pd.read_csv(
    "~/Documents/Barry/sideproject/Raw_Material_Price_Prediction/results/generated_feature/acid_feature.csv")

test_periods = [
    ('2019-01-01', '2019-07-01'),
    ('2019-07-01', '2020-01-01'),
    ('2020-01-01', '2020-07-01'),
    ('2020-07-01', '2021-01-01'),
    ('2021-01-01', '2021-07-01'),
    ('2021-07-01', '2022-01-01'),
    ('2022-01-01', '2022-07-01'),
    ('2022-07-01', '2023-01-01'),
    ('2023-01-01', '2023-07-01'),
    ('2023-07-01', '2024-01-01')
]

lags = [1, 3, 6]

alpha_bottom = 0.01

# Persistent Naive
for code in rm_codes:
    for lag in lags:
        mape_values = list()
        for period in test_periods:
            result = fc.persistence_Naive_MAPE(feature_df, code, lag, period)
            mape_values.append(result)

        assert len(mape_values) == len(test_periods), "len(mape_values)!=len(test_periods)"
        average_mape = np.mean(mape_values)
        print(f"{target} {code}, {lag}-month lag, Naive, average MAPE: {average_mape:.3f}")

# Lasso with autoregression features only
for code in rm_codes:
    for lag in lags:
        mape_values = list()
        for period in test_periods:
            result, coef = fc.train_model_AR(raw_df=feature_df,
                                       code=code,
                                       lag=lag,
                                       test_periods=period,
                                       alpha_bottom=alpha_bottom,
                                       return_coef=True)
            mape_values.append(result)
            print(coef)

        assert len(mape_values) == len(test_periods), "len(mape_values)!=len(test_periods)"
        average_mape = np.mean(mape_values)
        print(f"{target} {code}, {lag}-month lag, AR, average MAPE: {average_mape:.3f}")

# Lasso with autoregression features and external price drivers
for code in rm_codes:
    for lag in lags:
        mape_values = list()
        for period in test_periods:
            result, coef = fc.train_model_all_features(raw_df=feature_df,
                                                           code=code,
                                                           lag=lag,
                                                           test_periods=period,
                                                           alpha_bottom=alpha_bottom,
                                                           return_coef=True)
            mape_values.append(result)
            print(coef)

        assert len(mape_values) == len(test_periods), "len(mape_values)!=len(test_periods)"
        average_mape = np.mean(mape_values)
        print(f"{target} {code}, {lag}-month lag, all features, average MAPE: {average_mape:.3f}")
