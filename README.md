## TO-DOs
```
## Python 3.8
[] Import monthly average prices of external price drivers
[] Create rows and encoding with Key RM Codes, ex: Alkalis_RM02_0001, Alkalis_RM02_0002
[] To calculate the monthly average prices of the target variables
[] Create 12*N features, external factor prices from one-month before to 12-month before
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
```

| Group Description       | Price Driver                               |
|-------------------------|--------------------------------------------|
| Acid                    | Gas/Electricity/Ammonia/Wheat              |
| Alkalis                 | Gas/Electricity                            |
| Anionic Surfactants     | Gas/Electricity/Palm Oil/Ethylene oxide    |
| Bleaching Agents        | Gas/Electricity                            |
| Builder                 | Beets/Corn/Ethylene/Phosphate             |
| Fatty Acid              | Palm Oil/Transport                         |
| Non-ionic surfactants   | Gas/Electricity/Crude Oil/Palm Oil/Ammonia|
| Solvents                | Glycerol/Natural Oil                       |
