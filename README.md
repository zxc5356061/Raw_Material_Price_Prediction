## Approaches
```
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

| Group Description       | Price Driver                               |  Size  |
|-------------------------|--------------------------------------------|--------|
| Acid                    | Gas/Electricity/Ammonia/Wheat              | 3255   |
| Alkalis                 | Gas/Electricity                            | 6181   |
| Anionic Surfactants     | Gas/Electricity/Palm Oil/Ethylene oxide    | 1158   |
| Bleaching Agents        | Gas/Electricity                            | 4153   |
| Builder                 | Beets/Corn/Ethylene/Phosphate             |   1771  |
| Fatty Acid              | Palm Oil/Transport                         |  689   |
| Non-ionic surfactants   | Gas/Electricity/Crude Oil/Palm Oil/Ammonia| 3220    |
| Solvents                | Glycerol/Natural Oil                       |  710   |
