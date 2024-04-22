import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt


def naive_forest(df:pd.DataFrame, target:str, impute_list:pd.DataFrame):
    """
    To calculate monthly average prices per Key RM code and perform naive forecast
    
    Inputs:
    - df -> DataFrame with imputed(forward fill) target variable prices
    - target -> 'Group Description' of the target vaiable
    - impute_list -> second return of preprocessor.impute_pred_price_evo_csv(), the list of imputed rows
    
    Returns:
    - No return objects, the results of MAPE and MSE will be printed out.
    """

    target_df = df[(df['Group Description'] == target) & \
                 # To exclude records without real purchasing
                 # i.e. if there were purchases in Jan-2022 and Mar-2022, then we only consider the 
                    # performance of Naive forecast based on the predictions of Jan-2022 and Mar-2022,
                    # and exclude the imputed value in Feb-2022.
                 (~df[['Year','Month','Key RM code']].isin(impute_list).all(axis=1))]
    
    # To calculate monthly average prices of each RM code.
    avg_df = target_df.groupby(['Year','Month','Key RM code'])['PRICE (EUR/kg)']\
                    .mean()\
                    .to_frame()\
                    .reset_index()
    
    # Naive forecast
    avg_df['predicted_price'] = avg_df.groupby('Key RM code')['PRICE (EUR/kg)'].shift(periods=1)
    
    # Calculate MAPE and MSR
    # To make sure both y_true and y_pred have same shapes.
    na_rows = avg_df[avg_df['predicted_price'].isna()]
    true_filter = avg_df[(~avg_df[['Year','Month','Key RM code']].isin(na_rows).all(axis=1))]
    y_true = true_filter['PRICE (EUR/kg)']

    pred_filter = avg_df[avg_df['predicted_price'].notna()]
    y_pred = pred_filter['predicted_price']

    MAPE = mean_absolute_percentage_error(y_true,y_pred)
    MSR = mean_squared_error(y_true,y_pred)

    print(f"MAPE of Naive Forecast for {target}: {MAPE:.3f}")
    print(f"MSE of Naive Forecast for {target} {MSR:.3f}")
   
    
    # To match the predicted values with original df
    avg_df['year_month'] = avg_df['Year'].apply(round).astype('str') + "-" + avg_df['Month'].apply(round).astype('str')
    
    ## Draw all RM codes
    fig, ax = plt.subplots(figsize=[15,6])
    # Plot actual prices
    sns.lineplot(x=avg_df['year_month'], y=y_true, label='Actual_price',linestyle='dashed')
    
    # Plot predicted prices
    sns.lineplot(x=avg_df['year_month'], y=y_pred, label='Predicted_price')
    
    # Plot train-predict segamentation line
    ax.legend(loc='upper left')
    ax.set(title=f"Naive Forecast of {target}", ylabel='Price',xlabel='Time');
    
    # Filter data for specific months
    filter_df = avg_df[avg_df['year_month'].str.endswith(('3', '6', '9', '12'))]
    
    # Set x-axis ticks and labels
    plt.xticks(ticks=filter_df['year_month'], labels=filter_df['year_month'], rotation=45)
    
    # Display the plot
    plt.show()

