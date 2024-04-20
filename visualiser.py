import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def draw_graph(df:pd.DataFrame, x_col:str, y_col_actual:str, y_col_pred:str, prediction_cut:str, *args:str):
    """
    To draw a line graph based on provided parameters.
    
    Inputs ->
    df: Input data should be a Pandas DataFrame
    x_col: the column name indicating time in 'Year-Month' format. 
    y_col_actual: the column name indicating actual target variable prices
    y_col_pred: the column name indicating predicted target variable prices
    prediction_cut: the time point to distinguish trained data and predicted data
    *args: Key RM Codes corresponding to the target variable, start from the smallest number
    
    Return -> a line plot
    """
    # Draw all RM codes
    fig, ax = plt.subplots(figsize=[15,6])
    # Plot actual prices
    sns.lineplot(x=x_col, y=y_col_actual, data=df, label='Actual_price',linestyle='dashed',ax=ax)
    
    # Plot predicted prices
    sns.lineplot(x=x_col, y=y_col_pred, data=df, label='Predicted_price',ax=ax)
    
    # Plot train-predict segamentation line
    ax.axvline(x=prediction_cut, color='red', linestyle='--', label='Future predictions')
    ax.legend(loc='upper left')
    ax.set(title='Forecast_all_RM_codes', ylabel='Price',xlabel='Time');
    
    # Filter data for specific months
    filter_df = df[df[x_col].str.endswith(('3', '6', '9', '12'))]
    
    # Set x-axis ticks and labels
    plt.xticks(ticks=filter_df[x_col], labels=filter_df[x_col], rotation=45)
    
    # Display the plot
    plt.show()
    
    # Draw RM code which is not in the columns
    # Warn: unable to ensure all input RM codes are correct!
    with_dummy = [arg for arg in args if arg in df.columns]
    if [arg for arg in args] not in with_dummy:
        # eval() -> to evaluate the string as a boolean expression, then filt
        code_filters = eval("&".join([f"(df['{i}'] == 0)" for i in with_dummy]))  # <class 'pd.Series'>
        # print(type(filter_list)) 
        fig, ax = plt.subplots(figsize=[15, 6])
        # Plot actual prices
        sns.lineplot(x=x_col, y=y_col_actual, data=df[code_filters], label='Actual_price', linestyle='dashed', ax=ax)

        # Plot predicted prices
        sns.lineplot(x=x_col, y=y_col_pred, data=df[code_filters], label='Predicted_price', ax=ax)

        # Plot train-predict segamentation line
        ax.axvline(x=prediction_cut, color='red', linestyle='--', label='Future predictions')
        ax.legend(loc='upper left')
        ax.set(title=next(arg for arg in args if arg not in with_dummy), ylabel='Price', xlabel='Time');

        # Filter data for specific months
        filter_df = df[df[x_col].str.endswith(('3', '6', '9', '12'))]

        # Set x-axis ticks and labels
        plt.xticks(ticks=filter_df[x_col], labels=filter_df[x_col], rotation=45)

        # Display the plot
        plt.show()

    # Draw other RM codes
    for code in with_dummy:
        fig, ax = plt.subplots(figsize=[15, 6])
        # Plot actual prices
        sns.lineplot(x=x_col, y=y_col_actual, data=df[df[code] == 1], label='Actual_price', linestyle='dashed', ax=ax)

        # Plot predicted prices
        sns.lineplot(x=x_col, y=y_col_pred, data=df[df[code] == 1], label='Predicted_price', ax=ax)

        # Plot train-predict segamentation line
        ax.axvline(x=prediction_cut, color='red', linestyle='--', label='Future predictions')
        ax.legend(loc='upper left')
        ax.set(title=code, ylabel='Price', xlabel='Time');

        # Filter data for specific months
        filter_df = df[df[x_col].str.endswith(('3', '6', '9', '12'))]

        # Set x-axis ticks and labels
        plt.xticks(ticks=filter_df[x_col], labels=filter_df[x_col], rotation=45)

        # Display the plot
        plt.show()

