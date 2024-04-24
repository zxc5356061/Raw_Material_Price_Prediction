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
    
    Ex: draw_graph(acid_df_24,'year_month','Average_price','Predictions','2023-10','RM01/0001', 'RM01/0004', 'RM01/0006', 'RM01/0007')
    
    Return -> a line plot
    """
    ## Draw the RM code which is not in the columns
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
        sns.set_style("whitegrid")

        # Filter data for specific months
        filter_df = df[df[x_col].str.endswith(('3', '6', '9', '12'))]

        # Set x-axis ticks and labels
        plt.xticks(ticks=filter_df[x_col], labels=filter_df[x_col], rotation=45)

        # Display the plot
        plt.show()

    ## Draw other RM codes
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
        sns.set_style("whitegrid")

        # Filter data for specific months
        filter_df = df[df[x_col].str.endswith(('3', '6', '9', '12'))]

        # Set x-axis ticks and labels
        plt.xticks(ticks=filter_df[x_col], labels=filter_df[x_col], rotation=45)

        # Display the plot
        plt.show()
