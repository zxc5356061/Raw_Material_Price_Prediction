import pandas as pd

file_path = "/Users/huangp/Documents/Barry/sideproject/Raw_Material_Price_Prediction/data/raw/Dataset_Future_Predicting_Price_Evolutions_202403.csv"

df = pd.read_csv(file_path, index_col=0).dropna()

# print(df.info())

print(df[["Group Description", "Key RM code"]].groupby(["Group Description", "Key RM code"]).size())
