from io import StringIO
import pandas as pd

def json_to_dataframe(json_file, date_col:str):
    try:
        # Ensure 'body' key exists in json_file
        if 'body' not in json_file:
            raise ValueError("Missing 'body' key in the provided JSON file.")

        df = pd.read_json(StringIO(json_file['body']), orient="records")

        if date_col not in df.columns:
            raise KeyError(f"Column '{date_col}' not found in the DataFrame.")

        if df[date_col].isnull().any():
            raise ValueError(f"Invalid datetime format detected in column '{date_col}'.")
        df[date_col] = pd.to_datetime(df[date_col], format="ISO8601")

        return df

    except ValueError as ve:
        print(f"ValueError: {ve}")
    except KeyError as ke:
        print(f"KeyError: {ke}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return None
