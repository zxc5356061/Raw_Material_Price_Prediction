from io import StringIO
import pandas as pd

def json_to_dataframe(field: str, json_file:dict, date_col:str='Time'):
    """
    Convert a JSON file to a Pandas DataFrame and parse a specified date column.

    Parameters:
        json_file (dict): A dictionary with JSON string file under the 'body' key.
        date_col (str): The column name in the DataFrame to parse as datetime.

    Returns:
        pd.DataFrame: A DataFrame with the date column parsed, or None if an error occurs.

    Raises:
        ValueError: If required keys or data are missing.
        KeyError: If the specified date column is not found in the DataFrame.
        :param field:
    """
    try:
        # Ensure 'body' key exists in json_file
        if field not in json_file:
            raise ValueError(f"Missing {field} key in the provided JSON file.")

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
