import json

import numpy as np
import pandas as pd


def lambda_handler(event, context):
    try:
        # Parse inputs from event
        dummy_str = event.get("dummy_df", None)
        missing_str = event.get("missing", None)
        data = event.get("data", None)
        rm_code = event.get("rm_code", None)
        feature_duration_start = event.get("feature_duration_start", None)
        feature_duration_end = event.get("feature_duration_end", None)

        # Validate inputs
        required_fields = {
            "dummy_df": "Missing or invalid dummy_df",
            "missing": "Missing or invalid missing_df",
            "data": "Missing or invalid data",
            "rm_code": "Missing or invalid rm_code",
            "feature_duration_start": "Missing or invalid feature_duration_start",
            "feature_duration_end": "Missing or invalid feature_duration_end",
        }

        for field, error_message in required_fields.items():
            if event.get(field) is None:
                return {
                    "statusCode": 400,
                    "body": json.dumps({"input error": error_message})
                }

        # Handle dummy_df and missing_df
        dummy_json = json.loads(dummy_str)
        dummy_df = pd.DataFrame(data=dummy_json)
        dummy_df["Time"] = pd.to_datetime(dummy_df["Time"], format="ISO8601")

        missing_json = json.loads(missing_str)
        missing_df = pd.DataFrame(data=missing_json)


        # Get external price drivers
        external_drivers = {}
        for key in data.keys():
            if key != "price":
                current_df = pd.DataFrame(json.loads(data[key]))
                current_df["Time"] = pd.to_datetime(current_df["Time"], format="ISO8601")
                if "Unnamed: 0" in current_df.columns:
                    current_df.drop(["Unnamed: 0"], axis=1, inplace=True)

                external_drivers[f"{key}"] = current_df

        for key in external_drivers.keys():
            print(external_drivers.get(key).head())
        # Feature engineering
        # feature_df = generate_features(feature_duration_start, feature_duration_end, dummy_df, missing_df, *rm_code, **external_drivers)
        # print(feature_df.info())

        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Done"}),
        }

    except Exception as e:
        # Handle errors
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }


