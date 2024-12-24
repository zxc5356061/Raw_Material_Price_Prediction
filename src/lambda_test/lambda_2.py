import json


def lambda_handler(event, context):
    try:
        # Parse inputs from event
        data = event.get("data", None)
        target = event.get("target", None)
        rm_code = event.get("rm_code", None)

        # Validate inputs
        if data is None:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "input error": "Missing or invalid data"
                })
            }

        if target is None:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "input error": "Missing or invalid target"
                })
            }

        if rm_code is None:
            return {
                "statusCode": 400,
                "body": json.dumps({
                    "input error": "Missing or invalid rm_code"
                })
            }

        # Log the received inputs
        print("Received data:", data)
        print("Received target:", target)
        print("Received rm_code:", rm_code)

        # Return success response
        return {
            "statusCode": 200,
            "body": json.dumps({"message": "Inputs received and processed successfully from Lambda_1."})
        }

    except Exception as e:
        # Handle errors
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
