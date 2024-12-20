import json


def lambda_handler(event, context):
    # Log the input payload
    print(f"Input event: {event}")

    # Process the input and prepare the output
    output = {
        "message": "Processed by lambda_function_2",
        "originalMessage": event.get("message", ""),
        "status": "Completed"
    }

    return output
