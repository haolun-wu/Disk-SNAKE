import pandas as pd
import json

# Path to the JSON file
file_path = "date_dataset.json"

# Read the JSON file
with open(file_path, 'r') as file:
    json_data = file.read()

# Convert JSON string to dictionary
data_dict = json.loads(json_data)

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(data_dict, orient='index')


# Save the DataFrame to a CSV file
df.to_csv('date.csv', index=False)

# Optionally, print the DataFrame to see the output
print(df)
