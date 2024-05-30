import pandas as pd
import json
import numpy as np

# Path to the JSON file
file_path = "/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date/date_dataset.json"

# Read the JSON file
with open(file_path, 'r') as file:
    json_data = file.read()

# Convert JSON string to dictionary
data_dict = json.loads(json_data)

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(data_dict, orient='index')

# Save the DataFrame to a CSV file
df.to_csv("/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date/date_dataset.csv", index=False)


""" Set modeling for date """
# Selecting the first 9 columns
df = df.iloc[:, :9]
df.columns = ['date.{}'.format(str(i)) for i in range(len(df.columns))]
df.to_csv("/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date/date_dataset_order.csv", index=False)

def permute_rows(df):
    permuted_df = df.apply(lambda row: pd.Series(np.random.permutation(row)), axis=1)
    return permuted_df

# Apply the permutation function to the DataFrame
permuted_df = permute_rows(df)
permuted_df.columns = ['date.{}'.format(str(i)) for i in range(len(df.columns))]
permuted_df.to_csv("/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date/date_dataset_set.csv", index=False)



# # Optionally, print the DataFrame to see the output
# print(df)
