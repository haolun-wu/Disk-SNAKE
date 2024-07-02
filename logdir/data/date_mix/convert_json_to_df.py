import pandas as pd
import json
import numpy as np

# Path to the JSON file
file_path = (
    "/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_dataset.json"
)

# Read the JSON file
with open(file_path, "r") as file:
    json_data = file.read()

# Convert JSON string to dictionary
data_dict = json.loads(json_data)

# Remove the "Month DDth, YYYY" key from each entry, due to redundancy
for key, value in data_dict.items():
    if "Month DDth, YYYY" in value:
        del value["Month DDth, YYYY"]

# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(data_dict, orient="index")

# Reorder columns to have "Year", "Month", and "Day" as the first three columns
columns_order = ["Year", "Month", "DD"] + [col for col in df.columns if col not in ["Year", "Month", "DD"]]
df = df[columns_order]

# Save the DataFrame to a CSV file
df.to_csv(
    "/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_dataset.csv",
    index=False,
)

""" Set modeling for date """
# Selecting the first 11 columns (first three + next eight)
df = df.iloc[:, :11]
df.columns = ["date.{}".format(str(i)) for i in range(len(df.columns))]
df.to_csv(
    "/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_dataset_order.csv",
    index=False,
)

def permute_rows(df):
    # Separate the first three columns
    fixed_columns = df.iloc[:, :3]
    permutable_columns = df.iloc[:, 3:]
    
    # Permute the remaining columns
    # permuted_columns = permutable_columns.apply(lambda row: np.random.permutation(row), axis=1)
    permuted_columns = permutable_columns.apply(lambda row: list(np.random.permutation(row)), axis=1)

    
    # Concatenate the fixed and permuted columns back together
    permuted_df = pd.concat([fixed_columns, permuted_columns], axis=1)
    
    return permuted_df

# Apply the permutation function to the DataFrame
permuted_df = permute_rows(df)
permuted_df.columns = ["date.{}".format(str(i)) for i in range(len(permuted_df.columns))]
permuted_df.to_csv(
    "/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_dataset_set_mix.csv",
    index=False,
)


# # Optionally, print the DataFrame to see the output
print(permuted_df)
