import pandas as pd

# Load the data
file_path = "/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_dataset_set_mix.csv"
df = pd.read_csv(file_path)

# Convert the 'string' column from a string representation of a list to an actual list
df['string'] = df['string'].apply(eval)

# Initialize a list to store the new data
new_data = []

# Loop over the rows and create new samples with varying string list lengths
for idx, row in df.iterrows():
    # Determine the current sample length
    length = (idx // 1400) + 1
    new_row = row.copy()
    new_row['string'] = row['string'][:length]
    new_data.append(new_row)

# Create a new DataFrame from the new data
new_df = pd.DataFrame(new_data)

# Split the data into train, val, and test sets
train_data = []
val_data = []
test_data = []

for length in range(1, 9):
    subset = new_df[new_df['string'].apply(len) == length]
    train_data.append(subset.iloc[:1000])
    val_data.append(subset.iloc[1000:1200])
    test_data.append(subset.iloc[1200:1400])

train_df = pd.concat(train_data).sample(frac=1).reset_index(drop=True) # shuffle the training set
val_df = pd.concat(val_data)
test_df = pd.concat(test_data)

# Save the datasets to CSV files
train_df.to_csv("/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_mix_train.csv", index=False)
val_df.to_csv("/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_mix_val.csv", index=False)
test_df.to_csv("/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_mix_test.csv", index=False)

print("Data preprocessing and splitting completed successfully.")
