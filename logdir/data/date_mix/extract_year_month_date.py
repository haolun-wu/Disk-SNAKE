import pandas as pd
from dateutil import parser

# Sample data
file_path = (
    "/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_dataset_set.json"
)

# Read the JSON file
with open(file_path, "r") as file:
    df = file.read()


# Create DataFrame
# df = pd.DataFrame(data, columns=[f"date.{i}" for i in range(8)])

# Function to extract year, month, day
def extract_ymd(date_str):
    try:
        date = parser.parse(date_str, fuzzy=True)
        return date.year, date.month, date.day
    except:
        return None, None, None

# Extract year, month, day from the first column (assuming the first column always has a valid date)
df['year'], df['month'], df['day'] = zip(*df['date.0'].map(extract_ymd))

# Reorder columns
df = df[['year', 'month', 'day'] + [f"date.{i}" for i in range(8)]]

# Rename columns
df.columns = ['year', 'month', 'day'] + [f"date.{i}" for i in range(9)]

# Save to CSV
df.to_csv('/home/mila/h/haolun.wu/projects/Disk-SNAKE/logdir/data/date_mix/date_dataset_set_mix.csv', index=False)

print(df)
