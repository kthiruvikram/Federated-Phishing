import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the JSON data
with open('texts.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)
print(f"Label counts:\n{df['label'].value_counts()}")
print(df.head())

# Split into train/test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train size: {len(train_df)}")
print(f"Test size: {len(test_df)}")

# Save splits if you want
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)
