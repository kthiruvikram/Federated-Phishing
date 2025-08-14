import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

# Step 1: Load CSV data
df = pd.read_csv('enron-data/emails.csv')  # Adjust path/filename if needed

# Step 2: Convert labels ('ham' and 'spam') to numbers
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Optional: Drop any rows with missing values
df = df.dropna(subset=['text', 'label'])

# Step 3: Split into train & test (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print("Train ")
print(train_df.head())
print("\nTest ")
print(test_df.head())

# Step 4: Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Step 5: Tokenize text (email body)
def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,  # Can increase for longer emails
        return_tensors='pt'
    )

train_encodings = tokenize_texts(train_df['text'])
test_encodings = tokenize_texts(test_df['text'])

print("Train encodings shape:", train_encodings['input_ids'].shape)
