import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer

# Load phishing dataset from Hugging Face
dataset = load_dataset('ealvaradob/phishing-dataset')

# Convert to pandas DataFrame for easier manipulation
df = pd.DataFrame(dataset['train'])

# Inspect columns and labels
print("Columns:", df.columns)
print("Label distribution:\n", df['label'].value_counts())

# Map or confirm label format (usually 0 for benign, 1 for phishing)
# Here label is already numeric, so no need for mapping

# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

print("Train sample:")
print(train_df.head())
# Load tokenizer (DistilBERT variant)
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize text data (emails, SMS, URLs, etc.)
def tokenize_texts(texts):
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )

train_encodings = tokenize_texts(train_df['text'])
test_encodings = tokenize_texts(test_df['text'])

print("Train encodings shape:", train_encodings['input_ids'].shape)

from datasets import load_dataset

# Load the phishing dataset from Hugging Face
dataset = load_dataset("ealvaradob/phishing-dataset")

# Check dataset info and splits
print(dataset)

# Preview some samples from the training set
print(dataset['train'][0:3])
