import pandas as pd
from transformers import DistilBertTokenizer
import torch
import warnings

# Mute warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load CSVs
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Batched tokenization function
def tokenize_texts_in_batches(texts, batch_size=500):
    batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
    all_input_ids = []
    all_attention_mask = []
    for batch in batches:
        encodings = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        all_input_ids.append(encodings['input_ids'])
        all_attention_mask.append(encodings['attention_mask'])
    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Tokenize full data
train_encodings = tokenize_texts_in_batches(train_df['text'])
test_encodings = tokenize_texts_in_batches(test_df['text'])

print("Train input_ids shape:", train_encodings['input_ids'].shape)
print("Test input_ids shape:", test_encodings['input_ids'].shape)

# Save for later (optional)
torch.save(train_encodings, 'train_encodings.pt')
torch.save(test_encodings, 'test_encodings.pt')
