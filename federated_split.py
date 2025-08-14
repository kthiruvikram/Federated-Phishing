import torch
import pandas as pd
import numpy as np
from scipy.stats import dirichlet
from transformers import DistilBertTokenizer

def create_non_iid_splits(train_df, num_clients=10, alpha=0.5):
    """Create non-IID splits using Dirichlet distribution."""
    labels = train_df['label'].values
    client_indices = [[] for _ in range(num_clients)]
    
    for label in np.unique(labels):
        label_idx = np.where(labels == label)[0]
        proportions = dirichlet.rvs([alpha] * num_clients)[0]
        splits = (np.cumsum(proportions) * len(label_idx)).astype(int)[:-1]
        split_groups = np.split(np.random.permutation(label_idx), splits)
        for i, idxs in enumerate(split_groups):
            client_indices[i].extend(idxs.tolist())
    
    client_data = []
    for idxs in client_indices:
        sub_df = train_df.iloc[idxs].reset_index(drop=True)
        client_data.append(sub_df)
    return client_data

def compute_heterogeneity(client_data):
    """Compute label distribution variance across clients."""
    phishing_rates = []
    for df in client_dfs:
        rate = df['label'].mean() if len(df) > 0 else 0
        phishing_rates.append(rate)
    return float(np.std(phishing_rates))

# Load data
train_df = pd.read_csv('train.csv')

# Generate splits
client_dfs = create_non_iid_splits(train_df, num_clients=10, alpha=0.5)
hetero_score = compute_heterogeneity(client_dfs)

# Print summary
print(f"Non-IID Heterogeneity Score: {hetero_score:.4f}")
for i, df in enumerate(client_dfs):
    print(f"Client {i}: {len(df)} samples, Phishing ratio: {df['label'].mean():.2f}")

# Tokenize and save
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_df(df):
    if len(df) == 0:
        return {'input_ids': torch.empty(0), 'attention_mask': torch.empty(0), 'labels': torch.empty(0)}
    enc = tokenizer(df['text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
    labels = torch.tensor(df['label'].values)
    return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'], 'labels': labels}

client_encodings = [tokenize_df(df) for df in client_dfs]
torch.save(client_encodings, 'non_iid_client_data.pt')
print("Saved non-IID client data for federated training.")
