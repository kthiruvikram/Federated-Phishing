import warnings
from transformers import logging as hf_logging
import torch
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, Subset
import random

# Suppress warnings
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

def evaluate_accuracy(client_enc, client_labels, sample_size=500):
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('global_model.pt'))  # Load trained global model
    model.eval()
    dataset = TensorDataset(client_enc['input_ids'], client_enc['attention_mask'], client_labels)
    if len(dataset) < 128:
        return 0.0
    # Shuffle and split 80/20
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(dataset))
    val_indices = indices[split:]
    val_dataset = Subset(dataset, val_indices)
    loader = DataLoader(val_dataset, batch_size=16)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            outputs = model(input_ids=batch[0], attention_mask=batch[1])
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == batch[2]).sum().item()
            total += len(batch[2])
    return correct / total if total else 0.0

client_data = torch.load('non_iid_client_data.pt')
for idx, client in enumerate(client_data):
    acc = evaluate_accuracy(client, client['labels'])
    print(f"Client {idx}: Holdout Accuracy: {acc:.2f}")
