import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset

# Load global model and test data
model = DistilBertForSequenceClassification.from_pretrained('global_model')
test_encodings = torch.load('test_encodings.pt')
test_labels = torch.tensor(pd.read_csv('test.csv')['label'].values)

# Prep loader
dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)
loader = DataLoader(dataset, batch_size=16)

# Evaluate
model.eval()
correct = 0
with torch.no_grad():
    for batch in loader:
        outputs = model(input_ids=batch[0], attention_mask=batch[1])
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == batch[2]).sum().item()
accuracy = correct / len(test_labels)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
