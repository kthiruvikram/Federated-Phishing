import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW

from load_data import train_encodings, test_encodings, train_df, test_df

# Labels as tensors
train_labels = torch.tensor(train_df['label'].values)
test_labels = torch.tensor(test_df['label'].values)

# Create Dataset and DataLoader
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_labels)
train_loader = DataLoader(train_dataset, batch_size=2)
test_loader = DataLoader(test_dataset, batch_size=1)

# Load model
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.train()

optimizer = AdamW(model.parameters(), lr=5e-5)

# One quick epoch of training
for batch in train_loader:
    input_ids, attention_mask, labels = batch
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print("Batch loss:", loss.item())

model.eval()
for test_batch in test_loader:
    input_ids, attention_mask, labels = test_batch
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        print(f"True: {labels.tolist()} | Predicted: {preds.tolist()}")
