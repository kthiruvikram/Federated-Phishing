import torch
from transformers import DistilBertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

# Load client data
client_data = torch.load('client_data.pt')

# Train loop for all clients
for idx, (client_enc, client_labels) in enumerate(client_data):
    # Prep dataset and loader
    dataset = TensorDataset(client_enc['input_ids'], client_enc['attention_mask'], client_labels)
    loader = DataLoader(dataset, batch_size=16)
    
    # Load fresh model per client (or load global if aggregating rounds)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Train one epoch
    model.train()
    for batch in loader:
        outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print(f"Trained client {idx}! Final loss: {loss.item()}")
    model.save_pretrained(f'client_model_{idx}')
