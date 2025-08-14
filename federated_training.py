import torch
import random
from torch.utils.data import Subset
from transformers import DistilBertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import entropy  # For KL divergence; pip install scipy if needed
import numpy as np

def federated_round(client_data, global_model, noise_multiplier=0.1):
    global_sd = global_model.state_dict()
    client_updates = []
    privacy_tradeoffs = []  # Store per-client epsilon estimates

    for idx, client in enumerate(client_data):
        if len(client['input_ids']) < 128:
            continue

        # Load model for this client
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=2
        )
        model.load_state_dict(global_sd)

        # Random sampling for speed
        dataset = TensorDataset(client['input_ids'], client['attention_mask'], client['labels'])
        indices = random.sample(range(len(dataset)), min(500, len(dataset)))
        dataset = Subset(dataset, indices)

        loader = DataLoader(dataset, batch_size=16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

        # Train with manual DP
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
            loss = outputs.loss
            loss.backward()

            # Add Gaussian noise to gradients (local DP)
            for param in model.parameters():
                if param.grad is not None:
                    param.grad += torch.randn_like(param.grad) * noise_multiplier

            optimizer.step()

        client_updates.append(model.state_dict())

        # Simple epsilon estimate
        local_eps = noise_multiplier ** 2 / len(dataset)
        privacy_tradeoffs.append(round(local_eps, 4))

    # Average client models to update global model
    if client_updates:
        avg_sd = {
            k: torch.mean(torch.stack([sd[k] for sd in client_updates]), dim=0)
            for k in global_sd
        }
        global_model.load_state_dict(avg_sd)

    print(f"Per-client epsilon estimates: {privacy_tradeoffs}")

    # Measure drift
    drifts = []
    for sd in client_updates:
        drift = 0
        for k in global_sd:
            global_flat = global_sd[k].flatten().detach().cpu().numpy()
            client_flat = sd[k].flatten().detach().cpu().numpy()
            # Normalize to probabilities (add epsilon to avoid zeros)
            global_prob = np.abs(global_flat) + 1e-10
            global_prob /= global_prob.sum()
            client_prob = np.abs(client_flat) + 1e-10
            client_prob /= client_prob.sum()
            drift += entropy(global_prob, client_prob)
        drifts.append(round(drift / len(global_sd), 4))
    print(f"Per-client drifts: {drifts}")

    return global_model

# ========= Run the FL training ========= #

global_model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased', num_labels=2
)

client_data = torch.load('non_iid_client_data.pt')

for r in range(5):  # 5 rounds
    global_model = federated_round(client_data, global_model, noise_multiplier=0.1)
    print(f"Round {r+1} complete.")

torch.save(global_model.state_dict(), 'global_model.pt')
print("Global model saved to global_model.pt")
