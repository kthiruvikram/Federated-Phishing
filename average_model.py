import torch
from transformers import DistilBertForSequenceClassification

def average_models(client_models):
    avg_state = client_models[0].state_dict()
    for key in avg_state.keys():
        for model in client_models[1:]:
            avg_state[key] += model.state_dict()[key]
        avg_state[key] /= len(client_models)
    global_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    global_model.load_state_dict(avg_state)
    return global_model

# Load client models
client_models = [DistilBertForSequenceClassification.from_pretrained(f'client_model_{i}') for i in range(10)]

# Average and save
global_model = average_models(client_models)
global_model.save_pretrained('global_model')
print("Global model aggregated and saved!")
