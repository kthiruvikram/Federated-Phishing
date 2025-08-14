import torch
import sys
import pickle
from transformers import DistilBertForSequenceClassification

def measure_communication_cost(model_state_dict):
    return sys.getsizeof(pickle.dumps(model_state_dict)) / (1024 * 1024)  # Size in MB

def gradient_compression(gradients, ratio=0.5):
    compressed = {}
    total_params = kept_params = 0
    for name, grad in gradients.items():
        if grad is not None:
            flat = grad.flatten()
            total_params += len(flat)
            k = int(len(flat) * ratio)
            _, indices = torch.topk(torch.abs(flat), k)
            comp_flat = torch.zeros_like(flat)
            comp_flat[indices] = flat[indices]
            compressed[name] = comp_flat.reshape(grad.shape)
            kept_params += k
    return compressed, 1 - (kept_params / total_params) if total_params else 0

def analyze_communication(client_data, ratios=[1.0, 0.5, 0.1]):
    results = {}
    for r in ratios:
        total_cost = 0
        for enc in client_data[:3]:  # Test 3 clients
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
            dummy_grads = {n: torch.randn_like(p) * 0.01 for n, p in model.named_parameters() if p.requires_grad}
            if r < 1.0:
                comp_grads, comp_achieved = gradient_compression(dummy_grads, r)
                cost = measure_communication_cost(comp_grads)
            else:
                cost = measure_communication_cost(dummy_grads)
                comp_achieved = 0
            total_cost += cost
        results[r] = {'cost_mb': total_cost, 'compression': comp_achieved}
    return results

# Load and run
client_data = torch.load('non_iid_client_data.pt')
comm_results = analyze_communication(client_data)
print("Communication Efficiency:")
for r, m in comm_results.items():
    print(f"Ratio {r}: Cost {m['cost_mb']:.2f} MB, Compression {m['compression']:.2%}")
