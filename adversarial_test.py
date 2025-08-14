import torch
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import random
import string

# Load your trained model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained('global_model')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def generate_adversarial_emails(original_emails, attack_type="typo"):
    adversarial_emails = []
    for email in original_emails:
        if attack_type == "typo":
            # Character substitution attack
            words = email.split()
            if len(words) > 3:  # Only modify longer emails
                idx = random.randint(0, len(words)-1)
                word = words[idx]
                if len(word) > 3:
                    char_idx = random.randint(1, len(word)-2)
                    new_char = random.choice(string.ascii_lowercase)
                    words[idx] = word[:char_idx] + new_char + word[char_idx+1:]
            adversarial_emails.append(" ".join(words))
        elif attack_type == "insertion":
            # Character insertion attack
            words = email.split()
            for i in range(len(words)):
                if random.random() < 0.1:  # 10% chance to modify each word
                    pos = random.randint(0, len(words[i]))
                    words[i] = words[i][:pos] + random.choice(string.ascii_lowercase) + words[i][pos:]
            adversarial_emails.append(" ".join(words))
    return adversarial_emails

def test_adversarial_robustness(model, original_emails, labels):
    # Test different attack types
    attack_types = ["typo", "insertion"]
    results = {}
    
    for attack in attack_types:
        adversarial_emails = generate_adversarial_emails(original_emails, attack)
        
        # Tokenize and predict
        encodings = tokenizer(adversarial_emails, padding=True, truncation=True, 
                            max_length=128, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=1)
            
        # Calculate accuracy drop
        correct = (predictions == labels).sum().item()
        accuracy = correct / len(labels)
        results[attack] = accuracy
        
    return results

# Test on subset of test data
test_df = pd.read_csv('test.csv').sample(200)
test_emails = test_df['text'].tolist()
test_labels = torch.tensor(test_df['label'].values)

robustness_results = test_adversarial_robustness(model, test_emails, test_labels)
print("Adversarial Robustness Results:")
for attack, accuracy in robustness_results.items():
    print(f"{attack.capitalize()} Attack Accuracy: {accuracy*100:.2f}%")
