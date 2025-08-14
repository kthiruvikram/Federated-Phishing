# Federated Phishing Detection System

## 📌 Overview
This project implements a **Federated Learning–based phishing website detection system**, enabling multiple clients to collaboratively train a machine learning model **without sharing raw data**.  
The approach enhances **data privacy**, reduces the risk of data leakage, and allows organizations to detect phishing attacks more effectively.

Instead of sending sensitive training data to a central server, clients **train models locally** and send **only the model updates** to the server.  
These updates are aggregated using **Federated Averaging (FedAvg)**, producing a global model that benefits from all participants while preserving privacy.

---

## ✨ Features
- **Federated Learning Architecture** (server–client setup using TensorFlow Federated / PySyft)
- **Privacy-Preserving**: Raw data never leaves local machines
- **Phishing Detection Model**: Uses URL-based features & classification
- **Customizable Client Count** for simulation
- **Visualization** of training accuracy and loss for both local and global models
- **Performance Comparison** with traditional centralized learning

---

## 📂 Project Structure
federated-phishing/
│── data/ # Dataset files (e.g., phishing URLs dataset)
│── models/ # Saved model weights
│── server.py # Server-side federated learning script
│── client.py # Client-side training script
│── utils.py # Helper functions for preprocessing & metrics
│── requirements.txt # Python dependencies
│── README.md # Project documentation
│── results/ # Output figures, accuracy/loss plots


---

## 🔬 Methodology
1. **Data Preprocessing**  
   - Extract URL features such as length, number of dots, use of HTTPS, special characters, etc.  
   - Label URLs as **phishing** or **legitimate**.

2. **Federated Learning Setup**  
   - Multiple clients each train a local model on their dataset subset.
   - The server aggregates the updates using **Federated Averaging**.
   - This process repeats for several communication rounds.

3. **Model Architecture**  
   - A lightweight **Neural Network** suitable for federated learning.
   - Input: URL-based feature vector.
   - Output: Binary classification (Phishing / Legitimate).

4. **Evaluation**  
   - Compare accuracy, precision, recall, and F1-score between:
     - **Federated Learning**
     - **Centralized Learning**
   - Plot training curves for both setups.

---

## 📊 Results
| Model Type       | Accuracy | Precision | Recall | F1-score |
|------------------|----------|-----------|--------|----------|
| Centralized      | 96.8%    | 96.5%     | 97.1%  | 96.8%    |
| Federated (5 Clients) | 95.3%    | 95.0%     | 95.6%  | 95.3%    |

---

