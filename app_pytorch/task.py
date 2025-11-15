import torch
import os
import math
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score
import json

# -----------------------
# Model Definitions
# -----------------------

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class NeuralLogTransformer(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, ff_dim=2048, num_layers=1, max_len=75, num_classes=2, dropout=0.1):
        super().__init__()
        self.pos_encoding = PositionalEncoding(embed_dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                   dim_feedforward=ff_dim,
                                                   batch_first=True,
                                                   dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # GlobalAveragePooling1D
        out = self.classifier(x)
        return out

# -----------------------
# Data Loading
# -----------------------

def load_data(partition_file: str):
    """
    Load a client HDFS partition from a .pt file.
    Each file contains 'features' and 'labels' tensors.
    """
    data_dict = torch.load(partition_file)
    features = data_dict['features']  # [N, seq_len, embed_dim]
    labels = data_dict['labels']      # [N]

    dataset = TensorDataset(features, labels)

    # Split 80% train, 20% test
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_dataset, test_dataset = random_split(dataset, [n_train, n_test])

    trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=32)

    return trainloader, testloader

def load_centralized_dataset_from_pt(pt_file: str):
    """Load the full HDFS dataset from a .pt file for testing."""
    data_dict = torch.load(pt_file)
    dataset = TensorDataset(data_dict['features'], data_dict['labels'])
    return DataLoader(dataset, batch_size=32)

# -----------------------
# Training & Testing
# -----------------------

def train_model(model, trainloader, epochs=1, lr=3e-4, device='cpu'):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X_batch, Y_batch in trainloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == Y_batch).sum().item()
            total += X_batch.size(0)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/total:.4f}, Accuracy: {correct/total:.4f}")

def save_metrics_to_json(precision, recall, f1, accuracy, loss, filepath, iteration):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[str(iteration)] = {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def test(model, testloader, device, file_name, server_round):
    """Evaluate the model on HDFS tensor dataset."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct, loss_sum = 0, 0.0

    all_labels, all_preds = [], []

    with torch.no_grad():
        for X_batch, Y_batch in testloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            outputs = model(X_batch)

            loss_sum += criterion(outputs, Y_batch).item()
            preds = outputs.argmax(dim=1)
            correct += (preds == Y_batch).sum().item()

            all_labels.append(Y_batch.cpu())
            all_preds.append(preds.cpu())

    accuracy = correct / len(testloader.dataset)
    loss = loss_sum / len(testloader)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    save_metrics_to_json(precision, recall, f1, accuracy, loss, file_name, server_round)
    return loss, accuracy
