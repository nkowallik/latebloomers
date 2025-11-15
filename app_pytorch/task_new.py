"""app-pytorch: A Flower / PyTorch app."""

import torch
import os
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import json
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor

from sklearn.metrics import precision_score, recall_score, f1_score

class PositionalEncoding(torch.nn.Module):
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
        # x: [batch, seq_len, embed_dim]
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
        # x: [batch, seq_len, embed_dim]
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # GlobalAveragePooling1D
        out = self.classifier(x)
        return out


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch


def load_data(partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # Construct dataloaders
    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


def load_centralized_dataset():
    """Load test set and return dataloader."""
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=32)


def train_model(model, dataset, batch_size=64, epochs=1, lr=3e-4, device='cpu'):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for X_batch, Y_batch in loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
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
        return total_loss

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

def test(net, testloader, device, file_name, server_round):
    """Validate the model on the test set."""
    net.to(device)
    net.eval()  # make sure we're in eval mode

    criterion = torch.nn.CrossEntropyLoss()
    correct, loss_sum = 0, 0.0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)

            # loss and accuracy
            loss_sum += criterion(outputs, labels).item()
            preds = torch.max(outputs.data, 1)[1]
            correct += (preds == labels).sum().item()

            # store for precision/F1
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    # basic metrics
    accuracy = correct / len(testloader.dataset)
    loss = loss_sum / len(testloader)

    # stack all labels/preds and compute extra metrics
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    # multi-class: macro average; change to "binary" or "weighted" if needed
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    save_metrics_to_json(precision, recall, f1, accuracy, loss, file_name, server_round)

    return loss, accuracy
