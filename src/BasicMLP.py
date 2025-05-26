import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BasicMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, div_layer_value, device):
        super(BasicNN, self).__init__()
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.div_layer_value = div_layer_value

        self.network = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size // self.div_layer_value),
            nn.ReLU(),
            nn.Linear(self.hidden_size // self.div_layer_value, self.output_size)
        )

        self.val_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.device = device

    def forward(self, x):
        x = self.flatten(x)
        logits = self.network(x)
        return logits


    def fit(self, train_loader, val_loader, epochs, lr, early_stopping=False, patience=5):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.1 * lr)

        self.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self(data)
                loss = self.val_criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")


    def evaluate(self, data_loader, return_loss=False):
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.forward(x)
                loss = self.val_criterion(logits, y)

                batch_size = y.size(0)
                val_loss += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += batch_size

        avg_loss = val_loss / total
        acc = 100.0 * correct / total
        print(f"Validation — Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

        if return_loss:
            return avg_loss
        else:
            return acc