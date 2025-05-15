import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------
# Modified CounterPropagation Network
# ---------------------------


class ModifiedCPNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, div_layer_value, output_size, dropout_rate, temperature, neighborhood_function='gaussian', neighborhood_size=3, device=None):
        """
        input_size: Dimensionality of the input (e.g. 28*28 for MNIST)
        hidden_size: Number of neurons in the Kohonen layer.
        output_size: Number of classes.
        """
        super(ModifiedCPNN, self).__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.div_layer_value = div_layer_value
        self.output_size = output_size

        self.kohonen_weights = nn.Parameter(
            torch.empty(self.hidden_size, self.input_size))

        # Kohonen and Grossberg layers should be initialized randomly
        self.kohonen_weights.data.normal_(0, 1)
        self.kohonen_weights.data = F.normalize(
            self.kohonen_weights.data, p=2, dim=1)

        layers = []

        for i in range(self.hidden_layers):
            layers.append(
                nn.Linear(hidden_size, hidden_size // self.div_layer_value))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            hidden_size //= self.div_layer_value

        layers.append(nn.Linear(hidden_size, self.output_size))

        self.mlp = nn.Sequential(*layers)

        self.temperature = temperature
        self.neighborhood_function = neighborhood_function
        self.neighborhood_size = neighborhood_size

        self.kohonen_snapshots = []

        self.val_criterion = nn.CrossEntropyLoss(reduction='mean')

        self.to(self.device)

    def forward(self, x):
        x = self.flatten(x)
        # x is expected to be of shape (batch_size, input_size)
        # Compute distance between x and each kohonen weight vector.
        # (Using Euclidean distance here.)
        # shape: (batch, hidden_size)
        distances = torch.cdist(x, self.kohonen_weights)

        winners = torch.argmin(distances, dim=1)             # (B,)

        # (batch, hidden_size)
        soft_activations = F.softmax(-distances / self.temperature, dim=1)

        output = self.mlp(soft_activations)
        return output, winners, x.size(0)

    def update_kohonen(self, x, winner_indices, batch_size, optimizer, neighborhood_size):

        x = self.flatten(x)

        # Create a tensor for hidden indices, shape (batch_size, hidden_size)
        hidden_indices = torch.arange(
            self.hidden_size, device=x.device, dtype=torch.float32)
        hidden_indices = hidden_indices.unsqueeze(0).expand(batch_size, -1)

        # Expand winner indices to match hidden_indices dimensions: (batch_size, 1)
        winner_indices_exp = winner_indices.unsqueeze(1).float()
        # (batch_size, hidden_size)
        diff_indices = torch.abs(hidden_indices - winner_indices_exp)

        # Compute the influence for all samples in a vectorized way
        if self.neighborhood_function == 'gaussian':
            influence = torch.exp(- (diff_indices ** 2) /
                                  (2 * (neighborhood_size ** 2)))
        elif self.neighborhood_function == 'rectangular':
            influence = (diff_indices <= neighborhood_size).float()
        elif self.neighborhood_function == 'triangular':
            influence = torch.clamp(
                1 - diff_indices / neighborhood_size, min=0)
        else:
            influence = torch.exp(- (diff_indices ** 2) /
                                  (2 * (neighborhood_size ** 2)))
        x_expanded = x.unsqueeze(1)
        weights_expanded = self.kohonen_weights.unsqueeze(0)
        # (batch_size, hidden_size, input_size)
        diff = x_expanded - weights_expanded

        # Unsqueeze influence to match the diff dimensions: (batch_size, hidden_size, 1)
        influence = influence.unsqueeze(2)
        # Compute the per-sample gradients and then average over the batch
        grad = - influence * diff  # (batch_size, hidden_size, input_size)
        grad_accum = grad.mean(dim=0)  # (hidden_size, input_size)

        optimizer.zero_grad()
        self.kohonen_weights.grad = grad_accum
        optimizer.step()

        self.kohonen_weights.data = F.normalize(
            self.kohonen_weights.data, p=2, dim=1)

    def train_mlp(self, logits, y, optimizer):

        loss = self.val_criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def fit(self, train_loader, val_loader=None, epochs=20,
            kohonen_lr=0.01, mlp_lr=0.01,
            early_stopping=False, patience=None):

        optimizer_kh = optim.SGD(
            [self.kohonen_weights], lr=kohonen_lr, weight_decay=1e-4, momentum=0.95, nesterov=True)
        optimizer_mlp = optim.AdamW(self.mlp.parameters(), lr=mlp_lr)

        scheduler_kh = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_kh,
            T_max=epochs,
            eta_min=kohonen_lr * 0.1
        )

        scheduler_mlp = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_mlp,
            T_max=epochs,
            eta_min=mlp_lr * 0.1
        )

        sigma_0 = min(self.neighborhood_size, self.hidden_size/2)
        lambd = epochs / math.log(sigma_0)

        best_val_loss = float('inf')
        init_patience = patience

        for epoch in range(1, epochs + 1):

            sigma_t = sigma_0 * math.exp(-epoch / lambd)

            print(f"\n[Epoch {epoch}] Starting training...")

            self.train()

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device)

                output, winner_indices, batch_size = self.forward(batch_x)

                # Profile Grossberg update
                self.train_mlp(output, batch_y, optimizer_mlp)

                # Profile Kohonen update
                self.update_kohonen(batch_x, winner_indices, batch_size, optimizer_kh,
                                    neighborhood_size=sigma_t)

            scheduler_kh.step()
            scheduler_mlp.step()

            if val_loader is not None:
                val_loss = self.evaluate(val_loader, return_loss=True)
                print(f"[Epoch {epoch}] Validation loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = init_patience
                else:
                    if early_stopping:
                        if math.isnan(val_loss):
                            print("Early stopping triggered due to NaN loss.")
                            break
                        patience -= 1
                        if patience == 0:
                            print("Early stopping triggered.")
                            break

            with torch.no_grad():
                self.kohonen_snapshots.append(
                    self.kohonen_weights.cpu().clone())

        return self

    def evaluate(self, data_loader, return_loss=False):
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits, _, _ = self.forward(x)
                loss = self.val_criterion(logits, y)

                # accumulate
                batch_size = y.size(0)
                val_loss += loss.item() * batch_size  # sum of per-sample loss
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
