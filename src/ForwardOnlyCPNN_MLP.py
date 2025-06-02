import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------
# Modified CounterPropagation Network
# ---------------------------


class ExtendedForwardCPNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, div_layer_value,
                 output_size, dropout_rate, temperature,
                 neighborhood_function='gaussian', neighborhood_size=3, device=None):
        """
        Initialize the Modified CounterPropagation Neural Network.

        Args:
            input_size (int): Dimensionality of the input.

            hidden_size (int): Number of neurons in the Kohonen layer.

            hidden_layers (int): Number of layers in the MLP after Kohonen.

            div_layer_value (int): Factor to reduce the number of neurons 
            in each successive MLP layer.

            output_size (int): Number of output classes.

            dropout_rate (float): Dropout probability between MLP layers.

            temperature (float): Temperature for softmax in 
            Kohonen layer activation.

            neighborhood_function (str): Type of neighborhood function 
            ('gaussian', 'rectangular', 'triangular').

            neighborhood_size (float): Initial size of the 
            neighborhood function.

            device (torch.device, optional): Device to use ('cuda' or 'cpu').
        """
        super(ExtendedForwardCPNN, self).__init__()

        self.device = device or torch.device("cuda"
                                             if torch.cuda.is_available() else "cpu")

        assert input_size > 0, "Input size must be > 0"
        assert hidden_size > 0, "Hidden size must be > 0"
        assert div_layer_value >= 1, "div_layer_value must be >= 1"
        assert output_size > 0, "Output size must be > 0"
        assert hidden_layers >= 0, "Must have >= 0 hidden layers"

        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.div_layer_value = div_layer_value
        self.output_size = output_size

        self.kohonen_weights = nn.Parameter(torch.empty(self.hidden_size,
                                                        self.input_size))

        # Randomly initialize and normalize Kohonen weights
        self.kohonen_weights.data.normal_(0, 1)
        self.kohonen_weights.data = F.normalize(self.kohonen_weights.data,
                                                p=2, dim=1)

        # Build the MLP after the Kohonen layer
        layers = []
        for i in range(self.hidden_layers):
            new_size = max(1, hidden_size // self.div_layer_value)
            layers.append(nn.Linear(hidden_size,
                                    new_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            hidden_size = new_size

        layers.append(nn.Linear(hidden_size, self.output_size))
        self.mlp = nn.Sequential(*layers)

        self.temperature = temperature
        self.neighborhood_function = neighborhood_function
        self.neighborhood_size = neighborhood_size

        if self.neighborhood_size < 1:
            self.neighborhood_size = 1

        self.kohonen_snapshots = []

        self.val_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            output (Tensor): Output logits of shape (batch_size, output_size).

            winners (Tensor): Index of winning Kohonen neurons for each input.

            batch_size (int): Size of the current batch.
        """
        x = self.flatten(x)
        distances = torch.cdist(x, self.kohonen_weights)
        winners = torch.argmin(distances, dim=1)
        soft_activations = F.softmax(-distances / self.temperature, dim=1)
        output = self.mlp(soft_activations)
        return output, winners, x.size(0)

    def update_kohonen(self, x, winner_indices, batch_size,
                       optimizer, neighborhood_size):
        """
        Update the Kohonen weights using neighborhood-based learning.

        Args:
            x (Tensor): Input batch tensor.

            winner_indices (Tensor): Index of winning neurons.

            batch_size (int): Number of samples in the batch.

            optimizer (Optimizer): Optimizer for Kohonen weights.

            neighborhood_size (float): Current neighborhood size (sigma_t).
        """

        x = self.flatten(x)
        hidden_indices = torch.arange(self.hidden_size, device=x.device,
                                      dtype=torch.float32)
        hidden_indices = hidden_indices.unsqueeze(0).expand(batch_size, -1)
        winner_indices_exp = winner_indices.unsqueeze(1).float()
        diff_indices = torch.abs(hidden_indices - winner_indices_exp)

        # Neighborhood influence
        if self.neighborhood_function == 'gaussian':
            influence = torch.exp(-(diff_indices ** 2) /
                                  (2 * (neighborhood_size ** 2)))
        elif self.neighborhood_function == 'rectangular':
            influence = (diff_indices <= neighborhood_size).float()
        elif self.neighborhood_function == 'triangular':
            influence = torch.clamp(1 - diff_indices /
                                    neighborhood_size, min=0)
        else:
            influence = torch.exp(-(diff_indices ** 2) /
                                  (2 * (neighborhood_size ** 2)))

        x_expanded = x.unsqueeze(1)
        weights_expanded = self.kohonen_weights.unsqueeze(0)
        diff = x_expanded - weights_expanded
        influence = influence.unsqueeze(2)

        # Compute batch gradient
        grad = - influence * diff
        grad_accum = grad.mean(dim=0)

        optimizer.zero_grad()
        self.kohonen_weights.grad = grad_accum
        optimizer.step()

        # Normalize weights after update
        self.kohonen_weights.data = F.normalize(self.kohonen_weights.data,
                                                p=2, dim=1)

    def train_mlp(self, logits, y, optimizer):
        """
        Train the MLP using cross-entropy loss.

        Args:
            logits (Tensor): Predicted logits.

            y (Tensor): Ground truth labels.

            optimizer (Optimizer): Optimizer for the MLP.
        """

        loss = self.val_criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def fit(self, train_loader, val_loader=None, epochs=20,
            kohonen_lr=0.01, mlp_lr=0.01,
            early_stopping=False, patience=None):
        """
        Train the ModifiedCPNN on the given data.

        Args:
            train_loader (DataLoader): Training dataset loader.

            val_loader (DataLoader, optional): Validation dataset loader.

            epochs (int): Number of training epochs.

            kohonen_lr (float): Learning rate for Kohonen weights.

            mlp_lr (float): Learning rate for MLP.

            early_stopping (bool): Whether to use early stopping.

            patience (int, optional): Patience for early stopping.

        Returns:
            self: Trained model.
        """

        if len(train_loader) == 0:
            raise ValueError("Training loader is empty.")

        optimizer_kh = optim.SGD([self.kohonen_weights], lr=kohonen_lr,
                                 weight_decay=1e-4, momentum=0.95, nesterov=True)
        optimizer_mlp = optim.AdamW(self.mlp.parameters(), lr=mlp_lr)

        scheduler_kh = optim.lr_scheduler.CosineAnnealingLR(optimizer_kh,
                                                            T_max=epochs, eta_min=kohonen_lr * 0.1)
        scheduler_mlp = optim.lr_scheduler.CosineAnnealingLR(optimizer_mlp,
                                                             T_max=epochs, eta_min=mlp_lr * 0.1)

        sigma_0 = min(self.neighborhood_size, self.hidden_size / 2)
        if sigma_0 < 1:
            sigma_0 = 1

        decay_enabled = sigma_0 > 1

        if decay_enabled:
            lambd = epochs / math.log(sigma_0)

        best_val_loss = float('inf')
        init_patience = patience

        for epoch in range(1, epochs + 1):

            if decay_enabled:
                sigma_t = max(1.0, sigma_0 * math.exp(-epoch / lambd))
            else:
                sigma_t = sigma_0

            print(f"\n[Epoch {epoch}] Starting training...")

            self.train()

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device)
                output, winner_indices, batch_size = self.forward(batch_x)
                self.train_mlp(output, batch_y, optimizer_mlp)
                self.update_kohonen(batch_x, winner_indices, batch_size,
                                    optimizer_kh, neighborhood_size=sigma_t)

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
        """
        Evaluate the model on a validation or test set.

        Args:
            data_loader (DataLoader): Dataset loader to evaluate on.

            return_loss (bool): Whether to return loss or accuracy.

        Returns:
            float: Validation loss or accuracy.
        """

        if len(data_loader) == 0:
            raise ValueError("Evaluation loader is empty.")

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
