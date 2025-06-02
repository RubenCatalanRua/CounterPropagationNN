import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ForwardOnlyCPNN(nn.Module):
    """
    Base implementation of a Counter-Propagation Neural Network (CPNN), 
    combining a Kohonen self-organizing map (unsupervised) with a 
    Grossberg layer (supervised).

    Attributes:
        input_size (int): Size of the input features.

        hidden_size (int): Number of neurons in the Kohonen layer
        .
        output_size (int): Number of output classes.

        kohonen_weights (nn.Parameter): Weight vectors for Kohonen layer.

        grossberg_weights (nn.Parameter): Weight vectors for Grossberg layer.

        neighborhood_function (str): Type of neighborhood ('gaussian', 
        'triangular', or 'rectangular').

        neighborhood_size (float): Initial neighborhood size.

        kohonen_snapshots (List[Tensor]): History of Kohonen weights 
        after each epoch.

        device (torch.device): CUDA or CPU.
    """

    def __init__(self, input_size, hidden_size, output_size,
                 neighborhood_function='gaussian',
                 neighborhood_size=3, device=None):
        """
        Initializes the ForwardOnlyCPNN model with specified layer sizes 
        and neighborhood function.

        Args:
            input_size (int): Dimensionality of the input.

            hidden_size (int): Number of neurons in the Kohonen layer.

            output_size (int): Number of output classes.

            neighborhood_function (str): Neighborhood influence function 
            ('gaussian', 'triangular', 'rectangular').

            neighborhood_size (float): Size of the neighborhood.

            device (torch.device, optional): Computation device (CPU or CUDA). 
            Defaults to auto-detection.
        """

        super(ForwardOnlyCPNN, self).__init__()

        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        assert input_size > 0, "Input size must be > 0"
        assert hidden_size > 0, "Hidden size must be > 0"
        assert output_size > 0, "Output size must be > 0"

        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Kohonen and Grossberg weights
        self.kohonen_weights = nn.Parameter(
            torch.randn(self.hidden_size, self.input_size))
        with torch.no_grad():
            # Normalize each Kohonen weight vector to unit norm
            self.kohonen_weights.data = F.normalize(
                self.kohonen_weights.data, p=2, dim=1)

        self.grossberg_weights = nn.Parameter(
            torch.randn(self.output_size, self.hidden_size))

        with torch.no_grad():
            # Same normalizarion to Grossberg layer
            self.grossberg_weights.data = F.normalize(
                self.grossberg_weights.data, p=2, dim=1)

        self.neighborhood_function = neighborhood_function
        self.neighborhood_size = neighborhood_size

        if self.neighborhood_size < 1:
            self.neighborhood_size = 1

        # Used to create graphs showing the evolution 
        # of the Kohonen weight mappings
        self.kohonen_snapshots = []

        # This returns the average loss across the batch
        self.val_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            output (Tensor): Output logits from Grossberg layer.

            winners (Tensor): Indices of winning neurons in the Kohonen layer.

            batch_size (int): Number of samples in the batch.
        """

        x = self.flatten(x)
        distances = torch.cdist(x, self.kohonen_weights)  # Euclidean distances
        winners = torch.argmin(distances, dim=1)
        batch_size = x.size(0)
        winner_one_hot = torch.zeros(
            batch_size, self.kohonen_weights.size(0), device=x.device)
        winner_one_hot.scatter_(1, winners.unsqueeze(1), 1)
        output = torch.matmul(winner_one_hot, self.grossberg_weights.t())
        return output, winners, batch_size

    def update_kohonen(self, x, winner_indices, batch_size,
                       optimizer, neighborhood_size):
        """
        Updates Kohonen weights using the neighborhood function.

        Args:
            x (Tensor): Input batch.

            winner_indices (Tensor): Index of best-matching neuron per sample.

            batch_size (int): Size of the input batch.

            optimizer (torch.optim.Optimizer): Optimizer for Kohonen weights.

            neighborhood_size (float): Current neighborhood size (sigma).
        """

        x = self.flatten(x)

        hidden_indices = torch.arange(
            self.hidden_size, device=x.device, dtype=torch.float32)
        hidden_indices = hidden_indices.unsqueeze(0).expand(batch_size, -1)
        winner_indices_exp = winner_indices.unsqueeze(1).float()
        diff_indices = torch.abs(hidden_indices - winner_indices_exp)

        if self.neighborhood_function == 'rectangular':
            influence = (diff_indices <= neighborhood_size).float()
        elif self.neighborhood_function == 'triangular':
            influence = torch.clamp(
                1 - diff_indices / neighborhood_size, min=0)
        else:  # Gaussian
            influence = torch.exp(- (diff_indices ** 2) /
                                  (2 * (neighborhood_size ** 2)))

        x_expanded = x.unsqueeze(1)
        weights_expanded = self.kohonen_weights.unsqueeze(0)
        diff = x_expanded - weights_expanded
        influence = influence.unsqueeze(2)
        grad = - influence * diff
        grad_accum = grad.mean(dim=0)

        optimizer.zero_grad()
        self.kohonen_weights.grad = grad_accum
        optimizer.step()
        with torch.no_grad():
            self.kohonen_weights.data = F.normalize(
                self.kohonen_weights.data, p=2, dim=1)

    def update_grossberg(self, y, winner_indices, batch_size, optimizer):
        """
        Updates Grossberg weights using supervised labels.

        Args:
            x (Tensor): Input tensor.

            y (Tensor): Target class labels.

            winner_indices (Tensor): Indices of winning Kohonen neurons.

            batch_size (int): Batch size.

            optimizer (torch.optim.Optimizer): Optimizer for Grossberg weights.
        """

        optimizer.zero_grad()
        counts = torch.zeros(self.hidden_size, device=y.device)
        counts = counts.index_add(
            0, winner_indices, torch.ones(batch_size, device=y.device))
        y_onehot = F.one_hot(y, num_classes=self.output_size).float()

        y_sum = torch.zeros(
            self.output_size, self.hidden_size, device=x.device)
        y_sum = y_sum.index_add(1, winner_indices, y_onehot.transpose(0, 1))
        grad = self.grossberg_weights * counts.unsqueeze(0) - y_sum
        self.grossberg_weights.grad = grad
        optimizer.step()

    def fit(self, train_loader, val_loader=None, epochs=20,
            kohonen_lr=0.01, grossberg_lr=0.01,
            early_stopping=False, patience=None):
        """
        Trains the CPNN using Kohonen and Grossberg learning.

        Args:
            train_loader (DataLoader): Dataloader for training data.

            val_loader (DataLoader, optional): Dataloader for validation data.

            epochs (int): Number of training epochs.

            kohonen_lr (float): Learning rate for Kohonen weights.

            grossberg_lr (float): Learning rate for Grossberg weights.

            early_stopping (bool): Whether to use early stopping.

            patience (int, optional): Epochs to wait before stopping 
            after no improvement.
        """

        if len(train_loader) == 0:
            raise ValueError("Training loader is empty.")

        optimizer_kh = optim.SGD(
            [self.kohonen_weights], lr=kohonen_lr, weight_decay=1e-4,
            momentum=0.95, nesterov=True)
        optimizer_gr = optim.AdamW([self.grossberg_weights], lr=grossberg_lr)

        scheduler_kh = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_kh, T_max=epochs, eta_min=kohonen_lr * 0.1)
        scheduler_gr = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_gr, T_max=epochs, eta_min=grossberg_lr * 0.1)

        sigma_0 = min(self.neighborhood_size, self.hidden_size / 2)
        if sigma_0 < 1:
            sigma_0 = 1

        decay_enabled = sigma_0 > 1

        if decay_enabled:
            lambd = epochs / math.log(sigma_0)

        best_val_loss = float('inf')
        init_patience = patience

        for epoch in range(1, epochs + 1):

            # Guards added to avoid < 1 neighborhoods
            if decay_enabled:
                sigma_t = max(1.0, sigma_0 * math.exp(-epoch / lambd))
            else:
                sigma_t = sigma_0

            print(f"\n[Epoch {epoch}] Starting training...")

            self.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(
                    self.device), batch_y.to(self.device)
                _, winner_indices, batch_size = self.forward(batch_x)

                self.update_kohonen(
                    batch_x, winner_indices, batch_size, optimizer_kh,
                    neighborhood_size=sigma_t)
                self.update_grossberg(
                    batch_y, winner_indices, batch_size, optimizer_gr)

            scheduler_kh.step()
            scheduler_gr.step()

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
        Evaluates the model performance on given data.

        Args:
            data_loader (DataLoader): DataLoader for evaluation.

            return_loss (bool): If True, returns loss instead of accuracy.

        Returns:
            float: Loss or accuracy, depending on return_loss.
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

        return avg_loss if return_loss else acc
