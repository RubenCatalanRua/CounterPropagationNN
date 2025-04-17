import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------
# Base CounterPropagation Network
# ---------------------------
class BaseCPNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, neighborhood_function='gaussian', neighborhood_size=3):
        """
        input_size: Dimensionality of the input (e.g. 28*28 for MNIST)
        hidden_size: Number of neurons in the Kohonen layer.
        output_size: Number of classes.
        """
        super(BaseCPNN, self).__init__()
        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.kohonen_weights = nn.Parameter(torch.empty(hidden_size, input_size))

        self.grossberg_weights = nn.Parameter(torch.empty(output_size, hidden_size))

        # Kohonen and Grossberg layers should be initialized randomly
        self.kohonen_weights.data.normal_(0, 1)
        self.kohonen_weights.data = F.normalize(self.kohonen_weights.data, p=2, dim=1)

        self.grossberg_weights.data.normal_(0, 1)
        self.grossberg_weights.data = F.normalize(self.grossberg_weights.data, p=2, dim=1)


        self.neighborhood_function = neighborhood_function
        self.neighborhood_size = neighborhood_size

    def forward(self, x):
        x = self.flatten(x)
        # x is expected to be of shape (batch_size, input_size)
        # Compute distance between x and each kohonen weight vector.
        # (Using Euclidean distance here.)
        distances = torch.cdist(x, self.kohonen_weights)  # shape: (batch, hidden_size)
        # Winner-take-all: find index of closest neuron for each sample.
        winners = torch.argmin(distances, dim=1)  # shape: (batch,)
        # Create one-hot encoding of the winning neurons.
        batch_size = x.size(0)
        winner_one_hot = torch.zeros(batch_size, self.kohonen_weights.size(0), device=x.device)
        winner_one_hot.scatter_(1, winners.unsqueeze(1), 1)

        output = torch.matmul(winner_one_hot, self.grossberg_weights.t())
        return output, winners

    def update_kohonen(self, x, optimizer, lr, neighborhood_size):

        # Flatten the input: (batch_size, input_size)
        x = self.flatten(x)
        # Compute distances between each sample and each neuron's weight.
        # x: (batch_size, input_size), weights: (hidden_size, input_size)
        distances = torch.cdist(x, self.kohonen_weights, p=2)  # Shape: (batch_size, hidden_size
        winner_indices = torch.argmin(distances, dim=1)  # Shape: (batch_size,)

        batch_size = x.size(0)
        # Create a tensor for hidden indices, shape (batch_size, hidden_size)
        hidden_indices = torch.arange(self.hidden_size, device=x.device, dtype=torch.float32)
        hidden_indices = hidden_indices.unsqueeze(0).expand(batch_size, -1)

        # Expand winner indices to match hidden_indices dimensions: (batch_size, 1)
        winner_indices_exp = winner_indices.unsqueeze(1).float()
        diff_indices = torch.abs(hidden_indices - winner_indices_exp)  # (batch_size, hidden_size)

        # Compute the influence for all samples in a vectorized way
        if self.neighborhood_function == 'gaussian':
            influence = torch.exp(- (diff_indices ** 2) / (2 * (neighborhood_size ** 2)))
        elif self.neighborhood_function == 'rectangular':
            influence = (diff_indices <= neighborhood_size).float()
        elif self.neighborhood_function == 'triangular':
            influence = torch.clamp(1 - diff_indices / neighborhood_size, min=0)
        else:
            influence = torch.exp(- (diff_indices ** 2) / (2 * (neighborhood_size ** 2)))
        x_expanded = x.unsqueeze(1)
        weights_expanded = self.kohonen_weights.unsqueeze(0)
        diff = x_expanded - weights_expanded  # (batch_size, hidden_size, input_size)

        # Unsqueeze influence to match the diff dimensions: (batch_size, hidden_size, 1)
        influence = influence.unsqueeze(2)
        # Compute the per-sample gradients and then average over the batch
        grad = - influence * diff  # (batch_size, hidden_size, input_size)
        grad_accum = grad.mean(dim=0)  # (hidden_size, input_size)

        optimizer.zero_grad()
        self.kohonen_weights.grad = grad_accum * lr
        optimizer.step()
        self.kohonen_weights.data = F.normalize(self.kohonen_weights.data, p=2, dim=1)


    def train_grossberg(self, x, y, optimizer, lr):

        _, winner_indices = self.forward(x)
        batch_size = x.size(0)

        optimizer.zero_grad()
        counts = torch.zeros(self.hidden_size, device=x.device)
        counts = counts.index_add(0, winner_indices, torch.ones(batch_size, device=x.device))
        y_onehot = F.one_hot(y, num_classes=self.output_size).float()

        y_sum = torch.zeros(self.output_size, self.hidden_size, device=x.device)
        y_sum = y_sum.index_add(1, winner_indices, y_onehot.transpose(0, 1))
        grad = self.grossberg_weights * counts.unsqueeze(0) - y_sum
        self.grossberg_weights.grad = grad * lr
        optimizer.step()


    def fit(self, device, train_loader, val_loader=None, epochs=20,
            kohonen_lr=0.01, grossberg_lr=0.01,
            early_stopping=False, patience=None):

        optimizer_gr = optim.SGD([self.grossberg_weights], lr=grossberg_lr, momentum=0.8, nesterov=True)
        optimizer_kh = optim.SGD([self.kohonen_weights], lr=kohonen_lr, momentum=0.8, nesterov=True)

        scheduler_gr = optim.lr_scheduler.ReduceLROnPlateau(optimizer_gr, mode='min', factor=0.5, patience=3)
        scheduler_kh = optim.lr_scheduler.ExponentialLR(optimizer_kh, gamma=0.95)

        criterion = nn.CrossEntropyLoss()
        best_val_loss = float('inf')
        init_patience = patience

        for epoch in range(1, epochs + 1):
            kohonen_time = 0
            grossberg_time = 0

            print(f"\n[Epoch {epoch}] Starting training...")

            self.train()
            decayed_neighborhood = max(1, self.neighborhood_size - (self.neighborhood_size - 1) * (epoch - 1) / (epochs - 1))

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Profile Kohonen update
                self.update_kohonen(batch_x, optimizer_kh, lr=kohonen_lr,
                                    neighborhood_size=decayed_neighborhood)

                # Profile Grossberg update
                self.train_grossberg(batch_x, batch_y, optimizer_gr, grossberg_lr)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader, return_loss=True)
                print(f"[Epoch {epoch}] Validation loss: {val_loss:.4f}")
                scheduler_gr.step(val_loss)
                scheduler_kh.step()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = init_patience
                else:
                    if early_stopping:
                        patience -= 1
                        if patience == 0:
                            print("Early stopping triggered.")
                            break

        return self

    def evaluate(self, data_loader, return_loss=False):

        self.eval()
        device = next(self.parameters()).device
        correct = 0
        total = 0
        running_loss = 0.0
        criterion = nn.CrossEntropyLoss(reduction='sum')

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs, _ = self.forward(batch_x)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

                if return_loss:
                    running_loss += criterion(outputs, batch_y).item()

        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy:.2f}%")

        if return_loss:
            avg_loss = running_loss / total
            return avg_loss
        else:
            return accuracy
