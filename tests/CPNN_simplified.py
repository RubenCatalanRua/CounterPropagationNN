import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# ---------------------------
# Simplified CounterPropagation Network
# ---------------------------
class SimpleCPNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        input_size: Dimensionality of the input (e.g. 28*28 for MNIST)
        hidden_size: Number of neurons in the Kohonen layer.
        output_size: Number of classes.
        """
        super(SimpleCPNN, self).__init__()
        # Kohonen (winner-take-all) layer: unsupervised weights.
        self.kohonen_weights = nn.Parameter(torch.empty(hidden_size, input_size))
        # Grossberg mapping: a simple linear layer from hidden to output.
        # Note: We use our custom training update for kohonen_weights,
        # and we train grossberg_weights using standard gradient descent.
        self.grossberg_weights = nn.Parameter(torch.empty(output_size, hidden_size))
        
        # Initialize weights (using Xavier initialization)
        nn.init.xavier_uniform_(self.kohonen_weights)
        nn.init.xavier_uniform_(self.grossberg_weights)
        
        # Fixed setting for distance and neighborhood function.
        self.distance_metric = 'euclidean'
        self.neighborhood_function = 'gaussian'

    def forward(self, x):
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
        # Grossberg mapping: a simple linear mapping.
        # Output = one_hot * grossberg_weights^T
        output = torch.matmul(winner_one_hot, self.grossberg_weights.t())
        return output, winners

    def update_kohonen(self, x, lr, neighborhood_size):
        """
        Update the kohonen weights using a basic winner update.
        For each sample in the batch, only the winning neuron is updated.
        """
        # Compute distances and determine winners.
        distances = torch.cdist(x, self.kohonen_weights)
        winners = torch.argmin(distances, dim=1)
        # Loop over batch (for simplicity)
        for i in range(x.size(0)):
            win_idx = winners[i]
            # Simple update: move the winning neuron's weight toward the input.
            # (Neighborhood update could be added here if desired.)
            self.kohonen_weights.data[win_idx] += lr * (x[i] - self.kohonen_weights.data[win_idx])
            # Normalize the updated weight vector.
            self.kohonen_weights.data[win_idx] /= self.kohonen_weights.data[win_idx].norm() + 1e-8

    def train_grossberg(self, x, y, optimizer, criterion):
        """
        Standard supervised training update for the grossberg mapping.
        We run a forward pass and backpropagate the loss.
        """
        optimizer.zero_grad()
        outputs, _ = self.forward(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def fit(self, train_loader, val_loader=None, epochs=20, kohonen_lr=0.01, grossberg_lr=0.01, neighborhood_size=3):
        """
        Expects dataloaders that yield (inputs, labels).
        Labels should be integer class labels.
        """
        # Optimizer for grossberg weights only.
        optimizer = optim.SGD([self.grossberg_weights], lr=grossberg_lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(1, epochs + 1):
            self.train()
            running_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.view(batch_x.size(0), -1)  # flatten if needed
                batch_x, batch_y = batch_x.to(next(self.parameters()).device), batch_y.to(next(self.parameters()).device)
                # Update kohonen layer with an unsupervised rule.
                self.update_kohonen(batch_x, lr=kohonen_lr, neighborhood_size=neighborhood_size)
                # Update grossberg mapping with supervised learning.
                loss = self.train_grossberg(batch_x, batch_y, optimizer, criterion)
                running_loss += loss

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

            # Optionally evaluate on validation data.
            if val_loader is not None:
                self.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.view(batch_x.size(0), -1)
                        batch_x, batch_y = batch_x.to(next(self.parameters()).device), batch_y.to(next(self.parameters()).device)
                        outputs, _ = self.forward(batch_x)
                        predicted = outputs.argmax(dim=1)
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                print(f"Validation Accuracy: {100 * correct / total:.2f}%")
        return self

    def evaluate(self, data_loader):
        """
        Evaluate the model on a given dataset and print accuracy.
        """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.view(batch_x.size(0), -1)
                batch_x, batch_y = batch_x.to(next(self.parameters()).device), batch_y.to(next(self.parameters()).device)
                outputs, _ = self.forward(batch_x)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def predict(self, data_loader):
        """
        Predict labels for data coming from a dataloader.
        """
        self.eval()
        all_preds = []
        with torch.no_grad():
            for batch_x, _ in data_loader:
                batch_x = batch_x.view(batch_x.size(0), -1)
                batch_x = batch_x.to(next(self.parameters()).device)
                outputs, _ = self.forward(batch_x)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
        return torch.cat(all_preds)



transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model
input_size = 28 * 28
hidden_size = 100   # number of neurons in Kohonen layer
output_size = 10    # number of classes for MNIST
model = SimpleCPNN(input_size, hidden_size, output_size).to(device)

# Train the model
model.fit(train_loader, val_loader=val_loader, epochs=10, kohonen_lr=0.01, grossberg_lr=0.01, neighborhood_size=3)

# Predict on validation set
model.evaluate(val_loader)