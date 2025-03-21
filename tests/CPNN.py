# CPNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# --------------------------
# Autoencoder
# --------------------------
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(autoencoder, data, epochs=50, lr=0.001, batch_size=128, device=None, verbose=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(data, data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            decoded, _ = autoencoder(batch_x)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        if verbose:
            avg_loss = epoch_loss/len(data)
            print(f"Autoencoder Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}")
    return autoencoder

# --------------------------
# CounterPropagation Network (CPNN)
# --------------------------
class CounterPropagationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, init_method="xavier_uniform"):
        super(CounterPropagationNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Kohonen layer (Winner-Take-All)
        self.kohonen_weights = nn.Parameter(torch.empty(hidden_size, input_size))
        # Grossberg layer (Outstar)
        self.grossberg_weights = nn.Parameter(torch.empty(output_size, hidden_size))
        
        # Initialization method
        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(self.kohonen_weights)
            nn.init.xavier_uniform_(self.grossberg_weights)
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(self.kohonen_weights)
            nn.init.xavier_normal_(self.grossberg_weights)
        elif init_method == "kaiming_uniform":
            nn.init.kaiming_uniform_(self.kohonen_weights, nonlinearity="relu")
            nn.init.kaiming_uniform_(self.grossberg_weights, nonlinearity="relu")
        elif init_method == "kaiming_normal":
            nn.init.kaiming_normal_(self.kohonen_weights, nonlinearity="relu")
            nn.init.kaiming_normal_(self.grossberg_weights, nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(self.kohonen_weights)
            nn.init.xavier_uniform_(self.grossberg_weights)

    def forward(self, x):
        # Compute Euclidean distances and determine winners
        distances = torch.cdist(x, self.kohonen_weights)  # (batch_size, hidden_size)
        winner_indices = torch.argmin(distances, dim=1)     # (batch_size,)
        batch_size = x.size(0)
        # One-hot encode winners
        winner_one_hot = torch.zeros(batch_size, self.hidden_size, device=x.device)
        winner_one_hot.scatter_(1, winner_indices.unsqueeze(1), 1)
        output = torch.matmul(winner_one_hot, self.grossberg_weights.t())
        return output, winner_indices

    def train_kohonen(self, x, learning_rate=0.1, neighborhood_size=3):
        # Compute current neighborhood decay factor (linearly decaying to 1 over epochs)
        distances = torch.cdist(x, self.kohonen_weights)  # (batch_size, hidden_size)
        winner_indices = torch.argmin(distances, dim=1)     # (batch_size,)
        hidden_indices = torch.arange(self.hidden_size, device=x.device).float()
        
        # Compute influence per neuron based on distance from winner
        diff_indices = hidden_indices.unsqueeze(0) - winner_indices.unsqueeze(1).float()
        influence = torch.exp(- (diff_indices**2) / (2 * (neighborhood_size**2)))
        
        diff_x = x.unsqueeze(1) - self.kohonen_weights.unsqueeze(0)
        influence_expanded = influence.unsqueeze(2)
        
        update = learning_rate * (influence_expanded * diff_x).sum(dim=0)
        with torch.no_grad():
            self.kohonen_weights.add_(update)
            # Normalize each neuron's weight vector
            self.kohonen_weights.copy_(F.normalize(self.kohonen_weights, p=2, dim=1))

    def train_grossberg(self, x, y, learning_rate=0.1):
        # Forward pass to obtain winner indices
        _, winner_indices = self.forward(x)  # (batch_size,)
        batch_size = x.size(0)
        # Vectorized update: For each neuron j, accumulate the difference between the target and current weight
        with torch.no_grad():
            # Compute counts: how many times each neuron wins in the batch
            counts = torch.zeros(self.hidden_size, device=x.device)
            counts = counts.index_add(0, winner_indices, torch.ones(batch_size, device=x.device))
            
            # Sum target outputs for each neuron
            # y is (batch_size, output_size), we need to sum along batch dimension for each neuron index
            y_sum = torch.zeros(self.output_size, self.hidden_size, device=x.device)
            y_sum = y_sum.index_add(1, winner_indices, y.transpose(0, 1))
            
            # Update: subtract weight multiplied by counts and add the summed targets
            update = learning_rate * (y_sum - self.grossberg_weights * counts.unsqueeze(0))
            self.grossberg_weights.add_(update)

    def fit(self, train_loader, val_data=None, config=None):
        # Default training config
        if config is None:
            config = {
                "kohonen_lr": 0.5,
                "grossberg_lr": 0.5,
                "epochs": 300,
                "neighborhood_size": 3,
                "log_interval": 50,
                "verbose": False
            }
        verbose = config.get("verbose", False)
        epochs = config["epochs"]
        initial_neighborhood = config["neighborhood_size"]

        for epoch in range(1, epochs + 1):
            # Optionally decay neighborhood: here we linearly decay from the initial value to 1.
            decayed_neighborhood = max(1, initial_neighborhood - (initial_neighborhood - 1) * (epoch - 1) / (epochs - 1))
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.kohonen_weights.device)
                batch_y = batch_y.to(self.kohonen_weights.device)
                self.train_kohonen(batch_x, learning_rate=config["kohonen_lr"], neighborhood_size=decayed_neighborhood)
                self.train_grossberg(batch_x, batch_y, learning_rate=config["grossberg_lr"])
            if verbose and config["log_interval"] and epoch % config["log_interval"] == 0:
                if val_data is not None:
                    X_val, y_val = val_data
                    self.eval()
                    with torch.no_grad():
                        outputs, _ = self.forward(X_val)
                        # Compute simple accuracy metric on validation set
                        pred_labels = torch.argmax(outputs, dim=1)
                        true_labels = torch.argmax(y_val, dim=1)
                        acc = (pred_labels == true_labels).float().mean().item() * 100
                    print(f"Epoch {epoch}/{epochs} | Validation Accuracy: {acc:.2f}%")
                    self.train()
                else:
                    print(f"Epoch {epoch}/{epochs} completed.")

# --------------------------
# Wrapper Class for CPNNClassifier
# --------------------------
class CPNNClassifier:
    def __init__(self, hidden_size=100, init_method="xavier_uniform",
                 kohonen_lr=0.5, grossberg_lr=0.5, epochs=300, neighborhood_size=3,
                 batch_size=128, use_autoencoder=False, ae_dim=None, ae_epochs=50, ae_lr=0.001,
                 random_state=None, device=None, verbose=False):
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.kohonen_lr = kohonen_lr
        self.grossberg_lr = grossberg_lr
        self.epochs = epochs
        self.neighborhood_size = neighborhood_size
        self.batch_size = batch_size
        self.use_autoencoder = use_autoencoder
        self.ae_dim = ae_dim
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.random_state = random_state
        self.verbose = verbose

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print('LOADED CUDA')
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print('LOADED MPS')
        elif torch.xpu.is_available():
            self.device = torch.device("xpu")
            print('LOADED XPU')
        else:
            self.device = torch.device("cpu")
            print('CPU ROLLING')

        self.cpnn_ = None
        self.autoencoder_ = None
        self.input_size_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        # Optionally set random seed for reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Convert inputs (numpy arrays) to torch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if len(y.shape) == 1 or y.shape[1] == 1:
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
            self.classes_ = np.unique(y)
            y_tensor = torch.nn.functional.one_hot(y_tensor, num_classes=len(self.classes_)).float()
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            self.classes_ = np.arange(y_tensor.shape[1])
            
        self.input_size_ = X_tensor.shape[1]
        # Autoencoder transformation if enabled
        if self.use_autoencoder:
            if self.ae_dim is None:
                raise ValueError("ae_dim must be set if use_autoencoder is True.")
            self.autoencoder_ = SimpleAutoencoder(self.input_size_, self.ae_dim).to(self.device)
            self.autoencoder_ = train_autoencoder(self.autoencoder_, X_tensor,
                                                  epochs=self.ae_epochs, lr=self.ae_lr,
                                                  batch_size=self.batch_size, device=self.device,
                                                  verbose=self.verbose)
            with torch.no_grad():
                _, X_encoded = self.autoencoder_(X_tensor)
            X_train = X_encoded
            input_size = self.ae_dim
        else:
            X_train = X_tensor
            input_size = self.input_size_
        
        dataset = TensorDataset(X_train, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.cpnn_ = CounterPropagationNetwork(input_size, self.hidden_size, len(self.classes_),
                                                 init_method=self.init_method).to(self.device)
        config = {
            "kohonen_lr": self.kohonen_lr,
            "grossberg_lr": self.grossberg_lr,
            "epochs": self.epochs,
            "neighborhood_size": self.neighborhood_size,
            "log_interval": None if not self.verbose else max(1, self.epochs // 5),
            "verbose": self.verbose
        }
        self.cpnn_.fit(loader, val_data=None, config=config)
        return self
        
    def predict(self, X):
        self.cpnn_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.use_autoencoder:
            with torch.no_grad():
                _, X_tensor = self.autoencoder_(X_tensor)
        with torch.no_grad():
            outputs, _ = self.cpnn_(X_tensor)
            preds = torch.argmax(outputs, dim=1)
        return preds.cpu().numpy()
    
    def predict_proba(self, X):
        self.cpnn_.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if self.use_autoencoder:
            with torch.no_grad():
                _, X_tensor = self.autoencoder_(X_tensor)
        with torch.no_grad():
            outputs, _ = self.cpnn_(X_tensor)
            prob = torch.softmax(outputs, dim=1)
        return prob.cpu().numpy()
    
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        preds = self.predict(X)
        return accuracy_score(y, preds)
    
    def get_params(self, deep=True):
        return {
            "hidden_size": self.hidden_size,
            "init_method": self.init_method,
            "kohonen_lr": self.kohonen_lr,
            "grossberg_lr": self.grossberg_lr,
            "epochs": self.epochs,
            "neighborhood_size": self.neighborhood_size,
            "batch_size": self.batch_size,
            "use_autoencoder": self.use_autoencoder,
            "ae_dim": self.ae_dim,
            "ae_epochs": self.ae_epochs,
            "ae_lr": self.ae_lr,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "device": self.device,
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self