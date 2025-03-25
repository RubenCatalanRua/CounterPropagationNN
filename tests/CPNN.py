# CPNN.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time

# --------------------------
# Utility Functions
# --------------------------
def validate_hyperparameters(**kwargs):
    """
    Validate hyperparameters to ensure they meet expected conditions.
    Raises ValueError if any condition is not met.
    """
    if kwargs.get("kohonen_lr", 0.0) <= 0:
        raise ValueError("kohonen_lr must be positive.")
    if kwargs.get("grossberg_lr", 0.0) <= 0:
        raise ValueError("grossberg_lr must be positive.")
    if kwargs.get("max_epochs", 0) < 1:
        raise ValueError("max_epochs must be at least 1.")
    if kwargs.get("batch_size", 0) < 1:
        raise ValueError("batch_size must be at least 1.")
    if kwargs.get("use_autoencoder", False) and (kwargs.get("ae_dim", None) is None):
        raise ValueError("ae_dim must be set if use_autoencoder is True.")

    # TODO: ADD THE REST

# --------------------------
# Autoencoder
# --------------------------
class SimpleAutoencoder(nn.Module):
    """
    A simple autoencoder network with a single hidden encoding layer.
    
    Attributes:
        encoder (nn.Sequential): Maps input to a lower-dimensional space.
        decoder (nn.Sequential): Reconstructs input from the encoded representation.
    """
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
        """
        Forward pass through the autoencoder.
        
        Args:
            x (Tensor): Input data.
        
        Returns:
            decoded (Tensor): Reconstructed input.
            encoded (Tensor): Encoded representation.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(autoencoder, data, epochs=50, lr=0.001, batch_size=128, deviceT=None, verbose=False):
    """
    Trains the autoencoder using mean squared error loss.
    
    Moves the model to the specified device, optimizes using Adam, and logs training loss if verbose.
    
    Args:
        autoencoder (nn.Module): The autoencoder model.
        data (Tensor): Input data for training.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        device (torch.device): Device to train on.
        verbose (bool): If True, prints training loss.
    
    Returns:
        autoencoder (nn.Module): The trained autoencoder.
    """
    if lr <= 0:
        raise ValueError("Learning rate for autoencoder must be positive.")

    if deviceT is not None:
        device = deviceT
        print(f"Using provided device: {device}")
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("LOADED CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("LOADED MPS")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device = torch.device("xpu")
            print("LOADED XPU")
        else:
            device = torch.device("cpu")
            print("CPU ROLLING")

    autoencoder.to(device)
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(data, data)  # Input equals output for autoencoder.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs + 1):
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
            avg_loss = epoch_loss / len(data)
            print(f"Autoencoder Epoch {epoch}/{epochs}: Loss = {avg_loss:.6f}")
    return autoencoder

# --------------------------
# CounterPropagation Network (CPNN)
# --------------------------
class CounterPropagationNetwork(nn.Module):
    """
    CounterPropagation Network implementation combining a Kohonen layer for unsupervised clustering
    and a Grossberg layer for supervised mapping to output.
    
    Attributes:
        kohonen_weights (nn.Parameter): Weight matrix for the Kohonen layer.
        grossberg_weights (nn.Parameter): Weight matrix for the Grossberg layer.
    """
    def __init__(self, input_size, hidden_size, output_size, init_method="xavier_uniform"):
        super(CounterPropagationNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize Kohonen layer (for winner-take-all approach)
        self.kohonen_weights = nn.Parameter(torch.empty(hidden_size, input_size))
        # Initialize Grossberg layer (for mapping to output)
        self.grossberg_weights = nn.Parameter(torch.empty(output_size, hidden_size))
        
        # Initialize weights using the selected method.
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
        """
        Computes the forward pass by determining the winning neuron and mapping via Grossberg weights.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            output (Tensor): Network output.
            winner_indices (Tensor): Indices of the winning neurons.
        """
        distances = torch.cdist(x, self.kohonen_weights)  # Compute Euclidean distances.
        winner_indices = torch.argmin(distances, dim=1)     # Determine winners.
        batch_size = x.size(0)
        # Create a one-hot encoded tensor of winning neurons.
        winner_one_hot = torch.zeros(batch_size, self.hidden_size, device=x.device)
        winner_one_hot.scatter_(1, winner_indices.unsqueeze(1), 1)
        output = torch.matmul(winner_one_hot, self.grossberg_weights.t())
        return output, winner_indices

    def train_kohonen(self, x, learning_rate=0.1, neighborhood_size=3):
        """
        Updates the Kohonen weights using a Gaussian neighborhood function.
        
        Args:
            x (Tensor): Input data batch.
            learning_rate (float): Learning rate.
            neighborhood_size (float): Controls the spread of the update.
        """
        distances = torch.cdist(x, self.kohonen_weights)
        winner_indices = torch.argmin(distances, dim=1)
        hidden_indices = torch.arange(self.hidden_size, device=x.device).float()
        
        diff_indices = hidden_indices.unsqueeze(0) - winner_indices.unsqueeze(1).float()
        influence = torch.exp(- (diff_indices ** 2) / (2 * (neighborhood_size ** 2)))
        
        diff_x = x.unsqueeze(1) - self.kohonen_weights.unsqueeze(0)
        influence_expanded = influence.unsqueeze(2)
        
        update = learning_rate * (influence_expanded * diff_x).sum(dim=0)
        with torch.no_grad():
            self.kohonen_weights.add_(update)
            self.kohonen_weights.copy_(F.normalize(self.kohonen_weights, p=2, dim=1))

    def train_grossberg(self, x, y, learning_rate=0.1):
        """
        Updates the Grossberg weights based on the error between target and current output.
        
        Args:
            x (Tensor): Input data batch.
            y (Tensor): Target output batch.
            learning_rate (float): Learning rate for Grossberg weights.
        """
        _, winner_indices = self.forward(x)
        batch_size = x.size(0)
        with torch.no_grad():
            counts = torch.zeros(self.hidden_size, device=x.device)
            counts = counts.index_add(0, winner_indices, torch.ones(batch_size, device=x.device))
            
            y_sum = torch.zeros(self.output_size, self.hidden_size, device=x.device)
            y_sum = y_sum.index_add(1, winner_indices, y.transpose(0, 1))
            
            update = learning_rate * (y_sum - self.grossberg_weights * counts.unsqueeze(0))
            self.grossberg_weights.add_(update)

    def fit(self, train_loader, val_data=None, config=None):
        """
        Training loop for the CPNN that:
         - Iterates over epochs and batches.
         - Applies a decaying neighborhood function.
         - Optionally performs early stopping using validation data.
         
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_data (tuple, optional): Tuple (X_val, y_val) for validation.
            config (dict, optional): Configuration parameters.
            
        Returns:
            self
        """
        if config is None:
            config = {
                "kohonen_lr": 0.5,
                "grossberg_lr": 0.5,
                "max_epochs": 300,
                "neighborhood_size": 3,
                "log_interval": 50,
                "verbose": False,
                "early_stopping": False,
                "patience": 10,
            }
        verbose = config.get("verbose", False)
        max_epochs = config.get("max_epochs", 300)
        initial_neighborhood = config.get("neighborhood_size", 3)
        early_stopping = config.get("early_stopping", False)
        patience = config.get("patience", 10)

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(1, max_epochs + 1):
            # Compute decayed neighborhood factor (avoid division by zero if max_epochs==1)
            if max_epochs > 1:
                decayed_neighborhood = max(1, initial_neighborhood - (initial_neighborhood - 1) * (epoch - 1) / (max_epochs - 1))
            else:
                decayed_neighborhood = initial_neighborhood

            # Process each batch and update weights.
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.kohonen_weights.device)
                batch_y = batch_y.to(self.kohonen_weights.device)
                self.train_kohonen(batch_x, learning_rate=config["kohonen_lr"], neighborhood_size=decayed_neighborhood)
                self.train_grossberg(batch_x, batch_y, learning_rate=config["grossberg_lr"])

            # Optional logging and early stopping using validation data.
            if verbose and config.get("log_interval", None) and epoch % config["log_interval"] == 0:
                if val_data is not None:
                    X_val, y_val = val_data
                    self.eval()
                    with torch.no_grad():
                        outputs, _ = self.forward(X_val)
                        pred_labels = torch.argmax(outputs, dim=1)
                        true_labels = torch.argmax(y_val, dim=1)
                        acc = (pred_labels == true_labels).float().mean().item() * 100
                    print(f"Epoch {epoch}/{max_epochs} | Validation Accuracy: {acc:.2f}%")
                    if early_stopping:
                        if acc > best_val_acc:
                            best_val_acc = acc
                            epochs_no_improve = 0
                        else:
                            epochs_no_improve += 1
                        if acc < 0.5 * best_val_acc:
                            print("Early stopping triggered: validation accuracy dropped to less than half of the best value.")
                            break
                        if epochs_no_improve >= patience:
                            print(f"Early stopping triggered: no improvement for {patience} epochs.")
                            break
                    self.train()  # Reset model to training mode.
                else:
                    if verbose:
                        print(f"Epoch {epoch}/{max_epochs} completed.")
        return self

# --------------------------
# Wrapper Class for CPNNClassifier
# --------------------------
class CPNNClassifier:
    """
    A wrapper for the CounterPropagation Network that provides:
      - Optional autoencoder-based preprocessing.
      - Device selection (CPU, CUDA, MPS, XPU) via external parameter.
      - Hyperparameter configuration and validation.
    
    The wrapper integrates data preprocessing, training, prediction, and evaluation.
    """
    def __init__(self, hidden_size=100, init_method="xavier_uniform",
                 kohonen_lr=0.5, grossberg_lr=0.5, max_epochs=300, neighborhood_size=3,
                 batch_size=128, use_autoencoder=False, ae_dim=None, ae_epochs=50, ae_lr=0.001,
                 early_stopping=False, patience=10, random_state=None, device=None, verbose=False):
        # Validate hyperparameters.
        validate_hyperparameters(kohonen_lr=kohonen_lr, grossberg_lr=grossberg_lr,
                                 max_epochs=max_epochs, batch_size=batch_size,
                                 use_autoencoder=use_autoencoder, ae_dim=ae_dim)
        
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.kohonen_lr = kohonen_lr
        self.grossberg_lr = grossberg_lr
        self.max_epochs = max_epochs
        self.neighborhood_size = neighborhood_size
        self.batch_size = batch_size
        self.use_autoencoder = use_autoencoder
        self.ae_dim = ae_dim
        self.ae_epochs = ae_epochs
        self.ae_lr = ae_lr
        self.early_stopping = early_stopping
        self.patience = patience
        self.random_state = random_state
        self.verbose = verbose

        # Device selection: use provided device if available; otherwise, auto-select.
        if device is not None:
            self.device = device
            print(f"Using provided device: {self.device}")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("LOADED CUDA")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("LOADED MPS")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = torch.device("xpu")
                print("LOADED XPU")
            else:
                self.device = torch.device("cpu")
                print("CPU ROLLING")

        self.cpnn_ = None
        self.autoencoder_ = None
        self.input_size_ = None
        self.classes_ = None
        
    def fit(self, X, y):
        """
        Fits the CPNN model:
          - Converts input data to torch tensors.
          - Optionally pre-processes the data using an autoencoder.
          - Constructs the CPNN and trains it using the specified configuration.
          
        Args:
            X (ndarray): Input features.
            y (ndarray): Target labels.
            
        Returns:
            self
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
        
        # Convert input features and labels to tensors.
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if len(y.shape) == 1 or y.shape[1] == 1:
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
            self.classes_ = np.unique(y)
            # One-hot encode labels.
            y_tensor = torch.nn.functional.one_hot(y_tensor, num_classes=len(self.classes_)).float()
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            self.classes_ = np.arange(y_tensor.shape[1])
            
        self.input_size_ = X_tensor.shape[1]
        # If autoencoder is enabled, preprocess the input data.
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
        
        # Create dataset and DataLoader.
        dataset = torch.utils.data.TensorDataset(X_train, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Instantiate and train the CPNN model.
        self.cpnn_ = CounterPropagationNetwork(input_size, self.hidden_size, len(self.classes_),
                                                 init_method=self.init_method).to(self.device)
        config = {
            "kohonen_lr": self.kohonen_lr,
            "grossberg_lr": self.grossberg_lr,
            "max_epochs": self.max_epochs,
            "neighborhood_size": self.neighborhood_size,
            "log_interval": None if not self.verbose else max(1, self.max_epochs // 5),
            "verbose": self.verbose,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
        }
        self.cpnn_.fit(loader, val_data=None, config=config)
        return self
        
    def predict(self, X):
        """
        Predicts class labels for the given input.
        
        Args:
            X (ndarray): Input features.
            
        Returns:
            preds (ndarray): Predicted labels.
        """
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
        """
        Predicts class probabilities for the given input.
        
        Args:
            X (ndarray): Input features.
            
        Returns:
            prob (ndarray): Class probability distribution.
        """
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
        """
        Evaluates the model's accuracy on the provided data.
        
        Args:
            X (ndarray): Input features.
            y (ndarray): True labels.
            
        Returns:
            accuracy (float): Classification accuracy.
        """
        from sklearn.metrics import accuracy_score
        preds = self.predict(X)
        return accuracy_score(y, preds)
    
    def get_params(self, deep=True):
        """
        Returns the current hyperparameters as a dictionary.
        """
        return {
            "hidden_size": self.hidden_size,
            "init_method": self.init_method,
            "kohonen_lr": self.kohonen_lr,
            "grossberg_lr": self.grossberg_lr,
            "max_epochs": self.max_epochs,
            "neighborhood_size": self.neighborhood_size,
            "batch_size": self.batch_size,
            "use_autoencoder": self.use_autoencoder,
            "ae_dim": self.ae_dim,
            "ae_epochs": self.ae_epochs,
            "ae_lr": self.ae_lr,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "device": self.device,
        }
    
    def set_params(self, **params):
        """
        Update hyperparameters of the classifier.
        
        Args:
            params: Keyword arguments of hyperparameters.
            
        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self



# EXPLAIN HYPERPARAMETERS: WHY AND WHAT THEY DO
# WHY THE DEFAULTS
# ARTICLES?

# SPEED TEST

# HARDWARE EPOCH SELECTION

# EARLY STOPPING DIFERIDO (QUITAR EPOCHS)
# EPOCHS NO ES HIPERPARÁMETRO (EPOCHS MÁXIMOS)


# SI EARLY STOPPING DIFERIDO - HIPERPARÁMETRO 


# PACIENCIA DE TIEMPO O VALORES?



# TESTS UNITARIOS

# te amo
# yo más

# WRITE ORIGIN OF CODE USED (WHERE DID YOU READ HOW TO CREATE THE AUTOENCODER?)


# OMEGA 326 (PLANTA 3) - 4:15