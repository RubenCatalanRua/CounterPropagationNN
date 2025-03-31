import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import time
import logging
from abc import ABC, abstractmethod

# --------------------------
# Utility Functions
# --------------------------

def validate_hyperparameters(**kwargs):
    if not isinstance(kwargs.get("kohonen_lr", 0.0), (int, float)) or kwargs.get("kohonen_lr", 0.0) <= 0:
        raise ValueError("kohonen_lr must be a positive float.")
    if not isinstance(kwargs.get("grossberg_lr", 0.0), (int, float)) or kwargs.get("grossberg_lr", 0.0) <= 0:
        raise ValueError("grossberg_lr must be a positive float.")
    if not isinstance(kwargs.get("max_epochs", 0), int) or kwargs.get("max_epochs", 0) < 1:
        raise ValueError("max_epochs must be at least 1.")
    if not isinstance(kwargs.get("batch_size", 0), int) or kwargs.get("batch_size", 0) < 1:
        raise ValueError("batch_size must be at least 1.")
    if not isinstance(kwargs.get("patience", 0), int) or kwargs.get("patience", 0) < 1:
        raise ValueError("patience must be at least 1.")
    if not isinstance(kwargs.get("log_interval", 0), int) or kwargs.get("log_interval", 0) < 1:
        raise ValueError("log_interval must be at least 1.")
    if kwargs.get("distance_metric", 'euclidean') not in ['euclidean', 'manhattan', 'cosine']:
        raise ValueError("distance_metric must be one of 'euclidean', 'manhattan', or 'cosine'.")
    if kwargs.get("neighborhood_function", 'gaussian') not in ['gaussian', 'rectangular', 'triangular']:
        raise ValueError("neighborhood_function must be one of 'gaussian', 'rectangular', or 'triangular'.")
    if not isinstance(kwargs.get("hidden_size", 0), int) or kwargs.get("hidden_size", 0) < 1:
        raise ValueError("hidden_size must be at least 1.")

# --------------------------
# Autoencoder
# --------------------------

class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_dim, hidden_layers=1, hidden_size=128,
                 activation='relu', conv=False):
        super(Autoencoder, self).__init__()
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.hidden_size = hidden_size
        self.conv = conv

        # Set activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()  # Default to ReLU

        if conv:
            side = int(np.sqrt(input_size))
            if side * side != input_size:
                logging.warning(f"Input size {input_size} is not a perfect square. Reshaping to {side}x{side}.")
            self.side = side
            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
                self.activation,
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                self.activation,
                nn.Flatten(),
                nn.Linear(32 * (side // 4) * (side // 4), encoding_dim),
                self.activation
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_dim, 32 * (side // 4) * (side // 4)),
                self.activation,
                nn.Unflatten(1, (32, side // 4, side // 4)),
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
                self.activation,
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid()
            )
        else:
            encoder_layers = []
            decoder_layers = []
            if hidden_layers == 0:
                encoder_layers.append(nn.Linear(input_size, encoding_dim))
                encoder_layers.append(self.activation)
                decoder_layers.append(nn.Linear(encoding_dim, input_size))
                decoder_layers.append(self.activation)
            else:
                encoder_layers.append(nn.Linear(input_size, hidden_size))
                encoder_layers.append(self.activation)
                for _ in range(hidden_layers - 1):
                    encoder_layers.append(nn.Linear(hidden_size, hidden_size))
                    encoder_layers.append(self.activation)
                encoder_layers.append(nn.Linear(hidden_size, encoding_dim))
                encoder_layers.append(self.activation)

                decoder_layers.append(nn.Linear(encoding_dim, hidden_size))
                decoder_layers.append(self.activation)
                for _ in range(hidden_layers - 1):
                    decoder_layers.append(nn.Linear(hidden_size, hidden_size))
                    decoder_layers.append(self.activation)
                decoder_layers.append(nn.Linear(hidden_size, input_size))
                decoder_layers.append(nn.Sigmoid())

            self.encoder = nn.Sequential(*encoder_layers)
            self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        if self.conv:
            x = x.view(-1, 1, self.side, self.side)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if self.conv:
            decoded = decoded.view(-1, self.input_size)
        return decoded, encoded

def train_autoencoder(autoencoder, train_data, val_data=None, epochs=50, lr=0.001,
                      batch_size=128, device=None, verbose=False, patience=10):
    if lr <= 0:
        raise ValueError("Autoencoder learning rate must be positive.")
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    autoencoder.to(device)
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience // 2,
                                                           factor=0.5, verbose=verbose)

    train_dataset = TensorDataset(train_data, train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if val_data is not None:
        val_dataset = TensorDataset(val_data, val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        best_val_loss = float('inf')
        epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch_x, _ in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            decoded, _ = autoencoder(batch_x)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        avg_loss = epoch_loss / len(train_data)
        if verbose:
            logging.info(f"Autoencoder Epoch {epoch}/{epochs}: Training Loss = {avg_loss:.6f}")

        if val_data is not None:
            autoencoder.eval()
            with torch.no_grad():
                val_loss = 0.0
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    decoded, _ = autoencoder(batch_x)
                    loss = criterion(decoded, batch_x)
                    val_loss += loss.item() * batch_x.size(0)
                avg_val_loss = val_loss / len(val_data)
                logging.info(f"Autoencoder Epoch {epoch}/{epochs}: Validation Loss = {avg_val_loss:.6f}")
                scheduler.step(avg_val_loss)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        logging.info(f"Early stopping for autoencoder: no improvement for {patience} epochs.")
                        break
            autoencoder.train()
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

        # Initialize Kohonen (winner-take-all) weights.
        self.kohonen_weights = nn.Parameter(torch.empty(hidden_size, input_size))
        # Initialize Grossberg (output mapping) weights.
        self.grossberg_weights = nn.Parameter(torch.empty(output_size, hidden_size))

        # Weight initialization.
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

        self.distance_metric = 'euclidean'
        self.neighborhood_function = 'gaussian'

    def forward(self, x):
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(x, self.kohonen_weights)
        elif self.distance_metric == 'manhattan':
            distances = torch.cdist(x, self.kohonen_weights, p=1)
        elif self.distance_metric == 'cosine':
            x_norm = F.normalize(x, p=2, dim=1)
            weights_norm = F.normalize(self.kohonen_weights, p=2, dim=1)
            distances = 1 - torch.matmul(x_norm, weights_norm.t())
        else:
            distances = torch.cdist(x, self.kohonen_weights)

        winner_indices = torch.argmin(distances, dim=1)
        batch_size = x.size(0)
        winner_one_hot = torch.zeros(batch_size, self.hidden_size, device=x.device)
        winner_one_hot.scatter_(1, winner_indices.unsqueeze(1), 1)
        output = torch.matmul(winner_one_hot, self.grossberg_weights.t())
        return output, winner_indices

    def train_kohonen(self, x, optimizer, learning_rate=0.1, neighborhood_size=3):
        distances = torch.cdist(x, self.kohonen_weights)
        winner_indices = torch.argmin(distances, dim=1)
        hidden_indices = torch.arange(self.hidden_size, device=x.device).float()
        
        optimizer.zero_grad()
        grad_accum = torch.zeros_like(self.kohonen_weights)
        for i in range(x.size(0)):
            winner_index = winner_indices[i].item()
            input_vector = x[i]
            diff_indices = torch.abs(hidden_indices - winner_index)
            if self.neighborhood_function == 'gaussian':
                influence = torch.exp(- (diff_indices ** 2) / (2 * (neighborhood_size ** 2)))
            elif self.neighborhood_function == 'rectangular':
                influence = (diff_indices <= neighborhood_size).float()
            elif self.neighborhood_function == 'triangular':
                influence = torch.clamp(1 - diff_indices / neighborhood_size, min=0)
            else:
                influence = torch.exp(- (diff_indices ** 2) / (2 * (neighborhood_size ** 2)))
            grad_accum += - influence.unsqueeze(1) * (input_vector - self.kohonen_weights)
        grad_accum /= x.size(0)
        self.kohonen_weights.grad = grad_accum * learning_rate
        optimizer.step()
        self.kohonen_weights.data = F.normalize(self.kohonen_weights.data, p=2, dim=1)

    def train_grossberg(self, x, y, optimizer, learning_rate=0.1):
        _, winner_indices = self.forward(x)
        batch_size = x.size(0)
        optimizer.zero_grad()
        counts = torch.zeros(self.hidden_size, device=x.device)
        counts = counts.index_add(0, winner_indices, torch.ones(batch_size, device=x.device))
        y_sum = torch.zeros(self.output_size, self.hidden_size, device=x.device)
        y_sum = y_sum.index_add(1, winner_indices, y.transpose(0, 1))
        grad = self.grossberg_weights * counts.unsqueeze(0) - y_sum
        self.grossberg_weights.grad = grad * learning_rate
        optimizer.step()

    def fit(self, train_loader, val_data=None, config=None):
        if config is None:
            config = {
                "kohonen_lr": 0.1,
                "grossberg_lr": 0.1,
                "max_epochs": 300,
                "neighborhood_size": 3,
                "log_interval": 10,
                "verbose": False,
                "early_stopping": False,
                "neighborhood_function": 'gaussian',
                "kohonen_optimizer_type": "sgd",
                "grossberg_optimizer_type": "sgd",
            }
        verbose = config.get("verbose", False)
        max_epochs = config.get("max_epochs", 300)
        initial_neighborhood = config.get("neighborhood_size", 3)
        early_stopping = config.get("early_stopping", False)
        log_interval = config.get("log_interval", 10)
        self.neighborhood_function = config.get("neighborhood_function", 'gaussian')

        if val_data is not None:
            X_val, y_val = val_data
            criterion = nn.MSELoss()
        else:
            criterion = None

        best_val_acc = -float('inf')
        best_val_loss = float('inf')
        no_improve_counter = 0
        error_increase_counter = 0

        kohonen_lr = config.get("kohonen_lr", 0.1)
        grossberg_lr = config.get("grossberg_lr", 0.1)
        if config.get("kohonen_optimizer_type", "sgd").lower() == "adam":
            kohonen_optimizer = torch.optim.Adam([self.kohonen_weights], lr=kohonen_lr)
        else:
            kohonen_optimizer = torch.optim.SGD([self.kohonen_weights], lr=kohonen_lr)
        if config.get("grossberg_optimizer_type", "sgd").lower() == "adam":
            grossberg_optimizer = torch.optim.Adam([self.grossberg_weights], lr=grossberg_lr)
        else:
            grossberg_optimizer = torch.optim.SGD([self.grossberg_weights], lr=grossberg_lr)

        kohonen_scheduler = None
        grossberg_scheduler = None
        if config.get("kohonen_lr_scheduler") is not None:
            kohonen_scheduler = config["kohonen_lr_scheduler"](kohonen_optimizer)
        if config.get("grossberg_lr_scheduler") is not None:
            grossberg_scheduler = config["grossberg_lr_scheduler"](grossberg_optimizer)

        for epoch in range(1, max_epochs + 1):
            decayed_neighborhood = max(1, initial_neighborhood - (initial_neighborhood - 1) * (epoch - 1) / (max_epochs - 1))
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.kohonen_weights.device)
                batch_y = batch_y.to(self.kohonen_weights.device)
                self.train_kohonen(batch_x, optimizer=kohonen_optimizer,
                                   learning_rate=kohonen_lr, neighborhood_size=decayed_neighborhood)
                self.train_grossberg(batch_x, batch_y, optimizer=grossberg_optimizer,
                                     learning_rate=grossberg_lr)

            if kohonen_scheduler is not None:
                kohonen_scheduler.step()
            if grossberg_scheduler is not None:
                grossberg_scheduler.step()

            if verbose and log_interval and epoch % log_interval == 0 and val_data is not None:
                self.eval()
                with torch.no_grad():
                    outputs, _ = self.forward(X_val.to(self.kohonen_weights.device))
                    pred_labels = torch.argmax(outputs, dim=1)
                    true_labels = torch.argmax(y_val.to(self.kohonen_weights.device), dim=1)
                    acc = (pred_labels == true_labels).float().mean().item() * 100
                    val_loss = criterion(outputs, y_val.to(self.kohonen_weights.device)).item() if criterion is not None else 0.0
                logging.info(f"Epoch {epoch}/{max_epochs} | Validation Accuracy: {acc:.2f}% | Validation Loss: {val_loss:.6f}")

                if early_stopping:
                    if acc > best_val_acc:
                        best_val_acc = acc
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        error_increase_counter = 0
                    else:
                        error_increase_counter += 1

                    if no_improve_counter >= config.get("early_stopping_no_improve_patience", config["patience"]):
                        logging.warning(f"Early stopping triggered: No improvement in accuracy for {no_improve_counter} epochs.")
                        break
                    if error_increase_counter >= config.get("early_stopping_error_increase_patience", config["patience"]):
                        logging.warning(f"Early stopping triggered: Validation loss did not decrease for {error_increase_counter} epochs.")
                        break
                self.train()
            elif verbose and log_interval and epoch % log_interval == 0:
                logging.info(f"Epoch {epoch}/{max_epochs} completed.")
        return self

# --------------------------
# Abstract Base Class
# --------------------------

class CPNNBase(ABC):
    @abstractmethod
    def __init__(
        self,
        input_size,
        hidden_size,
        init_method,
        kohonen_lr,
        grossberg_lr,
        max_epochs,
        neighborhood_size,
        batch_size,
        use_autoencoder,
        ae_dim,
        ae_epochs,
        ae_lr,
        early_stopping,
        patience,
        early_stopping_no_improve_patience,
        early_stopping_error_increase_patience,
        random_state,
        device,
        verbose,
        log_interval,
        ae_hidden_layers,
        ae_activation,
        use_ae_conv,
        distance_metric,
        neighborhood_function,
        kohonen_lr_scheduler,
        grossberg_lr_scheduler,
        ae_lr_scheduler
    ):
        self.input_size = input_size
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
        self.log_interval = log_interval
        self.ae_hidden_layers = ae_hidden_layers
        self.ae_activation = ae_activation
        self.use_ae_conv = use_ae_conv
        self.distance_metric = distance_metric
        self.neighborhood_function = neighborhood_function
        self.kohonen_lr_scheduler = kohonen_lr_scheduler
        self.grossberg_lr_scheduler = grossberg_lr_scheduler
        self.ae_lr_scheduler = ae_lr_scheduler

        self.early_stopping_no_improve_patience = (early_stopping_no_improve_patience
                                                   if early_stopping_no_improve_patience is not None
                                                   else patience)
        self.early_stopping_error_increase_patience = (early_stopping_error_increase_patience
                                                       if early_stopping_error_increase_patience is not None
                                                       else patience)

        if device is not None:
            self.device = device
            logging.info(f"Using provided device: {self.device}")
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                logging.info("LOADED CUDA")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logging.info("LOADED MPS")
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                self.device = torch.device("xpu")
                logging.info("LOADED XPU")
            else:
                self.device = torch.device("cpu")
                logging.info("CPU IS BEING USED")

        self.cpnn_ = None
        self.autoencoder_ = None
        self.input_size_ = None
        self.classes_ = None

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict on new data. Must be implemented by subclasses.
        """
        pass

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        preds = self.predict(X)
        return accuracy_score(y, preds)

# --------------------------
# CPNNClassifier Subclass
# --------------------------

class CPNNClassifier(CPNNBase):
    def __init__(
        self,
        input_size,
        hidden_size=100,
        init_method="xavier_uniform",
        kohonen_lr=0.1,
        grossberg_lr=0.1,
        max_epochs=300,
        neighborhood_size=3,
        batch_size=128,
        use_autoencoder=False,
        ae_dim=None,
        ae_epochs=50,
        ae_lr=0.001,
        early_stopping=False,
        patience=10,
        early_stopping_no_improve_patience=None,
        early_stopping_error_increase_patience=None,
        random_state=None,
        device=None,
        verbose=False,
        log_interval=50,
        ae_hidden_layers=1,
        ae_activation='relu',
        use_ae_conv=False,
        distance_metric='euclidean',
        neighborhood_function='gaussian',
        kohonen_lr_scheduler=None,
        grossberg_lr_scheduler=None,
        ae_lr_scheduler=None
    ):
        super().__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            init_method=init_method,
            kohonen_lr=kohonen_lr,
            grossberg_lr=grossberg_lr,
            max_epochs=max_epochs,
            neighborhood_size=neighborhood_size,
            batch_size=batch_size,
            use_autoencoder=use_autoencoder,
            ae_dim=ae_dim,
            ae_epochs=ae_epochs,
            ae_lr=ae_lr,
            early_stopping=early_stopping,
            patience=patience,
            early_stopping_no_improve_patience=early_stopping_no_improve_patience,
            early_stopping_error_increase_patience=early_stopping_error_increase_patience,
            random_state=random_state,
            device=device,
            verbose=verbose,
            log_interval=log_interval,
            ae_hidden_layers=ae_hidden_layers,
            ae_activation=ae_activation,
            use_ae_conv=use_ae_conv,
            distance_metric=distance_metric,
            neighborhood_function=neighborhood_function,
            kohonen_lr_scheduler=kohonen_lr_scheduler,
            grossberg_lr_scheduler=grossberg_lr_scheduler,
            ae_lr_scheduler=ae_lr_scheduler
        )

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        if len(y.shape) == 1 or y.shape[1] == 1:
            y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
            self.classes_ = np.unique(y)
            y_tensor = torch.nn.functional.one_hot(y_tensor, num_classes=len(self.classes_)).float()
        else:
            y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
            self.classes_ = np.arange(y_tensor.shape[1])
        self.input_size_ = X_tensor.shape[1]

        validate_hyperparameters(
            input_size=self.input_size_,
            kohonen_lr=self.kohonen_lr,
            grossberg_lr=self.grossberg_lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            use_autoencoder=self.use_autoencoder,
            ae_dim=self.ae_dim,
            ae_epochs=self.ae_epochs,
            ae_lr=self.ae_lr,
            hidden_size=self.hidden_size,
            neighborhood_size=self.neighborhood_size,
            patience=self.patience,
            log_interval=self.log_interval,
            ae_hidden_layers=self.ae_hidden_layers,
            distance_metric=self.distance_metric,
            neighborhood_function=self.neighborhood_function
        )

        if self.use_autoencoder:
            if self.ae_dim is None:
                raise ValueError("ae_dim must be set if use_autoencoder is True.")
            self.autoencoder_ = Autoencoder(self.input_size_, self.ae_dim,
                                            hidden_layers=self.ae_hidden_layers,
                                            activation=self.ae_activation,
                                            conv=self.use_ae_conv).to(self.device)
            self.autoencoder_ = train_autoencoder(
                self.autoencoder_,
                X_tensor,
                epochs=self.ae_epochs,
                lr=self.ae_lr,
                batch_size=self.batch_size,
                device=self.device,
                verbose=self.verbose,
                patience=self.patience
            )
            with torch.no_grad():
                _, X_encoded = self.autoencoder_(X_tensor)
            X_train = X_encoded
            input_size = self.ae_dim
        else:
            X_train = X_tensor
            input_size = self.input_size_

        train_dataset = TensorDataset(X_train, y_tensor)
        loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.cpnn_ = CounterPropagationNetwork(
            input_size, self.hidden_size, len(self.classes_),
            init_method=self.init_method
        ).to(self.device)
        self.cpnn_.distance_metric = self.distance_metric
        self.cpnn_.neighborhood_function = self.neighborhood_function

        config = {
            "kohonen_lr": self.kohonen_lr,
            "grossberg_lr": self.grossberg_lr,
            "max_epochs": self.max_epochs,
            "neighborhood_size": self.neighborhood_size,
            "log_interval": self.log_interval,
            "verbose": self.verbose,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "early_stopping_no_improve_patience": self.early_stopping_no_improve_patience,
            "early_stopping_error_increase_patience": self.early_stopping_error_increase_patience,
            "neighborhood_function": self.neighborhood_function,
            "kohonen_optimizer_type": "sgd",
            "grossberg_optimizer_type": "sgd"
        }
        if self.kohonen_lr_scheduler is not None:
            config["kohonen_lr_scheduler"] = self.kohonen_lr_scheduler
        if self.grossberg_lr_scheduler is not None:
            config["grossberg_lr_scheduler"] = self.grossberg_lr_scheduler

        self.cpnn_.fit(loader, val_data=(X_train, y_tensor), config=config)
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
