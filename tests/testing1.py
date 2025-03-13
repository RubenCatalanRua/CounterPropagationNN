import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import optuna

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

def train_autoencoder(autoencoder, data, epochs=50, lr=0.001, batch_size=128):
    autoencoder.train()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(data, data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            decoded, _ = autoencoder(batch_x)
            loss = criterion(decoded, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        # Uncomment below to print autoencoder training loss if desired
        # print(f"Autoencoder Epoch {epoch}: Loss = {epoch_loss/len(data):.6f}")
    return autoencoder

# --------------------------
# CounterPropagation Network (CPNN)
# --------------------------
class CounterPropagationNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, init_method="xavier_uniform"):
        """
        Initialization of Kohonen and Grossberg layers.
        You can choose between Xavier and Kaiming methods.
        """
        super(CounterPropagationNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Kohonen layer (Winner-Take-All)
        self.kohonen_weights = nn.Parameter(torch.empty(hidden_size, input_size))
        # Grossberg layer (Outstar)
        self.grossberg_weights = nn.Parameter(torch.empty(output_size, hidden_size))
        
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
        Forward pass:
         - Computes the Euclidean distances between x and the Kohonen weights.
         - Determines the winning neuron and creates its one-hot encoding.
         - Computes the output from the Grossberg layer.
        """
        distances = torch.cdist(x, self.kohonen_weights)  # (batch_size, hidden_size)
        winner_indices = torch.argmin(distances, dim=1)     # (batch_size,)
        batch_size = x.size(0)
        # One-hot encoding for the winners
        winner_one_hot = torch.zeros(batch_size, self.hidden_size, device=x.device)
        winner_one_hot.scatter_(1, winner_indices.unsqueeze(1), 1)
        output = torch.matmul(winner_one_hot, self.grossberg_weights.t())
        return output, winner_indices

    def train_kohonen(self, x, learning_rate=0.1, neighborhood_size=3):
        """
        Update the Kohonen layer in a vectorized manner.
        For each sample, the Gaussian influence of each neuron is computed.
        """
        distances = torch.cdist(x, self.kohonen_weights)  # (batch_size, hidden_size)
        winner_indices = torch.argmin(distances, dim=1)     # (batch_size,)
        batch_size = x.size(0)
        hidden_indices = torch.arange(self.hidden_size, device=x.device).float()  # (hidden_size,)
        
        diff_indices = hidden_indices.unsqueeze(0) - winner_indices.unsqueeze(1).float()  # (batch_size, hidden_size)
        influence = torch.exp(- (diff_indices**2) / (2 * (neighborhood_size**2)))         # (batch_size, hidden_size)
        
        diff_x = x.unsqueeze(1) - self.kohonen_weights.unsqueeze(0)  # (batch_size, hidden_size, input_size)
        influence_expanded = influence.unsqueeze(2)                    # (batch_size, hidden_size, 1)
        
        update = learning_rate * (influence_expanded * diff_x).sum(dim=0)  # (hidden_size, input_size)
        with torch.no_grad():
            self.kohonen_weights.add_(update)
            self.kohonen_weights.copy_(torch.nn.functional.normalize(self.kohonen_weights, p=2, dim=1))

    def train_grossberg(self, x, y, learning_rate=0.1):
        """
        Update the Grossberg layer in a supervised manner.
        Samples are grouped by the winning neuron.
        """
        _, winner_indices = self.forward(x)
        with torch.no_grad():
            for j in range(self.hidden_size):
                mask = (winner_indices == j)
                if mask.sum() > 0:
                    diff = (y[mask] - self.grossberg_weights[:, j].unsqueeze(0)).sum(dim=0)
                    self.grossberg_weights[:, j] += learning_rate * diff

    def fit(self, train_loader, val_data=None, config=None):
        """
        Train the network using mini-batches.
        """
        if config is None:
            config = {
                "kohonen_lr": 0.5,
                "grossberg_lr": 0.5,
                "epochs": 300,
                "neighborhood_size": 3,
                "log_interval": 50
            }
            
        for epoch in range(1, config["epochs"] + 1):
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.kohonen_weights.device), batch_y.to(self.kohonen_weights.device)
                self.train_kohonen(batch_x, learning_rate=config["kohonen_lr"], neighborhood_size=config["neighborhood_size"])
                self.train_grossberg(batch_x, batch_y, learning_rate=config["grossberg_lr"])
                
            if epoch % config["log_interval"] == 0 and val_data is not None:
                X_val, y_val = val_data
                self.eval()
                with torch.no_grad():
                    outputs, _ = self.forward(X_val)
                    pred_labels = torch.argmax(outputs, dim=1)
                    true_labels = torch.argmax(y_val, dim=1)
                    acc = accuracy_score(true_labels.cpu().numpy(), pred_labels.cpu().numpy())
                    print(f"Epoch {epoch}: Validation Accuracy = {acc*100:.2f}%")
                self.train()

# --------------------------
# Load and preprocess data (MNIST)
# --------------------------
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Select a subset for quick tests (e.g., 1000 samples)
subset_size = 1000
X = X[:subset_size]
y = y[:subset_size]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
num_classes = len(torch.unique(y))
y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Move data to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)
X_val = X_val.to(device)
y_val = y_val.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
input_size_original = X_train.shape[1]

# --------------------------
# Optuna Objective Function
# --------------------------
def objective(trial):
    # Sample hyperparameters
    kohonen_lr = trial.suggest_float("kohonen_lr", 0.01, 0.5, log=True)
    grossberg_lr = trial.suggest_float("grossberg_lr", 0.01, 0.5, log=True)
    epochs = trial.suggest_int("epochs", 100, 300)
    neighborhood_size = trial.suggest_int("neighborhood_size", 3, 4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    hidden_size = trial.suggest_categorical("hidden_size", [50, 100, 250, 500, 1000])
    init_method = trial.suggest_categorical("init_method", ["xavier_uniform", "kaiming_uniform"])
    use_autoencoder = trial.suggest_categorical("use_autoencoder", [False, True])
    
    if use_autoencoder:
        ae_dim = trial.suggest_categorical("ae_dim", [50, 100])
        ae_epochs = trial.suggest_int("ae_epochs", 50, 100)
        ae_lr = trial.suggest_float("ae_lr", 0.01, 0.1, log=True)
    else:
        ae_dim = None

    # Train autoencoder if required
    if use_autoencoder:
        autoencoder = SimpleAutoencoder(input_size_original, ae_dim).to(device)
        autoencoder = train_autoencoder(autoencoder, X_train, epochs=ae_epochs, lr=ae_lr, batch_size=batch_size)
        with torch.no_grad():
            _, X_train_enc = autoencoder(X_train)
            _, X_val_enc   = autoencoder(X_val)
        new_input_size = ae_dim
    else:
        X_train_enc, X_val_enc = X_train, X_val
        new_input_size = input_size_original

    # Create DataLoader for training
    train_dataset = TensorDataset(X_train_enc, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Instantiate and train the CPNN
    cp_net = CounterPropagationNetwork(new_input_size, hidden_size, num_classes, init_method=init_method).to(device)
    config = {
        "kohonen_lr": kohonen_lr,
        "grossberg_lr": grossberg_lr,
        "epochs": epochs,
        "neighborhood_size": neighborhood_size,
        "log_interval": max(1, epochs // 5)
    }
    cp_net.fit(train_loader, val_data=(X_val_enc, y_val), config=config)
    
    # Evaluate on the validation set
    cp_net.eval()
    with torch.no_grad():
        outputs, _ = cp_net(X_val_enc)
        pred_labels = torch.argmax(outputs, dim=1)
        true_labels = torch.argmax(y_val, dim=1)
    val_acc = accuracy_score(true_labels.cpu().numpy(), pred_labels.cpu().numpy())
    print(f"Trial completed with Validation Accuracy: {val_acc*100:.2f}%")
    return val_acc

# --------------------------
# Run Optuna Study
# --------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("\nBest trial:")
trial = study.best_trial
print("  Validation Accuracy: {:.2f}%".format(trial.value * 100))
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# --------------------------
# Re-train Final Model Using Best Hyperparameters and Evaluate on Test Set
# --------------------------
best_params = trial.params
if best_params["use_autoencoder"]:
    autoencoder = SimpleAutoencoder(input_size_original, best_params["ae_dim"]).to(device)
    autoencoder = train_autoencoder(autoencoder, X_train, epochs=best_params["ae_epochs"], lr=best_params["ae_lr"], batch_size=best_params["batch_size"])
    with torch.no_grad():
        _, X_test_enc = autoencoder(X_test)
    final_input = X_test_enc
    final_input_size = best_params["ae_dim"]
else:
    final_input = X_test
    final_input_size = input_size_original

final_model = CounterPropagationNetwork(final_input_size, best_params["hidden_size"], num_classes, init_method=best_params["init_method"]).to(device)
train_dataset = TensorDataset(X_train if not best_params["use_autoencoder"] else autoencoder(X_train)[1], y_train)
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
final_model.fit(train_loader, val_data=(X_val if not best_params["use_autoencoder"] else autoencoder(X_val)[1], y_val), config={
    "kohonen_lr": best_params["kohonen_lr"],
    "grossberg_lr": best_params["grossberg_lr"],
    "epochs": best_params["epochs"],
    "neighborhood_size": best_params["neighborhood_size"],
    "log_interval": max(1, best_params["epochs"] // 5)
})
final_model.eval()
with torch.no_grad():
    outputs, _ = final_model(final_input)
    predicted_labels = torch.argmax(outputs, dim=1)
    true_labels = torch.argmax(y_test, dim=1)
test_acc = accuracy_score(true_labels.cpu().numpy(), predicted_labels.cpu().numpy())
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# --------------------------
# Visualize the Confusion Matrix
# --------------------------
conf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted_labels.cpu().numpy())
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()