import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import time

# Assuming CPNN_v3.py is in the same directory
from CPNN import CPNNClassifier  # Import the updated wrapper class

def preprocess_data(X, y, subset_size=2000, random_state=42):
    """
    Preprocesses the MNIST data:
        - Uses only a subset for speed.
        - Scales features using StandardScaler.
        - Splits the data into train, validation, and test sets.
    """
    # Use a subset of the data
    X = X[:subset_size]
    y = y[:subset_size]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data: first into training and temporary set,
    # then split the temporary set equally into validation and test sets.
    X_temp, X_test, y_temp, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=random_state, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test

# --- Load and Preprocess MNIST ---
print("Loading Digits dataset...")
digits = load_digits()
X, y = digits.data, digits.target

# Adjust the subset size if desired
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(X, y, subset_size=3000)

# --- External Device Selection ---
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- Hardware Epoch Speed Test (Run Once) ---
use_hardware_epochs = False
hardware_time_limit = 10.0  # seconds

print('Input feature size:', X_train.shape[1])
input_size = X_train.shape[1]  # For MNIST, 784

if use_hardware_epochs:
    dummy_clf = CPNNClassifier(
        input_size=input_size,
        hidden_size=50,
        init_method="xavier_uniform",
        kohonen_lr=0.1,
        grossberg_lr=0.1,
        max_epochs=1,
        neighborhood_size=3,
        batch_size=32,
        use_autoencoder=False,
        random_state=42,
        device=device,
        verbose=False
    )
    speed_test_size = max(1, int(0.1 * X_train.shape[0]))
    X_speed = X_train[:speed_test_size]
    y_speed = y_train[:speed_test_size]
    start_time = time.time()
    dummy_clf.fit(X_speed, y_speed)
    epoch_time = time.time() - start_time
    computed_hardware_max_epochs = int((hardware_time_limit / epoch_time) / 2)
    print(f"Hardware speed test: epoch time = {epoch_time:.4f}s, computed max_epochs = {computed_hardware_max_epochs}")
else:
    computed_hardware_max_epochs = None

# --- Define the Objective Function for Optuna ---
def objective(trial):
    # Hyperparameter sampling
    hidden_size = trial.suggest_int("hidden_size", 50, 1000)
    init_method = trial.suggest_categorical("init_method", ["xavier_uniform", "kaiming_uniform", "xavier_normal", "kaiming_normal"])
    kohonen_lr = trial.suggest_float("kohonen_lr", 0.001, 0.5, log=True)
    grossberg_lr = trial.suggest_float("grossberg_lr", 0.001, 0.5, log=True)
    sampled_max_epochs = trial.suggest_int("max_epochs", 100, 300)
    if use_hardware_epochs and computed_hardware_max_epochs is not None:
        max_epochs = min(sampled_max_epochs, computed_hardware_max_epochs)
    else:
        max_epochs = sampled_max_epochs
    neighborhood_size = trial.suggest_int("neighborhood_size", 3, 5)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    use_autoencoder = trial.suggest_categorical("use_autoencoder", [True, False])
    early_stopping = trial.suggest_categorical("early_stopping", [True, False])
    patience = trial.suggest_int("patience", 5, 40) if early_stopping else 10
    ae_hidden_layers = trial.suggest_int("ae_hidden_layers", 0, 3)
    ae_activation = trial.suggest_categorical("ae_activation", ['relu', 'tanh', 'sigmoid'])
    use_ae_conv = trial.suggest_categorical("use_ae_conv", [False, True])
    distance_metric = trial.suggest_categorical("distance_metric", ['euclidean', 'manhattan', 'cosine'])
    neighborhood_function = trial.suggest_categorical("neighborhood_function", ['gaussian', 'rectangular', 'triangular'])
    optimizer_type = trial.suggest_categorical("optimizer", ['SGD', 'Adam', 'AdamW'])
    scheduler_type = trial.suggest_categorical("scheduler", [None, 'StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    hidden_layers = trial.suggest_int("hidden_layers", 1, 6)

    # Optimizer specific parameters
    kohonen_optimizer_params = {}
    grossberg_optimizer_params = {}
    if optimizer_type in ['Adam', 'AdamW']:
        kohonen_optimizer_params['betas'] = (trial.suggest_float("kohonen_beta1", 0.9, 0.999), trial.suggest_float("kohonen_beta2", 0.9, 0.999))
        grossberg_optimizer_params['betas'] = (trial.suggest_float("grossberg_beta1", 0.9, 0.999), trial.suggest_float("grossberg_beta2", 0.9, 0.999))
    if optimizer_type == 'AdamW':
        kohonen_optimizer_params['weight_decay'] = trial.suggest_float("kohonen_weight_decay", 1e-5, 1e-2, log=True)
        grossberg_optimizer_params['weight_decay'] = trial.suggest_float("grossberg_weight_decay", 1e-5, 1e-2, log=True)

    # Scheduler specific parameters
    scheduler_params = {}
    if scheduler_type == 'StepLR':
        scheduler_params['step_size'] = trial.suggest_int('step_size', 5, 50)
        scheduler_params['gamma'] = trial.suggest_float('gamma', 0.1, 0.9)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler_params['T_max'] = trial.suggest_int('t_max', 50, max_epochs)
        scheduler_params['eta_min'] = trial.suggest_float('eta_min', 0, 0.1)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler_params['factor'] = trial.suggest_float('factor', 0.1, 0.5)
        scheduler_params['patience'] = trial.suggest_int('plateau_patience', 2, 10)
        scheduler_params['min_lr'] = trial.suggest_float('min_lr', 1e-6, 1e-4, log=True)
    if use_autoencoder:
        ae_dim = trial.suggest_int("ae_dim", 1, input_size - 1)
        ae_epochs = trial.suggest_int("ae_epochs", 50, 100)
        ae_lr = trial.suggest_float("ae_lr", 0.01, 0.1, log=True)
    else:
        ae_dim = None
        ae_epochs = None
        ae_lr = None

    cpnn_clf = CPNNClassifier(
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
        random_state=42,
        device=device,
        verbose=False,
        log_interval=50,
        ae_hidden_layers=ae_hidden_layers,
        ae_activation=ae_activation,
        use_ae_conv=use_ae_conv,
        distance_metric=distance_metric,
        neighborhood_function=neighborhood_function,
        hidden_layers=hidden_layers
    )

    cpnn_clf.fit(X_train, y_train)
    preds = cpnn_clf.predict(X_val)
    val_acc = accuracy_score(y_val, preds)
    print(f"Trial completed with Validation Accuracy: {val_acc * 100:.2f}%")
    return val_acc

# --- Optimize Hyperparameters with Optuna ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=30)

print("\nBest trial:")
trial = study.best_trial
print("  Validation Accuracy: {:.2f}%".format(trial.value * 100))
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# --- Re-train Final Model on Training + Validation Data ---
best_params = trial.params
sampled_max_epochs = best_params["max_epochs"]
if use_hardware_epochs and computed_hardware_max_epochs is not None:
    final_max_epochs = min(sampled_max_epochs, computed_hardware_max_epochs)
else:
    final_max_epochs = sampled_max_epochs

final_clf = CPNNClassifier(
    input_size=input_size,
    hidden_size=best_params["hidden_size"],
    init_method=best_params["init_method"],
    kohonen_lr=best_params["kohonen_lr"],
    grossberg_lr=best_params["grossberg_lr"],
    max_epochs=final_max_epochs,
    neighborhood_size=best_params["neighborhood_size"],
    batch_size=best_params["batch_size"],
    use_autoencoder=best_params.get("use_autoencoder", False),
    ae_dim=best_params.get("ae_dim", None),
    ae_epochs=best_params.get("ae_epochs", None),
    ae_lr=best_params.get("ae_lr", None),
    early_stopping=best_params.get("early_stopping", False),
    patience=best_params.get("patience", 10),
    random_state=42,
    device=device,
    verbose=True,
    log_interval=50,
    ae_hidden_layers=best_params.get("ae_hidden_layers", 1),
    ae_activation=best_params.get("ae_activation", 'relu'),
    use_ae_conv=best_params.get("use_ae_conv", False),
    distance_metric=best_params.get("distance_metric", 'euclidean'),
    neighborhood_function=best_params.get("neighborhood_function", 'gaussian'),
    hidden_layers=best_params.get("hidden_layers", 1)
)
# Combine training and validation data for final training
X_train_val = np.concatenate([X_train, X_val], axis=0)
y_train_val = np.concatenate([y_train, y_val], axis=0)

final_clf.fit(X_train_val, y_train_val)
predicted_labels = final_clf.predict(X_test)
test_acc = accuracy_score(y_test, predicted_labels)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

conf_matrix = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix on Test Set")
plt.show()
