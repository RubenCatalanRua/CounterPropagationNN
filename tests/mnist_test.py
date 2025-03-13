# mnist_test.py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from CPNN import SimpleAutoencoder, train_autoencoder, CounterPropagationNetwork

print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

subset_size = 1000
X = X[:subset_size]
y = y[:subset_size]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convert data to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)
num_classes = len(torch.unique(y))
y_one_hot = torch.nn.functional.one_hot(y, num_classes=num_classes).float()

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, y_train = X_train.to(device), y_train.to(device)
X_val, y_val = X_val.to(device), y_val.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)
input_size_original = X_train.shape[1]

def objective(trial):
    # Hyperparameter sampling
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

    # Train autoencoder if needed
    if use_autoencoder:
        autoencoder = SimpleAutoencoder(input_size_original, ae_dim).to(device)
        autoencoder = train_autoencoder(autoencoder, X_train, epochs=ae_epochs, lr=ae_lr, batch_size=batch_size, device=device)
        with torch.no_grad():
            _, X_train_enc = autoencoder(X_train)
            _, X_val_enc   = autoencoder(X_val)
        new_input_size = ae_dim
    else:
        X_train_enc, X_val_enc = X_train, X_val
        new_input_size = input_size_original

    train_dataset = TensorDataset(X_train_enc, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    cp_net = CounterPropagationNetwork(new_input_size, hidden_size, num_classes, init_method=init_method).to(device)
    config = {
        "kohonen_lr": kohonen_lr,
        "grossberg_lr": grossberg_lr,
        "epochs": epochs,
        "neighborhood_size": neighborhood_size,
        "log_interval": max(1, epochs // 5),
        "verbose": True
    }
    cp_net.fit(train_loader, val_data=(X_val_enc, y_val), config=config)
    
    cp_net.eval()
    with torch.no_grad():
        outputs, _ = cp_net(X_val_enc)
        pred_labels = torch.argmax(outputs, dim=1)
        true_labels = torch.argmax(y_val, dim=1)
    val_acc = accuracy_score(true_labels.cpu().numpy(), pred_labels.cpu().numpy())
    print(f"Trial completed with Validation Accuracy: {val_acc*100:.2f}%")
    return val_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("\nBest trial:")
trial = study.best_trial
print("  Validation Accuracy: {:.2f}%".format(trial.value * 100))
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Re-train Final Model Using Best Hyperparameters and Evaluate on Test Set
best_params = trial.params
if best_params["use_autoencoder"]:
    autoencoder = SimpleAutoencoder(input_size_original, best_params["ae_dim"]).to(device)
    autoencoder = train_autoencoder(autoencoder, X_train, epochs=best_params["ae_epochs"], lr=best_params["ae_lr"], batch_size=best_params["batch_size"], device=device)
    with torch.no_grad():
        _, X_test_enc = autoencoder(X_test)
    final_input = X_test_enc
    final_input_size = best_params["ae_dim"]
else:
    final_input = X_test
    final_input_size = input_size_original

final_model = CounterPropagationNetwork(final_input_size, best_params["hidden_size"], num_classes, init_method=best_params["init_method"]).to(device)
train_dataset = TensorDataset(X_train if not best_params["use_autoencoder"] 
                              else autoencoder(X_train)[1], y_train)
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
final_model.fit(train_loader, val_data=(X_val if not best_params["use_autoencoder"] 
                                         else autoencoder(X_val)[1], y_val), 
                config={
                    "kohonen_lr": best_params["kohonen_lr"],
                    "grossberg_lr": best_params["grossberg_lr"],
                    "epochs": best_params["epochs"],
                    "neighborhood_size": best_params["neighborhood_size"],
                    "log_interval": max(1, best_params["epochs"] // 5),
                    "verbose": True
                })

final_model.eval()
with torch.no_grad():
    outputs, _ = final_model(final_input)
    predicted_labels = torch.argmax(outputs, dim=1)
    true_labels = torch.argmax(y_test, dim=1)
test_acc = accuracy_score(true_labels.cpu().numpy(), predicted_labels.cpu().numpy())
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# Visualize the Confusion Matrix
conf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted_labels.cpu().numpy())
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(num_classes), yticklabels=range(num_classes))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix on Test Set")
plt.show()