# mnist_test.py
import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

from CPNN import CPNNClassifier  # Import the wrapper

print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Use a subset for speed
subset_size = 1000
X = X[:subset_size]
y = y[:subset_size]

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the objective function for hyperparameter tuning using optuna
def objective(trial):
    # Hyperparameter sampling
    hidden_size = trial.suggest_categorical("hidden_size", [50, 100, 250, 500, 1000])
    init_method = trial.suggest_categorical("init_method", ["xavier_uniform", "kaiming_uniform"])
    kohonen_lr = trial.suggest_float("kohonen_lr", 0.01, 0.5, log=True)
    grossberg_lr = trial.suggest_float("grossberg_lr", 0.01, 0.5, log=True)
    epochs = trial.suggest_int("epochs", 100, 300)
    neighborhood_size = trial.suggest_int("neighborhood_size", 3, 4)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    use_autoencoder = trial.suggest_categorical("use_autoencoder", [False, True])
    
    if use_autoencoder:
        ae_dim = trial.suggest_categorical("ae_dim", [50, 100])
        ae_epochs = trial.suggest_int("ae_epochs", 50, 100)
        ae_lr = trial.suggest_float("ae_lr", 0.01, 0.1, log=True)
    else:
        ae_dim = None
        ae_epochs = None
        ae_lr = None

    # Instantiate the classifier with sampled hyperparameters
    cpnn_clf = CPNNClassifier(
        hidden_size=hidden_size,
        init_method=init_method,
        kohonen_lr=kohonen_lr,
        grossberg_lr=grossberg_lr,
        epochs=epochs,
        neighborhood_size=neighborhood_size,
        batch_size=batch_size,
        use_autoencoder=use_autoencoder,
        ae_dim=ae_dim,
        ae_epochs=ae_epochs,
        ae_lr=ae_lr,
        random_state=42,
        verbose=False  # Set to True for detailed logging
    )
    
    # Fit the classifier on the training set
    cpnn_clf.fit(X_train, y_train)
    
    # Predict on the validation set and calculate accuracy
    preds = cpnn_clf.predict(X_val)
    val_acc = accuracy_score(y_val, preds)
    print(f"Trial completed with Validation Accuracy: {val_acc*100:.2f}%")
    return val_acc

# Optimize hyperparameters with optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("\nBest trial:")
trial = study.best_trial
print("  Validation Accuracy: {:.2f}%".format(trial.value * 100))
print("  Best hyperparameters: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# Re-train final model using the best hyperparameters and evaluate on test set
best_params = trial.params
final_clf = CPNNClassifier(
    hidden_size=best_params["hidden_size"],
    init_method=best_params["init_method"],
    kohonen_lr=best_params["kohonen_lr"],
    grossberg_lr=best_params["grossberg_lr"],
    epochs=best_params["epochs"],
    neighborhood_size=best_params["neighborhood_size"],
    batch_size=best_params["batch_size"],
    use_autoencoder=best_params.get("use_autoencoder", False),
    ae_dim=best_params.get("ae_dim", None),
    ae_epochs=best_params.get("ae_epochs", None),
    ae_lr=best_params.get("ae_lr", None),
    random_state=42,
    verbose=True  # Enable verbose for final training
)

final_clf.fit(X_train, y_train)
predicted_labels = final_clf.predict(X_test)
test_acc = accuracy_score(y_test, predicted_labels)
print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")

# Visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix on Test Set")
plt.show()