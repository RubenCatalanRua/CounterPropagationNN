import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import optuna

from BaseCPNN import BaseCPNN


transform = transforms.Compose([transforms.ToTensor()])

# Load full training dataset
full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)


# Split into 15,000 training and 5,000 validation
train_size = int(0.4 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# Load test dataset (already predefined in MNIST)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(full_train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

print(f"Training samples: {len(train_dataset)}")
print(f"Validation samples: {len(val_dataset)}")
print(f"Testing samples: {len(test_dataset)}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model
sample_image, _ = full_train_dataset[0]  # Get first image
input_size = sample_image.numel()
output_size = len(full_train_dataset.classes)    # number of classes for MNIST

print("Input size:", input_size)
print("Output size:", output_size)

# Train the model

def objective(trial):
    # Hyperparameters to tune
    kohonen_lr = trial.suggest_float("kohonen_lr", 0.1, 0.6, log=True)
    grossberg_lr = trial.suggest_float("grossberg_lr", 0.1, 0.4, log=True)
    neighborhood_size = trial.suggest_int("neighborhood_size", 12, 16)
    neighborhood_function = trial.suggest_categorical("neighborhood_function", ['gaussian', 'triangular', 'rectangular'])
    hidden_size = trial.suggest_int("hidden_size", 200, 2000, step=50)
    epochs = trial.suggest_int("epochs", 50, 50, step=5)  # Adjust if needed
    early_stopping = trial.suggest_categorical("early_stopping", [True, False])
    patience = trial.suggest_int("patience", 10, 10, step=5)
    # Fixed batch size through cache

    # Create the model with the current hyperparameters.
    # Assumes input_size and output_size are defined globally (e.g., based on your dataset).
    model = BaseCPNN(input_size, hidden_size, output_size).to(device)

    # Train the model using the current hyperparameters.
    model.fit(train_loader=test_loader,
    		  device=device,
              val_loader=test_loader,
              epochs=epochs,
              kohonen_lr=kohonen_lr,
              grossberg_lr=grossberg_lr,
              neighborhood_size=neighborhood_size,
              neighborhood_function=neighborhood_function,
              early_stopping=early_stopping,
              patience=patience)

    # Evaluate the model on the test set (or validation set).
    val_loss = model.evaluate(test_loader, return_loss=True)
    print(f"Validation loss: {val_loss}")
    return val_loss

# Create and run the Optuna study.
study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=1)

print("Best trial:")
trial = study.best_trial
print("  Accuracy: ", trial.value)
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")