class BasicNN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_layers, div_layer_value, output_size, dropout_rate, device):
        super(BasicNN, self).__init__()

        assert hidden_layers > 0, "Number of hidden layers must be greater than 0"

        self.flatten = nn.Flatten()
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.div_layer_value = div_layer_value


        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout_rate))

        for i in range(self.hidden_layers - 1):
            new_size = max(1, hidden_size // self.div_layer_value)
            layers.append(nn.Linear(hidden_size,
                                    new_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            hidden_size = new_size

        layers.append(nn.Linear(hidden_size, self.output_size))
        self.mlp = nn.Sequential(*layers)


        self.val_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.device = device

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits


    def fit(self, train_loader, val_loader, epochs, lr, early_stopping=False, patience=5):
        optimizer = optim.AdamW(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.1 * lr)

        best_val_loss = float('inf')
        init_patience = patience

        self.train()

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self(data)
                loss = self.val_criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()

            if val_loader is not None:
              val_loss = self.evaluate(val_loader, return_loss=True)
              print(f"[Epoch {epoch}] Validation loss: {val_loss:.4f}")

              if val_loss < best_val_loss:
                  best_val_loss = val_loss
                  patience = init_patience
              else:
                  if early_stopping:
                      if math.isnan(val_loss):
                          print("Early stopping triggered due to NaN loss.")
                          break
                      patience -= 1
                      if patience == 0:
                          print("Early stopping triggered.")
                          break


    def evaluate(self, data_loader, return_loss=False):
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.forward(x)
                loss = self.val_criterion(logits, y)

                batch_size = y.size(0)
                val_loss += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += batch_size

        avg_loss = val_loss / total
        acc = 100.0 * correct / total
        print(f"Validation — Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%")

        if return_loss:
            return avg_loss
        else:
            return acc