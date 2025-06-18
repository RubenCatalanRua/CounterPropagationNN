import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


class FullCPNN(nn.Module):
    """
    A five-layer Counter-Propagation Neural Network (CPNN) with:
      1) Input → Kohonen (unsupervised clustering)
      2) Kohonen → forward Grossberg → class logits (supervised classification)
      3) Kohonen → reverse Grossberg → input reconstructions
      (autoencoder-style reconstruction)

    Attributes:
        input_size (int): Dimensionality of the (flattened) input features.

        hidden_size (int): Number of neurons in the Kohonen layer.

        output_size (int): Number of output classes.

        kohonen_weights (nn.Parameter): Weight matrix
        for Kohonen layer (shape: hidden_size × input_size).

        G_fwd (nn.Parameter): Forward Grossberg weight matrix
        (shape: output_size × hidden_size).

        G_rev (nn.Parameter): Reverse Grossberg weight matrix
        (shape: input_size × hidden_size).

        neighborhood_function (str): Type of neighborhood influence
        ('gaussian', 'triangular', or 'rectangular')
        .
        neighborhood_size (float): Initial size (σ) of the
        neighborhood function (minimum 1).

        val_criterion (nn.Module): CrossEntropyLoss for classification.

        recon_criterion (nn.Module): MSELoss for reconstruction.

        device (torch.device): Computation device (CPU or CUDA).
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 neighborhood_function: str = 'gaussian',
                 neighborhood_size: float = 3,
                 device: torch.device = None):
        """
        Initializes the FullCPNN model with specified layer sizes
        and neighborhood function.

        Args:
            input_size (int): Dimensionality of the (flattened) input.

            hidden_size (int): Number of neurons in the Kohonen layer.

            output_size (int): Number of output classes.

            neighborhood_function (str): Neighborhood influence function
                                    ('gaussian', 'triangular', 'rectangular').

            neighborhood_size (float): Initial size of
            the neighborhood function (σ).

            device (torch.device, optional): Computation device (CPU or CUDA).
                                             Defaults to auto-detection.
        """
        super(FullCPNN, self).__init__()

        # Device setup
        self.device = device or torch.device('cuda'
            if torch.cuda.is_available() else 'cpu')

        # Assertions to ensure positive dimension sizes
        assert input_size > 0, "Input size must be > 0"
        assert hidden_size > 0, "Hidden size must be > 0"
        assert output_size > 0, "Output size must be > 0"

        self.flatten = nn.Flatten()
        # Model dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize Kohonen weights
        self.kohonen_weights = nn.Parameter(torch.randn(
            hidden_size, input_size))
        with torch.no_grad():
            # Normalize each Kohonen weight vector
            self.kohonen_weights.data = F.normalize(self.kohonen_weights.data,
                p=2, dim=1)

        # Initialize Grossberg weights:
        #   - G_fwd: maps from hidden (Kohonen) to output classes
        #   (shape: output_size × hidden_size)
        #   - G_rev: maps from hidden (Kohonen) to reconstruct inputs
        #   (shape: input_size × hidden_size)
        self.G_fwd = nn.Parameter(torch.randn(output_size, hidden_size))
        self.G_rev = nn.Parameter(torch.randn(input_size, hidden_size))
        with torch.no_grad():
            # Normalize rows of Grossberg weights
            self.G_fwd.data = F.normalize(self.G_fwd.data, p=2, dim=1)
            self.G_rev.data = F.normalize(self.G_rev.data, p=2, dim=1)

        # Neighborhood settings for Kohonen learning
        self.neighborhood_function = neighborhood_function
        self.neighborhood_size = max(1.0, float(neighborhood_size))

        # Loss functions: classification (forward Grossberg)
        # and reconstruction (reverse Grossberg)
        self.val_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.recon_criterion = nn.MSELoss(reduction='mean')

        # Move parameters to the specified device
        self.to(self.device)

    def forward(self, x):
        """
        Forward pass through the FullCPNN.

        1) Compute best-matching unit (BMU) in Kohonen layer
        via Euclidean distance.
        2) Produce class logits via forward Grossberg: one-hot BMU → G_fwd.
        3) Produce reconstructions via reverse Grossberg: one-hot BMU → G_rev.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size) or any
            shape that flattens to input_size.

        Returns:
            output (Tensor): Logits for classification,
            shape (batch_size, output_size).

            recos (Tensor): Reconstructed inputs,
            shape (batch_size, input_size).

            winners (Tensor): Indices of winning Kohonen neurons,
            shape (batch_size,).
        """
        x = self.flatten(x)

        distances = torch.cdist(x, self.kohonen_weights)  # Euclidean distances
        winners = torch.argmin(distances, dim=1)
        batch_size = x.size(0)
        winner_one_hot = torch.zeros(
            batch_size, self.kohonen_weights.size(0), device=x.device)
        winner_one_hot.scatter_(1, winners.unsqueeze(1), 1)


        output = torch.matmul(winner_one_hot, self.G_fwd.t())

        recos = torch.matmul(winner_one_hot, self.G_rev.t())

        return output, recos, winners

    def update_kohonen(self, x, winner_indices, batch_size,
                       optimizer, neighborhood_size):
        """
        Updates Kohonen weights using the neighborhood function.

        Args:
            x (Tensor): Input batch.

            winner_indices (Tensor): Index of best-matching neuron per sample.

            batch_size (int): Size of the input batch.

            optimizer (torch.optim.Optimizer): Optimizer for Kohonen weights.

            neighborhood_size (float): Current neighborhood size (sigma).
        """

        x = self.flatten(x)

        hidden_indices = torch.arange(
            self.hidden_size, device=x.device, dtype=torch.float32)
        hidden_indices = hidden_indices.unsqueeze(0).expand(batch_size, -1)
        winner_indices_exp = winner_indices.unsqueeze(1).float()
        diff_indices = torch.abs(hidden_indices - winner_indices_exp)

        if self.neighborhood_function == 'rectangular':
            influence = (diff_indices <= neighborhood_size).float()
        elif self.neighborhood_function == 'triangular':
            influence = torch.clamp(
                1 - diff_indices / neighborhood_size, min=0)
        else:  # Gaussian
            influence = torch.exp(- (diff_indices ** 2) /
                                  (2 * (neighborhood_size ** 2)))

        x_expanded = x.unsqueeze(1)
        weights_expanded = self.kohonen_weights.unsqueeze(0)
        diff = x_expanded - weights_expanded
        influence = influence.unsqueeze(2)
        grad = - influence * diff
        grad_accum = grad.mean(dim=0)

        optimizer.zero_grad()
        self.kohonen_weights.grad = grad_accum
        optimizer.step()
        with torch.no_grad():
            self.kohonen_weights.data = F.normalize(
                self.kohonen_weights.data, p=2, dim=1)

    def update_grossberg(self, x, class_logits, recos, labels, winners,
                         opt_fwd, opt_rev):
        """
        Update both Grossberg weight matrices (forward and reverse)
        using supervised losses.

        Args:
            class_logits (Tensor): Predicted logits from forward Grossberg,
            shape (batch_size, output_size).

            recos (Tensor): Reconstructed inputs from reverse Grossberg,
            shape (batch_size, input_size).

            labels (Tensor): Ground-truth class labels, shape (batch_size,).

            winners (Tensor): BMU indices, shape (batch_size,).

            opt_fwd (Optimizer): Optimizer for
            forward Grossberg weights (G_fwd).

            opt_rev (Optimizer): Optimizer for
            reverse Grossberg weights (G_rev).

        """

        # Zero gradients for both forward and reverse optimizers

        opt_fwd.zero_grad()
        opt_rev.zero_grad()

        x = self.flatten(x)

        counts = torch.zeros(self.hidden_size, device=self.device)
        counts = counts.index_add(
            0, winners, torch.ones(x.size(0), device=self.device))
        y_onehot = F.one_hot(labels, num_classes=self.output_size).float()

        y_sum = torch.zeros(
            self.output_size, self.hidden_size, device=self.device)

        y_sum = y_sum.index_add(1, winners, y_onehot.transpose(0, 1))

        x_sum = torch.zeros(
            self.input_size, self.hidden_size, device=self.device)

        x_sum = x_sum.index_add(1, winners, x.transpose(0, 1))

        grad_fwd = self.G_fwd * counts.unsqueeze(0) - y_sum
        self.G_fwd.grad = grad_fwd

        grad_rev = self.G_rev * counts.unsqueeze(0) - x_sum
        self.G_rev.grad = grad_rev

        opt_fwd.step()
        opt_rev.step()

        """

        Placeholder backprop style updates

        opt_fwd.zero_grad()
        opt_rev.zero_grad()

        x = self.flatten(x)

        # Classification loss (forward Grossberg)
        loss_fwd = self.val_criterion(class_logits, labels)
        # Reconstruction loss (reverse Grossberg)
        loss_rev = self.recon_criterion(recos, x)

         # Combined loss
        total_loss = loss_fwd + loss_rev
        total_loss.backward()

        # Step both optimizers
        opt_fwd.step()
        opt_rev.step()

        """

    def fit(self,
            train_loader,
            val_loader=None,
            kohonen_epochs: int = 5,
            grossberg_epochs: int = 10,
            kohonen_lr: float = 1e-2,
            grossberg_lr: float = 1e-3,
            early_stopping: bool = False,
            patience: int = 5):
        """
        Trains the FullCPNN in two phases:
         1) Unsupervised Kohonen training for kohonen_epochs.
         2) Supervised Grossberg training (both forward and reverse)
         for grossberg_epochs.

        Learning rate schedulers are applied in each phase
        using CosineAnnealingLR.

        Args:
            train_loader (DataLoader): Dataloader for training data,
            yielding (x, y) or (x, _) for Kohonen phase.

            val_loader (DataLoader, optional): Dataloader for validation data.
            Used in Grossberg phase if provided.

            kohonen_epochs (int): Number of epochs for Kohonen-only training.

            grossberg_epochs (int): Number of epochs
            for Grossberg joint training.

            kohonen_lr (float): Initial learning rate for Kohonen SGD optimizer.

            grossberg_lr (float): Initial learning rate
            for both Grossberg Adam optimizers.

            early_stopping (bool): Whether to apply early stopping
            during Grossberg phase.

            patience (int): Number of epochs with no improvement
            before stopping early.

        Returns:
            self: The trained FullCPNN instance.
        """
        # --- Phase 1: Train Kohonen (unsupervised) ---
        opt_k = optim.SGD([self.kohonen_weights], lr=kohonen_lr, momentum=0.9)
        # Cosine annealing scheduler for Kohonen learning rate
        scheduler_kh = optim.lr_scheduler.CosineAnnealingLR(
            opt_k, T_max=kohonen_epochs, eta_min=kohonen_lr * 0.1
        )

        sigma_0 = min(self.neighborhood_size, self.hidden_size / 2)
        if sigma_0 < 1.0:
            sigma_0 = 1.0

        decay_enabled = sigma_0 > 1

        if decay_enabled:
            lambd = kohonen_epochs / math.log(sigma_0)

        #print('--- Phase 1: Training Kohonen (Unsupervised) ---')
        self.train()
        for epoch in range(1, kohonen_epochs + 1):
            #print(f"[Kohonen Epoch {epoch}/{kohonen_epochs}]")


            if decay_enabled:
                sigma_t = max(1.0, sigma_0 * math.exp(-epoch / lambd))
            else:
                sigma_t = sigma_0

            for batch_x,_ in train_loader:
                # In Kohonen phase, labels are not used
                batch_x = batch_x.to(self.device)

                # Find BMUs for this batch
                _, _, winner_indices = self.forward(batch_x)
                # Update Kohonen weights
                self.update_kohonen(
                    batch_x, winner_indices, batch_x.size(0), opt_k,
                    neighborhood_size=sigma_t)

            # Step learning rate scheduler after each epoch
            scheduler_kh.step()

        # --- Phase 2: Train Grossberg (supervised) ---
        opt_f = optim.AdamW([self.G_fwd], lr=grossberg_lr)
        opt_r = optim.AdamW([self.G_rev], lr=grossberg_lr)
        # Cosine annealing schedulers for Grossberg learning rates
        scheduler_fwd = optim.lr_scheduler.CosineAnnealingLR(
            opt_f, T_max=grossberg_epochs, eta_min=grossberg_lr * 0.1
        )
        scheduler_rev = optim.lr_scheduler.CosineAnnealingLR(
            opt_r, T_max=grossberg_epochs, eta_min=grossberg_lr * 0.1
        )

        best_val_loss = float('inf')
        init_patience = patience

        #print('\n--- Phase 2: Training Grossberg (Supervised) ---')
        for epoch in range(1, grossberg_epochs + 1):
            self.train()
            #print(f"[Grossberg Epoch {epoch}/{grossberg_epochs}]")
            epoch_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass: get logits, reconstructions, and BMUs
                cls_logits, recos, winners = self.forward(batch_x)
                # Compute and apply gradients for Grossberg layers
                self.update_grossberg(
                    batch_x, cls_logits, recos, batch_y, winners, opt_f, opt_r
                )

            # Step Grossberg schedulers
            scheduler_fwd.step()
            scheduler_rev.step()

            # Validation (if provided)
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, return_loss=True, combined=True)

                # Early stopping logic
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

        return self

    def evaluate(self, data_loader, return_loss = False, combined = False):
        """
        Evaluates the FullCPNN on provided data (validation or test).

        Computes combined classification + reconstruction loss
        and classification accuracy.

        Args:
            data_loader (DataLoader): DataLoader yielding (x, y) pairs.

            return_loss (bool): If True, returns combined loss;
            otherwise returns accuracy (%).

        Returns:
            float: Combined loss (if return_loss=True)
            or accuracy (%) otherwise.
        """
        if len(data_loader) == 0:
            raise ValueError("Evaluation loader is empty.")

        self.eval()
        total_loss = 0.0
        total_l1 = 0.0
        total_l2 = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                B = x.size(0)
                flat = self.flatten(x)

                # Forward pass
                logits, recos, _ = self.forward(x)
                # Compute losses
                l1 = self.val_criterion(logits, y)
                l2 = self.recon_criterion(recos, flat)

                total_l1 += l1.item() * B
                total_l2 += l2.item() * B
                total_loss += (l1.item() + l2.item()) * B

                # Classification accuracy
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += B

        avg_loss = total_loss / total
        avg_l1 = total_l1 / total
        avg_l2 = total_l2 / total
        acc = 100.0 * correct / total

        print(f"Eval — l1={avg_l1:.4f}, l2={avg_l2:.4f}")
        print(f"Combined_loss={avg_loss:.4f}, acc={acc:.2f}%")

        if return_loss:
            if combined:
                return avg_loss
            else:
                return avg_l1
        else:
            return acc