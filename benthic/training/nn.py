r"""A collection of tools to train a Neural Network."""

import numpy as np
import torch
import torch.nn as nn
import wandb

from benthic.data.tools import Statistics


# fmt: off
class FullyConnectedNeuralNetwork(nn.Module):
    r"""Fully connected neural network with Gaussian output (mean and variance).

    This network predicts both the mean and variance of a Gaussian distribution
    for each output dimension. The variance is constrained to be positive using
    the log-exponential trick.

    Arguments:
        input_size: Dimension of input features.
        output_size: Dimension of output predictions.
        layers: List of hidden layer sizes (e.g., [64, 64] creates two 64-neuron layers).
        dropout: Dropout probability applied after each hidden layer (default: 0.05).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        layers: list[int],
        dropout: float = 0.05,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        # Build hidden layers
        hidden_layers = []
        prev_size = input_size

        for hidden_size in layers:
            hidden_layers.append(nn.Linear(prev_size, hidden_size))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.hidden = nn.Sequential(*hidden_layers)

        # Output heads: one for mean, one for log-variance
        self.mean_head = nn.Linear(prev_size, output_size)
        self.logvar_head = nn.Linear(prev_size, output_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Forward pass.

        Arguments:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            mean: Predicted mean of shape (batch_size, output_size).
            variance: Predicted variance of shape (batch_size, output_size).
        """
        h = self.hidden(x)

        # Predict mean directly
        mean = self.mean_head(h)

        # Predict log-variance and convert to variance (log-exp trick ensures positivity)
        log_variance = self.logvar_head(h)
        variance = torch.exp(log_variance)

        return mean, variance


def loss(
    y_true: torch.Tensor,
    y_pred_mean: torch.Tensor,
    y_pred_variance: torch.Tensor,
) -> torch.Tensor:
    r"""Negative log-likelihood loss for Gaussian predictions.

    Mathematical Formula:
    ---------------------

        NLL = (1/2) * log(2π * σ²) + (1/2) * [(y - μ)² / σ²]

    Arguments:
        y_true: True target values of shape (batch_size, output_size).
        y_pred_mean: Predicted means of shape (batch_size, output_size).
        y_pred_variance: Predicted variances of shape (batch_size, output_size).

    Returns:
        Scalar loss value.
    """
    return (
        0.5 * (torch.log(y_pred_variance) + (y_true - y_pred_mean) ** 2 / y_pred_variance)
    ).mean()


def training_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_folder: str,
    fold_idx: int,
    config: dict,
):
    r"""Train a Neural Network.

    Arguments:
        X_train: Input data for training.
        y_train: Output data for training.
        X_val: Input data for validation.
        y_val: Output data for validation.
        model_name: Name of the model to be trained, used for logging and saving.
        model_folder: Path to the folder where the model will be saved.
        config: Training configuration dictionary.
    """

    # Standardizing data (using training set statistics to leak no information from validation set)
    stats_X, stats_y = Statistics(), Statistics()
    stats_X.fit(X_train), stats_y.fit(y_train)
    X_train, y_train, X_val, y_val = (
        stats_X.standardize(X_train),
        stats_y.standardize(y_train),
        stats_X.standardize(X_val),
        stats_y.standardize(y_val),
    )

    # Converting numpy arrays to PyTorch tensors
    X_train, y_train, X_val, y_val = (
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train),
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val),
    )

    # Initializing the model
    model = FullyConnectedNeuralNetwork(
        input_size=X_train.shape[1],
        output_size=y_train.shape[1],
        layers=config["config_nn"]["layers"],
        dropout=config["config_nn"]["dropout"],
    )

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["config_nn"]["learning_rate"])

    # Storing losses and best validation loss for early stopping
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(config["config_nn"]["epochs"]):
        #
        # ===================
        #      TRAINING
        # ===================
        model.train()

        # Forward pass
        y_mean_train, y_var_train = model(X_train)

        # Compute loss
        train_loss = loss(y_train, y_mean_train, y_var_train)

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # ===================
        #      VALIDATION
        # ===================
        model.eval()
        with torch.no_grad():
            y_mean_val, y_var_val = model(X_val)
            val_loss = loss(y_val, y_mean_val, y_var_val)

        # Store losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Check for improvement and save model if it's the best so far
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            torch.save(model.state_dict(), model_folder / "best_model.pth")

        # Logging to Weights & Biases
        wandb.log({
            "Training & Validation / Epoch": epoch + 1,
            "Training & Validation / Loss (Training)": train_loss.item(),
            "Training & Validation / Loss (Validation)": val_loss.item(),
        })

    # Logging the best validation loss for the fold
    wandb.log({"Training & Validation / Best Loss (Validation)": best_val_loss})

    # Loading the best model for final evaluation
    model.load_state_dict(torch.load(model_folder / "best_model.pth", weights_only=True))
    model.eval()

    # Predictions on validation set for final evaluation
    with torch.no_grad():
        y_mean_val, y_var_val = model(X_val)

    # Unscaling the data
    y_mean_val, y_var_val = (
        stats_y.unstandardize(y_mean_val.numpy()),
        stats_y.unstandardize(np.sqrt(y_var_val.numpy())) ** 2,
    )

    # Saving results and validation set
    torch.save(y_val,      model_folder / f"f{fold_idx}_y_true.pt")
    torch.save(y_mean_val, model_folder / f"f{fold_idx}_y_pred_mean.pt")
    torch.save(y_var_val,  model_folder / f"f{fold_idx}_y_pred_var.pt")
