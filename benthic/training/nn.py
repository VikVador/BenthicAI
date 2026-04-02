r"""A collection of tools to train a Neural Network."""

import numpy as np
import torch
import torch.nn as nn
import wandb

from benthic.data.tools import Statistics


# fmt: off
class FullyConnectedNeuralNetwork(nn.Module):
    r"""Fully connected neural network with Gaussian Mixture Model (GMM) output.

    Arguments:
        input_size: Dimension of input features.
        output_size: Dimension of output predictions.
        layers: List of hidden layer sizes (e.g., [64, 64] creates two 64-neuron layers).
        dropout: Dropout probability applied after each hidden layer (default: 0.05).
        num_components: Number of Gaussian components in the mixture (default: 1).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        layers: list[int],
        dropout: float = 0.05,
        num_components: int = 1,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.num_components = num_components

        # Build hidden layers
        hidden_layers = []
        prev_size = input_size

        for hidden_size in layers:
            hidden_layers.append(nn.Linear(prev_size, hidden_size))
            hidden_layers.append(nn.GELU())
            hidden_layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        self.hidden = nn.Sequential(*hidden_layers)

        # Output heads: one for weights, one for means, one for log-variances
        self.weights_head = nn.Linear(prev_size, output_size * num_components)
        self.means_head   = nn.Linear(prev_size, output_size * num_components)
        self.logvars_head = nn.Linear(prev_size, output_size * num_components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass.

        Arguments:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            weights: Mixture weights of shape (batch_size, output_size, num_components).
            means: Predicted means of shape (batch_size, output_size, num_components).
            variances: Predicted variances of shape (batch_size, output_size, num_components).
        """
        h = self.hidden(x)
        batch_size = x.shape[0]

        # Predict GMM parameters (flat outputs)
        means          = self.means_head(h)    # (batch, output_size * K)
        logvars        = self.logvars_head(h)  # (batch, output_size * K)
        weights_logits = self.weights_head(h)  # (batch, output_size * K)

        # Reshape to (batch, output_size, K)
        means          = means.view(batch_size, self.output_size, self.num_components)
        logvars        = logvars.view(batch_size, self.output_size, self.num_components)
        weights_logits = weights_logits.view(batch_size, self.output_size, self.num_components)

        # Apply constraints
        weights   = torch.nn.functional.softmax(weights_logits, dim=-1)
        variances = torch.exp(logvars)

        return weights, means, variances

    def inference(self, x: torch.Tensor, num_samples: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Sample from the Gaussian mixture and compute empirical statistics.

        This method draws num_samples from the predicted mixture distribution
        and returns the empirical mean and variance of these samples.

        Arguments:
            x: Input tensor of shape (batch_size, input_size).
            num_samples: Number of Monte Carlo samples to draw (default: 100).

        Returns:
            empirical_mean: Mean of samples of shape (batch_size, output_size).
            empirical_var: Variance of samples of shape (batch_size, output_size).
        """
        weights, means, variances = self.forward(x)  # (batch, output_size, K)
        batch_size = x.shape[0]

        # Storage for samples: (num_samples, batch, output_size)
        samples = torch.zeros(
            num_samples, batch_size, self.output_size,
            device=x.device, dtype=x.dtype
        )

        # Sample num_samples times from the mixture
        for s in range(num_samples):
            # Sample component indices from categorical distribution
            component_dist = torch.distributions.Categorical(weights)
            component_indices = component_dist.sample()  # (batch, output_size)

            # Gather parameters for selected components
            selected_means = torch.gather(means, dim=2, index=component_indices.unsqueeze(-1)).squeeze(-1)  # (batch, output_size)

            selected_vars = torch.gather(
                variances, dim=2,
                index=component_indices.unsqueeze(-1)
            ).squeeze(-1)  # (batch, output_size)

            # Sample from N(selected_means, selected_vars)
            epsilon = torch.randn_like(selected_means)
            samples[s] = selected_means + torch.sqrt(selected_vars) * epsilon

        # Compute empirical statistics
        empirical_mean = samples.mean(dim=0)  # (batch, output_size)
        empirical_var = samples.var(dim=0)    # (batch, output_size)

        return empirical_mean, empirical_var


def loss(
    y_true: torch.Tensor,
    weights: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Negative log-likelihood loss for Gaussian Mixture Model predictions.

    Mathematical Formula:
    ---------------------

        NLL = -log[ Σ_k π_k * N(y | μ_k, σ²_k) ]
            = -log[ Σ_k exp(log π_k - 0.5*log(σ²_k) - 0.5*(y-μ_k)²/σ²_k) ]
            = -logsumexp_k[ log π_k - 0.5*log(σ²_k) - 0.5*(y-μ_k)²/σ²_k ]

    Arguments:
        y_true:    True target values of shape (batch_size, output_size).
        weights:   Mixture weights of shape (batch_size, output_size, num_components).
        means:     Predicted means of shape (batch_size, output_size, num_components).
        variances: Predicted variances of shape (batch_size, output_size, num_components).

    Returns:
        nll:      Scalar NLL averaged over batch and output dimensions.
        mse_mean: MSE between the GMM predicted mean and y_true (for comparison with GP).
        mean_var: Mean predicted aleatoric variance Σ_k π_k σ²_k (calibration indicator).
    """
    # Expand y_true for broadcasting: (batch, output_size) -> (batch, output_size, 1)
    y_true_exp = y_true.unsqueeze(-1)

    log_weights   = torch.log(weights + 1e-8)
    log_variances = torch.log(variances + 1e-8)
    squared_diff  = (y_true_exp - means) ** 2

    log_probs        = -0.5 * log_variances - 0.5 * squared_diff / variances
    log_mixture_probs = log_weights + log_probs
    log_likelihood   = torch.logsumexp(log_mixture_probs, dim=-1)
    nll              = -log_likelihood.mean()

    # MSE on the GMM mean: E[μ] = Σ_k π_k μ_k
    gmm_mean = (weights * means).sum(dim=-1)           # (batch, output_size)
    mse_mean = ((y_true - gmm_mean) ** 2).mean()

    # Mean aleatoric variance: E[σ²] = Σ_k π_k σ²_k
    mean_var = (weights * variances).sum(dim=-1).mean()

    return nll, mse_mean, mean_var


def training_nn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    coords_val: np.ndarray,
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
        coords_val: Coordinates (lat, lon, depth) of the validation stations, shape (n_val, 3).
        model_folder: Path to the folder where the model will be saved.
        fold_idx: Index of the fold for k-fold cross-validation (inner loop).
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
        num_components=config["config_nn"].get("num_components", 1),
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
        weights_train, means_train, vars_train = model(X_train)
        train_nll, train_mse, train_var = loss(y_train, weights_train, means_train, vars_train)

        # Backward pass
        optimizer.zero_grad()
        train_nll.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            weights_val, means_val, vars_val = model(X_val)
            val_nll, val_mse, val_var = loss(y_val, weights_val, means_val, vars_val)

        train_losses.append(train_nll.item())
        val_losses.append(val_nll.item())

        if val_nll.item() < best_val_loss:
            best_val_loss = val_nll.item()
            torch.save(model.state_dict(), model_folder / f"best_model_{fold_idx}.pth")

        if epoch % 1000 == 0:
            wandb.log({
                "Training & Validation / Epoch":              epoch + 1,
                "Training & Validation / NLL (Training)":     train_nll.item(),
                "Training & Validation / NLL (Validation)":   val_nll.item(),
                "Training & Validation / MSE (Training)":     train_mse.item(),
                "Training & Validation / MSE (Validation)":   val_mse.item(),
                "Training & Validation / Var (Training)":     train_var.item(),
                "Training & Validation / Var (Validation)":   val_var.item(),
            })

    wandb.log({"Training & Validation / Best Validation NLL (NN)": best_val_loss})

    # Loading the best model for final evaluation
    model.load_state_dict(torch.load(model_folder / f"best_model_{fold_idx}.pth", weights_only=True))
    model.eval()

    # Predictions on validation set for final evaluation (using inference sampling)
    num_inference_samples = config["config_nn"].get("num_inference_samples", 1000)
    with torch.no_grad():
        y_mean_val, y_var_val = model.inference(X_val, num_samples=num_inference_samples)

    # Saving results and validation set (standardized)
    np.save(model_folder / f"{fold_idx}_standardized_y_true.npy",      np.array(y_val))
    np.save(model_folder / f"{fold_idx}_standardized_y_pred_mean.npy", np.array(y_mean_val))
    np.save(model_folder / f"{fold_idx}_standardized_y_pred_var.npy",  np.array(y_var_val))

    # Unscaling the data
    y_val, y_mean_val, y_var_val = (
        stats_y.unstandardize(y_val.numpy()),
        stats_y.unstandardize(y_mean_val.numpy()),
        stats_y.unstandardize(np.sqrt(y_var_val.numpy())) ** 2,
    )

    # Saving results and validation set
    np.save(model_folder / f"{fold_idx}_y_true.npy",      np.array(y_val))
    np.save(model_folder / f"{fold_idx}_y_pred_mean.npy", np.array(y_mean_val))
    np.save(model_folder / f"{fold_idx}_y_pred_var.npy",  np.array(y_var_val))
    np.save(model_folder / f"{fold_idx}_coords.npy",      coords_val)
