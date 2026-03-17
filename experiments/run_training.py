r"""Script to launch a training pipeline."""

import argparse
import wandb

from dawgz import job, schedule

# isort: split
# from benthic.training.gp import training_gp
from benthic.training.nn import training_nn
from benthic.training.tools import load_configuration

# fmt: off
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Launch a training pipeline.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the training .yml configuration file.",
    )

    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="nn",
        choices=["gp", "nn"],
        help="Model type, 'gp' for Gaussian Process and 'nn' for Neural Network.",
    )

    args           = parser.parse_args()
    configs        = load_configuration(args.config)
    config_cluster = configs[0].get("Cluster").copy()

    # Creating model name
    model_name = wandb.util.generate_id()

    # Looping over all models to be trained
    for cfg in configs:

        # Extracting training configuration
        cfg = cfg["Training"]

        # Extracting number of models given k-fold cross-validation
        num_models = cfg["config_dataset"]["r_validation"] * cfg["config_dataset"]["k_validation"]

        # Creating a job array for parallel training
        @job(array=num_models, **config_cluster)
        def train(i: int):
            training_nn(name = model_name, fold_idx = i, config = cfg)

        # Launching the training jobs
        schedule(
            train,
            name="BENTHIC-TRAINING",
            backend="slurm",
            export="ALL",
        )
