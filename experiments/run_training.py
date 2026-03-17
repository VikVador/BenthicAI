r"""Script to launch a training pipeline."""

import argparse
import wandb

from benthic.training import training
from dawgz import job, schedule

# isort: split
from benthic.config import FOLDER_MODEL_GP, FOLDER_MODEL_NN
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

    args           = parser.parse_args()
    configs        = load_configuration(args.config)
    config_cluster = configs[0].get("Cluster").copy()

    # Looping over all models to be trained
    for cfg in configs:

        # Extracting training configuration
        cfg = cfg["Training"]

        # Creating model name
        model_name   = wandb.util.generate_id()
        model_folder = FOLDER_MODEL_NN / f"{model_name}" if cfg["config_model"] == "nn" else FOLDER_MODEL_GP / f"{model_name}"
        if not model_folder.exists():
            model_folder.mkdir(parents=True)

        # Extracting number of models given k-fold cross-validation
        num_models = cfg["config_dataset"]["r_validation"] * cfg["config_dataset"]["k_validation"]

        # Launching parallel training
        @job(array=num_models, **config_cluster)
        def train(i: int, model_name=model_name, cfg=cfg):
            training(model_name = model_name, fold_idx = i, config = cfg)

        # Launching the training jobs
        schedule(
            train,
            name="BENTHIC-TRAINING",
            backend="slurm",
            export="ALL",
        )
