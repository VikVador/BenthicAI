r"""Script to launch a training pipeline."""

import argparse
import wandb

from benthic.training import training
from dawgz import job, schedule

# fmt: off
#
# isort: split
from benthic.config import FOLDER_MODEL_GP, FOLDER_MODEL_NN
from benthic.training.tools import load_configuration

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a training pipeline.")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the training .yml configuration file.",
    )

    args = parser.parse_args()
    configs = load_configuration(args.config)
    config_cluster_train = configs[0]["Cluster"]["Train"].copy()
    config_cluster_evaluate = configs[0]["Cluster"]["Evaluate"].copy()

    for cfg in configs:

        # Extracting the training configuration
        cfg = cfg["Training"]

        # Creating model name
        model_name = wandb.util.generate_id()
        model_folder = (
            FOLDER_MODEL_NN / f"{model_name}"
            if cfg["config_model"]["model_type"] == "nn"
            else FOLDER_MODEL_GP / f"{model_name}"
        )
        if not model_folder.exists():
            model_folder.mkdir(parents=True)

        # Extracting number of models given k-fold cross-validation
        num_models = cfg["config_dataset"]["r_validation"] * cfg["config_dataset"]["k_validation"]

        # Launching parallel training
        @job(array=num_models, **config_cluster_train)
        def train(i: int, model_name=model_name, model_folder=model_folder, config=cfg):
            training(
                fold_idx=i,
                model_name=model_name,
                model_folder=model_folder,
                config=config,
            )

        # Launching the training jobs
        schedule(
            train,
            name="BENTHIC-TRAINING",
            backend="slurm",
            export="ALL",
        )
