r"""Script to launch a training pipeline."""

import argparse

# isort: split
# from benthic.training.gp import training_gp
# from benthic.training.nn import training_nn

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
        "--backend",
        "-b",
        type=str,
        default="slurm",
        choices=["slurm", "async"],
        help="Computation backend, 'slurm' for cluster-based scheduling and 'async' for local execution.",
    )

    args           = parser.parse_args()
    configs        = load_configuration(args.config)
    config_cluster = configs[0].get("Cluster").copy()

    # Run locally
    if args.backend == "async":
        """
        TO BE DONE. EXAMPLE:
        training_gp(DICT_TRAINING_CONFIG[0])
        """

    # Launching jobs on cluster
    else:

        """
        TO BE DONE. EXAMPLE:

        # Defining function to be executed in parallel on cluster
        @job(array=len(configs), **config_cluster)
        def train(i: int):
            training_gp(DICT_TRAINING_CONFIG[i])

        # Launching the training jobs on cluster with DAWGZ
        schedule(
            train,
            name="BENTHIC-TRAINING",
            backend="slurm",
            interpreter=interpreter,
            export="ALL",
        )
        """
