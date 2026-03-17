<hr>
<p  align="center">
<b  style="font-size:1vw;">Benthic Traits Mapping</b>
</p>
<hr>

This repository contains the code for the paper **Predicting benthic functions and sensitivity across the Black Sea seascapes**.

<hr>
<p  align="center">
<b  style="font-size:1vw;">Installation</b>
</p>
<hr>

To set-up everything, it is necessary to have access to a [Slurm](https://slurm.schedmd.com) cluster, to login to a [Weights & Biases](https://wandb.ai) account and to install the [benthic](benthic) module as a package. First, create a new Python environment, for example with [conda](https://docs.conda.io).

```
conda create -n benthic python=3.11
conda activate benthic
```

Then, install the [benthic](benthic) module as an [editable](https://pip.pypa.io/en/latest/topics/local-project-installs) package with its dependencies.

```
pip install --editable .[all] --extra-index-url https://download.pytorch.org/whl/cpu
```

Optionally, we provide [pre-commit hooks](pre-commit.yml) to automatically detect code issues.

```
pre-commit install --config pre-commit.yml
```

Don't forget to tell Keras to use Torch as backend:
```
export KERAS_BACKEND="torch"
```