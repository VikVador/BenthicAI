r"""Script to test whether learning the outputs jointly helps (exact multitask GP, no WandB).

Every output is made unconstrained, then one exact multitask GP is fitted on all of them:
    - each trait (K modalities summing to 1, or to the station total) -> K-1 isometric log-ratios,
    - the station total of the effect groups -> log(total),
    - the independent indices (de, mi, ir, dep) -> logit.

The kernel is k(x, x') * B[t, t'], with B a learned task covariance (low rank + diagonal).
Three models are compared under the same grouped cross-validation:
    - joint:    B = W W^T + diag(v), outputs inform each other. With noise_rank > 0, the noise
                is correlated across tasks too (F F^T + s^2 I), so B only has to explain how
                outputs move together with the environment, not the noise shared within a station,
    - diagonal: B = diag(v) and independent noise, the same model with independent outputs,
    - sklearn:  the independent sklearn GP on the raw (standardized) outputs, as a reference.

Predictions are samples pushed back to the original scale (softmax, exp, sigmoid), from which
we read R^2 (per output and per trait, as in run_ceiling.py), 90% coverage and CRPS. Coverage
is also computed on the transformed scale, where the Gaussian assumption is made, to tell a
miscalibrated GP apart from a distortion of the back-transform.
"""

import argparse
import gpytorch
import numpy as np
import os
import pandas as pd
import time
import torch
import xarray as xr

from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

# isort: split
from benthic.config import PATH_DATASET_TRAIN_VAL
from benthic.data.const import (
    DATASET_INPUT_FEATURES,
    DATASET_INPUT_VARIABLES,
    DATASET_OUTPUT_VARIABLES,
    OUTPUT_EFFECT_GROUPS,
    OUTPUT_INDEPENDENT_VARIABLES,
    OUTPUT_PROPORTION_GROUPS,
)
from benthic.data.dataset import get_datasets, input_groups
from benthic.data.tools import (
    Statistics,
    check_output_groups,
    from_zarr_to_numpy,
    input_feature_names,
    select_features,
)
from benthic.diagnostics.metrics import r2_score

MODELS = ("sklearn", "diagonal", "joint")
INDEX = {name: i for i, name in enumerate(DATASET_OUTPUT_VARIABLES)}


# ===================
#      TRANSFORM
# ===================
def ilr_basis(k: int) -> np.ndarray:
    r"""Orthonormal basis (k, k-1) of the vectors summing to 0 (Helmert), for isometric log-ratios."""
    V = np.zeros((k, k - 1))
    for j in range(1, k):
        V[:j, j - 1] = 1 / j
        V[j, j - 1] = -1
        V[:, j - 1] *= np.sqrt(j / (j + 1))
    return V


def replace_zeros(p: np.ndarray, delta: float) -> np.ndarray:
    r"""Multiplicative zero replacement: zeros become delta, the other parts shrink to keep the sum at 1."""
    p = p / p.sum(axis=1, keepdims=True)
    zero = p <= 0
    return np.where(zero, delta, p * (1 - delta * zero.sum(axis=1, keepdims=True)))


def softmax(a: np.ndarray) -> np.ndarray:
    a = np.exp(a - a.max(axis=-1, keepdims=True))
    return a / a.sum(axis=-1, keepdims=True)


class OutputTransform:
    r"""Maps the outputs (ordered as DATASET_OUTPUT_VARIABLES) to unconstrained coordinates and back.

    Coordinates are ordered as: log-ratios of the proportion groups, log-ratios of the effect
    groups (divided by the station total), log(total), logit of the independent outputs.

    Arguments:
        delta: Value replacing the zeros of the proportions before taking logs.
        eps:   The independent outputs are clipped to [eps, 1 - eps] before the logit.
    """

    def __init__(self, delta: float = 1e-3, eps: float = 1e-3):
        self.delta, self.eps = delta, eps
        self.groups = {**OUTPUT_PROPORTION_GROUPS, **OUTPUT_EFFECT_GROUPS}
        self.bases = {name: ilr_basis(len(modalities)) for name, modalities in self.groups.items()}

        self.slices, self.names, start = {}, [], 0
        for name, modalities in self.groups.items():
            self.slices[name] = slice(start, start + len(modalities) - 1)
            self.names += [f"{name}_ilr{j}" for j in range(1, len(modalities))]
            start += len(modalities) - 1
        self.i_total = start
        self.names += ["log_total"] + [f"logit_{v}" for v in OUTPUT_INDEPENDENT_VARIABLES]
        self.num_tasks = len(self.names)

    def forward(self, y: np.ndarray) -> np.ndarray:
        r"""(n, n_outputs) -> (n, num_tasks)."""
        total = check_output_groups(y)
        z = np.empty((len(y), self.num_tasks))

        for name, modalities in self.groups.items():
            p = y[:, [INDEX[m] for m in modalities]]
            if name in OUTPUT_EFFECT_GROUPS:
                p = p / total[:, None]
            log_p = np.log(replace_zeros(p, self.delta))
            z[:, self.slices[name]] = (log_p - log_p.mean(axis=1, keepdims=True)) @ self.bases[name]

        z[:, self.i_total] = np.log(total)
        for j, v in enumerate(OUTPUT_INDEPENDENT_VARIABLES):
            a = np.clip(y[:, INDEX[v]], self.eps, 1 - self.eps)
            z[:, self.i_total + 1 + j] = np.log(a / (1 - a))
        return z

    def inverse(self, z: np.ndarray) -> np.ndarray:
        r"""(..., num_tasks) -> (..., n_outputs). Proportions sum to 1, effect groups to the total."""
        y = np.empty(z.shape[:-1] + (len(DATASET_OUTPUT_VARIABLES),))
        total = np.exp(z[..., self.i_total])

        for name, modalities in self.groups.items():
            p = softmax(z[..., self.slices[name]] @ self.bases[name].T)
            if name in OUTPUT_EFFECT_GROUPS:
                p = p * total[..., None]
            y[..., [INDEX[m] for m in modalities]] = p

        for j, v in enumerate(OUTPUT_INDEPENDENT_VARIABLES):
            y[..., INDEX[v]] = 1 / (1 + np.exp(-z[..., self.i_total + 1 + j]))
        return y


def trait_blocks(y: np.ndarray) -> dict:
    r"""Same blocks as run_ceiling.trait_blocks, for any leading dimensions and without the sum check.

    Effect groups are divided by the station total (mean of their sums), which becomes its own
    block, so predictions of a model that does not respect the constraints can be scored too.
    """
    sums = np.stack([y[..., [INDEX[m] for m in mods]].sum(axis=-1) for mods in OUTPUT_EFFECT_GROUPS.values()])
    total = sums.mean(axis=0)

    blocks = {name: y[..., [INDEX[m] for m in mods]] for name, mods in OUTPUT_PROPORTION_GROUPS.items()}
    blocks |= {name: y[..., [INDEX[m] for m in mods]] / total[..., None] for name, mods in OUTPUT_EFFECT_GROUPS.items()}
    blocks["total_abundance"] = total[..., None]
    blocks |= {v: y[..., [INDEX[v]]] for v in OUTPUT_INDEPENDENT_VARIABLES}
    return blocks


# ===================
#       MODELS
# ===================
class MultitaskGP(gpytorch.models.ExactGP):
    r"""Exact GP with covariance k(x, x') * B[t, t'] over all tasks."""

    def __init__(self, X, Z, likelihood, rank: int, kernel: str):
        super().__init__(X, Z, likelihood)
        num_tasks, num_inputs = Z.shape[1], X.shape[1]
        if kernel == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=num_inputs)
        else:
            base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=num_inputs)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ConstantMean(), num_tasks=num_tasks)
        self.covar_module = gpytorch.kernels.MultitaskKernel(base, num_tasks=num_tasks, rank=rank)

    def forward(self, x):
        return gpytorch.distributions.MultitaskMultivariateNormal(self.mean_module(x), self.covar_module(x))

    def task_correlation(self) -> np.ndarray:
        r"""Correlation between tasks of the signal, from B."""
        return as_correlation(self.covar_module.task_covar_module.covar_matrix.to_dense())

    def noise_correlation(self) -> np.ndarray:
        r"""Correlation between tasks of the noise, F F^T + s^2 I (identity if noise_rank = 0)."""
        likelihood = self.likelihood
        if likelihood.rank == 0:
            return np.eye(self.covar_module.num_tasks)
        F = likelihood.task_noise_covar_factor
        return as_correlation(F @ F.T + likelihood.noise * torch.eye(len(F), dtype=F.dtype))


def as_correlation(covariance: torch.Tensor) -> np.ndarray:
    covariance = covariance.detach().numpy()
    d = np.sqrt(np.diag(covariance))
    return covariance / np.outer(d, d)


def fit_predict_multitask(X_train, Z_train, X_val, Z_val, joint: bool, args) -> tuple:
    r"""Fit the multitask GP on standardized data and sample the predictive distribution at X_val.

    Returns:
        samples: Predictive samples of shape (num_samples, n_val, num_tasks), observation noise included.
        nll:     Negative log-likelihood of Z_val per station (joint over tasks, standardized scale).
        model:   The fitted model.
    """
    X_train, Z_train, X_val, Z_val = (torch.as_tensor(a, dtype=torch.float64) for a in (X_train, Z_train, X_val, Z_val))
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=Z_train.shape[1], rank=args.noise_rank if joint else 0
    )
    model = MultitaskGP(X_train, Z_train, likelihood, rank=args.rank, kernel=args.kernel).double()

    # Diagonal model: no low-rank part, so B = diag(v) and tasks are independent
    if not joint:
        factor = model.covar_module.task_covar_module.covar_factor
        factor.data.zero_()
        factor.requires_grad_(False)

    model.train(), likelihood.train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(args.iterations):
        optimizer.zero_grad()
        loss = -mll(model(X_train), Z_train)
        loss.backward()
        optimizer.step()

    # The predictive covariance (n_val * num_tasks rows) has no Kronecker structure. Above
    # max_cholesky_size (800 by default) gpytorch would use approximate iterative solves.
    model.eval(), likelihood.eval()
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(10_000):
        predictive = likelihood(model(X_val))
        nll = -predictive.log_prob(Z_val).item() / len(X_val)
        samples = predictive.sample(torch.Size([args.samples])).numpy()
    return samples, nll, model


def fit_predict_sklearn(X_train, y_train, X_val, args, rng) -> np.ndarray:
    r"""Independent sklearn GP on standardized raw outputs, sampled as independent Gaussians."""
    num_inputs = X_train.shape[1]
    base = RBF(np.ones(num_inputs)) if args.kernel == "rbf" else Matern(np.ones(num_inputs), nu=2.5)
    gpr = GaussianProcessRegressor(kernel=ConstantKernel() * base + WhiteKernel(), n_restarts_optimizer=0)
    gpr.fit(X_train, y_train)
    mean, std = gpr.predict(X_val, return_std=True)
    if std.ndim == 1:
        std = np.repeat(std[:, None], mean.shape[1], axis=1)
    return mean + std * rng.standard_normal((args.samples,) + mean.shape)


# ===================
#      METRICS
# ===================
def crps(samples: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    r"""Sample CRPS, E|X - y| - E|X - X'| / 2, with samples on axis 0. Lower is better."""
    S = samples.shape[0]
    weights = (2 * np.arange(1, S + 1) - S - 1).reshape((S,) + (1,) * y_true.ndim)
    spread = 2 * (weights * np.sort(samples, axis=0)).sum(axis=0) / S**2
    return np.abs(samples - y_true).mean(axis=0) - spread / 2


def block_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r"""R^2 of a block of columns, variances summed over the columns (as in noise_ceiling)."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum()
    return 1 - ss_res / ss_tot


def evaluate(samples: np.ndarray, y_true: np.ndarray) -> dict:
    r"""Everything needed for the metrics of one fold, from samples (S, n, n_outputs) on the original scale."""
    blocks_samples, blocks_true = trait_blocks(samples), trait_blocks(y_true)
    low, high = np.percentile(samples, [5, 95], axis=0)
    return {
        "y_true": y_true,
        "y_pred": samples.mean(axis=0),
        "inside": (y_true >= low) & (y_true <= high),
        "width": high - low,
        "crps": crps(samples, y_true),
        "blocks_true": blocks_true,
        "blocks_pred": {name: b.mean(axis=0) for name, b in blocks_samples.items()},
    }


def pooled_metrics(folds: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    r"""Per-output and per-trait metrics on the validation folds of one repetition, pooled together."""
    cat = lambda key: np.concatenate([f[key] for f in folds])
    y_true = cat("y_true")
    outputs = pd.DataFrame(
        {
            "r2": r2_score(y_true, cat("y_pred")),
            "coverage90": cat("inside").mean(axis=0),
            "crps": cat("crps").mean(axis=0),
            "crps_norm": cat("crps").mean(axis=0) / y_true.std(axis=0),
            "width90_norm": cat("width").mean(axis=0) / y_true.std(axis=0),
        },
        index=pd.Index(DATASET_OUTPUT_VARIABLES, name="output"),
    )
    traits = pd.DataFrame(
        {
            "r2": [
                block_r2(
                    np.concatenate([f["blocks_true"][name] for f in folds]),
                    np.concatenate([f["blocks_pred"][name] for f in folds]),
                )
                for name in folds[0]["blocks_true"]
            ]
        },
        index=pd.Index(list(folds[0]["blocks_true"]), name="trait"),
    )
    return outputs, traits


# ===================
#        MAIN
# ===================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Joint vs independent outputs with an exact multitask GP.")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "matern"], help="Kernel over the inputs (ARD).")
    parser.add_argument("--rank", type=int, default=5, help="Rank of the low-rank part of B (joint model).")
    parser.add_argument("--noise_rank", type=int, default=0, help="Rank of the noise correlation across tasks (joint model, 0 = independent noise).")
    parser.add_argument("--iterations", type=int, default=300, help="Adam steps on the marginal likelihood.")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--samples", type=int, default=1000, help="Predictive samples per station.")
    parser.add_argument("--delta", type=float, default=1e-3, help="Replacement of zero proportions.")
    parser.add_argument("--eps", type=float, default=1e-3, help="Clipping of the indices before the logit.")
    parser.add_argument("--r_validation", type=int, default=3)
    parser.add_argument("--k_validation", type=int, default=5)
    parser.add_argument("--k_test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1, help="Seed of the splits and of the sampling.")
    parser.add_argument("--ceiling", type=str, default="ceiling.csv_traits.csv", help="Per-trait ceiling from run_ceiling.py.")
    parser.add_argument("--output", "-o", type=str, default="multitask_gp", help="Prefix of the CSV outputs.")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), help="CPU threads of torch.")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Same steps as training() up to the split
    ds = xr.open_zarr(PATH_DATASET_TRAIN_VAL)
    X = from_zarr_to_numpy(ds, DATASET_INPUT_VARIABLES)
    X = select_features(X, input_feature_names(ds, DATASET_INPUT_VARIABLES), DATASET_INPUT_FEATURES)
    y = from_zarr_to_numpy(ds, DATASET_OUTPUT_VARIABLES)
    check_output_groups(y)

    X_train, y_train, X_val, y_val, _, _, _, _, _ = get_datasets(
        X,
        y,
        groups=input_groups(X),
        r_validation=args.r_validation,
        k_validation=args.k_validation,
        k_test=args.k_test,
        seed_validation=args.seed,
        seed_test=args.seed,
    )

    transform = OutputTransform(delta=args.delta, eps=args.eps)
    print(f"{len(X)} stations, {X.shape[1]} inputs, {len(DATASET_OUTPUT_VARIABLES)} outputs -> {transform.num_tasks} tasks")
    print(f"{len(X_train)} folds ({args.r_validation} x {args.k_validation}), models: {', '.join(args.models)}\n")

    results = {m: [] for m in args.models}
    fold_rows, task_rows, correlations, noise_correlations = [], [], [], []

    for k in range(len(X_train)):
        stats_X = Statistics().fit(X_train[k])
        Xs_train, Xs_val = stats_X.standardize(X_train[k]), stats_X.standardize(X_val[k])

        Z_train, Z_val = transform.forward(y_train[k]), transform.forward(y_val[k])
        stats_Z = Statistics().fit(Z_train)

        for name in args.models:
            start = time.time()
            nll, coverage_z = np.nan, np.nan
            if name == "sklearn":
                stats_y = Statistics().fit(y_train[k])
                samples = stats_y.unstandardize(fit_predict_sklearn(Xs_train, stats_y.standardize(y_train[k]), Xs_val, args, rng))
            else:
                samples, nll, model = fit_predict_multitask(
                    Xs_train, stats_Z.standardize(Z_train), Xs_val, stats_Z.standardize(Z_val), name == "joint", args
                )
                samples = stats_Z.unstandardize(samples)

                # Coverage on the transformed scale, before the back-transform
                low, high = np.percentile(samples, [5, 95], axis=0)
                inside_z = ((Z_val >= low) & (Z_val <= high)).mean(axis=0)
                coverage_z = inside_z.mean()
                task_rows += [
                    {"model": name, "fold": k, "task": task, "coverage90": c} for task, c in zip(transform.names, inside_z)
                ]

                samples = transform.inverse(samples)
                if name == "joint":
                    correlations.append(model.task_correlation())
                    noise_correlations.append(model.noise_correlation())

            results[name].append(evaluate(samples, y_val[k]))
            fold_rows.append({"model": name, "fold": k, "nll": nll, "coverage90_z": coverage_z, "seconds": time.time() - start})
            print(f"fold {k:2d}  {name:8s}  nll {nll:7.2f}  coverage_z {coverage_z:4.2f}  ({time.time() - start:.0f} s)", flush=True)

    # Pooling the validation folds of every repetition, so each station is predicted once
    tables_outputs, tables_traits = [], []
    for name, folds in results.items():
        for r in range(args.r_validation):
            outputs, traits = pooled_metrics(folds[r * args.k_validation : (r + 1) * args.k_validation])
            tables_outputs.append(outputs.assign(model=name, repetition=r).reset_index())
            tables_traits.append(traits.assign(model=name, repetition=r).reset_index())

    outputs = pd.concat(tables_outputs, ignore_index=True)
    traits = pd.concat(tables_traits, ignore_index=True)
    folds = pd.DataFrame(fold_rows)

    outputs.to_csv(f"{args.output}_outputs.csv", index=False)
    traits.to_csv(f"{args.output}_traits.csv", index=False)
    folds.to_csv(f"{args.output}_folds.csv", index=False)
    pd.DataFrame(task_rows).to_csv(f"{args.output}_tasks.csv", index=False)
    for label, matrices in (("task", correlations), ("noise", noise_correlations)):
        if matrices:
            pd.DataFrame(np.mean(matrices, axis=0), index=transform.names, columns=transform.names).to_csv(
                f"{args.output}_{label}_correlation.csv"
            )

    # Summary: per-trait R^2 (mean over repetitions) next to the ceiling
    summary = traits.pivot_table(index="trait", columns="model", values="r2", aggfunc="mean")[args.models]
    if Path(args.ceiling).exists():
        summary.insert(0, "ceiling", pd.read_csv(args.ceiling, index_col="trait")["r2_max"])
    if {"joint", "diagonal"} <= set(args.models):
        wins = traits.pivot_table(index=["trait", "repetition"], columns="model", values="r2")
        summary["joint - diag"] = summary["joint"] - summary["diagonal"]
        summary["joint wins"] = (wins["joint"] > wins["diagonal"]).groupby("trait").sum().astype(str) + f"/{args.r_validation}"

    print("\nPer-trait R^2 (mean over repetitions)\n" + summary.sort_values(args.models[-1]).round(3).to_string())

    overall = outputs.groupby("model")[["r2", "coverage90", "crps_norm"]].mean()
    overall["coverage90_z"] = folds.groupby("model")["coverage90_z"].mean()
    overall["nll"] = folds.groupby("model")["nll"].mean()
    overall["seconds/fold"] = folds.groupby("model")["seconds"].mean()
    print("\nMean over outputs (coverage target 0.90, _z = transformed scale, NLL on the transformed scale, lower CRPS/NLL is better)")
    print(overall.loc[args.models].round(3).to_string())
