r"""Script to evaluate one GP classifier per trait (variational GP, softmax likelihood, no WandB).

Every trait is a vector of proportions (a soft classification) and gets its own model:
    - proportion groups:  K modalities summing to 1,
    - effect groups:      K modalities divided by the station total, plus log(total) on its own,
    - independent indices (de, mi, ir, dep): two classes [v, 1 - v].

A composition is modelled with K latent GPs f_1..f_K and the probabilities q = softmax(f). The
likelihood is a multinomial with N trials and fractional counts N * p:

    log p(p | f) = N * sum_k p_k log q_k + const

so a modality at 0% simply adds nothing (no zero replacement). N (the dispersion) is estimated
per trait from the stations with identical inputs, as Var_within(p_k) = q_k (1 - q_k) / N.
log(total) has one latent GP and a Gaussian likelihood.

Two options reduce the flexibility that lets the model learn the training stations by heart:
    - --share_kernel trait: the modalities of a trait share their lengthscales (one per input)
      instead of each having its own, so 11 lengthscales per trait instead of 11 * K,
    - --nugget: a learned per-station offset on the latents, q = softmax(f(x) + eps), eps ~ N(0, tau^2),
      tau learned per trait (logistic-normal multinomial). The scatter between stations that the
      environment does not explain goes into tau instead of bending the smooth function f,
    - --priors: log-normal priors on the lengthscales (median --lengthscale_prior, sqrt(#inputs) by
      default, on standardized inputs) and on the amplitudes (median --outputscale_prior), with
      log-standard deviation --prior_width. The hyperparameters are then MAP estimates instead of
      maximum likelihood: short lengthscales and large amplitudes (wiggly functions) need more
      evidence from the data. Hyperparameters start at the prior medians.

Inference is variational, with all distinct training inputs as inducing points (no sparse
approximation). Predictions are samples: q from the latent GPs, then observed proportions
~ Multinomial(N, q) / N, so intervals include the noise of the data. Metrics are the same as
in run_multitask_gp.py (R^2 per output and per trait, 90% coverage, CRPS), with the sklearn GP
as reference.

Two checks that the models learn a relationship rather than the training stations by heart:
    - R^2 on the training stations next to the validation one (a large gap means memorization),
    - --shuffle: outputs shuffled across stations, so R^2 should drop to about 0.
The uncertainty is reported as the mean width of the 90% interval, divided by the spread of the data.

With --wandb_mode online, every fold is a WandB run (grouped by --output) with, for every trait,
the learning curves (loss, R^2 on training and validation every --log_every steps) and the final
lengthscales, and a last run holds the summary tables. The test folds are never used.
"""

import argparse
import gpytorch
import numpy as np
import os
import pandas as pd
import time
import torch
import wandb
import xarray as xr

from pathlib import Path

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
from run_multitask_gp import INDEX, block_r2, evaluate, fit_predict_sklearn, pooled_metrics

MODELS = ("sklearn", "classifier")

# Output columns of every trait, to summarize per-output metrics by trait
TRAIT_OUTPUTS = {**OUTPUT_PROPORTION_GROUPS, **OUTPUT_EFFECT_GROUPS, **{v: [v] for v in OUTPUT_INDEPENDENT_VARIABLES}}
OUTPUT_TRAIT = {output: trait for trait, outputs in TRAIT_OUTPUTS.items() for output in outputs}


# ===================
#       TARGETS
# ===================
def compositions(y: np.ndarray) -> tuple[dict, np.ndarray]:
    r"""Split the outputs into one composition (n, K) per trait, and log(total) (n,)."""
    total = check_output_groups(y)
    comps = {}
    for name, modalities in {**OUTPUT_PROPORTION_GROUPS, **OUTPUT_EFFECT_GROUPS}.items():
        p = np.clip(y[:, [INDEX[m] for m in modalities]], 0, None)
        comps[name] = p / p.sum(axis=1, keepdims=True)
    for name in OUTPUT_INDEPENDENT_VARIABLES:
        v = np.clip(y[:, INDEX[name]], 0, 1)
        comps[name] = np.stack([v, 1 - v], axis=1)
    return comps, np.log(total)


def assemble(comps: dict, log_total: np.ndarray) -> np.ndarray:
    r"""Inverse of compositions, for any leading dimensions: (..., n_outputs) ordered as DATASET_OUTPUT_VARIABLES."""
    y = np.empty(log_total.shape + (len(DATASET_OUTPUT_VARIABLES),))
    total = np.exp(log_total)
    for name, modalities in OUTPUT_PROPORTION_GROUPS.items():
        y[..., [INDEX[m] for m in modalities]] = comps[name]
    for name, modalities in OUTPUT_EFFECT_GROUPS.items():
        y[..., [INDEX[m] for m in modalities]] = comps[name] * total[..., None]
    for name in OUTPUT_INDEPENDENT_VARIABLES:
        y[..., INDEX[name]] = comps[name][..., 0]
    return y


def dispersion(p: np.ndarray, groups: np.ndarray, max_trials: float) -> float:
    r"""Number of trials N of a multinomial matching the spread within groups of identical inputs.

    For a multinomial, sum_k Var(p_k) = sum_k q_k (1 - q_k) / N, with q the group mean. Falls back
    to max_trials when there is no spread (or no replicated group).
    """
    members = [m for m in (np.where(groups == g)[0] for g in np.unique(groups)) if len(m) > 1]
    if not members:
        return max_trials
    var_within = sum(((p[m] - p[m].mean(axis=0)) ** 2).sum() for m in members) / sum(len(m) - 1 for m in members)
    q = np.concatenate([np.repeat(p[m].mean(axis=0, keepdims=True), len(m), axis=0) for m in members])
    expected = (q * (1 - q)).sum(axis=1).mean()
    return float(np.clip(expected / var_within, 1, max_trials)) if var_within > 0 else max_trials


# ===================
#        MODEL
# ===================
class LatentGP(gpytorch.models.ApproximateGP):
    r"""num_latents independent GPs, variational with inducing points fixed at the inputs.

    Every latent has its own amplitude and constant mean. With share_kernel, they share one set of
    ARD lengthscales; otherwise each latent has its own.

    Arguments:
        priors: Optional (lengthscale median, outputscale median, log-standard deviation) of
                log-normal priors on the hyperparameters, which also start at the medians.
    """

    def __init__(self, inducing_points: torch.Tensor, num_latents: int, kernel: str, share_kernel: bool = False, priors: tuple | None = None):
        batch = torch.Size([num_latents])
        distribution = gpytorch.variational.CholeskyVariationalDistribution(len(inducing_points), batch_shape=batch)
        strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points.expand(num_latents, *inducing_points.shape).clone(),
                distribution,
                learn_inducing_locations=False,
            ),
            num_tasks=num_latents,
        )
        super().__init__(strategy)

        num_inputs = inducing_points.shape[1]
        base_batch = torch.Size([]) if share_kernel else batch
        lengthscale_prior = outputscale_prior = None
        if priors is not None:
            lengthscale, outputscale, width = priors
            lengthscale_prior = gpytorch.priors.LogNormalPrior(float(np.log(lengthscale)), width)
            outputscale_prior = gpytorch.priors.LogNormalPrior(float(np.log(outputscale)), width)
        if kernel == "rbf":
            base = gpytorch.kernels.RBFKernel(ard_num_dims=num_inputs, batch_shape=base_batch, lengthscale_prior=lengthscale_prior)
        else:
            base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=num_inputs, batch_shape=base_batch, lengthscale_prior=lengthscale_prior)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch)
        self.covar_module = gpytorch.kernels.ScaleKernel(base, batch_shape=batch, outputscale_prior=outputscale_prior)
        if priors is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale
            self.covar_module.outputscale = outputscale

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class SoftLabelLikelihood(gpytorch.likelihoods.Likelihood):
    r"""Multinomial with num_trials trials and fractional counts: N * sum_k p_k log softmax(f + eps)_k.

    With a nugget, eps ~ N(0, tau^2) is an offset per station and latent, with tau learned. The
    expectation over q(f) and eps is a Monte Carlo average over num_samples draws.
    """

    def __init__(self, num_trials: float, num_samples: int, nugget: bool = False, nugget_init: float = 0.5):
        super().__init__()
        self.num_trials, self.num_samples, self.has_nugget = num_trials, num_samples, nugget
        if nugget:
            # softplus(raw) = nugget_init
            self.raw_nugget = torch.nn.Parameter(torch.tensor(float(np.log(np.expm1(nugget_init)))))

    @property
    def nugget(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_nugget) if self.has_nugget else torch.tensor(0.0)

    def add_nugget(self, f: torch.Tensor) -> torch.Tensor:
        return f + self.nugget * torch.randn_like(f) if self.has_nugget else f

    def forward(self, function_samples, *args, **kwargs):
        return torch.distributions.Categorical(logits=self.add_nugget(function_samples))

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        f = self.add_nugget(function_dist.rsample(torch.Size([self.num_samples])))
        return self.num_trials * (observations * torch.log_softmax(f, dim=-1)).sum(dim=-1).mean(dim=0)


def fit_latent_gp(X: np.ndarray, target: np.ndarray, likelihood, mean_init: np.ndarray, args, callback=None) -> tuple:
    r"""Maximize the ELBO of a LatentGP with one latent per column of target.

    Arguments:
        callback: Optional function (iteration, model, loss) called every args.log_every steps and at the end.

    Returns:
        model, likelihood (in eval mode) and the final loss (negative ELBO per station).
    """
    X, target = torch.as_tensor(X), torch.as_tensor(target)
    inducing_points = torch.as_tensor(np.unique(X.numpy(), axis=0))
    priors = None
    if args.priors:
        lengthscale = args.lengthscale_prior if args.lengthscale_prior > 0 else np.sqrt(X.shape[1])
        priors = (lengthscale, args.outputscale_prior, args.prior_width)
    model = LatentGP(inducing_points, target.shape[1], args.kernel, share_kernel=args.share_kernel == "trait", priors=priors).double()
    likelihood = likelihood.double()
    model.mean_module.raw_constant.data.copy_(torch.as_tensor(mean_init))

    model.train(), likelihood.train()
    optimizer = torch.optim.Adam([*model.parameters(), *likelihood.parameters()], lr=args.lr)
    elbo = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X))
    for iteration in range(args.iterations):
        optimizer.zero_grad()
        loss = -elbo(model(X), target)
        loss.backward()
        optimizer.step()
        if callback is not None and (iteration % args.log_every == 0 or iteration == args.iterations - 1):
            callback(iteration, model, loss.item())

    model.eval(), likelihood.eval()
    return model, likelihood, loss.item()


def learning_curve(name: str, X_train, target_train, X_val, target_val, link):
    r"""Callback logging to WandB the loss and the R^2 of the posterior mean on training and validation.

    Arguments:
        name: Trait name, used as prefix of the WandB keys (each trait has its own iteration axis).
        link: Maps the latent mean (n, K) to the scale of the targets (softmax for compositions).
    """
    wandb.define_metric(f"{name}/*", step_metric=f"{name}/iteration")
    X_train, X_val = torch.as_tensor(X_train), torch.as_tensor(X_val)

    def callback(iteration: int, model, loss: float):
        model.eval()
        with torch.no_grad():
            pred_train, pred_val = (link(model(x).mean.numpy()) for x in (X_train, X_val))
        model.train()
        wandb.log({
            f"{name}/iteration": iteration,
            f"{name}/loss":      loss,
            f"{name}/r2_train":  block_r2(target_train, pred_train),
            f"{name}/r2_val":    block_r2(target_val, pred_val),
        })

    return callback


def log_lengthscales(name: str, model):
    r"""Final lengthscale of every input (mean over the latents) in the WandB summary. Short = the input matters."""
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy().reshape(-1, len(DATASET_INPUT_FEATURES)).mean(axis=0)
    for feature, value in zip(DATASET_INPUT_FEATURES, lengthscale):
        wandb.run.summary[f"{name}/lengthscale/{feature}"] = float(value)


def fit_predict_classifier(X_train, y_train, X_pred, args, rng, y_val=None) -> tuple[np.ndarray, list]:
    r"""Fit one model per trait plus one for log(total), and sample the outputs at X_pred.

    Arguments:
        y_val: Outputs of the first len(y_val) rows of X_pred. If given and WandB is on, the
               learning curves are logged (R^2 of the posterior mean on training and validation).

    Returns:
        samples: (num_samples, n_pred, n_outputs) on the original scale, noise of the data included.
        rows:    Per-trait diagnostics (dispersion N, final loss).
    """
    comps, log_total = compositions(y_train)
    groups = input_groups(X_train)
    stats_X = Statistics().fit(X_train)
    Xs_train, Xs_pred = stats_X.standardize(X_train), stats_X.standardize(X_pred)
    size = torch.Size([args.samples])

    monitor = y_val is not None and args.wandb_mode != "disabled"
    if monitor:
        comps_true_val, log_total_true_val = compositions(y_val)
        Xs_monitor = Xs_pred[: len(y_val)]
        softmax = lambda f: np.exp(f - f.max(axis=-1, keepdims=True)) / np.exp(f - f.max(axis=-1, keepdims=True)).sum(axis=-1, keepdims=True)

    comps_pred, rows = {}, []
    for name, p in comps.items():
        num_trials = args.trials if args.trials > 0 else dispersion(p, groups, args.max_trials)
        callback = learning_curve(name, Xs_train, p, Xs_monitor, comps_true_val[name], softmax) if monitor else None
        model, likelihood, loss = fit_latent_gp(
            Xs_train,
            p,
            SoftLabelLikelihood(num_trials, args.mc_samples, nugget=args.nugget, nugget_init=args.nugget_init),
            mean_init=np.log(np.clip(p.mean(axis=0), 1e-3, None)),
            args=args,
            callback=callback,
        )
        nugget = likelihood.nugget.item()
        if monitor:
            log_lengthscales(name, model)
            wandb.run.summary[f"{name}/nugget"] = nugget
        with torch.no_grad():
            f = likelihood.add_nugget(model(torch.as_tensor(Xs_pred)).sample(size))
            q = torch.softmax(f, dim=-1).numpy()
        q = q / q.sum(axis=-1, keepdims=True)
        comps_pred[name] = rng.multinomial(max(1, round(num_trials)), q) / max(1, round(num_trials))
        rows.append({"trait": name, "num_trials": num_trials, "nugget": nugget, "loss": loss})

    # log(total): one latent GP with a Gaussian likelihood, on the standardized scale
    stats_t = Statistics().fit(log_total[:, None])
    target = stats_t.standardize(log_total[:, None])
    callback = None
    if monitor:
        target_val = stats_t.standardize(log_total_true_val[:, None])
        callback = learning_curve("total_abundance", Xs_train, target, Xs_monitor, target_val, lambda f: f)
    model, likelihood, loss = fit_latent_gp(
        Xs_train,
        target,
        gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1),
        mean_init=np.zeros(1),
        args=args,
        callback=callback,
    )
    if monitor:
        log_lengthscales("total_abundance", model)
    with torch.no_grad():
        log_total_pred = stats_t.unstandardize(likelihood(model(torch.as_tensor(Xs_pred))).sample(size).numpy())[..., 0]
    rows.append({"trait": "total_abundance", "num_trials": np.nan, "loss": loss})

    return assemble(comps_pred, log_total_pred), rows


def wandb_table(df: pd.DataFrame) -> wandb.Table:
    r"""A DataFrame (index and possibly two-level columns) as a WandB table."""
    df = df.copy()
    df.columns = ["/".join(c) if isinstance(c, tuple) else str(c) for c in df.columns]
    return wandb.Table(dataframe=df.reset_index())


# ===================
#        MAIN
# ===================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One variational GP classifier per trait, softmax likelihood.")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=MODELS)
    parser.add_argument("--kernel", default="rbf", choices=["rbf", "matern"], help="Kernel over the inputs (ARD).")
    parser.add_argument("--share_kernel", default="modality", choices=["modality", "trait"], help="Lengthscales per modality, or shared by the modalities of a trait.")
    parser.add_argument("--nugget", action="store_true", help="Learned per-station offset on the latents (overdispersion).")
    parser.add_argument("--nugget_init", type=float, default=0.5, help="Initial value of the nugget tau.")
    parser.add_argument("--priors", action="store_true", help="Log-normal priors on lengthscales and amplitudes (MAP instead of maximum likelihood).")
    parser.add_argument("--lengthscale_prior", type=float, default=0, help="Median of the lengthscale prior, on standardized inputs (0 = sqrt(#inputs)).")
    parser.add_argument("--outputscale_prior", type=float, default=1.0, help="Median of the amplitude (latent variance) prior.")
    parser.add_argument("--prior_width", type=float, default=0.5, help="Log-standard deviation of the priors (smaller = stronger).")
    parser.add_argument("--trials", type=float, default=0, help="Dispersion N of every trait (0 = estimated from replicates).")
    parser.add_argument("--max_trials", type=float, default=1000, help="Upper bound of the estimated N.")
    parser.add_argument("--iterations", type=int, default=500, help="Adam steps on the ELBO.")
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--mc_samples", type=int, default=16, help="Monte Carlo samples of the expected log-likelihood.")
    parser.add_argument("--samples", type=int, default=1000, help="Predictive samples per station.")
    parser.add_argument("--r_validation", type=int, default=3)
    parser.add_argument("--k_validation", type=int, default=5)
    parser.add_argument("--k_test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1, help="Seed of the splits and of the sampling.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the outputs across stations (sanity check, R^2 should drop to about 0).")
    parser.add_argument("--ceiling", type=str, default="ceiling.csv_traits.csv", help="Per-trait ceiling from run_ceiling.py.")
    parser.add_argument("--output", "-o", type=str, default="gp_classifier", help="Prefix of the CSV outputs.")
    parser.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), help="CPU threads of torch.")
    parser.add_argument("--wandb_mode", default="disabled", choices=["online", "offline", "disabled"])
    parser.add_argument("--wandb_project", default="GP-Classifier")
    parser.add_argument("--wandb_team", default="benthicAI")
    parser.add_argument("--log_every", type=int, default=10, help="Steps between two points of the learning curves.")
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

    # Sanity check: whole output rows are shuffled, so they no longer match their inputs
    if args.shuffle:
        y = y[rng.permutation(len(y))]
        print("Outputs shuffled across stations: R^2 should be about 0.\n")

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
    print(f"{len(X)} stations, {X.shape[1]} inputs, {len(DATASET_OUTPUT_VARIABLES)} outputs")
    print(f"{len(X_train)} folds ({args.r_validation} x {args.k_validation}), models: {', '.join(args.models)}\n")

    results = {m: [] for m in args.models}
    fold_rows, trait_rows, train_outputs, train_traits = [], [], [], []

    wandb_kwargs = dict(mode=args.wandb_mode, project=args.wandb_project, entity=args.wandb_team, group=args.output)

    for k in range(len(X_train)):
        # Predicting the validation and the training stations at once, the latter to measure memorization
        X_pred, n_val = np.concatenate([X_val[k], X_train[k]]), len(X_val[k])
        run = wandb.init(name=f"{args.output}_fold{k}", config={**vars(args), "fold": k}, **wandb_kwargs)

        for name in args.models:
            start = time.time()
            if name == "sklearn":
                stats_X, stats_y = Statistics().fit(X_train[k]), Statistics().fit(y_train[k])
                samples = stats_y.unstandardize(
                    fit_predict_sklearn(
                        stats_X.standardize(X_train[k]), stats_y.standardize(y_train[k]), stats_X.standardize(X_pred), args, rng
                    )
                )
            else:
                samples, rows = fit_predict_classifier(X_train[k], y_train[k], X_pred, args, rng, y_val=y_val[k])
                trait_rows += [{"fold": k, **row} for row in rows]

            results[name].append(evaluate(samples[:, :n_val], y_val[k]))
            outputs, traits = pooled_metrics([evaluate(samples[:, n_val:], y_train[k])])
            train_outputs.append(outputs.assign(model=name, fold=k).reset_index())
            train_traits.append(traits.assign(model=name, fold=k).reset_index())

            # Final metrics of the fold in the WandB summary
            val_outputs, val_traits = pooled_metrics([results[name][-1]])
            for trait in val_traits.index:
                run.summary[f"{name}/{trait}/r2_val"] = float(val_traits.loc[trait, "r2"])
                run.summary[f"{name}/{trait}/r2_train"] = float(traits.loc[trait, "r2"])
            for metric in ("r2", "coverage90", "width90_norm", "crps_norm"):
                run.summary[f"{name}/overall/{metric}_val"] = float(val_outputs[metric].median())

            fold_rows.append({"model": name, "fold": k, "seconds": time.time() - start})
            print(f"fold {k:2d}  {name:10s}  ({time.time() - start:.0f} s)", flush=True)

        run.finish()

    # Pooling the validation folds of every repetition, so each station is predicted once
    tables_outputs, tables_traits = [], []
    for name, folds in results.items():
        for r in range(args.r_validation):
            outputs, traits = pooled_metrics(folds[r * args.k_validation : (r + 1) * args.k_validation])
            tables_outputs.append(outputs.assign(model=name, repetition=r).reset_index())
            tables_traits.append(traits.assign(model=name, repetition=r).reset_index())

    outputs = pd.concat(tables_outputs, ignore_index=True)
    traits = pd.concat(tables_traits, ignore_index=True)
    train_outputs = pd.concat(train_outputs, ignore_index=True)
    train_traits = pd.concat(train_traits, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    diagnostics = pd.DataFrame(trait_rows)

    outputs.to_csv(f"{args.output}_outputs.csv", index=False)
    traits.to_csv(f"{args.output}_traits.csv", index=False)
    train_outputs.to_csv(f"{args.output}_train_outputs.csv", index=False)
    train_traits.to_csv(f"{args.output}_train_traits.csv", index=False)
    folds.to_csv(f"{args.output}_folds.csv", index=False)
    diagnostics.to_csv(f"{args.output}_diagnostics.csv", index=False)

    # Summary: per-trait R^2 (mean over repetitions) next to the ceiling and the dispersion
    summary = traits.pivot_table(index="trait", columns="model", values="r2", aggfunc="mean")[args.models]
    if Path(args.ceiling).exists():
        summary.insert(0, "ceiling", pd.read_csv(args.ceiling, index_col="trait")["r2_max"])
    if {"sklearn", "classifier"} <= set(args.models):
        wins = traits.pivot_table(index=["trait", "repetition"], columns="model", values="r2")
        summary["clf - skl"] = summary["classifier"] - summary["sklearn"]
        summary["clf wins"] = (wins["classifier"] > wins["sklearn"]).groupby("trait").sum().astype(str) + f"/{args.r_validation}"
    if not diagnostics.empty:
        summary["N"] = diagnostics.groupby("trait")["num_trials"].median()
        if args.nugget:
            summary["nugget"] = diagnostics.groupby("trait")["nugget"].median()

    print("\nPer-trait R^2 on validation (mean over repetitions; N = median estimated dispersion, nugget = median tau)")
    print(summary.sort_values(args.models[-1]).round(3).to_string())

    # Memorization: training R^2 (mean over folds) next to validation R^2
    val_r2 = traits.groupby(["trait", "model"])["r2"].mean().unstack()[args.models]
    train_r2 = train_traits.groupby(["trait", "model"])["r2"].mean().unstack()[args.models]
    memorization = pd.concat({"val": val_r2, "train": train_r2, "gap": train_r2 - val_r2}, axis=1).swaplevel(axis=1)
    memorization = memorization[[(m, c) for m in args.models for c in ("val", "train", "gap")]]
    print("\nPer-trait R^2, validation vs training (a large gap means the training stations are learned by heart)")
    print(memorization.sort_values((args.models[-1], "val")).round(3).to_string())

    # Uncertainty: 90% interval width relative to the spread of the data, and its coverage
    by_trait = outputs.assign(trait=outputs["output"].map(OUTPUT_TRAIT))
    uncertainty = by_trait.groupby(["trait", "model"])[["width90_norm", "coverage90"]].mean().unstack()
    uncertainty = uncertainty.swaplevel(axis=1)[[(m, c) for m in args.models for c in ("width90_norm", "coverage90")]]
    print("\nPer-trait uncertainty: 90% interval width / std of the data, and coverage (target 0.90)")
    print(uncertainty.round(3).to_string())

    overall = outputs.groupby("model")[["r2", "coverage90", "width90_norm", "crps_norm"]].mean()
    overall.insert(1, "r2_median", outputs.groupby("model")["r2"].median())
    overall.insert(2, "r2_train_median", train_outputs.groupby("model")["r2"].median())
    overall["seconds/fold"] = folds.groupby("model")["seconds"].mean()
    print("\nOver outputs (coverage target 0.90, narrower width at the same coverage is better, lower CRPS is better)")
    print(overall.loc[args.models].round(3).to_string())

    # Summary tables of all folds in one last WandB run
    run = wandb.init(name=f"{args.output}_summary", config=vars(args), **wandb_kwargs)
    run.log({
        "summary/r2_val":       wandb_table(summary),
        "summary/memorization": wandb_table(memorization),
        "summary/uncertainty":  wandb_table(uncertainty),
        "summary/overall":      wandb_table(overall.loc[args.models]),
    })
    run.finish()
