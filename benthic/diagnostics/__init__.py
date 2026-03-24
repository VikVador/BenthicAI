import numpy as np
import torch

from benthic.config import FOLDER_MODEL_GP, FOLDER_MODEL_NN
from benthic.diagnostics.error_metrics import bias, r2_score, rmse
from benthic.training.tools import load_configuration


def compute_diagnostics(idx, model_name, config):
    if config["config_model"]["model_type"] == "nn":
        model_folder = FOLDER_MODEL_NN / f"{model_name}"
        y_pred_file = f"{idx}_y_pred_mean.pt"
        y_true_file = f"{idx}_y_true.pt"
        y_var_file = f"{idx}_y_pred_var.pt"

        y_pred = torch.load(model_folder / y_pred_file).detach().cpu().numpy()
        y_true = torch.load(model_folder / y_true_file).detach().cpu().numpy()
        y_var = torch.load(model_folder / y_var_file).detach().cpu().numpy()

    elif config["config_model"]["model_type"] == "gp":
        model_folder = FOLDER_MODEL_GP / f"{model_name}"
        pred_file = f"gp_fold_{idx:03d}_predictions.npz"

        predictions = np.load(model_folder / pred_file)
        y_pred = np.asarray(predictions["y_pred_va"])
        y_true = np.asarray(predictions["y_val"])
        y_var = np.asarray(predictions["y_std_va"])

    bias_array = bias(y_true, y_pred)
    r2_array = r2_score(y_true, y_pred)
    rmse_array = rmse(y_true, y_pred)

    np.save(model_folder / f"bias_{idx}.npy", bias_array)
    np.save(model_folder / f"r2_{idx}.npy", r2_array)
    np.save(model_folder / f"rmse_{idx}.npy", rmse_array)



    


