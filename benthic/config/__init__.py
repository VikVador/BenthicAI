r"""Global paths and configuration helpers."""

from pathlib import Path

# fmt: off
#
LOCAL           = Path("/gpfs/projects/acad/bsmfc/benthicAI/")
FOLDER_DATASETS = LOCAL / "datasets"
FOLDER_MODELS   = LOCAL / "models"

FOLDER_DATASET_AI         = FOLDER_DATASETS / "ai"
FOLDER_DATASET_MODEL      = FOLDER_DATASETS / "model"
FOLDER_DATASET_RAW        = FOLDER_DATASETS / "raw"
FOLDER_DATASET_STATISTICS = FOLDER_DATASETS / "statistics"

# --- Datasets ---
#
# Path to datasets for training, validation and testing
PATH_DATASET_TRAIN_VAL   = FOLDER_DATASET_AI / "training_validation.zarr"
PATH_DATASET_TEST        = FOLDER_DATASET_AI / "test.zarr"
PATH_DATASET_COORDINATES = FOLDER_DATASET_AI / "station_coordinates.pt" # (STATIONS, 3) with lat, lon, depth

# Path to information abour numerical domain
PATH_MODEL_BATHYMETRY = FOLDER_DATASET_MODEL / "bathymetry.nc"
PATH_MODEL_MESH_MASK  = FOLDER_DATASET_MODEL / "mesh_mask.nc"

# Path to benthic traits
PATH_DATA = {
    "seabed"        : FOLDER_DATASET_RAW / "blacksea_seabed_annotated.nc",
    "effect_mod"    : FOLDER_DATASET_RAW / "effect_mod.txt",
    "response_mod"  : FOLDER_DATASET_RAW / "resp_mod.txt",
    "ind_ecol"      : FOLDER_DATASET_RAW / "ind_ecol.txt",
    "environmental" : FOLDER_DATASET_RAW / "environement.txt",
    "pose_grp"      : FOLDER_DATASET_RAW / "pose_grp.csv",
    "labels"        : FOLDER_DATASET_RAW / "labels.csv",
}

# --- Models ---
#
# Path to models trained on the benthic data
FOLDER_MODEL_GP = FOLDER_MODELS / "gp"
FOLDER_MODEL_NN = FOLDER_MODELS / "nn"
