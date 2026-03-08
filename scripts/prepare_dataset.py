import os
import sys
import numpy as np
from tqdm import tqdm
from scipy import io

from src.utils.config import load_config

# -----------------------
# Load config
# -----------------------

cfg = load_config("configs/prepare_dataset.yaml")
RAW_PATH  = cfg.paths.raw_path

# -----------------------
# Load min–max stats
# -----------------------

min_max = io.loadmat(cfg.paths.min_max_file)
all_features = cfg.features.met_variables_raw + cfg.features.emission_variables_raw

min_vals = {f: min_max[f"{f}_min"].item() for f in all_features}
max_vals = {f: min_max[f"{f}_max"].item() for f in all_features}

# -----------------------
# Helper Functions
# -----------------------

def train_val_split(samples, val_frac=0.2):
    N = samples.shape[0]
    n_val = int(val_frac * N)
    
    # Strictly chronological split
    train_idx = N - n_val
    
    train_samples = samples[:train_idx]
    val_samples = samples[train_idx:]

    return train_samples, val_samples


def create_timeseries_samples(
    month,
    feature_list,
    train_save_dir,
    val_save_dir, 
    val_frac,
    seed,
    horizon,
    stride,
):
    train_data = {}
    val_data = {}

    for feat in tqdm(feature_list):

        file_path = os.path.join(RAW_PATH, month, f"{feat}.npy")
        arr = np.load(file_path).astype(np.float32)


        minn = min_vals[feat]
        maxx = max_vals[feat]
        den  = maxx - minn

        arr = (arr - minn) / den

        if feat in ["u10", "v10"]:
            arr = 2.0 * arr - 1.0

        if feat in cfg.features.emission_variables_raw:
            arr = np.clip(arr, 0, 1)


        print("Original shape:", arr.shape)

        T = arr.shape[0]
        idx = range(0, T - horizon + 1, stride)

        samples = np.stack([arr[i:i+horizon] for i in idx], axis=0)

        print("Total samples created -", samples.shape[0])

        train_samples, val_samples = train_val_split(
            samples, val_frac=val_frac, seed=seed
        )

        train_data[feat] = train_samples
        val_data[feat]   = val_samples

        del arr, samples

    return train_data, val_data


# -----------------------
# Run
# -----------------------

os.makedirs(cfg.paths.train_savepath, exist_ok=True)
os.makedirs(cfg.paths.val_savepath, exist_ok=True)

print(f"\n==============================")
print(f"Train Save path: {cfg.paths.train_savepath}")
print(f"Val Save path: {cfg.paths.val_savepath}") 
print(f"Horizon={cfg.data.horizon}, Stride={cfg.data.stride}")
print(f"==============================\n")

for feat in all_features:

    print("\n===================================")
    print("Processing feature:", feat)
    print("===================================\n")

    train_chunks = []
    val_chunks   = []

    for month in cfg.data.months:

        print(f"Month: {month}")
        print(f"==============================\n")

        train_m, val_m = create_timeseries_samples(
            month=month,
            feature_list=[feat],
            train_save_dir=cfg.paths.train_savepath,
            val_save_dir=cfg.paths.val_savepath,
            val_frac=cfg.data.val_frac,
            seed=cfg.data.seed,
            horizon=cfg.data.horizon,
            stride=cfg.data.stride,
        )

        train_chunks.append(train_m[feat])
        val_chunks.append(val_m[feat])

        del train_m, val_m

    train_merged = np.concatenate(train_chunks, axis=0)
    val_merged   = np.concatenate(val_chunks, axis=0)

    print(
        f"\nFinal {feat} -> Train:", train_merged.shape,
        "Val:", val_merged.shape
    )

    np.save(
        os.path.join(cfg.paths.train_savepath, f"train_{feat}.npy"),
        train_merged.astype(np.float32)
    )
    np.save(
        os.path.join(cfg.paths.val_savepath, f"val_{feat}.npy"),
        val_merged.astype(np.float32)
    )

    del train_chunks, val_chunks, train_merged, val_merged

print("\nStacking all features into single files...")

all_train = []
all_val = []

for feat in all_features:
    train_path = os.path.join(cfg.paths.train_savepath, f"train_{feat}.npy")
    val_path = os.path.join(cfg.paths.val_savepath, f"val_{feat}.npy")
    all_train.append(np.load(train_path))
    all_val.append(np.load(val_path))

train_stacked = np.stack(all_train, axis=-1)
val_stacked   = np.stack(all_val,   axis=-1)

print("Train stacked shape:", train_stacked.shape)
print("Val stacked shape:",   val_stacked.shape)

np.save(os.path.join(cfg.paths.train_savepath, "train_all.npy"), train_stacked.astype(np.float32))
np.save(os.path.join(cfg.paths.val_savepath,   "val_all.npy"),   val_stacked.astype(np.float32))

print("Done.")