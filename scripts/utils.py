import os
import json
import time
import torch
import random
import numpy as np
import yaml
from datetime import datetime

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_experiment_folder(config):
    out_dir = config["experiment"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Saving config snapshot
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Creating metadata
    meta = {
        "start_time": str(datetime.now()),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def finalize_experiment(out_dir, metrics: dict, start_time):
    end_time = time.time()

    summary = {
        "metrics": metrics,
        "runtime_seconds": end_time - start_time,
        "end_time": str(datetime.now())
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
