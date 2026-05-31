import os
import json
import torch
import statistics
import argparse
import numpy as np
import pickle
import random


_epsilon = 1e-08


def safe_div(x, y):
    return torch.div(x, (y + _epsilon))


def safe_log(x):
    return torch.log(x + _epsilon)


def safe_sqrt(x):
    x = torch.clamp(x, min=_epsilon)
    try:
        return torch.sqrt(x)
    except (AttributeError, TypeError):
        return x ** 0.5


def check_bad(x: torch.Tensor):
    assert not torch.isnan(x).any() or torch.isinf(x).any(), "Nan or Inf in tensor"


def set_seed(seed, device):
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_params(
        config: argparse.Namespace
) -> str:
    """
    Saves args for reproducing results
    """
    dir_ = os.getcwd()
    path = f"{dir_}/runs/{config.data}/{config.model}/{config.timestamp}"

    if not os.path.exists(path):
        os.makedirs(path)

    config_dict = config.as_dict() if hasattr(config, "as_dict") else vars(config)
    with open(f'{path}/commandline_args.txt', 'w') as f:
        json.dump(config_dict, f, indent=2)

    return path


def print_performance(
        path: str = None,
        **kwargs
) -> None:
    """
    Print performance and save it locally.
    """
    prf = f""
    for k, v in kwargs.items():
        if k == "dcal_pvalues":
            # Count number of p-values > 0.05 (calibrated)
            count = sum([1 for x in v if x > 0.05])
            prf += f"{k}: {count}/{len(v)}\n"
            continue

        if len(v) == 0 or None in v:
            continue

        if isinstance(v, list):
            mean = statistics.mean(v)
            std = statistics.stdev(v) if len(v) > 1 else 0.0   # sample standard deviation (n-1)
            prf += f"{k}: {mean:.3f} +/- {std:.3f}\n"
        else:
            prf += f"{k}: {v:.3f}\n"
    print(prf)

    if path is not None:
        prf_dict = {k: v for k, v in kwargs.items()}
        with open(f"{path}/performance.pkl", 'wb') as f:
            pickle.dump(prf_dict, f)

        with open(f"{path}/performance.txt", 'w') as f:
            f.write(prf)


def pad_tensor(
        logits: torch.Tensor,
        val: float = 0,
        where: str = 'end'
) -> torch.Tensor:
    """Add a column of `val` at the start of end of `input`."""
    if len(logits.shape) == 1:
        pad = torch.tensor([val], dtype=logits.dtype, device=logits.device)

        if where == 'end':
            return torch.cat([logits, pad])
        elif where == 'start':
            return torch.cat([pad, logits])
        else:
            raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")
    elif len(logits.shape) == 2:
        pad = torch.zeros(logits.size(0), 1, dtype=logits.dtype, device=logits.device) + val

        if where == 'end':
            return torch.cat([logits, pad], dim=1)
        elif where == 'start':
            return torch.cat([pad, logits], dim=1)
        else:
            raise ValueError(f"Need `where` to be 'start' or 'end', got {where}")
    else:
        raise ValueError("The logits must be either a 1D or 2D tensor")


def df2np(df):
    x = df.drop(["time", "event"], axis=1).values
    t, e = df["time"].values, df["event"].values
    return x, t, e
