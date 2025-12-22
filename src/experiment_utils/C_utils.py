import argparse
import os
import sys
import json
import yaml
import random
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split

import wandb

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _require_dir(path, what):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{what} directory not found: {path}")


def _require_file(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _print_access(label, path):
    print(f"[data] {label}: {os.path.abspath(path)}")


def resolve_paths(data_root):
    """
    Returns all dataset roots + risk json paths following the new structure.
    """
    return {
        # TRAIN
        "l2d_train_graphs": os.path.join(data_root, "training_data", "L2D", "graphs"),
        "l2d_train_risk": os.path.join(data_root, "training_data", "L2D", "risk_scores_L2D.json"),
        "nuplan_train_graphs": os.path.join(data_root, "training_data", "NuPlan", "graphs"),
        "nuplan_train_risk": os.path.join(data_root, "training_data", "NuPlan", "risk_scores_NuPlan.json"),

        # EVAL
        "l2d_eval_graphs": os.path.join(data_root, "evaluation_data", "L2D", "graphs"),
        "l2d_eval_risk": os.path.join(data_root, "evaluation_data", "L2D", "risk_scores_L2D_true.json"),
        "nuplan_eval_graphs": os.path.join(data_root, "evaluation_data", "NuPlan", "graphs"),
        "nuplan_eval_risk": os.path.join(data_root, "evaluation_data", "NuPlan", "risk_scores_NuPlan.json"),
    }


def diag_gaussian_kl(mu_p, logvar_p, mu_q, logvar_q, reduction):
    var_p = torch.exp(logvar_p)
    var_q = torch.exp(logvar_q)
    kl_per_dim = 0.5 * (
        (logvar_q - logvar_p) + (var_p + (mu_p - mu_q) ** 2) / (var_q + 1e-8) - 1.0
    )
    kl = kl_per_dim.sum(dim=-1)  # [B]
    if reduction == "mean":
        return kl.mean()
    if reduction == "sum":
        return kl.sum()
    return kl


def kl_from_projected(zp, zq, reduction):
    """
    If proj outputs [mu|logvar] (dim even), computes KL between diagonal Gaussians.
    If proj outputs plain embedding (dim odd), assumes unit variance.
    """
    if zp.ndim != 2 or zq.ndim != 2:
        raise ValueError(f"Expected 2D tensors [B, D]; got {zp.shape} and {zq.shape}")
    if zp.shape != zq.shape:
        raise ValueError(f"Shape mismatch for KL: {zp.shape} vs {zq.shape}")

    d = zp.shape[-1]
    if d % 2 == 0:
        k = d // 2
        mu_p, logvar_p = zp[:, :k], zp[:, k:]
        mu_q, logvar_q = zq[:, :k], zq[:, k:]
        return diag_gaussian_kl(mu_p, logvar_p, mu_q, logvar_q, reduction=reduction)
    else:
        mu_p, mu_q = zp, zq
        zeros_p = torch.zeros_like(mu_p)
        zeros_q = torch.zeros_like(mu_q)
        return diag_gaussian_kl(mu_p, zeros_p, mu_q, zeros_q, reduction=reduction)


def _targets_from_batch(batch, prediction_mode):
    """
    Prefer using batch.y (already aligned with the dataset's risk_scores_path).
    This avoids brittle episode-id lookups and mismatched JSON schemas.

    - regression: returns float tensor [B,1]
    - classification: returns long tensor [B]
    """
    if not hasattr(batch, "y"):
        raise AttributeError("Batch has no attribute 'y'. get_graph_dataset should attach risk labels as batch.y.")

    if prediction_mode == "regression":
        return batch.y.view(-1, 1).float().to(device)
    else:
        y = batch.y.view(-1).to(device)
        if y.dtype.is_floating_point:
            raise RuntimeError(
                "Classification mode but batch.y is floating point. "
                "Either preprocess labels into classes, or enable --class_thresholds to bin."
            )
        return y.long()
