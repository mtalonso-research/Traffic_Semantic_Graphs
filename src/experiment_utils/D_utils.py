import argparse
import os
import sys
import json
import random
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

import wandb
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph_encoding.autoencoder import batched_graph_embeddings

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


def _require_dir(path, what):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{what} directory not found: {path}")


def _require_file(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _print_access(label, path):
    print(f"[data] {label}: {os.path.abspath(path)}")


def episode_id_from_episode_path(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return base.split("_")[0]


def episode_ids_from_batch(data):
    paths = data["window_meta"].episode_path
    return [episode_id_from_episode_path(p) for p in paths]


def risk_to_class_safe(risk_scores):
    r = risk_scores.view(-1).float()
    y = torch.zeros_like(r, dtype=torch.long)
    y[r > 0.0043] = 1
    y[r > 0.1008] = 2
    y[r > 0.3442] = 3
    return y


def compute_confusion_matrix(num_classes, y_true, y_pred):
    """
    y_true, y_pred: 1D Long tensors on CPU or GPU
    returns [C,C] where rows=true, cols=pred
    """
    if y_true.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.long)
    y_true = y_true.view(-1).long()
    y_pred = y_pred.view(-1).long()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=y_true.device)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def apply_yaml_overrides(parser, args):
    if args.config is None:
        return args
    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"--config not found: {args.config}")
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f) or {}
    defaults = parser.parse_args([])
    for k, v in cfg.items():
        if not hasattr(args, k):
            print(f"[warn] config key '{k}' not in args; ignoring.")
            continue
        if getattr(args, k) == getattr(defaults, k):
            setattr(args, k, v)
    return args


def resolve_paths(data_root, dataset_name):
    """
    TRAIN:
      data/training_data/<DATASET>/graphs/
      data/training_data/<DATASET>/risk_scores_<DATASET>.json

    EVAL:
      data/evaluation_data/<DATASET>/graphs/
      L2D:    data/evaluation_data/L2D/risk_scores_L2D_true.json
      NuPlan: data/evaluation_data/NuPlan/risk_scores_NuPlan.json
    """
    train_graph_root = os.path.join(data_root, "training_data", dataset_name, "graphs")
    train_risk_path = os.path.join(data_root, "training_data", dataset_name, f"risk_scores_{dataset_name}.json")

    eval_graph_root = os.path.join(data_root, "evaluation_data", dataset_name, "graphs")
    if dataset_name == "L2D":
        eval_risk_path = os.path.join(data_root, "evaluation_data", "L2D", "risk_scores_L2D_true.json")
    else:
        eval_risk_path = os.path.join(data_root, "evaluation_data", "NuPlan", "risk_scores_NuPlan.json")

    return {
        "train_graph_root": train_graph_root,
        "train_risk_path": train_risk_path,
        "eval_graph_root": eval_graph_root,
        "eval_risk_path": eval_risk_path,
    }


def infer_graph_emb_dim(encoder, quantizer, loader, metadata, embed_dim_per_type):
    encoder.eval()
    with torch.no_grad():
        batch0 = next(iter(loader))
        batch0 = quantizer.transform_inplace(batch0).to(device)
        z_dict, _, _ = encoder(batch0)
        g = batched_graph_embeddings(z_dict, batch0, metadata, embed_dim_per_type=embed_dim_per_type)
        return int(g.shape[-1])


def make_risk_targets_from_batch(
    batch,
    prediction_mode,
    class_thresholds,
):
    """
    This expects batch.y to be present (since get_graph_dataset uses risk_scores_path).
    """
    if prediction_mode == "regression":
        return batch.y.view(-1, 1).float()
    return risk_to_class_safe(batch.y, thresholds=class_thresholds)
