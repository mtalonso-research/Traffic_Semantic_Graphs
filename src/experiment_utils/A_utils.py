import argparse
import os
import json
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.loader import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph_encoding.data_loaders import get_graph_dataset  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _print_access(label, path):
    print(f"[data] {label}: {os.path.abspath(path)}")


def _require_file(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _require_dir(path, what):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{what} directory not found: {path}")


def risk_to_class_safe(risk_scores):
    r = risk_scores.view(-1).float()
    y = torch.zeros_like(r, dtype=torch.long)
    y[r > 0.0043] = 1
    y[r > 0.1008] = 2
    y[r > 0.3442] = 3
    return y


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


def load_dataset(graph_root, risk_path, mode):
    _print_access("graphs root", graph_root)
    _print_access("risk scores", risk_path)
    _require_dir(graph_root, "Graphs")
    _require_file(risk_path, "Risk scores JSON")

    return get_graph_dataset(
        root_dir=graph_root,
        mode=mode,
        side_information_path=None,  # baseline ignores side info
        risk_scores_path=risk_path,
    )


def compute_train_constant(train_loader, prediction_mode, num_classes):
    """
    Returns:
      - mean risk (float) if regression
      - most frequent class (int) if classification
    For convenience, returns both:
      (mean_risk, mode_class)
    where one is meaningful depending on prediction_mode.
    """
    ys = []
    class_counts = np.zeros(num_classes, dtype=np.int64)

    for batch in tqdm(train_loader, desc="[fit] scanning training y"):
        y = batch.y.detach().cpu().view(-1).numpy().astype(np.float64)
        ys.append(y)

        y_cls = risk_to_class_safe(batch.y).detach().cpu().numpy()
        for c in y_cls:
            if 0 <= int(c) < num_classes:
                class_counts[int(c)] += 1

    y_all = np.concatenate(ys) if ys else np.array([0.0], dtype=np.float64)
    mean_risk = float(np.mean(y_all))
    mode_class = int(np.argmax(class_counts)) if class_counts.sum() > 0 else 0

    if prediction_mode == "regression":
        print(f"[fit] learned constant mean risk = {mean_risk:.6f} from {y_all.size} samples")
    else:
        print(f"[fit] learned constant mode class = {mode_class} from {class_counts.sum()} samples")
        print(f"[fit] class counts = {class_counts.tolist()}")

    return mean_risk, mode_class


def evaluate_constant(
    loader,
    prediction_mode,
    mean_risk,
    mode_class,
    num_classes,
):
    """
    Computes:
      regression: MSE vs y
      classification: CE loss + accuracy vs thresholded y
    """
    if prediction_mode == "regression":
        loss_fn = nn.MSELoss(reduction="mean")
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(loader, desc="[eval]"):
                target = batch.y.view(-1, 1).float().to(device)
                pred = torch.full_like(target, float(mean_risk))
                loss = loss_fn(pred, target)
                total_loss += float(loss.item())
                n_batches += 1

        return {
            "mse": total_loss / max(n_batches, 1),
            "mean_pred": mean_risk,
        }

    else:
        # CrossEntropy expects logits [B, C] and targets [B]
        loss_fn = nn.CrossEntropyLoss(reduction="mean")
        total_loss = 0.0
        correct = 0
        total = 0
        n_batches = 0

        # Build constant logits that always pick mode_class.
        # Using big positive for mode class and 0 for others.
        big = 10.0

        with torch.no_grad():
            for batch in tqdm(loader, desc="[eval]"):
                target = risk_to_class_safe(batch.y).to(device)  # [B]
                bsz = int(target.shape[0])

                logits = torch.zeros((bsz, num_classes), device=device, dtype=torch.float32)
                logits[:, mode_class] = big

                loss = loss_fn(logits, target)
                total_loss += float(loss.item())
                n_batches += 1

                pred = torch.argmax(logits, dim=-1)
                correct += int((pred == target).sum().item())
                total += bsz

        return {
            "cross_entropy": total_loss / max(n_batches, 1),
            "accuracy": correct / max(total, 1),
            "mode_class": mode_class,
        }