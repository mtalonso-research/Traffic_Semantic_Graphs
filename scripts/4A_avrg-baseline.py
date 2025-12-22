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
from src.experiment_utils.A_utils import resolve_paths, load_dataset, compute_train_constant, evaluate_constant


def main():
    parser = argparse.ArgumentParser(description="Constant baseline: predict training mean (or mode class).")

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--l2d", action="store_true")
    parser.add_argument("--nup", action="store_true")

    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--prediction_mode", type=str, default="classification", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=4)

    args = parser.parse_args()

    if args.l2d == args.nup:
        raise SystemExit("Specify exactly one dataset: --l2d or --nup (not both, not neither).")

    dataset_name = "L2D" if args.l2d else "NuPlan"
    paths = resolve_paths(args.data_root, dataset_name)

    print(f"[info] dataset = {dataset_name}")
    print("[info] loading TRAIN...")
    train_ds = load_dataset(paths["train_graph_root"], paths["train_risk_path"], mode=args.mode)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,  # baseline fit is just a scan; shuffle irrelevant
        num_workers=args.num_workers,
    )

    mean_risk, mode_class = compute_train_constant(train_loader, args.prediction_mode, args.num_classes)

    print("[info] loading EVAL...")
    eval_ds = load_dataset(paths["eval_graph_root"], paths["eval_risk_path"], mode=args.mode)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metrics = evaluate_constant(
        eval_loader,
        prediction_mode=args.prediction_mode,
        mean_risk=mean_risk,
        mode_class=mode_class,
        num_classes=args.num_classes,
    )

    print("\n========== Baseline results ==========")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")
    print("=====================================\n")


if __name__ == "__main__":
    main()
