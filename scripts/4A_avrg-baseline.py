import argparse
import os
import json
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment_utils.A_utils import (
    resolve_paths,
    load_dataset,
    compute_train_constant,
    extract_graph_targets,
    confusion_matrix,
    classification_metrics_from_cm,
    ordinal_mae,
    quadratic_weighted_kappa,
    log_loss_from_hard_preds,
    regression_metrics,
    tail_regression_metrics,
    sklearn_expanded_classification_metrics,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(args) -> Dict[str, Any]:
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
        shuffle=False,
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

    y_true_all = []
    for batch in tqdm(eval_loader, desc="[eval] scanning graph-level targets"):
        y = extract_graph_targets(batch, args.prediction_mode, args.num_classes)
        y_true_all.append(y)

    y_true = np.concatenate(y_true_all, axis=0)
    metrics: Dict[str, Any] = {
        "dataset": dataset_name,
        "mode": args.mode,
        "prediction_mode": args.prediction_mode,
        "num_eval_samples": int(len(y_true)),
    }

    if args.prediction_mode == "classification":
        y_true_cls = y_true.astype(np.int64)
        y_pred_cls = np.full_like(y_true_cls, int(mode_class), dtype=np.int64)

        cm = confusion_matrix(y_true_cls, y_pred_cls, args.num_classes)
        cls = classification_metrics_from_cm(cm)
        cls["constant_predicted_class"] = int(mode_class)
        cls["ordinal_mae_bins"] = ordinal_mae(y_true_cls, y_pred_cls)
        cls["qwk"] = quadratic_weighted_kappa(y_true_cls, y_pred_cls, args.num_classes)
        cls["log_loss_hard"] = log_loss_from_hard_preds(y_true_cls, y_pred_cls)
        cls.update(sklearn_expanded_classification_metrics(y_true_cls, y_pred_cls))

        metrics.update(cls)

    else:
        y_true_reg = y_true.astype(np.float64)
        y_pred_reg = np.full_like(y_true_reg, float(mean_risk), dtype=np.float64)

        reg = regression_metrics(y_true_reg, y_pred_reg)
        metrics.update(reg)
        metrics["constant_predicted_mean"] = float(mean_risk)

        t90 = tail_regression_metrics(y_true_reg, y_pred_reg, q=0.90)
        t95 = tail_regression_metrics(y_true_reg, y_pred_reg, q=0.95)
        metrics.update({f"tail90_{k}": v for k, v in t90.items()})
        metrics.update({f"tail95_{k}": v for k, v in t95.items()})

    print("\n========== Baseline results (expanded) ==========")
    for k in sorted(metrics.keys()):
        v = metrics[k]
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    if "confusion_matrix" in metrics:
        print("confusion_matrix:")
        for row in metrics["confusion_matrix"]:
            print(row)

    print("===============================================\n")

    if args.save_metrics_json:
        os.makedirs(os.path.dirname(args.save_metrics_json) or ".", exist_ok=True)
        with open(args.save_metrics_json, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[info] saved metrics json -> {args.save_metrics_json}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Constant baseline: predict training mean (or mode class).")

    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--l2d", action="store_true")
    parser.add_argument("--nup", action="store_true")

    parser.add_argument("--mode", type=str, default="all")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--prediction_mode", type=str, default="classification", choices=["regression", "classification"])
    parser.add_argument("--num_classes", type=int, default=4)

    parser.add_argument("--save_metrics_json", type=str, default="")

    args = parser.parse_args()
    run(args)
