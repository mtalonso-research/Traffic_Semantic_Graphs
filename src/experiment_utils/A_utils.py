import os
from typing import Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    mean_absolute_error,
    cohen_kappa_score,
)

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


# -----------------------------
# Utils / robust target extraction
# -----------------------------
def _to_numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def _is_integer_like(arr: np.ndarray, num_classes: int, tol: float = 1e-6) -> bool:
    """True if values are integers (or very close) and within [0, num_classes-1]."""
    if arr.size == 0:
        return False
    a = arr.astype(np.float64)
    if np.any(np.isnan(a)) or np.any(np.isinf(a)):
        return False
    if a.min() < -tol or a.max() > (num_classes - 1) + tol:
        return False
    return np.max(np.abs(a - np.round(a))) <= tol


def extract_graph_targets(batch, prediction_mode: str, num_classes: int) -> np.ndarray:
    """
    Extract graph-level targets from a PyG Batch robustly.
    - For classification: prefer integer-like class labels.
    - For regression: prefer float-like risk scores.
    """
    keys = []
    try:
        keys = list(batch.keys())
    except Exception:
        keys = [k for k in dir(batch) if not k.startswith("_")]

    num_graphs = getattr(batch, "num_graphs", None)

    preferred_cls = [
        "risk_class", "risk_bin", "risk_label", "y_class", "y_cls",
        "class_label", "labels_cls", "target_cls"
    ]

    preferred_reg = [
        "risk_score", "risk_scores", "risk", "y_reg", "y_score",
        "target", "targets"
    ]

    def get_attr(name: str) -> Optional[np.ndarray]:
        if hasattr(batch, name):
            return _to_numpy(getattr(batch, name))
        return None

    if prediction_mode == "classification":
        for name in preferred_cls:
            arr = get_attr(name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr.reshape(-1)
            if num_graphs is not None and arr.shape[0] != num_graphs:
                continue
            if _is_integer_like(arr, num_classes):
                return np.round(arr).astype(np.int64)

        for name in keys:
            if not hasattr(batch, name):
                continue
            arr = get_attr(name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 0:
                continue
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr.reshape(-1)

            if num_graphs is not None and arr.shape[0] != num_graphs:
                continue

            if arr.dtype.kind in ("i", "u", "b"):
                if arr.min() >= 0 and arr.max() < num_classes:
                    return arr.astype(np.int64)

            if arr.dtype.kind == "f" and _is_integer_like(arr, num_classes):
                return np.round(arr).astype(np.int64)

        for name in ["y", "label", "labels"]:
            arr = get_attr(name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr.reshape(-1)
            if num_graphs is not None and arr.shape[0] != num_graphs:
                continue
            if _is_integer_like(arr, num_classes):
                return np.round(arr).astype(np.int64)

            if name == "y" and arr.dtype.kind == "f":
                return risk_to_class_safe(torch.from_numpy(arr)).numpy()

            raise RuntimeError(
                f"Found '{name}' but it does NOT look like class labels "
                f"(min={arr.min()}, max={arr.max()}, dtype={arr.dtype}). "
                f"Likely you’re pointing at a continuous risk score."
            )

        raise RuntimeError(
            "Could not find classification targets on batch. "
            "Expected a graph-level integer label field like risk_class/risk_bin/y_class."
        )

    else:
        for name in preferred_reg:
            arr = get_attr(name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr.reshape(-1)
            if num_graphs is not None and arr.shape[0] != num_graphs:
                continue
            return arr.astype(np.float64)

        for name in keys:
            if not hasattr(batch, name):
                continue
            arr = get_attr(name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 0:
                continue
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr.reshape(-1)
            if num_graphs is not None and arr.shape[0] != num_graphs:
                continue
            if arr.dtype.kind == "f":
                return arr.astype(np.float64)

        arr = get_attr("y")
        if arr is not None:
            arr = np.asarray(arr)
            if arr.ndim > 1 and arr.shape[-1] == 1:
                arr = arr.reshape(-1)
            if num_graphs is not None and arr.shape[0] != num_graphs:
                raise RuntimeError("Found y but it is not graph-level (length != num_graphs).")
            return arr.astype(np.float64)

        raise RuntimeError(
            "Could not find regression targets on batch. Expected graph-level float field like risk_score/risk/y."
        )


# -----------------------------
# Metrics: classification
# -----------------------------
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def classification_metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    K = cm.shape[0]
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)

    tp = np.diag(cm).astype(np.float64)
    fn = support - tp
    fp = pred_count - tp

    precision = np.array([_safe_div(tp[k], tp[k] + fp[k]) for k in range(K)], dtype=np.float64)
    recall = np.array([_safe_div(tp[k], tp[k] + fn[k]) for k in range(K)], dtype=np.float64)
    f1 = np.array([_safe_div(2 * precision[k] * recall[k], precision[k] + recall[k]) for k in range(K)], dtype=np.float64)

    total = float(cm.sum())
    acc = _safe_div(tp.sum(), total)

    macro_precision = float(np.mean(precision)) if K else 0.0
    macro_recall = float(np.mean(recall)) if K else 0.0
    macro_f1 = float(np.mean(f1)) if K else 0.0

    weights = support / total if total > 0 else np.zeros_like(support)
    weighted_precision = float(np.sum(weights * precision))
    weighted_recall = float(np.sum(weights * recall))
    weighted_f1 = float(np.sum(weights * f1))

    balanced_acc = macro_recall
    micro_f1 = acc  # single-label multiclass

    return {
        "accuracy": acc,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "support_per_class": support.astype(int).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(np.float64) - y_pred.astype(np.float64)))) if len(y_true) else 0.0


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    K = num_classes
    if len(y_true) == 0:
        return 0.0
    O = confusion_matrix(y_true, y_pred, K).astype(np.float64)
    N = O.sum()
    if N == 0:
        return 0.0

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / N

    denom = (K - 1) ** 2 if K > 1 else 1.0
    W = np.zeros((K, K), dtype=np.float64)
    for i in range(K):
        for j in range(K):
            W[i, j] = ((i - j) ** 2) / denom

    num = (W * O).sum()
    den = (W * E).sum()
    return float(1.0 - _safe_div(num, den))


def log_loss_from_hard_preds(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    if len(y_true) == 0:
        return 0.0
    p = np.full(len(y_true), eps, dtype=np.float64)
    p[y_pred == y_true] = 1.0 - eps
    return float(-np.mean(np.log(p)))


# -----------------------------
# Metrics: regression
# -----------------------------
def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    if np.std(yt) == 0 or np.std(yp) == 0:
        return 0.0
    return float(np.corrcoef(yt, yp)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    if len(y_true) < 2:
        return {
            "mae": 0.0,
            "mse": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "spearman_p": 1.0,
        }

    yt = y_true.astype(np.float64)
    yp = y_pred.astype(np.float64)
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - _safe_div(ss_res, ss_tot)) if ss_tot != 0 else 0.0

    rho, p = spearmanr(yt, yp)
    if not np.isfinite(rho):
        rho = 0.0
    if not np.isfinite(p):
        p = 1.0

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson_r": _pearson_r(yt, yp),
        "spearman_rho": rho,
        "spearman_p": p,
    }


def tail_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> Dict[str, Any]:
    if len(y_true) == 0:
        return {"tail_q": q, "tail_count": 0, "tail_mae": 0.0, "tail_rmse": 0.0, "tail_threshold": 0.0}
    thresh = float(np.quantile(y_true.astype(np.float64), q))
    mask = y_true.astype(np.float64) >= thresh
    if not np.any(mask):
        return {"tail_q": q, "tail_count": 0, "tail_mae": 0.0, "tail_rmse": 0.0, "tail_threshold": thresh}
    yt = y_true[mask].astype(np.float64)
    yp = y_pred[mask].astype(np.float64)
    err = yp - yt
    return {
        "tail_q": q,
        "tail_threshold": thresh,
        "tail_count": int(np.sum(mask)),
        "tail_mae": float(np.mean(np.abs(err))),
        "tail_rmse": float(np.sqrt(np.mean(err ** 2))),
    }


# -----------------------------
# sklearn-expanded classification metrics 
# -----------------------------
def sklearn_expanded_classification_metrics(
    y_true_cls: np.ndarray,
    y_pred_cls: np.ndarray,
) -> Dict[str, Any]:
    return {
        "f1_score": f1_score(y_true_cls, y_pred_cls, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true_cls, y_pred_cls),
        "mean_absolute_error": mean_absolute_error(y_true_cls, y_pred_cls),
        "quadratic_weighted_kappa": cohen_kappa_score(y_true_cls, y_pred_cls, weights="quadratic"),
    }
