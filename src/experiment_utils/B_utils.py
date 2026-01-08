import os
import sys
import time
import uuid
import yaml
import random
from typing import Dict, Any

import numpy as np
import torch

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


def episode_id_from_episode_path(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return base.split("_")[0]


def risk_to_class_safe(risk_scores):
    r = risk_scores.view(-1).float()
    y = torch.zeros_like(r, dtype=torch.long)
    y[r > 0.0043] = 1
    y[r > 0.1008] = 2
    y[r > 0.3442] = 3
    return y


def infer_graph_emb_dim(encoder, quantizer, loader, metadata, embed_dim_per_type):
    encoder.eval()
    with torch.no_grad():
        batch0 = next(iter(loader))
        batch0 = quantizer.transform_inplace(batch0).to(device)
        z_dict, _, _ = encoder(batch0)
        g = batched_graph_embeddings(z_dict, batch0, metadata, embed_dim_per_type=embed_dim_per_type)
        return int(g.shape[-1])


def _unwrap_wandb_value(v):
    if isinstance(v, dict) and "value" in v and len(v) == 1:
        return v["value"]
    return v


def apply_yaml_overrides(parser, args):
    if args.load_config is None:
        return args

    if not os.path.isfile(args.load_config):
        raise FileNotFoundError(f"--load_config file not found: {args.load_config}")

    with open(args.load_config, "r") as f:
        cfg = yaml.safe_load(f) or {}

    defaults = parser.parse_args([])

    for k, raw_v in cfg.items():
        if k == "_wandb":
            continue
        if not hasattr(args, k):
            continue

        v = _unwrap_wandb_value(raw_v)
        current = getattr(args, k)
        default = getattr(defaults, k)

        if current == default:
            setattr(args, k, v)

    return args


def enforce_dataset_cli_wins(args):
    argv = sys.argv
    cli_nup = "--nup" in argv
    cli_l2d = "--l2d" in argv

    if cli_nup and cli_l2d:
        raise SystemExit("Specify exactly one dataset: --l2d or --nup (not both).")

    if cli_nup:
        args.nup = True
        args.l2d = False
    elif cli_l2d:
        args.l2d = True
        args.nup = False


def _require_file(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _require_dir(path, what):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{what} directory not found: {path}")


def _print_access(label, path):
    print(f"[data] {label}: {os.path.abspath(path)}")


def resolve_paths(args, dataset_name):
    base = args.data_root  # defaults to "data"

    train_graph_root = os.path.join(base, "training_data", dataset_name, "graphs")
    train_risk_path = os.path.join(base, "training_data", dataset_name, f"risk_scores_{dataset_name}.json")

    eval_graph_root = os.path.join(base, "evaluation_data", dataset_name, "graphs")
    if dataset_name == "L2D":
        eval_risk_path = os.path.join(base, "evaluation_data", "L2D", "risk_scores_L2D_true.json")
    else:
        eval_risk_path = os.path.join(base, "evaluation_data", "NuPlan", "risk_scores_NuPlan.json")

    return {
        "train_graph_root": train_graph_root,
        "train_risk_path": train_risk_path,
        "eval_graph_root": eval_graph_root,
        "eval_risk_path": eval_risk_path,
    }


def _format_confusion_matrix(cm):
    """
    Pretty-print confusion matrix with rows=true, cols=pred.
    """
    n = cm.shape[0]
    header = "true\\pred | " + " ".join([f"{i:>6d}" for i in range(n)])
    sep = "-" * len(header)
    lines = [header, sep]
    for i in range(n):
        row = " ".join([f"{cm[i, j]:>6d}" for j in range(n)])
        lines.append(f"{i:>9d} | {row}")
    return "\n".join(lines)


# -----------------------------
# helpers
# -----------------------------
def _make_fallback_run_id() -> str:
    # Safe unique ID even without wandb: timestamp + pid + random suffix
    return f"local-{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else 0.0


def classification_metrics_from_cm(cm: np.ndarray) -> Dict[str, Any]:
    K = cm.shape[0]
    support = cm.sum(axis=1).astype(np.float64)
    pred_count = cm.sum(axis=0).astype(np.float64)

    tp = np.diag(cm).astype(np.float64)
    fn = support - tp
    fp = pred_count - tp

    precision = np.array([safe_div(tp[k], tp[k] + fp[k]) for k in range(K)], dtype=np.float64)
    recall = np.array([safe_div(tp[k], tp[k] + fn[k]) for k in range(K)], dtype=np.float64)
    f1 = np.array([safe_div(2 * precision[k] * recall[k], precision[k] + recall[k]) for k in range(K)], dtype=np.float64)

    total = float(cm.sum())
    acc = safe_div(tp.sum(), total)

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
