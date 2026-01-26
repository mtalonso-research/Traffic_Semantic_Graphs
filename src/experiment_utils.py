import os
import sys
import time
import uuid
import yaml
import random
from typing import Dict, Any, Optional
import argparse
import os
import sys
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import  Dataset
from scipy.stats import spearmanr
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

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph_encoding.data_loaders import get_graph_dataset
from src.graph_encoding.autoencoder import batched_graph_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# Reproducibility
# =====================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# =====================================================================
# Filesystem helpers
# =====================================================================
def _print_access(label, path):
    print(f"[data] {label}: {os.path.abspath(path)}")


def _require_file(path, what):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{what} not found: {path}")


def _require_dir(path, what):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{what} directory not found: {path}")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _make_fallback_run_id() -> str:
    return f"local-{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"

# =====================================================================
# Dataset paths & loading
# =====================================================================
def resolve_paths(data_root_or_args, dataset_name: str):
    """
    Constructs train/eval paths.

    Train:
      - uses dataset_name literally

    Eval:
      - noisy_*  -> evaluated literally, uses risk_scores_true.json
      - clean*   -> evaluated literally, uses risk_scores.json
    """
    if hasattr(data_root_or_args, "data_root"):
        base = data_root_or_args.data_root
    else:
        base = str(data_root_or_args)

    name_lower = dataset_name.lower()

    train_graph_root = os.path.join(base, "training_data", dataset_name, "graphs")
    train_risk_path = os.path.join(base, "training_data", dataset_name, "risk_scores.json")

    if "noisy" in name_lower:
        eval_dataset = dataset_name
        eval_risk_file = "risk_scores_true.json"
    elif name_lower.startswith("clean"):
        eval_dataset = dataset_name
        eval_risk_file = "risk_scores.json"
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

    eval_graph_root = os.path.join(base, "evaluation_data", eval_dataset, "graphs")
    eval_risk_path = os.path.join(base, "evaluation_data", eval_dataset, eval_risk_file)

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
        side_information_path=None,
        risk_scores_path=risk_path,
    )

# =====================================================================
# Risk → class
# =====================================================================
def risk_to_class_safe(risk_scores):
    r = risk_scores.view(-1).float()
    y = torch.zeros_like(r, dtype=torch.long)
    y[r > 0.0043] = 1
    y[r > 0.1008] = 2
    y[r > 0.3442] = 3
    return y

# =====================================================================
# Training constants
# =====================================================================
def compute_train_constant(train_loader, prediction_mode, num_classes):
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

    return mean_risk, mode_class

# =====================================================================
# Robust target extraction
# =====================================================================
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
    if arr.size == 0:
        return False
    if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
        return False
    if arr.min() < -tol or arr.max() > (num_classes - 1) + tol:
        return False
    return np.max(np.abs(arr - np.round(arr))) <= tol


def extract_graph_targets(batch, prediction_mode: str, num_classes: int) -> np.ndarray:
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
            arr = np.asarray(arr).reshape(-1)
            if num_graphs is not None and arr.shape[0] != num_graphs:
                continue
            if _is_integer_like(arr, num_classes):
                return np.round(arr).astype(np.int64)

        for name in keys:
            arr = get_attr(name)
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 0:
                continue
            arr = arr.reshape(-1)
            if num_graphs is not None and arr.shape[0] != num_graphs:
                continue
            if arr.dtype.kind in ("i", "u", "b"):
                if arr.min() >= 0 and arr.max() < num_classes:
                    return arr.astype(np.int64)
            if arr.dtype.kind == "f" and _is_integer_like(arr, num_classes):
                return np.round(arr).astype(np.int64)

        arr = get_attr("y")
        if arr is not None:
            arr = np.asarray(arr).reshape(-1)
            if _is_integer_like(arr, num_classes):
                return np.round(arr).astype(np.int64)
            return risk_to_class_safe(torch.from_numpy(arr)).numpy()

        raise RuntimeError("Could not find classification targets.")

    else:
        for name in preferred_reg:
            arr = get_attr(name)
            if arr is None:
                continue
            arr = np.asarray(arr).reshape(-1)
            if num_graphs is not None and arr.shape[0] != num_graphs:
                continue
            return arr.astype(np.float64)

        arr = get_attr("y")
        if arr is not None:
            arr = np.asarray(arr).reshape(-1)
            return arr.astype(np.float64)

        raise RuntimeError("Could not find regression targets.")

# =====================================================================
# Metrics — classification
# =====================================================================
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

    precision = np.array([_safe_div(tp[k], tp[k] + fp[k]) for k in range(K)])
    recall = np.array([_safe_div(tp[k], tp[k] + fn[k]) for k in range(K)])
    f1 = np.array([_safe_div(2 * precision[k] * recall[k], precision[k] + recall[k]) for k in range(K)])

    total = float(cm.sum())
    acc = _safe_div(tp.sum(), total)

    return {
        "accuracy": acc,
        "micro_f1": acc,
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "confusion_matrix": cm.tolist(),
    }


def ordinal_mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else 0.0


def quadratic_weighted_kappa(y_true, y_pred, num_classes):
    O = confusion_matrix(y_true, y_pred, num_classes).astype(np.float64)
    N = O.sum()
    if N == 0:
        return 0.0

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred) / N

    W = np.zeros((num_classes, num_classes))
    denom = (num_classes - 1) ** 2 if num_classes > 1 else 1.0
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / denom

    return float(1.0 - _safe_div((W * O).sum(), (W * E).sum()))

# =====================================================================
# Metrics — regression
# =====================================================================
def _pearson_r(y_true, y_pred):
    if len(y_true) < 2:
        return 0.0
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true, y_pred):
    if len(y_true) < 2:
        return dict(mae=0.0, mse=0.0, rmse=0.0, r2=0.0,
                    pearson_r=0.0, spearman_rho=0.0, spearman_p=1.0)

    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - _safe_div(ss_res, ss_tot)) if ss_tot != 0 else 0.0

    rho, p = spearmanr(y_true, y_pred)
    return dict(
        mae=mae,
        mse=mse,
        rmse=rmse,
        r2=r2,
        pearson_r=_pearson_r(y_true, y_pred),
        spearman_rho=float(rho) if np.isfinite(rho) else 0.0,
        spearman_p=float(p) if np.isfinite(p) else 1.0,
    )

# =====================================================================
# sklearn-expanded metrics
# =====================================================================
def sklearn_expanded_classification_metrics(y_true_cls, y_pred_cls):
    return {
        "f1_score": f1_score(y_true_cls, y_pred_cls, average="macro"),
        "balanced_accuracy": balanced_accuracy_score(y_true_cls, y_pred_cls),
        "mean_absolute_error": mean_absolute_error(y_true_cls, y_pred_cls),
        "quadratic_weighted_kappa": cohen_kappa_score(
            y_true_cls, y_pred_cls, weights="quadratic"
        ),
    }

# =====================================================================
# Embeddings
# =====================================================================
def infer_graph_emb_dim(encoder, quantizer, loader, metadata, embed_dim_per_type):
    encoder.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        batch = quantizer.transform_inplace(batch).to(device)
        z_dict, _, _ = encoder(batch)
        g = batched_graph_embeddings(
            z_dict, batch, metadata, embed_dim_per_type=embed_dim_per_type
        )
        return int(g.shape[-1])

# =====================================================================
# Config helpers
# =====================================================================
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
        if getattr(args, k) == getattr(defaults, k):
            setattr(args, k, v)

    return args


def episode_id_from_episode_path(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].split("_")[0]

class ProjectionHead(nn.Module):
    """
    Same-dim projection head: in_dim -> in_dim.
    Default is a small residual MLP so it's "near identity" but still flexible.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: str = "relu",
        residual: bool = True
    ):
        super().__init__()
        h = hidden_dim if hidden_dim is not None else dim
        act = nn.ReLU() if activation == "relu" else nn.GELU()
        self.residual = residual
        self.net = nn.Sequential(
            nn.Linear(dim, h),
            act,
            nn.Dropout(dropout),
            nn.Linear(h, dim),
        )

        # Bias towards identity at init if residual.
        if self.residual:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + y if self.residual else y


# ------------------------------- anchor utilities -------------------------------

def _episode_stem(pathlike: str) -> str:
    return os.path.splitext(os.path.basename(str(pathlike)))[0]


def _get_item_episode_stem(ds, idx: int) -> str:
    """
    Reads episode_path from a SINGLE dataset item and returns stem.
    Assumes each item has data["window_meta"].episode_path set.
    """
    data = ds[idx]
    if hasattr(data, "node_types") and ("window_meta" in data.node_types):
        wm = data["window_meta"]
        if hasattr(wm, "episode_path"):
            ep = wm.episode_path
            if isinstance(ep, (list, tuple)):
                ep = ep[0]
            return _episode_stem(str(ep))

    raise SystemExit(
        "Dataset item does not have window_meta.episode_path. "
        "Either your dataset is different than expected or you need to adjust the key extraction."
    )


class PairedAnchorDataset(Dataset):
    """
    Returns (clean_item, noisy_item) for the same episode stem.

    If an episode has multiple windows in a split, we keep them all.
    We pair windows deterministically by index up to min(count_clean, count_noisy).
    """
    def __init__(self, clean_subset, noisy_subset):
        self.clean_subset = clean_subset
        self.noisy_subset = noisy_subset

        c_map: Dict[str, List[int]] = {}
        for pos in range(len(clean_subset)):
            stem = _get_item_episode_stem(clean_subset, pos)
            c_map.setdefault(stem, []).append(pos)

        n_map: Dict[str, List[int]] = {}
        for pos in range(len(noisy_subset)):
            stem = _get_item_episode_stem(noisy_subset, pos)
            n_map.setdefault(stem, []).append(pos)

        self.stems = sorted(set(c_map.keys()) & set(n_map.keys()))
        if len(self.stems) == 0:
            raise SystemExit(
                "No anchor overlap found between the chosen clean/noisy SPLITS. "
                "This means your train split intersection is empty. "
                "Try changing seed/val_fraction OR check that episode_path stems actually match across datasets."
            )

        pairs: List[Tuple[int, int]] = []
        for s in self.stems:
            c_list = c_map[s]
            n_list = n_map[s]
            k = min(len(c_list), len(n_list))
            for j in range(k):
                pairs.append((c_list[j], n_list[j]))

        if len(pairs) == 0:
            raise SystemExit(
                "Anchor overlap stems exist, but no usable pairs were constructed. "
                "Please check dataset integrity."
            )

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i: int):
        ci, ni = self.pairs[i]
        return self.clean_subset[ci], self.noisy_subset[ni]


def _paired_alignment_loss(p_clean: torch.Tensor, p_noisy: torch.Tensor, loss_kind: str = "l2") -> torch.Tensor:
    if loss_kind == "cosine":
        return (1.0 - F.cosine_similarity(p_clean, p_noisy, dim=-1)).mean()
    elif loss_kind == "smoothl1":
        return F.smooth_l1_loss(p_clean, p_noisy, reduction="mean")
    else:
        return torch.mean((p_clean - p_noisy) ** 2)


def _paired_consistency_loss(
    logits_teacher: torch.Tensor,
    logits_student: torch.Tensor,
    mode: str,
    kind: str = "kl",   # "kl" or "mse"
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Per-pair consistency (NOT batch-mean matching).
    - Classification: teacher probs vs student probs (KL or MSE)
    - Regression: MSE on scalar predictions
    """
    if mode == "classification":
        p_t = torch.softmax(logits_teacher, dim=-1).clamp(eps, 1.0)          # [B,C]
        p_s = torch.softmax(logits_student, dim=-1).clamp(eps, 1.0)          # [B,C]
        if kind == "mse":
            return torch.mean((p_s - p_t) ** 2)
        # KL(teacher || student): sum p_t * (log p_t - log p_s)
        return torch.mean(torch.sum(p_t * (torch.log(p_t) - torch.log(p_s)), dim=-1))

    # regression
    lt = logits_teacher.view(-1)
    ls = logits_student.view(-1)
    return torch.mean((ls - lt) ** 2)


# ------------------------------- args / checkpoints -------------------------------


def _adopt_args_from_ckpt(args: argparse.Namespace, ckpt: Dict[str, Any], keys: List[str], label: str) -> None:
    if "args" not in ckpt or not isinstance(ckpt["args"], dict):
        print(f"[warn] {label} checkpoint has no 'args' dict; not adopting args.")
        return
    saved = ckpt["args"]
    for k in keys:
        if k in saved and hasattr(args, k):
            setattr(args, k, saved[k])


def _set_dataset_flags(ns: argparse.Namespace, dataset_name: str) -> None:
    for k in ["clean", "clean1", "clean2", "clean5", "noisy10", "noisy20", "noisy40", "noisy60"]:
        if hasattr(ns, k):
            setattr(ns, k, False)

    if dataset_name in ("clean1", "clean2", "clean5"):
        setattr(ns, dataset_name, True)
        ns.clean = True
    elif dataset_name in ("noisy_10", "noisy10"):
        ns.noisy10 = True
    elif dataset_name in ("noisy_20", "noisy20"):
        ns.noisy20 = True
    elif dataset_name in ("noisy_40", "noisy40"):
        ns.noisy40 = True
    elif dataset_name in ("noisy_60", "noisy60"):
        ns.noisy60 = True
    else:
        raise SystemExit(f"Unknown dataset name: {dataset_name}.")


def _canonical_dataset_name(name: str) -> str:
    x = name.strip().lower()
    if x in ("clean1", "clean2", "clean5"):
        return x
    if x in ("noisy10", "noisy_10"):
        return "noisy_10"
    if x in ("noisy20", "noisy_20"):
        return "noisy_20"
    if x in ("noisy40", "noisy_40"):
        return "noisy_40"
    if x in ("noisy60", "noisy_60"):
        return "noisy_60"
    raise SystemExit(f"Invalid dataset name: {name}. Allowed: clean1, clean2, clean5, noisy_10, noisy_20, noisy_40, noisy_60.")


def _set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag

def log_loss_from_hard_preds(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    if len(y_true) == 0:
        return 0.0
    p = np.full(len(y_true), eps, dtype=np.float64)
    p[y_pred == y_true] = 1.0 - eps
    return float(-np.mean(np.log(p)))

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

def log_annotations(file_path: str, script_name: str, anchor_pct: int, noise_pct: int, seed: int, metrics: Dict[str, Any], domain: Optional[str] = None):
    """
    Logs experiment metrics to a CSV file.
    Finds the correct row based on script, anchor_pct, noise_pct, and seed.
    """
    import pandas as pd

    _require_file(file_path, "Annotations CSV file")

    df = pd.read_csv(file_path)

    # Find the row to update
    mask = (
        (df["script"] == script_name) &
        (df["anchor_pct"] == anchor_pct) &
        (df["noise_pct"] == noise_pct) &
        (df["seed"] == seed)
    )
    
    if not mask.any():
        print(f"Error: No row found in {file_path} for script={script_name}, anchor_pct={anchor_pct}, noise_pct={noise_pct}, seed={seed}. Skipping logging for this run.")
        return

    # Update the metrics
    for metric_key, metric_value in metrics.items():
        col_name = None
        # Case 1: Metric from 5A with explicit domain (e.g., "eval/clean/accuracy")
        if 'eval/' in metric_key:
            parts = metric_key.split('/')
            if len(parts) == 3: # eval/domain/metric
                metric_domain, metric_suffix = parts[1], parts[2]
                col_name = f"{metric_domain}_{metric_suffix}"
        # Case 2: Metric from 3A or 4A where domain is passed explicitly
        elif domain:
            col_name = f"{domain}_{metric_key}"

        if col_name and col_name in df.columns:
            df.loc[mask, col_name] = metric_value
        elif col_name:
            pass

    df.loc[mask, "status"] = "completed"

    df.to_csv(file_path, index=False)
    print(f"Logged metrics for {script_name} (anchor_pct={anchor_pct}, noise_pct={noise_pct}, seed={seed}) to {file_path}")


