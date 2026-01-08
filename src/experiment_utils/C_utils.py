import argparse
import os
import sys
import time
import uuid
import random
from typing import Dict, Any, List, Set

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.graph_encoding.autoencoder import batched_graph_embeddings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Repro / IO helpers
# -------------------------
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


def _make_fallback_run_id() -> str:
    return f"local-{int(time.time())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _running_under_wandb_agent_or_sweep() -> bool:
    for k in ("WANDB_SWEEP_ID", "WANDB_AGENT_ID", "WANDB_RUN_ID"):
        if os.environ.get(k):
            return True
    return False


def _str2bool(v):
    if v is None:
        return True
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


# -------------------------
# Episode / tags helpers
# -------------------------
def episode_id_from_episode_path(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return base.split("_")[0]


def episode_ids_from_batch(data):
    paths = data["window_meta"].episode_path
    return [episode_id_from_episode_path(p) for p in paths]


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a.intersection(b))
    uni = len(a) + len(b) - inter
    return float(inter) / float(uni) if uni > 0 else 0.0


def semantic_sim_cross(tags_a: List[Set[str]], tags_b: List[Set[str]], device_: torch.device) -> torch.Tensor:
    """
    Build cross-domain semantic similarity matrix S of shape [len(tags_a), len(tags_b)]
    using Jaccard similarity on tag sets.
    """
    A, B = len(tags_a), len(tags_b)
    S = torch.zeros((A, B), dtype=torch.float32, device=device_)
    for i in range(A):
        ai = tags_a[i]
        for j in range(B):
            S[i, j] = _jaccard(ai, tags_b[j])
    return S


def cosine_sim_cross(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: [A, D], y: [B, D] => cosine similarity matrix [A, B]
    """
    x_n = F.normalize(x, p=2, dim=1)
    y_n = F.normalize(y, p=2, dim=1)
    return x_n @ y_n.t()


def tag_weighted_teacher_probs(
    teacher_probs_hq: torch.Tensor,  # [B_hq, C]
    tags_hq: List[Set[str]],
    tags_lq: List[Set[str]],
    tag_softmax_temp: float,
) -> torch.Tensor:
    """
    For each LQ sample i, compute weights over HQ samples j using softmax(Jaccard(i,j)/temp),
    then return mixture of HQ teacher probs: p_i = sum_j w_ij * teacher_probs_hq[j].
    """
    B_hq, C = teacher_probs_hq.shape
    B_lq = len(tags_lq)
    if B_hq == 0 or B_lq == 0:
        return torch.zeros((B_lq, C), device=teacher_probs_hq.device, dtype=teacher_probs_hq.dtype)

    W = torch.zeros((B_lq, B_hq), device=teacher_probs_hq.device, dtype=torch.float32)
    for i in range(B_lq):
        for j in range(B_hq):
            W[i, j] = _jaccard(tags_lq[i], tags_hq[j])

    temp = float(tag_softmax_temp)
    if temp <= 0:
        idx = torch.argmax(W, dim=1)
        return teacher_probs_hq[idx]
    W = F.softmax(W / temp, dim=1)  # [B_lq, B_hq]
    return W @ teacher_probs_hq  # [B_lq, C]


# -------------------------
# Risk label helpers
# -------------------------
def risk_to_class_safe(risk_scores):
    r = risk_scores.view(-1).float()
    y = torch.zeros_like(r, dtype=torch.long)
    y[r > 0.0043] = 1
    y[r > 0.1008] = 2
    y[r > 0.3442] = 3
    return y


# -------------------------
# Config / dataset paths
# -------------------------
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


# -------------------------
# Schedules / misc math
# -------------------------
def effective_weight(epoch: int, base: float, warmup: int, ramp: int) -> float:
    base = float(base)
    warmup = int(max(warmup, 0))
    ramp = int(max(ramp, 0))

    if base <= 0.0:
        return 0.0
    if warmup == 0 and ramp == 0:
        return base
    if epoch <= warmup:
        return 0.0
    if ramp == 0:
        return base
    t = (epoch - warmup) / float(ramp)
    t = max(0.0, min(1.0, t))
    return base * t


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


def _masked_alignment_mse(E: torch.Tensor, S: torch.Tensor, tags_hq: List[Set[str]], tags_lq: List[Set[str]]) -> torch.Tensor:
    if E.numel() == 0:
        return torch.tensor(0.0, device=E.device)
    hq_nonempty = torch.tensor([len(t) > 0 for t in tags_hq], device=E.device, dtype=torch.bool)  # [B_hq]
    lq_nonempty = torch.tensor([len(t) > 0 for t in tags_lq], device=E.device, dtype=torch.bool)  # [B_lq]
    mask = hq_nonempty[:, None] & lq_nonempty[None, :]  # [B_hq, B_lq]
    if not bool(mask.any().item()):
        return torch.tensor(0.0, device=E.device)
    diff2 = (E - S) ** 2
    return diff2[mask].mean()


# -------------------------
# Checkpoint compatibility helpers
# -------------------------
def _torch_load_compat(path: str, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_single_encoder_ckpt_flexible(encoder, ckpt_path: str) -> None:
    _require_file(ckpt_path, "encoder checkpoint")
    ck = _torch_load_compat(ckpt_path, map_location=device)

    state_dict = None
    if isinstance(ck, dict):
        if "state_dict" in ck and isinstance(ck["state_dict"], dict):
            state_dict = ck["state_dict"]
        elif "model_state_dict" in ck and isinstance(ck["model_state_dict"], dict):
            state_dict = ck["model_state_dict"]
        elif "encoder_state_dict" in ck and isinstance(ck["encoder_state_dict"], dict):
            state_dict = ck["encoder_state_dict"]
        elif "encoder" in ck and isinstance(ck["encoder"], dict):
            state_dict = ck["encoder"]
        elif all(isinstance(v, torch.Tensor) for v in ck.values()):
            state_dict = ck

    if state_dict is None:
        raise RuntimeError(
            f"Could not find a usable state_dict in {ckpt_path}. "
            f"Keys found: {list(ck.keys())[:50] if isinstance(ck, dict) else type(ck)}"
        )

    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[warn] Loaded encoder ckpt with key mismatch from {ckpt_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})."
        )


def _override_model_args_from_encoder_ckpt(args, enc_ckpt_or_args: Dict[str, Any]) -> None:
    if not isinstance(enc_ckpt_or_args, dict):
        return

    ck_args = enc_ckpt_or_args.get("args", None)
    src = ck_args if isinstance(ck_args, dict) else enc_ckpt_or_args

    must_match = (
        "hidden_dim",
        "embed_dim",
        "num_encoder_layers",
        "num_decoder_layers",
        "activation",
        "dropout_rate",
        "proj_dim",
        "quant_bins",
    )
    for k in must_match:
        if k in src and hasattr(args, k):
            setattr(args, k, src[k])


def _preload_eval_encoder_args(args) -> None:
    need = bool(args.evaluate_risk) or bool(args.train_proj_risk_jointly)
    if not need:
        return
    if not args.l2d_4b_ckpt_path or not args.nup_4b_ckpt_path:
        return

    if os.path.exists(args.l2d_4b_ckpt_path):
        ck = _torch_load_compat(args.l2d_4b_ckpt_path, map_location="cpu")
        if isinstance(ck, dict) and isinstance(ck.get("args", None), dict):
            _override_model_args_from_encoder_ckpt(args, ck)

    if os.path.exists(args.nup_4b_ckpt_path):
        ck = _torch_load_compat(args.nup_4b_ckpt_path, map_location="cpu")
        if isinstance(ck, dict) and isinstance(ck.get("args", None), dict):
            _override_model_args_from_encoder_ckpt(args, ck)


def _format_confusion_matrix(cm):
    n = cm.shape[0]
    header = "true\\pred | " + " ".join([f"{i:>6d}" for i in range(n)])
    sep = "-" * len(header)
    lines = [header, sep]
    for i in range(n):
        row = " ".join([f"{cm[i, j]:>6d}" for j in range(n)])
        lines.append(f"{i:>9d} | {row}")
    return "\n".join(lines)
