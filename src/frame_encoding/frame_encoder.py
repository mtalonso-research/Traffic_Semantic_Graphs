import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Optional
import numpy as np
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
from torchvision import models


# -------------------------------------------------------------------------
# Helper: build an MLP
# -------------------------------------------------------------------------
def build_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
):
    layers = []
    dim = input_dim

    act = {
        "relu": nn.ReLU(inplace=True),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }.get(activation, nn.ReLU(inplace=True))

    for h in hidden_dims:
        layers.append(nn.Linear(dim, h))
        layers.append(act)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        dim = h

    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


# -------------------------------------------------------------------------
# Pooling modules (unchanged)
# -------------------------------------------------------------------------

class MeanPool(nn.Module):
    def forward(self, x):
        return x.mean(dim=0)


class MaxPool(nn.Module):
    def forward(self, x):
        return x.max(dim=0).values


class MeanMaxPool(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.proj = nn.Linear(2 * z_dim, z_dim)

    def forward(self, x):
        m = x.mean(dim=0)
        M = x.max(dim=0).values
        return self.proj(torch.cat([m, M], dim=0))


class LSEPool(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.temp = nn.Parameter(torch.tensor(init_temp))

    def forward(self, x):
        scale = F.softplus(self.temp) + 1e-6
        return (x * scale).logsumexp(dim=0) / scale


class AttnPool(nn.Module):
    def __init__(self, z_dim, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(z_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        w = (x @ self.q) / (x.shape[-1] ** 0.5)
        w = F.softmax(w, dim=0)
        w = self.dropout(w)
        return (w.unsqueeze(1) * x).sum(dim=0)


class MultiHeadAttnPool(nn.Module):
    def __init__(self, z_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.q = nn.Parameter(torch.randn(num_heads, z_dim))
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(num_heads * z_dim, z_dim)

    def forward(self, x):
        w = (self.q @ x.T) / (x.shape[-1] ** 0.5)
        w = F.softmax(w, dim=1)
        w = self.dropout(w)
        pooled = torch.einsum("ht,td->hd", w, x)
        return self.proj(pooled.reshape(-1))


class GatedPool(nn.Module):
    def __init__(self, z_dim, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(2 * z_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        gm = x.mean(dim=0, keepdim=True).expand_as(x)
        g = torch.cat([x, gm], dim=-1)
        w = self.gate(g).squeeze(-1)
        w = F.softmax(w, dim=0)
        return (w.unsqueeze(1) * x).sum(dim=0)


# -------------------------------------------------------------------------
# Main Model
# -------------------------------------------------------------------------
class MultiTaskClipModel(nn.Module):
    def __init__(
        self,
        num_classes_per_task: List[int],
        z_dim: int = 256,
        encoder_name: str = "resnet18",
        pretrained: bool = True,

        # Frame head config
        frame_head_hidden_dims: List[int] = [512],
        frame_head_activation: str = "relu",
        frame_head_dropout: float = 0.0,

        # Pooling config
        pooling: str = "mean",
        num_pool_heads: int = 4,
        pool_hidden_dim: int = 128,
        pool_dropout: float = 0.0,
        pool_temperature_init: float = 1.0,

        # Task head config
        task_head_hidden_dims: List[int] = [],
        task_head_activation: str = "relu",
        task_head_dropout: float = 0.0,

        # NEW: Encoder training control
        encoder_trainable: str = "all",  # "all", "none", "partial:N"
    ):
        super().__init__()

        self.num_classes_per_task = num_classes_per_task
        self.z_dim = z_dim
        self.encoder_trainable = encoder_trainable

        # -------------------------------------------------------------
        # Backbone
        # -------------------------------------------------------------
        if encoder_name == "resnet18":
            backbone = (
                models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                if pretrained else models.resnet18(weights=None)
            )
            enc_out_dim = backbone.fc.in_features

            # Modules in correct freeze order
            self.backbone_stages = [
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ]

        elif encoder_name == "resnet50":
            backbone = (
                models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                if pretrained else models.resnet50(weights=None)
            )
            enc_out_dim = backbone.fc.in_features

            self.backbone_stages = [
                backbone.conv1,
                backbone.bn1,
                backbone.relu,
                backbone.maxpool,
                backbone.layer1,
                backbone.layer2,
                backbone.layer3,
                backbone.layer4,
            ]

        else:
            raise ValueError(f"Unsupported encoder_name: {encoder_name}")

        # Remove fc
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        self.enc_out_dim = enc_out_dim

        # -------------------------------------------------------------
        # Apply freezing rules
        # -------------------------------------------------------------
        self._configure_encoder_trainability()

        # -------------------------------------------------------------
        # Frame embedding head
        # -------------------------------------------------------------
        self.frame_head = build_mlp(
            input_dim=enc_out_dim,
            hidden_dims=frame_head_hidden_dims,
            output_dim=z_dim,
            activation=frame_head_activation,
            dropout=frame_head_dropout,
        )

        # -------------------------------------------------------------
        # Pooling
        # -------------------------------------------------------------
        if pooling == "mean":
            self.pool = MeanPool()
        elif pooling == "max":
            self.pool = MaxPool()
        elif pooling == "meanmax":
            self.pool = MeanMaxPool(z_dim)
        elif pooling == "attn":
            self.pool = AttnPool(z_dim, dropout=pool_dropout)
        elif pooling == "multihead_attn":
            self.pool = MultiHeadAttnPool(z_dim, num_heads=num_pool_heads, dropout=pool_dropout)
        elif pooling == "lse":
            self.pool = LSEPool(init_temp=pool_temperature_init)
        elif pooling == "gated":
            self.pool = GatedPool(z_dim, hidden_dim=pool_hidden_dim, dropout=pool_dropout)
        else:
            raise ValueError(f"Unknown pooling: {pooling}")

        # -------------------------------------------------------------
        # Task heads
        # -------------------------------------------------------------
        self.task_heads = nn.ModuleList([
            build_mlp(
                input_dim=z_dim,
                hidden_dims=task_head_hidden_dims,
                output_dim=C_k,
                activation=task_head_activation,
                dropout=task_head_dropout,
            )
            for C_k in num_classes_per_task
        ])


    # -----------------------------------------------------------------
    # Freezing logic
    # -----------------------------------------------------------------
    def _configure_encoder_trainability(self):
        mode = self.encoder_trainable

        if mode == "all":
            return  # nothing to freeze

        elif mode == "none":
            # Freeze everything
            for p in self.encoder.parameters():
                p.requires_grad = False
            # BN layers in eval mode
            self.encoder.apply(self._freeze_bn)
            return

        elif mode.startswith("partial:"):
            try:
                N = int(mode.split(":")[1])
            except:
                raise ValueError(f"Invalid partial mode: {mode}")

            # Clamp N to available stages
            N = min(N, len(self.backbone_stages))

            # Freeze first N stages
            for stage in self.backbone_stages[:N]:
                for p in stage.parameters():
                    p.requires_grad = False
                stage.apply(self._freeze_bn)

            return

        else:
            raise ValueError(f"Unknown encoder_trainable mode: {mode}")

    @staticmethod
    def _freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------
    def forward(self, frames_list: List[torch.Tensor]):
        device = next(self.parameters()).device

        lengths = [f.shape[0] for f in frames_list]
        all_frames = torch.cat(frames_list, dim=0).to(device)

        feats = self.encoder(all_frames).flatten(start_dim=1)

        z_frames = self.frame_head(feats)
        z_split = torch.split(z_frames, lengths, dim=0)

        clip_embs = torch.stack([self.pool(z) for z in z_split], dim=0)
        logits_per_task = [head(clip_embs) for head in self.task_heads]

        return logits_per_task, clip_embs

def multitask_categorical_loss(
    logits_per_task,
    targets,
    label_smoothing = 0.0,
    task_weights = None,
    sample_weights = None,
):
    """
    Args:
        logits_per_task: list of length K, each (B, C_k)
        targets: (B, K) long tensor of class indices
        label_smoothing: epsilon in [0, 1); 0 = no smoothing
        task_weights: optional list of length K, weight per task (defaults to 1.0)
        sample_weights: optional (B,) tensor of per-sample weights

    Returns:
        scalar loss
    """
    K = len(logits_per_task)
    B = targets.shape[0]

    if task_weights is None:
        task_weights = [1.0] * K

    device = logits_per_task[0].device
    total_loss = torch.zeros((), device=device)

    for k in range(K):
        logits_k = logits_per_task[k]          # (B, C_k)
        target_k = targets[:, k]               # (B,)
        C_k = logits_k.size(-1)

        if C_k == 1:  # Binary classification
            loss_k = F.binary_cross_entropy_with_logits(
                logits_k.squeeze(1),
                target_k.float(),
                reduction="none",
            )
        else:  # Multi-class classification
            if label_smoothing > 0.0:
                eps = label_smoothing
                # log probabilities
                log_probs = F.log_softmax(logits_k, dim=-1)      # (B, C_k)
                # one-hot with smoothing
                with torch.no_grad():
                    true_dist = torch.zeros_like(log_probs)
                    true_dist.fill_(eps / (C_k - 1))
                    true_dist.scatter_(1, target_k.unsqueeze(1), 1.0 - eps)
                loss_k = -(true_dist * log_probs).sum(dim=-1)    # (B,)
            else:
                loss_k = F.cross_entropy(
                    logits_k,
                    target_k,
                    reduction="none",
                )                                                # (B,)

        if sample_weights is not None:
            loss_k = loss_k * sample_weights.to(device)      # (B,)

        # mean over batch
        loss_k = loss_k.mean()
        total_loss = total_loss + task_weights[k] * loss_k

    return total_loss

def collate_episodes(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]):
    """
    batch: list of (frames, target, episode_id)
        frames: (T_i, C, H, W)
        target: (K,)
        episode_id: str
    """
    frames_list, targets_list, ids_list = zip(*batch)
    frames_list = list(frames_list)
    targets = torch.stack(targets_list, dim=0)
    episode_ids = list(ids_list)
    return frames_list, targets, episode_ids

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    label_smoothing: float = 0.0,
    task_weights=None,
):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for frames_list, targets_batch, _episode_ids in tqdm(loader,desc='Training'):
        targets_batch = targets_batch.to(device)

        logits_per_task, clip_embs = model(frames_list)

        loss = multitask_categorical_loss(
            logits_per_task,
            targets_batch,
            label_smoothing=label_smoothing,
            task_weights=task_weights,
            sample_weights=None,  # all LQ for now; we can add this later
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(1, num_batches)

def evaluate(
    model,
    loader,
    device,
    label_smoothing: float = 0.0,
    task_weights=None,
):
    model.eval()
    running_loss = 0.0
    num_batches = 0

    K = len(model.num_classes_per_task)
    correct_per_task = [0 for _ in range(K)]
    total_samples = 0

    with torch.no_grad():
        for frames_list, targets_batch, _episode_ids in tqdm(loader,desc='Evaluating'):
            targets_batch = targets_batch.to(device)
            B = targets_batch.size(0)

            logits_per_task, clip_embs = model(frames_list)

            # loss
            loss = multitask_categorical_loss(
                logits_per_task,
                targets_batch,
                label_smoothing=label_smoothing,
                task_weights=task_weights,
                sample_weights=None,
            )
            running_loss += loss.item()
            num_batches += 1

            # per-task accuracy
            for k in range(K):
                logits_k = logits_per_task[k]       # (B, C_k)
                if logits_k.size(-1) == 1:
                    preds_k = (logits_k > 0).float().squeeze(1)
                else:
                    preds_k = logits_k.argmax(dim=1)    # (B,)
                correct_per_task[k] += (preds_k == targets_batch[:, k]).sum().item()

            total_samples += B

    avg_loss = running_loss / max(1, num_batches)
    acc_per_task = [c / max(1, total_samples) for c in correct_per_task]
    mean_acc = float(np.mean(acc_per_task))

    return avg_loss, acc_per_task, mean_acc

def encode_batch_episodes(model, frames_list):
    """
    Args:
        model: trained MultiTaskClipModel
        frames_list: list of length B, each (T_i, C, H, W)

    Returns:
        frame_embs_per_episode: list length B, each (T_i, z_dim)
        clip_embs: (B, z_dim)
    """
    device = next(model.parameters()).device
    model.eval()

    lengths = [f.shape[0] for f in frames_list]
    all_frames = torch.cat(frames_list, dim=0).to(device)   # (sum_T, C, H, W)

    with torch.no_grad():
        feats = model.encoder(all_frames).flatten(start_dim=1)  # (sum_T, D)
        z_frames = model.frame_head(feats)                      # (sum_T, z_dim)

    # Split back per episode
    z_split = torch.split(z_frames, lengths, dim=0)         # list of (T_i, z_dim)

    # Clip-level mean pooling
    clip_embs = torch.stack(
        [model.pool(z) for z in z_split],
        dim=0,
    )                                                     # (B, z_dim)

    return z_split, clip_embs

import os
import json
import re
from pathlib import Path

import torch


def augment_graphs_with_frame_embeddings(
    graph_dir,
    frame_embeddings_path,
    output_dir=None,
    feature_name="frame_embedding",
):
    """
    Add per-frame embeddings as a feature on ego nodes in graph JSONs.

    Args:
        graph_dir: directory containing graph JSON files.
        frame_embeddings_path: path to frame_embeddings.pt
            (dict: episode_id -> (T_i, z_dim) tensor).
        output_dir: where to write updated JSONs.
            - None (default): overwrite in place in graph_dir.
            - otherwise: write to that directory (created if needed).
        feature_name: key name under node['features'] to store the embedding.
    """
    graph_dir = Path(graph_dir)
    if output_dir is None:
        output_dir = graph_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading frame embeddings from {frame_embeddings_path}...")
    frame_embeddings = torch.load(frame_embeddings_path, map_location="cpu")

    # For convenience, allow keys like "0", "L2D:0", "NuPlan:0"
    def get_embedding_for_episode(ep_idx: int):
        candidates = [
            str(ep_idx),
            f"L2D:{ep_idx}",
            f"NuPlan:{ep_idx}",
            f"episode_{ep_idx:06d}",  # just in case you used that naming
        ]
        for key in candidates:
            if key in frame_embeddings:
                return frame_embeddings[key]
        return None

    json_files = sorted(graph_dir.glob("*.json"))
    print(f"Found {len(json_files)} graph files in {graph_dir}")

    for json_path in json_files:
        with json_path.open("r") as f:
            g = json.load(f)

        # ---- Infer episode index from metadata["source_files"] ----
        src_files = g.get("metadata", {}).get("source_files", {})

        episode_indices = []
        for v in src_files.values():
            if not isinstance(v, str):
                continue
            # Look for "...Episode000000/..." pattern
            m = re.search(r"Episode(\d+)", v)
            if m:
                episode_indices.append(int(m.group(1)))

        if not episode_indices:
            print(f"[WARN] Could not infer episode index from {json_path}, skipping.")
            continue

        # Assume all entries are the same episode; pick the first
        ep_idx = episode_indices[0]

        emb = get_embedding_for_episode(ep_idx)
        if emb is None:
            print(
                f"[WARN] No embedding found for episode index {ep_idx} "
                f"(json: {json_path.name}), skipping."
            )
            continue

        # Convert to numpy for easier indexing, then to Python lists later
        if isinstance(emb, torch.Tensor):
            emb_np = emb.cpu().numpy()
        else:
            emb_np = emb  # assume already array-like

        T, z_dim = emb_np.shape

        ego_nodes = g.get("nodes", {}).get("ego", [])
        if not ego_nodes:
            print(f"[WARN] No ego nodes in {json_path}, skipping.")
            continue

        # Optional sanity check: number of ego nodes vs frames
        if len(ego_nodes) != T:
            print(
                f"[WARN] Mismatch in {json_path.name}: "
                f"{len(ego_nodes)} ego nodes vs {T} frame embeddings. "
                f"Proceeding by matching ego_k with embedding[k]."
            )

        # ---- Attach embeddings to ego nodes ----
        for node in ego_nodes:
            node_id = node.get("id", "")
            m = re.match(r"ego_(\d+)", node_id)
            if not m:
                # If the naming convention ever changes, you can handle it here
                print(f"[WARN] Unexpected ego id '{node_id}' in {json_path.name}, skipping this node.")
                continue

            k = int(m.group(1))
            if k >= T:
                print(
                    f"[WARN] ego index {k} out of range for episode {ep_idx} "
                    f"(T={T}) in {json_path.name}, skipping this node."
                )
                continue

            vec = emb_np[k].tolist()
            node.setdefault("features", {})[feature_name] = vec

        # ---- Write updated JSON ----
        out_path = output_dir / json_path.name
        with out_path.open("w") as f:
            json.dump(g, f)

        print(f"Updated {out_path} (episode {ep_idx}, z_dim={z_dim})")
