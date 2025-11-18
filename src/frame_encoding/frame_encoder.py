import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple
import numpy as np
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm

class MultiTaskClipModel(nn.Module):
    def __init__(
        self,
        num_classes_per_task: List[int],
        z_dim: int = 256,
        encoder_name: str = "resnet18",
        pretrained: bool = True,
    ):
        """
        Args:
            num_classes_per_task: list of length K, C_k classes per task.
            z_dim: dimension of clip embedding z.
            encoder_name: which torchvision resnet to use.
            pretrained: use ImageNet weights if True.
        """
        super().__init__()

        self.num_classes_per_task = num_classes_per_task
        self.z_dim = z_dim

        # ----- Encoder backbone -----
        if encoder_name == "resnet18":
            if pretrained:
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                backbone = models.resnet18(weights=None)
            enc_out_dim = backbone.fc.in_features  # 512
        elif encoder_name == "resnet50":
            if pretrained:
                backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                backbone = models.resnet50(weights=None)
            enc_out_dim = backbone.fc.in_features  # 2048
        else:
            raise ValueError(f"Unsupported encoder_name: {encoder_name}")

        # Drop the classifier; keep everything up to global average pooling
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # (N, D, 1, 1)
        self.enc_out_dim = enc_out_dim

        # ----- Frame embedding head (enc_out_dim -> z_dim) -----
        self.frame_head = nn.Sequential(
            nn.Flatten(),                      # (N, D, 1, 1) -> (N, D)
            nn.Linear(self.enc_out_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.z_dim),
        )

        # ----- Clip-level heads: one Linear per task -----
        self.task_heads = nn.ModuleList()
        for C_k in num_classes_per_task:
            self.task_heads.append(nn.Linear(self.z_dim, C_k))

    def forward(self, frames_list: List[torch.Tensor]):
        """
        Args:
            frames_list: list of length B
                each element: tensor of shape (T_i, C, H, W)

        Returns:
            logits_per_task: list of length K,
                each tensor shape (B, C_k)
            clip_embs: tensor of shape (B, z_dim)
        """
        device = next(self.parameters()).device

        # 1) Concatenate all frames across batch
        lengths = [f.shape[0] for f in frames_list]          # T_i per episode
        all_frames = torch.cat(frames_list, dim=0).to(device)  # (sum_T, C, H, W)

        # 2) Encode all frames in one shot
        feats = self.encoder(all_frames)                     # (sum_T, D, 1, 1)
        z_frames = self.frame_head(feats)                    # (sum_T, z_dim)

        # 3) Split back per episode and mean-pool over time
        z_split = torch.split(z_frames, lengths, dim=0)      # list of (T_i, z_dim)
        clip_embs = []
        for z_clip in z_split:
            clip_embs.append(z_clip.mean(dim=0))             # (z_dim,)

        clip_embs = torch.stack(clip_embs, dim=0)            # (B, z_dim)

        # 4) Per-task logits
        logits_per_task = []
        for head in self.task_heads:
            logits_per_task.append(head(clip_embs))          # (B, C_k)

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
        feats = model.encoder(all_frames)                   # (sum_T, D, 1, 1)
        z_frames = model.frame_head(feats)                  # (sum_T, z_dim)

    # Split back per episode
    z_split = torch.split(z_frames, lengths, dim=0)         # list of (T_i, z_dim)

    # Clip-level mean pooling
    clip_embs = torch.stack(
        [z.mean(dim=0) for z in z_split],
        dim=0,
    )                                                       # (B, z_dim)

    return z_split, clip_embs