# student_core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from torch_geometric.nn import HeteroConv, SAGEConv, global_mean_pool

# =======================
# Normalization utilities
# =======================

@dataclass
class Normalizer:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    def to(self, device: torch.device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std.clamp_min(self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std.clamp_min(self.eps) + self.mean


@torch.no_grad()
def compute_feature_stats(
    loader,
    node_types: List[str],
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Normalizer]:
    """
    Compute mean/std per node type over x features using sums/sumsq.
    Returns dict: ntype -> Normalizer(mean[D], std[D]).
    """
    sums: Dict[str, torch.Tensor] = {}
    sumsq: Dict[str, torch.Tensor] = {}
    counts: Dict[str, float] = {}
    seen_dims: Dict[str, int] = {}

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        for ntype in node_types:
            if hasattr(batch, ntype) and hasattr(batch[ntype], "x") and batch[ntype].x is not None:
                x = batch[ntype].x
                if device is not None:
                    x = x.to("cpu")
                if ntype not in sums:
                    D = x.size(-1)
                    sums[ntype] = torch.zeros(D)
                    sumsq[ntype] = torch.zeros(D)
                    counts[ntype] = 0.0
                    seen_dims[ntype] = D
                if x.size(-1) != seen_dims[ntype]:
                    continue
                sums[ntype] += x.sum(dim=0)
                sumsq[ntype] += (x * x).sum(dim=0)
                counts[ntype] += x.size(0)

    norms: Dict[str, Normalizer] = {}
    for ntype, s in sums.items():
        n = max(counts[ntype], 1.0)
        mean = s / n
        var = sumsq[ntype] / n - mean * mean
        var.clamp_(min=0.0)
        std = var.sqrt()
        norms[ntype] = Normalizer(mean=mean, std=std)
    return norms


@torch.no_grad()
def compute_target_stats(
    loader,
    device: Optional[torch.device] = None,
    max_batches: Optional[int] = None,
) -> Normalizer:
    """
    Compute mean/std for targets batch.y over the dataset.
    Returns a Normalizer(mean[Dy], std[Dy]).
    """
    sum_y = None
    sumsq_y = None
    count_y = 0

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        y = batch.y
        if device is not None:
            y = y.to("cpu")
        if sum_y is None:
            Dy = y.size(-1)
            sum_y = torch.zeros(Dy)
            sumsq_y = torch.zeros(Dy)
        sum_y += y.sum(dim=0)
        sumsq_y += (y * y).sum(dim=0)
        count_y += y.size(0)

    n = max(count_y, 1)
    mean = sum_y / n
    var = sumsq_y / n - mean * mean
    var.clamp_(min=0.0)
    std = var.sqrt()
    return Normalizer(mean=mean, std=std)

# ---------- tiny utils (no torch_scatter required) ----------

def mlp(in_dim: int, out_dim: int, hidden_mult: int = 2, p: float = 0.1) -> nn.Sequential:
    h = max(out_dim * hidden_mult, out_dim)
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, h),
        nn.GELU(),
        # nn.Dropout(p),  # (disabled per your code)
        nn.Linear(h, out_dim),
    )

def _segment_max_1d(values: torch.Tensor, index: torch.Tensor, K: int) -> torch.Tensor:
    """
    values: [N]  (int/long or float)
    index:  [N]  group ids in [0, K-1]
    returns: [K] max per group
    """
    if values.dtype.is_floating_point:
        out = torch.full((K,), float("-inf"), dtype=values.dtype, device=values.device)
    else:
        out = torch.full((K,), torch.iinfo(values.dtype).min, dtype=values.dtype, device=values.device)
    out.index_reduce_(0, index, values, reduce="amax", include_self=True)
    return out

def _segment_mean(values: torch.Tensor, index: torch.Tensor, K: int) -> torch.Tensor:
    """
    values: [N, D]
    index:  [N]
    returns: [K, D] mean per group (safe for empty groups)
    """
    D = values.size(-1)
    out = torch.zeros((K, D), dtype=values.dtype, device=values.device)
    out.index_add_(0, index, values)  # sum per group
    counts = torch.bincount(index, minlength=K).clamp_min(1).to(values.dtype).unsqueeze(1)
    return out / counts

@torch.no_grad()
def select_last_ego_mask(batch) -> torch.Tensor:
    """Boolean mask selecting the last-frame ego node(s) per graph/window."""
    frame = batch["ego"].frame_id.long()   # [N_ego]
    bvec  = batch["ego"].batch.long()      # [N_ego]
    B = int(batch["ego"].ptr.shape[0] - 1)
    max_frame = _segment_max_1d(frame, bvec, K=B)  # [B]
    return frame.eq(max_frame[bvec])               # [N_ego]

@torch.no_grad()
def aggregate_last_ego(h_ego_all: torch.Tensor, batch) -> torch.Tensor:
    """Average last-frame ego embeddings per graph. Returns [B, H]."""
    mask   = select_last_ego_mask(batch)             # [N_ego]
    h_last = h_ego_all[mask]                         # [N_last, H]
    b_last = batch["ego"].batch[mask].long()         # [N_last]
    B = int(batch["ego"].ptr.shape[0] - 1)
    return _segment_mean(h_last, b_last, K=B)        # [B, H]

# ---------- targets & model ----------

TARGET_OUT_DIMS = {
    "ego_next_pos": 2,
    "ego_next_heading": 1,
    "ego_next_speed": 1,
    "ego_next_controls": 5,
    "min_dist_vehicle": 1,
    "min_dist_pedestrian": 1,
    "closest_vehicle_relvel_y": 1,
    "ego_traj_5": 10,
}

class StudentModel(nn.Module):
    """
    Per-type MLP encoders -> HeteroConv (GraphSAGE) -> ego selector -> task head.
    Also computes a graph embedding (z_graph) ready for future distillation.
    """
    def __init__(
        self,
        feature_dims: Dict[str, int],           # {'ego': d_e, 'vehicle': d_v, ...}
        metadata: Tuple[List[str], List[Tuple[str, str, str]]],  # (node_types, edge_types)
        target_spec: str,                        # key in TARGET_OUT_DIMS
        hidden_dim: int = 128,
        gnn_layers: int = 2,
        dropout: float = 0.0,
        projector_dim: int = 128,               # z_graph dim for future distillation
        use_graph_context: bool = False,        # if True, concat z_graph into ego head
    ):
        super().__init__()
        assert target_spec in TARGET_OUT_DIMS, f"Unknown target_spec: {target_spec}"
        self.target_spec = target_spec
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.hidden_dim = hidden_dim
        self.use_graph_context = use_graph_context

        # per-type encoders -> hidden_dim
        self.encoders = nn.ModuleDict()
        for ntype in self.node_types:
            in_dim = feature_dims.get(ntype, 0)
            if in_dim > 0:
                self.encoders[ntype] = mlp(in_dim, hidden_dim, hidden_mult=2, p=dropout)

        # hetero GNN stack (GraphSAGE)
        convs = []
        for _ in range(gnn_layers):
            rel_layers = {}
            for (src, rel, dst) in self.edge_types:
                rel_layers[(src, rel, dst)] = SAGEConv((-1, -1), hidden_dim)  # infer dims
            convs.append(HeteroConv(rel_layers, aggr="mean"))
        self.convs = nn.ModuleList(convs)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # graph readout + projector (for future distillation)
        self.readout_proj = mlp(hidden_dim, projector_dim, hidden_mult=2, p=dropout)
        self.projector_dim = projector_dim

        # task head
        head_in = hidden_dim + (projector_dim if use_graph_context else 0)
        out_dim = TARGET_OUT_DIMS[target_spec]
        self.head = mlp(head_in, out_dim, hidden_mult=2, p=dropout)

    def forward(self, batch):
        """
        Returns dict:
          - y_pred:  [B, target_dim]
          - h_ego:   [B, hidden_dim]
          - z_graph: [B, projector_dim]  (L2-normalized)
        """
        # 1) encode per node type
        x_dict = {}
        for ntype in batch.node_types:
            enc = self.encoders[ntype] if ntype in self.encoders else None
            if enc is not None and hasattr(batch[ntype], "x"):
                x_dict[ntype] = enc(batch[ntype].x)

        # 2) hetero message passing
        h = x_dict
        for conv in self.convs:
            h = conv(h, batch.edge_index_dict)
            for k in h:
                h[k] = self.act(h[k])
                h[k] = self.drop(h[k])

        # 3) ego selection (last frame per graph)
        h_ego_all = h["ego"]                             # [N_ego, H]
        h_ego     = aggregate_last_ego(h_ego_all, batch) # [B, H]
        B = int(batch["ego"].ptr.shape[0] - 1)           # number of graphs/windows in the batch

        # 4) graph readout (per-type mean over graphs, robust to missing types per graph)
        z_sum = None
        z_w   = None
        for ntype, hx in h.items():
            bvec = batch[ntype].batch                     # [N_type]
            # Sum features per graph
            pooled_sum = torch.zeros((B, hx.size(-1)), device=hx.device, dtype=hx.dtype)
            pooled_sum.index_add_(0, bvec, hx)
            # Count nodes per graph for this type
            counts = torch.bincount(bvec, minlength=B).to(hx.device).unsqueeze(1).to(hx.dtype)  # [B,1]
            # Mean per graph; avoid div by zero
            pooled_mean = pooled_sum / counts.clamp_min(1.0)
            # Type-present weights (1 if graph has >=1 node of this type, else 0)
            w = (counts > 0).to(hx.dtype)                 # [B,1]
            if z_sum is None:
                z_sum = pooled_mean * w
                z_w   = w
            else:
                z_sum = z_sum + pooled_mean * w
                z_w   = z_w + w

        z_graph = z_sum / z_w.clamp_min(1.0)              # [B, H]
        z_graph = self.readout_proj(z_graph)              # [B, P]
        z_graph = F.normalize(z_graph, p=2, dim=1)

        # 5) task head
        head_in = h_ego if not self.use_graph_context else torch.cat([h_ego, z_graph], dim=1)
        y_pred  = self.head(head_in)                       # [B, out_dim]

        return {"y_pred": y_pred, "h_ego": h_ego, "z_graph": z_graph}

# ----------      super-light epoch helpers       ----------

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: Optional[str] = None,
    feature_norms: Optional[Dict[str, Normalizer]] = None,
    target_norm: Optional[Normalizer] = None,
) -> float:
    """Returns average loss over the loader."""
    if device is None:
        device = next(model.parameters()).device
    # <-- ensure the normalizer lives on the same device as the tensors we compare
    if target_norm is not None:
        target_norm = target_norm.to(device)

    model.train(True)
    total, count = 0.0, 0

    for batch in loader:
        # normalize features per node type on CPU (before .to(device))
        if feature_norms:
            for ntype, norm in feature_norms.items():
                if hasattr(batch, ntype) and hasattr(batch[ntype], "x") and batch[ntype].x is not None:
                    batch[ntype].x = norm.normalize(batch[ntype].x)

        batch = batch.to(device)
        out = model(batch)

        # targets
        y = batch.y.to(device)
        if target_norm is not None:
            y_norm = target_norm.normalize(y)
            y_pred_norm = target_norm.normalize(out["y_pred"])
            loss = criterion(y_pred_norm, y_norm)
        else:
            loss = criterion(out["y_pred"], y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = out["y_pred"].size(0)
        total += loss.item() * bs
        count += bs

    return total / max(count, 1)

@torch.no_grad()
def eval_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: Optional[str] = None,
    feature_norms: Optional[Dict[str, Normalizer]] = None,
    target_norm: Optional[Normalizer] = None,
) -> float:
    """Returns average loss over the loader."""
    if device is None:
        device = next(model.parameters()).device
    # <-- ensure the normalizer lives on the same device as the tensors we compare
    if target_norm is not None:
        target_norm = target_norm.to(device)

    model.train(False)
    total, count = 0.0, 0

    for batch in loader:
        # normalize features per node type on CPU (before .to(device))
        if feature_norms:
            for ntype, norm in feature_norms.items():
                if hasattr(batch, ntype) and hasattr(batch[ntype], "x") and batch[ntype].x is not None:
                    batch[ntype].x = norm.normalize(batch[ntype].x)

        batch = batch.to(device)
        out = model(batch)

        y = batch.y.to(device)
        if target_norm is not None:
            y_norm = target_norm.normalize(y)
            y_pred_norm = target_norm.normalize(out["y_pred"])
            loss = criterion(y_pred_norm, y_norm)
        else:
            loss = criterion(out["y_pred"], y)

        bs = out["y_pred"].size(0)
        total += loss.item() * bs
        count += bs

    return total / max(count, 1)
