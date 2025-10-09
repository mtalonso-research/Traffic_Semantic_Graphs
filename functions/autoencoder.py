import torch
from torch_geometric.data import Data, HeteroData
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn.dense import Linear
from torch_geometric.nn import global_mean_pool, global_max_pool
import math
from collections import defaultdict

def compute_feature_entropy_over_loader(loader, node_types, quantizer, base=2):
    # Frequency tables per node type and per feature
    freq_tables = defaultdict(lambda: defaultdict(lambda: torch.zeros(0, dtype=torch.long)))

    # Accumulate frequencies across all batches
    for batch in loader:
        node_types = node_types 
        batch = quantizer.transform_inplace(batch)
        for nt in node_types:
            if "x" not in batch[nt]:
                continue
            x = batch[nt].x.long()  # [N, F]
            N, F = x.shape
            for f in range(F):
                vals = x[:, f]
                unique, counts = vals.unique(return_counts=True)
                # expand storage if needed
                table = freq_tables[nt][f]
                if table.numel() < unique.max().item() + 1:
                    new_table = torch.zeros(unique.max().item() + 1, dtype=torch.long)
                    new_table[:table.numel()] = table
                    table = new_table
                table[unique] += counts
                freq_tables[nt][f] = table

    # Compute entropy from aggregated frequencies
    feature_entropy = {}
    for nt, feats in freq_tables.items():
        entropies = []
        for f, table in feats.items():
            probs = table.float()
            probs = probs[probs > 0]
            probs = probs / probs.sum()
            H = -(probs * (probs + 1e-12).log()).sum().item()
            if base != math.e:
                H /= math.log(base)
            entropies.append(H)
        feature_entropy[nt] = entropies

    return feature_entropy

class QuantileFeatureQuantizer:
    def __init__(self, bins=32, node_types=("ego", "vehicle", "environment"), key="x"):
        assert bins >= 2
        self.bins = int(bins)
        self.node_types = tuple(node_types)
        self.key = key
        self.edges = {nt: None for nt in self.node_types}    # dict[node_type] -> (F, bins-1) tensor
        self.bin_centers = {nt: None for nt in self.node_types}  # dict[node_type] -> (F, bins) tensor
        self.fitted = False

    @torch.no_grad()
    def fit(self, dataset_or_loader, max_samples_per_type=2_000_000):
        # First pass: infer feature dims and gather samples
        buffers = {nt: [] for nt in self.node_types}
        feat_dim = {nt: None for nt in self.node_types}

        def _accum(x, nt):
            if x is None: return
            if feat_dim[nt] is None:
                feat_dim[nt] = x.size(-1)
            else:
                if feat_dim[nt] != x.size(-1):
                    raise ValueError(f"Feature dim mismatch for {nt}: seen {feat_dim[nt]} vs {x.size(-1)}")
            # Flatten to (N, F)
            buffers[nt].append(x.detach())

        for batch in dataset_or_loader:
            # Support HeteroDataBatch and HeteroData
            for nt in self.node_types:
                x = batch[nt].get(self.key, None)
                if x is not None:
                    _accum(x, nt)

        for nt in self.node_types:
            if not buffers[nt]:
                continue
            X = torch.cat(buffers[nt], dim=0)  # (N, F)
            if torch.isnan(X).any():
                # Simple imputation: median per feature
                med = torch.nanmedian(X, dim=0).values
                mask = torch.isnan(X)
                X[mask] = med.repeat(X.size(0), 1)[mask]
            if X.size(0) > max_samples_per_type:
                idx = torch.randperm(X.size(0))[:max_samples_per_type]
                X = X[idx]

            F = X.size(1)
            q = torch.linspace(0, 1, steps=self.bins + 1, device=X.device)
            # Remove 0 and 1 to get interior cut points
            qs = q[1:-1]

            # Compute edges per feature dim (F, bins-1)
            edges = torch.stack([
                torch.quantile(X[:, f], qs, interpolation="linear") for f in range(F)
            ], dim=0)

            mid_ps = (q[:-1] + q[1:]) / 2.0  # length = bins
            centers = torch.stack([
                torch.quantile(X[:, f], mid_ps, interpolation="linear") for f in range(F)
            ], dim=0)

            self.edges[nt] = edges.cpu()
            self.bin_centers[nt] = centers.cpu()

        self.fitted = True
        return self

    @torch.no_grad()
    def transform_inplace(self, hetero):
        if not self.fitted:
            raise RuntimeError("Quantizer not fitted. Call fit() first.")
        for nt in self.node_types:
            x = hetero[nt].get(self.key, None)
            if x is None: continue
            edges = self.edges[nt].to(x.device)          # (F, B-1)
            # bucketize expects (...,) and boundaries (..., B-1). Do per-feature to be safe.
            xs = []
            for f in range(x.size(1)):
                # indices in [0..B-1]
                xi = torch.bucketize(x[:, f], boundaries=edges[f], right=False)
                xs.append(xi.view(-1, 1))
            hetero[nt][self.key] = torch.cat(xs, dim=1).long()
        return hetero

    @torch.no_grad()
    def inverse(self, nt, idx):
        centers = self.bin_centers[nt].to(idx.device)  # (F, B)
        # Gather per feature
        out_cols = []
        for f in range(idx.size(-1)):
            out_cols.append(centers[f].gather(0, idx[..., f]))
        return torch.stack(out_cols, dim=-1)

    def spec(self):
        """Return a tiny spec useful to build decoder heads."""
        return {nt: {"feat_dim": None if self.edges[nt] is None else self.edges[nt].shape[0],
                     "bins": self.bins}
                for nt in self.node_types}

class DiscreteFeatureEncoder(nn.Module):
    def __init__(self, spec, embed_dim=16, use_feature_mask=False,
                feature_entropy=None, trainable_gates=False):
        super().__init__()
        self.embeddings = nn.ModuleDict()
        self.in_dims = {}  # nt -> F * embed_dim
        self.use_feature_mask = use_feature_mask
        self._trainable_gates = trainable_gates  # remember the flag

        # ----- Build per-feature embeddings -----
        for nt, nt_spec in spec.items():
            Fdim, bins = nt_spec["feat_dim"], nt_spec["bins"]
            if Fdim is None or Fdim == 0:
                continue
            self.embeddings[nt] = nn.ModuleList([
                nn.Embedding(bins, embed_dim) for _ in range(Fdim)
            ])
            self.in_dims[nt] = Fdim * embed_dim

        # ----- Optional feature masks (entropy-aware initialization) -----
        if self.use_feature_mask:
            # store gates in ParameterDict if trainable, else plain dict
            self.feature_gates = nn.ParameterDict() if trainable_gates else {}
            for nt in self.embeddings.keys():
                Fdim = spec[nt]["feat_dim"]

                if feature_entropy and nt in feature_entropy:
                    ent = torch.tensor(feature_entropy[nt], dtype=torch.float32)
                    # normalize entropy to [0,1]
                    ent = (ent - ent.min()) / (ent.max() - ent.min() + 1e-8)
                    # convert to logits so sigmoid(logit(ent)) ≈ ent
                    logits = torch.logit(ent.clamp(1e-3, 1 - 1e-3))
                else:
                    logits = torch.randn(Fdim) * 0.01  # fallback init near 0

                param = nn.Parameter(logits, requires_grad=trainable_gates) \
                        if trainable_gates else logits
                self.feature_gates[nt] = param
        else:
            self.feature_gates = None

    def train_gates(self, trainable: bool = True):
        if not self.use_feature_mask or self.feature_gates is None:
            return  # nothing to do

        new_gates = nn.ParameterDict()
        for nt, g in self.feature_gates.items():
            if isinstance(g, nn.Parameter):
                g.requires_grad = trainable
                new_gates[nt] = g
            else:
                # convert tensor to Parameter if now trainable
                if trainable:
                    new_gates[nt] = nn.Parameter(g.clone(), requires_grad=True)
                else:
                    new_gates[nt] = g.clone().detach()
        self.feature_gates = new_gates
        self._trainable_gates = trainable

    def forward(self, data):
        out = {}
        for nt, emb_list in self.embeddings.items():
            if "x" not in data[nt]:
                continue
            x = data[nt].x  # [N, F] Long

            # Embedding lookup first
            embs = [emb_list[f](x[:, f]) for f in range(x.size(1))]  # list of [N, embed_dim]
            embs = torch.stack(embs, dim=1)  # [N, F, embed_dim]

            # Apply feature mask after embeddings
            if self.use_feature_mask:
                gates = torch.sigmoid(self.feature_gates[nt])  # [F]
                embs = embs * gates.view(1, -1, 1)  # broadcast over N and embed_dim

            # Flatten per node 
            out[nt] = embs.view(embs.size(0), -1).to(x.device)  # [N, F*embed_dim]
        return out

class HeteroGraphAutoencoder(nn.Module):
    def __init__(self, metadata, hidden_dim=64, embed_dim=32, quantizer_spec=None, feat_emb_dim=16, 
                 use_feature_mask=False, feature_entropy=None, trainable_gates=False, conv_layer_cls=SAGEConv):
        super().__init__()
        node_types, edge_types = metadata
        self.metadata = metadata
        self.quantizer_spec = quantizer_spec

        # Discrete -> float embeddings
        self.feature_encoder = DiscreteFeatureEncoder(quantizer_spec, embed_dim=feat_emb_dim, use_feature_mask=use_feature_mask, feature_entropy=feature_entropy, trainable_gates=trainable_gates)

        # Pre-projection per node type to hidden_dim 
        self.pre_lin = nn.ModuleDict()
        for nt in node_types:
            in_dim = self.feature_encoder.in_dims.get(nt, 0)
            if in_dim and in_dim > 0:
                self.pre_lin[nt] = Linear(in_dim, hidden_dim)
            else:
                # If a node type has no features in spec, give it a dummy 1-dim -> hidden mapper
                self.pre_lin[nt] = Linear(1, hidden_dim)

        # Hetero encoder: message passing
        self.encoder = HeteroConv({
            etype: conv_layer_cls((-1, -1), hidden_dim) for etype in edge_types
        }, aggr='mean')

        # Project to latent per node type
        self.lin_proj = nn.ModuleDict({nt: Linear(hidden_dim, embed_dim) for nt in node_types})

        # Feature decoders (per node type)
        self.feature_heads = nn.ModuleDict()
        for nt, spec in quantizer_spec.items():
            Fdim = spec["feat_dim"]
            if Fdim is None or Fdim == 0:
                continue
            bins = spec["bins"]
            self.feature_heads[nt] = nn.Linear(embed_dim, Fdim * bins)

        # Edge decoders (relation-specific bilinear)
        self.edge_decoders = nn.ModuleDict()
        for srctype, rel, dsttype in self.metadata[1]:
            self.edge_decoders[str((srctype, rel, dsttype))] = nn.Bilinear(embed_dim, embed_dim, 1)

    def encode(self, x_dict, edge_index_dict):
        # Pre-project (and ReLU) per node type to hidden_dim
        x0 = {}
        for nt, pre in self.pre_lin.items():
            if nt in x_dict:
                x0[nt] = F.relu(pre(x_dict[nt]))
            else:
                continue

        # Message passing 
        h_mp = self.encoder(x0, edge_index_dict)
        # if no messages for a node type, use pre-projected features x0
        h_dict = {}
        for nt in x0.keys():
            h_dict[nt] = h_mp.get(nt, x0[nt])

        # Final projection to latent
        z_dict = {nt: F.relu(self.lin_proj[nt](h)) for nt, h in h_dict.items()}
        return z_dict

    def decode_features(self, z_dict):
        out = {}
        for nt, head in self.feature_heads.items():
            if nt not in z_dict:
                # This node type had zero nodes this batch.
                continue
            logits = head(z_dict[nt])  # [N, F*B]
            Fdim, bins = self.quantizer_spec[nt]["feat_dim"], self.quantizer_spec[nt]["bins"]
            logits = logits.view(-1, Fdim, bins)
            out[nt] = logits
        return out

    def decode_edges(self, z_dict, edge_index_dict):
        out = {}
        for key, edge_index in edge_index_dict.items():
            srctype, rel, dsttype = key
            if srctype not in z_dict or dsttype not in z_dict:
                continue  # this batch has no nodes of a required type
            if edge_index.numel() == 0:
                continue  # no edges for this relation in this batch
            z_src = z_dict[srctype][edge_index[0]]
            z_dst = z_dict[dsttype][edge_index[1]]
            score = self.edge_decoders[str(key)](z_src, z_dst).squeeze(-1)
            out[key] = score
        return out

    def forward(self, data):
        x_embed = self.feature_encoder(data)              # nt -> float embeddings
        z_dict = self.encode(x_embed, data.edge_index_dict)
        feat_logits = self.decode_features(z_dict)
        edge_logits = self.decode_edges(z_dict, data.edge_index_dict)
        return z_dict, feat_logits, edge_logits
    
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=128, proj_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.proj(x)


def feature_loss(feat_logits, data):
    total_loss, total_dims = 0.0, 0
    for nt, logits in feat_logits.items():
        if "x" not in data[nt] or logits.numel() == 0:
            continue
        targets = data[nt].x  # Long [N, F]
        Fdim = logits.size(1)
        for f in range(Fdim):
            total_loss += F.cross_entropy(logits[:, f, :], targets[:, f])
            total_dims += 1
    if total_dims == 0:
        return torch.tensor(0.0, device=next(iter(feat_logits.values())).device if feat_logits else "cpu")
    return total_loss / total_dims


def edge_loss(edge_logits, z_dict, edge_decoders, num_neg=1):
    if not edge_logits:
        any_device = next(iter(z_dict.values())).device if z_dict else "cpu"
        return torch.tensor(0.0, device=any_device)

    loss = 0.0
    count = 0
    for key, pos_score in edge_logits.items():
        srctype, rel, dsttype = key
        if srctype not in z_dict or dsttype not in z_dict:
            continue
        if pos_score.numel() == 0:
            continue

        pos_label = torch.ones_like(pos_score)

        # Negative samples: same count as positives
        num_src = z_dict[srctype].size(0)
        num_dst = z_dict[dsttype].size(0)
        if num_src == 0 or num_dst == 0:
            continue

        neg_src = torch.randint(0, num_src, (pos_score.size(0),), device=pos_score.device)
        neg_dst = torch.randint(0, num_dst, (pos_score.size(0),), device=pos_score.device)
        neg_score = edge_decoders[str(key)](z_dict[srctype][neg_src], z_dict[dsttype][neg_dst]).squeeze(-1)
        neg_label = torch.zeros_like(neg_score)

        all_scores = torch.cat([pos_score, neg_score], dim=0)
        all_labels = torch.cat([pos_label, neg_label], dim=0)
        loss += F.binary_cross_entropy_with_logits(all_scores, all_labels)
        count += 1

    if count == 0:
        any_device = next(iter(z_dict.values())).device if z_dict else "cpu"
        return torch.tensor(0.0, device=any_device)
    return loss / count

def kl_divergence_between_gaussians(z1, z2, eps=1e-6):
    mu1, mu2 = z1.mean(0), z2.mean(0)
    var1 = z1.var(0) + eps
    var2 = z2.var(0) + eps

    kl = 0.5 * torch.sum(
        torch.log(var2 / var1)
        + (var1 + (mu1 - mu2).pow(2)) / var2
        - 1
    )
    return kl

def batched_graph_embeddings(z_dict, batch_data, metadata, pooling="mean", embed_dim_per_type=32):

    B = 0
    pooled = {}
    for nt in metadata[0]:
        if nt in z_dict and "batch" in batch_data[nt]:
            z = z_dict[nt]
            batch_vec = batch_data[nt].batch
            B_nt = batch_vec.max().item() + 1
            B = max(B, B_nt)
            if pooling == "mean":
                pooled_nt = global_mean_pool(z, batch_vec)
            elif pooling == "max":
                pooled_nt = global_max_pool(z, batch_vec)
            else:
                raise ValueError(pooling)
            pooled[nt] = pooled_nt
        else:
            pooled[nt] = None  # Mark missing type

    # Compute max number of graphs across node types
    if B == 0:
        return torch.zeros(1, len(batch_data.node_types) * embed_dim_per_type, device=z.device)

    out_blocks = []
    for nt in metadata[0]:
        if pooled[nt] is None:
            out_blocks.append(torch.zeros(B, embed_dim_per_type, device=z.device))
        else:
            t = pooled[nt]
            B_nt, D = t.shape
            if B_nt < B:
                pad = torch.zeros(B - B_nt, D, device=t.device)
                t = torch.cat([t, pad], dim=0)
            out_blocks.append(t)

    return torch.cat(out_blocks, dim=-1)

