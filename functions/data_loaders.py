import os
import json
import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform
import torch.nn.functional as F
import datetime as dt

class NormalizeNodeFeatures(BaseTransform):
    def __init__(self, method="zscore"):
        self.method = method

    def __call__(self, data):
        for node_type in data.node_types:
            if "x" not in data[node_type]:
                continue

            x = data[node_type].x

            # Replace NaNs/Infs first
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Normalize if requested
            if self.method == "zscore":
                mean = x.mean(dim=0, keepdim=True)
                std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
                x = (x - mean) / std
            elif self.method == "l2":
                x = F.normalize(x, p=2, dim=1)

            # Replace NaNs/Infs again (edge case: std=0 etc.)
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            data[node_type].x = x

        return data

# =====================================================
# Base dataset (common helpers)
# =====================================================
class BaseGraphJsonDataset(Dataset):
    def __init__(self, root_dir, fixed_dim=6, transform=None, pre_transform=None):
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.fixed_dim = fixed_dim
        self.graph_filenames = sorted(
            [f for f in os.listdir(root_dir) if f.endswith(".json")],
            key=lambda f: int(f.split("_")[0]) if f.split("_")[0].isdigit() else f
        )

    def len(self):
        return len(self.graph_filenames)


# =====================================================
# 1. Full dataset (all nodes + edges)
# =====================================================
class GraphJsonDatasetAll(BaseGraphJsonDataset):
    def get(self, idx):
        fname = self.graph_filenames[idx]
        path = os.path.join(self.root_dir, fname)

        with open(path, 'r') as f:
            data_dict = json.load(f)

        data = HeteroData()
        id_maps = {}

        # Nodes
        for node_type, nodes in data_dict.get("nodes", {}).items():
            id_map = {n["id"]: i for i, n in enumerate(nodes)}
            id_maps[node_type] = id_map

            x = []
            for node in nodes:
                vals = [float(v) if isinstance(v, (int, float)) else -1.0
                        for v in node.get("features", {}).values()]
                if len(vals) < self.fixed_dim:
                    vals += [-1.0] * (self.fixed_dim - len(vals))
                else:
                    vals = vals[:self.fixed_dim]
                x.append(torch.tensor(vals, dtype=torch.float))
            if x:
                data[node_type].x = torch.stack(x)

        # Edges
        for edge_type, edge_list in data_dict.get("edges", {}).items():
            parts = edge_type.split("_to_")
            if len(parts) != 2:
                continue
            src_type, dst_type = parts
            rel = (src_type, "to", dst_type)

            src_map = id_maps.get(src_type, {})
            dst_map = id_maps.get(dst_type, {})
            src_idx, dst_idx = [], []
            for edge in edge_list:
                s_id, t_id = edge.get("source"), edge.get("target")
                if s_id in src_map and t_id in dst_map:
                    src_idx.append(src_map[s_id])
                    dst_idx.append(dst_map[t_id])
            if src_idx:
                data[rel].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

        data["window_meta"].episode_path = fname
        return data


# =====================================================
# 2. Ego + Vehicle dataset
# =====================================================
class GraphJsonDatasetEgoVeh(BaseGraphJsonDataset):
    def get(self, idx):
        fname = self.graph_filenames[idx]
        path = os.path.join(self.root_dir, fname)

        with open(path, 'r') as f:
            data_dict = json.load(f)

        data = HeteroData()
        id_maps = {}

        for node_type in ["ego", "vehicle"]:
            nodes = data_dict.get("nodes", {}).get(node_type, [])
            id_map = {n["id"]: i for i, n in enumerate(nodes)}
            id_maps[node_type] = id_map

            x = []
            for node in nodes:
                vals = [float(v) if isinstance(v, (int, float)) else -1.0
                        for v in node.get("features", {}).values()]
                if len(vals) < self.fixed_dim:
                    vals += [-1.0] * (self.fixed_dim - len(vals))
                else:
                    vals = vals[:self.fixed_dim]
                x.append(torch.tensor(vals, dtype=torch.float))
            if x:
                data[node_type].x = torch.stack(x)

        for edge_type, edge_list in data_dict.get("edges", {}).items():
            parts = edge_type.split("_to_")
            if len(parts) != 2:
                continue
            src_type, dst_type = parts
            if src_type not in ["ego", "vehicle"] or dst_type not in ["ego", "vehicle"]:
                continue
            rel = (src_type, "to", dst_type)

            src_map = id_maps.get(src_type, {})
            dst_map = id_maps.get(dst_type, {})
            src_idx, dst_idx = [], []
            for edge in edge_list:
                s_id, t_id = edge.get("source"), edge.get("target")
                if s_id in src_map and t_id in dst_map:
                    src_idx.append(src_map[s_id])
                    dst_idx.append(dst_map[t_id])
            if src_idx:
                data[rel].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

        data["window_meta"].episode_path = fname
        return data


# =====================================================
# 3. Ego-only dataset
# =====================================================
class GraphJsonDatasetEgoOnly(BaseGraphJsonDataset):
    def get(self, idx):
        fname = self.graph_filenames[idx]
        path = os.path.join(self.root_dir, fname)

        with open(path, 'r') as f:
            data_dict = json.load(f)

        data = HeteroData()

        ego_nodes = data_dict.get("nodes", {}).get("ego", [])
        id_map = {n["id"]: i for i, n in enumerate(ego_nodes)}

        x = []
        for node in ego_nodes:
            vals = [float(v) if isinstance(v, (int, float)) else -1.0
                    for v in node.get("features", {}).values()]
            if len(vals) < self.fixed_dim:
                vals += [-1.0] * (self.fixed_dim - len(vals))
            else:
                vals = vals[:self.fixed_dim]
            x.append(torch.tensor(vals, dtype=torch.float))
        if x:
            data["ego"].x = torch.stack(x)

        edge_list = data_dict.get("edges", {}).get("ego_to_ego", [])
        src_idx, dst_idx = [], []
        for edge in edge_list:
            s_id, t_id = edge.get("source"), edge.get("target")
            if s_id in id_map and t_id in id_map:
                src_idx.append(id_map[s_id])
                dst_idx.append(id_map[t_id])
        if src_idx:
            data[("ego", "to", "ego")].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

        data["window_meta"].episode_path = fname
        return data


# =====================================================
# 4. Ego + Environment dataset
# =====================================================
class GraphJsonDatasetEgoEnv(BaseGraphJsonDataset):
    def get(self, idx):
        fname = self.graph_filenames[idx]
        path = os.path.join(self.root_dir, fname)

        with open(path, 'r') as f:
            data_dict = json.load(f)

        data = HeteroData()
        id_maps = {}

        for node_type in ["ego", "environment"]:
            nodes = data_dict.get("nodes", {}).get(node_type, [])
            id_map = {n["id"]: i for i, n in enumerate(nodes)}
            id_maps[node_type] = id_map

            x = []
            for node in nodes:
                vals = [float(v) if isinstance(v, (int, float)) else -1.0
                        for v in node.get("features", {}).values()]
                if len(vals) < self.fixed_dim:
                    vals += [-1.0] * (self.fixed_dim - len(vals))
                else:
                    vals = vals[:self.fixed_dim]
                x.append(torch.tensor(vals, dtype=torch.float))
            if x:
                data[node_type].x = torch.stack(x)

        # Edges only between ego/env
        for edge_type, edge_list in data_dict.get("edges", {}).items():
            parts = edge_type.split("_to_")
            if len(parts) != 2:
                continue
            src_type, dst_type = parts
            if src_type not in ["ego", "environment"] or dst_type not in ["ego", "environment"]:
                continue
            rel = (src_type, "to", dst_type)

            src_map = id_maps.get(src_type, {})
            dst_map = id_maps.get(dst_type, {})
            src_idx, dst_idx = [], []
            for edge in edge_list:
                s_id, t_id = edge.get("source"), edge.get("target")
                if s_id in src_map and t_id in dst_map:
                    src_idx.append(src_map[s_id])
                    dst_idx.append(dst_map[t_id])
            if src_idx:
                data[rel].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

        data["window_meta"].episode_path = fname
        return data
    

# =====================================================
# Edge-free variants
# =====================================================

class GraphJsonDatasetAllNoEdges(BaseGraphJsonDataset):
    def get(self, idx):
        fname = self.graph_filenames[idx]
        path = os.path.join(self.root_dir, fname)
        with open(path, 'r') as f:
            data_dict = json.load(f)

        data = HeteroData()
        for node_type, nodes in data_dict.get("nodes", {}).items():
            x = []
            for node in nodes:
                vals = [float(v) if isinstance(v, (int, float)) else -1.0
                        for v in node.get("features", {}).values()]
                if len(vals) < self.fixed_dim:
                    vals += [-1.0] * (self.fixed_dim - len(vals))
                else:
                    vals = vals[:self.fixed_dim]
                x.append(torch.tensor(vals, dtype=torch.float))
            if x:
                data[node_type].x = torch.stack(x)
        data["window_meta"].episode_path = fname
        return data


class GraphJsonDatasetEgoVehNoEdges(BaseGraphJsonDataset):
    def get(self, idx):
        fname = self.graph_filenames[idx]
        path = os.path.join(self.root_dir, fname)
        with open(path, 'r') as f:
            data_dict = json.load(f)

        data = HeteroData()
        for node_type in ["ego", "vehicle"]:
            nodes = data_dict.get("nodes", {}).get(node_type, [])
            x = []
            for node in nodes:
                vals = [float(v) if isinstance(v, (int, float)) else -1.0
                        for v in node.get("features", {}).values()]
                if len(vals) < self.fixed_dim:
                    vals += [-1.0] * (self.fixed_dim - len(vals))
                else:
                    vals = vals[:self.fixed_dim]
                x.append(torch.tensor(vals, dtype=torch.float))
            if x:
                data[node_type].x = torch.stack(x)
        data["window_meta"].episode_path = fname
        return data


class GraphJsonDatasetEgoOnlyNoEdges(BaseGraphJsonDataset):
    def get(self, idx):
        fname = self.graph_filenames[idx]
        path = os.path.join(self.root_dir, fname)
        with open(path, 'r') as f:
            data_dict = json.load(f)

        data = HeteroData()
        ego_nodes = data_dict.get("nodes", {}).get("ego", [])
        x = []
        for node in ego_nodes:
            vals = [float(v) if isinstance(v, (int, float)) else -1.0
                    for v in node.get("features", {}).values()]
            if len(vals) < self.fixed_dim:
                vals += [-1.0] * (self.fixed_dim - len(vals))
            else:
                vals = vals[:self.fixed_dim]
            x.append(torch.tensor(vals, dtype=torch.float))
        if x:
            data["ego"].x = torch.stack(x)
        data["window_meta"].episode_path = fname
        return data


class GraphJsonDatasetEgoEnvNoEdges(BaseGraphJsonDataset):
    def get(self, idx):
        fname = self.graph_filenames[idx]
        path = os.path.join(self.root_dir, fname)
        with open(path, 'r') as f:
            data_dict = json.load(f)

        data = HeteroData()
        for node_type in ["ego", "environment"]:
            nodes = data_dict.get("nodes", {}).get(node_type, [])
            x = []
            for node in nodes:
                vals = [float(v) if isinstance(v, (int, float)) else -1.0
                        for v in node.get("features", {}).values()]
                if len(vals) < self.fixed_dim:
                    vals += [-1.0] * (self.fixed_dim - len(vals))
                else:
                    vals = vals[:self.fixed_dim]
                x.append(torch.tensor(vals, dtype=torch.float))
            if x:
                data[node_type].x = torch.stack(x)
        data["window_meta"].episode_path = fname
        return data


# =====================================================
# Loader factory
# =====================================================
def get_graph_dataset(root_dir, mode="all", fixed_dim=6, 
                      normalize=False, norm_method="zscore"):
    """
    Factory function to return the right dataset loader.

    mode:
      With edges: "all" | "ego_veh" | "ego" | "ego_env"
      No edges:   "all_no_edges" | "ego_veh_no_edges" | "ego_no_edges" | "ego_env_no_edges"

    normalize:
      If True, applies NormalizeNodeFeatures transform.
    norm_method:
      "zscore" or "l2"
    """
    transform = None
    if normalize:
        transform = NormalizeNodeFeatures(method=norm_method)

    if mode == "all":
        return GraphJsonDatasetAll(root_dir, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_veh":
        return GraphJsonDatasetEgoVeh(root_dir, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego":
        return GraphJsonDatasetEgoOnly(root_dir, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_env":
        return GraphJsonDatasetEgoEnv(root_dir, fixed_dim=fixed_dim, transform=transform)
    elif mode == "all_no_edges":
        return GraphJsonDatasetAllNoEdges(root_dir, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_veh_no_edges":
        return GraphJsonDatasetEgoVehNoEdges(root_dir, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_no_edges":
        return GraphJsonDatasetEgoOnlyNoEdges(root_dir, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_env_no_edges":
        return GraphJsonDatasetEgoEnvNoEdges(root_dir, fixed_dim=fixed_dim, transform=transform)
    else:
        raise ValueError(
            f"Unknown mode {mode}. Choose from: all, ego_veh, ego, ego_env, "
            f"all_no_edges, ego_veh_no_edges, ego_no_edges, ego_env_no_edges"
        )
