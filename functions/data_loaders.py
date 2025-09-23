import os
import json
import torch
from torch_geometric.data import Dataset, HeteroData
from torch_geometric.transforms import BaseTransform
import torch.nn.functional as F


# =====================================================
# Utility: normalize node features
# =====================================================
class NormalizeNodeFeatures(BaseTransform):
    def __init__(self, method="zscore"):
        self.method = method

    def __call__(self, data):
        for node_type in data.node_types:
            if "x" not in data[node_type]:
                continue

            x = data[node_type].x
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            if self.method == "zscore":
                mean = x.mean(dim=0, keepdim=True)
                std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
                x = (x - mean) / std
            elif self.method == "l2":
                x = F.normalize(x, p=2, dim=1)

            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            data[node_type].x = x

        return data


# =====================================================
# Base dataset (common helpers)
# =====================================================
class BaseGraphJsonDataset(Dataset):
    def __init__(self, root_dir, tags_root=None, fixed_dim=6, transform=None, pre_transform=None):
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.tags_root = tags_root
        self.fixed_dim = fixed_dim

        self.graph_filenames = sorted(
            [f for f in os.listdir(root_dir) if f.endswith(".json")],
            key=lambda f: int(f.split("_")[0]) if f.split("_")[0].isdigit() else f
        )

        # Tag vocabularies (built once if tags_root provided)
        self.action2id = {}
        self.control2id = {}
        self.road2id = {}
        self.env2id = {}
        self.off_action = 0
        self.off_control = 0
        self.off_road = 0
        self.off_env = 0
        self.y_dim = 0

        if self.tags_root is not None:
            self._build_tag_vocabs()

    def len(self):
        return len(self.graph_filenames)

    # ---------------------------
    # Tag handling
    # ---------------------------
    def _build_tag_vocabs(self):
        actions, controls, roads, envs = set(), set(), set(), set()

        # Scan all tag jsons in tags_root
        for fname in os.listdir(self.tags_root):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(self.tags_root, fname)
            try:
                with open(path, "r") as f:
                    tags = json.load(f)
            except Exception:
                continue

            a = tags.get("action_tag", None)
            c = tags.get("traffic_control_tag", None)
            r = tags.get("road_feature_tags", []) or []
            e = tags.get("environment_tags", []) or []

            if a is not None:
                actions.add(a)
            if c is not None:
                controls.add(c)
            for t in r:
                if t is not None:
                    roads.add(t)
            for t in e:
                if t is not None:
                    envs.add(t)

        self.action2id = {t: i for i, t in enumerate(sorted(actions))}
        self.control2id = {t: i for i, t in enumerate(sorted(controls))}
        self.road2id = {t: i for i, t in enumerate(sorted(roads))}
        self.env2id = {t: i for i, t in enumerate(sorted(envs))}

        self.off_action = 0
        self.off_control = self.off_action + len(self.action2id)
        self.off_road = self.off_control + len(self.control2id)
        self.off_env = self.off_road + len(self.road2id)
        self.y_dim = self.off_env + len(self.env2id)

    def _map_graph_to_tags(self, graph_fname):
        """
        Resolve the corresponding tag file for a given graph filename.
        Tries both conventions:
        1) Same filename in tags_root (e.g., '123_graph.json')
        2) 'episode_000NNN.json' where NNN is the numeric id from the graph fname
        """
        if self.tags_root is None:
            return None

        # 1) Same-name convention
        same_name = os.path.join(self.tags_root, graph_fname)
        if os.path.exists(same_name):
            return same_name

        # Extract leading token if it's digits, else extract first number substring
        base_token = graph_fname.split("_")[0]
        if base_token.isdigit():
            num = base_token
        else:
            # fallback: pull first contiguous digit run from the fname
            digits = "".join(ch if ch.isdigit() else " " for ch in graph_fname).split()
            num = digits[0] if digits else None

        if num is not None:
            padded = str(num).zfill(6)
            alt_name = f"episode_{padded}.json"
            alt_path = os.path.join(self.tags_root, alt_name)
            if os.path.exists(alt_path):
                return alt_path

        raise FileNotFoundError(f"No tag file found for graph '{graph_fname}' under '{self.tags_root}'")

    def _encode_tags(self, tag_dict):
        if self.y_dim == 0:
            # No vocabs or no tags_root: return empty tensor to keep API stable
            return torch.empty(0, dtype=torch.float32)

        y = torch.zeros(self.y_dim, dtype=torch.float32)

        # Action (one-hot)
        a = tag_dict.get("action_tag", None)
        if a in self.action2id:
            y[self.off_action + self.action2id[a]] = 1.0

        # Control (one-hot)
        c = tag_dict.get("traffic_control_tag", None)
        if c in self.control2id:
            y[self.off_control + self.control2id[c]] = 1.0

        # Road features (multi-hot)
        for t in tag_dict.get("road_feature_tags", []) or []:
            if t in self.road2id:
                y[self.off_road + self.road2id[t]] = 1.0

        # Environment (multi-hot)
        for t in tag_dict.get("environment_tags", []) or []:
            if t in self.env2id:
                y[self.off_env + self.env2id[t]] = 1.0

        return y.unsqueeze(0)  # shape [1, D]

    def _load_and_attach_tags(self, data, graph_fname):
        if self.tags_root is None:
            return data
        tag_path = self._map_graph_to_tags(graph_fname)
        with open(tag_path, "r") as f:
            tag_dict = json.load(f)
        data.y = self._encode_tags(tag_dict)
        return data

    # ---------------------------
    # Graph helpers
    # ---------------------------
    def _pack_node_features(self, nodes):
        x = []
        for node in nodes:
            vals = [float(v) if isinstance(v, (int, float)) else -1.0
                    for v in node.get("features", {}).values()]
            if len(vals) < self.fixed_dim:
                vals += [-1.0] * (self.fixed_dim - len(vals))
            else:
                vals = vals[:self.fixed_dim]
            x.append(torch.tensor(vals, dtype=torch.float))
        return torch.stack(x) if x else None


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
            x = self._pack_node_features(nodes)
            if x is not None:
                data[node_type].x = x

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
        data = self._load_and_attach_tags(data, fname)
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

            x = self._pack_node_features(nodes)
            if x is not None:
                data[node_type].x = x

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
        data = self._load_and_attach_tags(data, fname)
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

        x = self._pack_node_features(ego_nodes)
        if x is not None:
            data["ego"].x = x

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
        data = self._load_and_attach_tags(data, fname)
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

            x = self._pack_node_features(nodes)
            if x is not None:
                data[node_type].x = x

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
        data = self._load_and_attach_tags(data, fname)
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
            x = self._pack_node_features(nodes)
            if x is not None:
                data[node_type].x = x

        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
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
            x = self._pack_node_features(nodes)
            if x is not None:
                data[node_type].x = x

        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
        return data


class GraphJsonDatasetEgoOnlyNoEdges(BaseGraphJsonDataset):
    def get(self, idx):
        fname = self.graph_filenames[idx]
        path = os.path.join(self.root_dir, fname)

        with open(path, 'r') as f:
            data_dict = json.load(f)

        data = HeteroData()
        ego_nodes = data_dict.get("nodes", {}).get("ego", [])
        x = self._pack_node_features(ego_nodes)
        if x is not None:
            data["ego"].x = x

        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
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
            x = self._pack_node_features(nodes)
            if x is not None:
                data[node_type].x = x

        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
        return data


# =====================================================
# Loader factory
# =====================================================
def get_graph_dataset(root_dir, mode="all", tags_root=None, fixed_dim=6,
                      normalize=False, norm_method="zscore"):
    """
    Factory function to return the right dataset loader.

    mode:
      With edges: "all" | "ego_veh" | "ego" | "ego_env"
      No edges:   "all_no_edges" | "ego_veh_no_edges" | "ego_no_edges" | "ego_env_no_edges"

    tags_root:
      Directory containing per-episode tag JSONs. If provided, vocabularies are built at init
      and each sample will have a flat tag vector in data.y.

    normalize:
      If True, applies NormalizeNodeFeatures transform with method in {zscore, l2}.
    """
    transform = None
    if normalize:
        transform = NormalizeNodeFeatures(method=norm_method)

    if mode == "all":
        return GraphJsonDatasetAll(root_dir, tags_root=tags_root, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_veh":
        return GraphJsonDatasetEgoVeh(root_dir, tags_root=tags_root, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego":
        return GraphJsonDatasetEgoOnly(root_dir, tags_root=tags_root, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_env":
        return GraphJsonDatasetEgoEnv(root_dir, tags_root=tags_root, fixed_dim=fixed_dim, transform=transform)
    elif mode == "all_no_edges":
        return GraphJsonDatasetAllNoEdges(root_dir, tags_root=tags_root, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_veh_no_edges":
        return GraphJsonDatasetEgoVehNoEdges(root_dir, tags_root=tags_root, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_no_edges":
        return GraphJsonDatasetEgoOnlyNoEdges(root_dir, tags_root=tags_root, fixed_dim=fixed_dim, transform=transform)
    elif mode == "ego_env_no_edges":
        return GraphJsonDatasetEgoEnvNoEdges(root_dir, tags_root=tags_root, fixed_dim=fixed_dim, transform=transform)
    else:
        raise ValueError(
            f"Unknown mode {mode}. Choose from: all, ego_veh, ego, ego_env, "
            f"all_no_edges, ego_veh_no_edges, ego_no_edges, ego_env_no_edges"
        )
