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
    def __init__(self, root_dir, tags_root=None, side_information_path=None, node_features_to_exclude=None, fixed_dim=10, transform=None, pre_transform=None, risk_scores_path=None):
        super().__init__(root_dir, transform, pre_transform)
        self.root_dir = root_dir
        self.tags_root = tags_root
        self.side_information_path = side_information_path
        self.node_features_to_exclude = node_features_to_exclude
        self.fixed_dim = fixed_dim
        self.risk_scores_path = risk_scores_path
        
        if self.risk_scores_path is not None:
            with open(self.risk_scores_path, 'r') as f:
                self.risk_scores = json.load(f)

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
        if self.tags_root is None or (hasattr(self, 'risk_scores_path') and self.risk_scores_path is not None):
            return data
        tag_path = self._map_graph_to_tags(graph_fname)
        with open(tag_path, "r") as f:
            tag_dict = json.load(f)
        data.y = self._encode_tags(tag_dict)
        return data

    def _load_and_attach_risk(self, data, graph_fname):
        if not hasattr(self, 'risk_scores_path') or self.risk_scores_path is None:
            return data
        
        episode_num = graph_fname.split('_')[0]
        risk_score = self.risk_scores.get(episode_num, 0.0) # Default to 0.0 if not found
        data.y = torch.tensor([risk_score], dtype=torch.float).unsqueeze(0)
        return data

    def _load_and_attach_side_information(self, data, graph_fname):
        if self.side_information_path is not None:
            episode_id = graph_fname.split('_')[0]
            embedding_path = os.path.join(self.side_information_path, f"{episode_id}.pt")
            if os.path.exists(embedding_path):
                data.side_information = torch.load(embedding_path)
        return data

    # ---------------------------
    # Graph helpers
    # ---------------------------
    def _pack_node_features(self, nodes):
        x = []
        for node in nodes:
            if self.node_features_to_exclude is not None:
                vals = [float(v) if isinstance(v, (int, float)) else -1.0
                        for k, v in node.get("features", {}).items() if k not in self.node_features_to_exclude]
            else:
                vals = [float(v) if isinstance(v, (int, float)) else -1.0
                        for v in node.get("features", {}).values()]
            
            if len(vals) < self.fixed_dim:
                vals += [-1.0] * (self.fixed_dim - len(vals))
            else:
                vals = vals[:self.fixed_dim]
            x.append(torch.tensor(vals, dtype=torch.float))
        return torch.stack(x) if x else None

    def get_metadata(self):
        node_types = ['ego', 'vehicle', 'pedestrian', 'environment']
        edge_types = []
        for source_type in node_types:
            for dest_type in node_types:
                edge_types.append((source_type, 'to', dest_type))
        return (node_types, edge_types)

    def get_quantizer_spec(self):
        # This is a dummy spec. In a real scenario, you would inspect the data to get the feature dimensions.
        spec = {}
        for nt in ['ego', 'vehicle', 'pedestrian', 'environment']:
            spec[nt] = {'feat_dim': self.fixed_dim, 'bins': 32}
        return spec



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
        all_node_types = self.get_metadata()[0]
        for node_type in all_node_types:
            nodes = data_dict.get("nodes", {}).get(node_type, [])
            id_map = {n["id"]: i for i, n in enumerate(nodes)}
            id_maps[node_type] = id_map
            x = self._pack_node_features(nodes)
            if x is not None:
                data[node_type].x = x
            else:
                # Ensure the node type is present, even if empty
                data[node_type].x = torch.empty((0, self.fixed_dim), dtype=torch.float)


        # Edges
        all_edge_types = [('ego', 'to', 'ego'), ('ego', 'to', 'pedestrian'), ('ego', 'to', 'vehicle'), ('ego', 'to', 'environment')]
        for edge_type in all_edge_types:
            src_type, _, dst_type = edge_type
            edge_list = data_dict.get("edges", {}).get(f"{src_type}_to_{dst_type}", [])
            
            src_map = id_maps.get(src_type, {})
            dst_map = id_maps.get(dst_type, {})
            src_idx, dst_idx = [], []
            for edge in edge_list:
                s_id, t_id = edge.get("source"), edge.get("target")
                if s_id in src_map and t_id in dst_map:
                    src_idx.append(src_map[s_id])
                    dst_idx.append(dst_map[t_id])
            
            if src_idx:
                data[edge_type].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            else:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)


        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
        data = self._load_and_attach_risk(data, fname)
        data = self._load_and_attach_side_information(data, fname)
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
            else:
                data[node_type].x = torch.empty((0, self.fixed_dim), dtype=torch.float)

        all_edge_types = [('ego', 'to', 'ego'), ('ego', 'to', 'vehicle')]
        for edge_type in all_edge_types:
            src_type, _, dst_type = edge_type
            edge_list = data_dict.get("edges", {}).get(f"{src_type}_to_{dst_type}", [])

            src_map = id_maps.get(src_type, {})
            dst_map = id_maps.get(dst_type, {})
            src_idx, dst_idx = [], []
            for edge in edge_list:
                s_id, t_id = edge.get("source"), edge.get("target")
                if s_id in src_map and t_id in dst_map:
                    src_idx.append(src_map[s_id])
                    dst_idx.append(dst_map[t_id])
            
            if src_idx:
                data[edge_type].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            else:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
        data = self._load_and_attach_risk(data, fname)
        data = self._load_and_attach_side_information(data, fname)
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
        data = self._load_and_attach_risk(data, fname)
        data = self._load_and_attach_side_information(data, fname)
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
            else:
                data[node_type].x = torch.empty((0, self.fixed_dim), dtype=torch.float)

        # Edges only between ego/env
        all_edge_types = [('ego', 'to', 'ego'), ('ego', 'to', 'environment')]
        for edge_type in all_edge_types:
            src_type, _, dst_type = edge_type
            edge_list = data_dict.get("edges", {}).get(f"{src_type}_to_{dst_type}", [])
            
            src_map = id_maps.get(src_type, {})
            dst_map = id_maps.get(dst_type, {})
            src_idx, dst_idx = [], []
            for edge in edge_list:
                s_id, t_id = edge.get("source"), edge.get("target")
                if s_id in src_map and t_id in dst_map:
                    src_idx.append(src_map[s_id])
                    dst_idx.append(dst_map[t_id])
            
            if src_idx:
                data[edge_type].edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)
            else:
                data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
        data = self._load_and_attach_risk(data, fname)
        data = self._load_and_attach_side_information(data, fname)
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
        all_node_types = self.get_metadata()[0]
        for node_type in all_node_types:
            nodes = data_dict.get("nodes", {}).get(node_type, [])
            x = self._pack_node_features(nodes)
            if x is not None:
                data[node_type].x = x
            else:
                data[node_type].x = torch.empty((0, self.fixed_dim), dtype=torch.float)

        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
        data = self._load_and_attach_risk(data, fname)
        data = self._load_and_attach_side_information(data, fname)
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
            else:
                data[node_type].x = torch.empty((0, self.fixed_dim), dtype=torch.float)

        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
        data = self._load_and_attach_risk(data, fname)
        data = self._load_and_attach_side_information(data, fname)
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
        data = self._load_and_attach_risk(data, fname)
        data = self._load_and_attach_side_information(data, fname)
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
            else:
                data[node_type].x = torch.empty((0, self.fixed_dim), dtype=torch.float)

        data["window_meta"].episode_path = fname
        data = self._load_and_attach_tags(data, fname)
        data = self._load_and_attach_risk(data, fname)
        data = self._load_and_attach_side_information(data, fname)
        return data


# =====================================================
# Loader factory
# =====================================================
def get_graph_dataset(root_dir, mode="all", tags_root=None, side_information_path=None, 
                      node_features_to_exclude=None, fixed_dim=10,
                      normalize=False, norm_method="zscore", risk_scores_path=None):
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
        return GraphJsonDatasetAll(root_dir, tags_root=tags_root, side_information_path=side_information_path, node_features_to_exclude=node_features_to_exclude, fixed_dim=fixed_dim, transform=transform, risk_scores_path=risk_scores_path)
    elif mode == "ego_veh":
        return GraphJsonDatasetEgoVeh(root_dir, tags_root=tags_root, side_information_path=side_information_path, node_features_to_exclude=node_features_to_exclude, fixed_dim=fixed_dim, transform=transform, risk_scores_path=risk_scores_path)
    elif mode == "ego":
        return GraphJsonDatasetEgoOnly(root_dir, tags_root=tags_root, side_information_path=side_information_path, node_features_to_exclude=node_features_to_exclude, fixed_dim=fixed_dim, transform=transform, risk_scores_path=risk_scores_path)
    elif mode == "ego_env":
        return GraphJsonDatasetEgoEnv(root_dir, tags_root=tags_root, side_information_path=side_information_path, node_features_to_exclude=node_features_to_exclude, fixed_dim=fixed_dim, transform=transform, risk_scores_path=risk_scores_path)
    elif mode == "all_no_edges":
        return GraphJsonDatasetAllNoEdges(root_dir, tags_root=tags_root, side_information_path=side_information_path, node_features_to_exclude=node_features_to_exclude, fixed_dim=fixed_dim, transform=transform, risk_scores_path=risk_scores_path)
    elif mode == "ego_veh_no_edges":
        return GraphJsonDatasetEgoVehNoEdges(root_dir, tags_root=tags_root, side_information_path=side_information_path, node_features_to_exclude=node_features_to_exclude, fixed_dim=fixed_dim, transform=transform, risk_scores_path=risk_scores_path)
    elif mode == "ego_no_edges":
        return GraphJsonDatasetEgoOnlyNoEdges(root_dir, tags_root=tags_root, side_information_path=side_information_path, node_features_to_exclude=node_features_to_exclude, fixed_dim=fixed_dim, transform=transform, risk_scores_path=risk_scores_path)
    elif mode == "ego_env_no_edges":
        return GraphJsonDatasetEgoEnvNoEdges(root_dir, tags_root=tags_root, side_information_path=side_information_path, node_features_to_exclude=node_features_to_exclude, fixed_dim=fixed_dim, transform=transform, risk_scores_path=risk_scores_path)
    else:
        raise ValueError(
            f"Unknown mode {mode}. Choose from: all, ego_veh, ego, ego_env, "
            f"all_no_edges, ego_veh_no_edges, ego_no_edges, ego_env_no_edges"
        )