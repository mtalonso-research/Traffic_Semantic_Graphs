from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any, Optional
import json
import pandas as pd
import numpy as np
import os
from geopy.distance import geodesic
from datetime import datetime

import networkx as nx
import ipycytoscape
import networkx as nx
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

from functions.utils_L2D import load_and_restore_parquet

def generate_graphs(min_ep,max_ep=-1,
                    source_data_dir='../data/processed/L2D',
                    processed_frame_dir='../data/processed_frames/L2D',
                    output_dir='../data/graphical/L2D'):
    
    if max_ep == -1: max_ep = min_ep + 1

    for ep_num in range(min_ep,max_ep):
        parquet_path,aux_path = directory_lookup(ep_num,source_data_dir,processed_frame_dir)
        graph_dict = build_graph_json(
            parquet_path=parquet_path,
            aux_paths=aux_path,
            image_lookup=image_lookup,
            graph_id=f"scene_{ep_num:06d}"
        )

        json_path = os.path.join(output_dir, f"{ep_num}_graph.json")
        with open(json_path, 'w') as f:
            json.dump(graph_dict, f, indent=2)

def directory_lookup(ep_num,source_data_dir,processed_frame_dir):
    parquet_path = os.path.join(source_data_dir,f'episode_{ep_num:06d}.parquet')
    df = load_and_restore_parquet(parquet_path)
    aux_path = {}
    for frame_idx in range(len(df)):
        aux_path[str(frame_idx)] = os.path.join(processed_frame_dir,f"Episode{ep_num:06d}/front_left_Annotations",f'frame_{frame_idx:05d}.json')
    return parquet_path,aux_path

def image_lookup(frame_idx: str) -> str:
    return f"L2D_data/camera/chunk-000/episode_000000/observation.images.front_left/frame_{frame_idx:05}.jpg"

"""Graph Builder – draft skeleton

Builds a heterogeneous traffic‑scene graph as JSON suitable for
PyTorch Geometric (`HeteroData`) and the companion visualizer.

Node types: ego, vehicle, pedestrian, environment
Edge types:
  • ego_to_ego      – continuity between ego nodes across frames
  • ego_to_vehicle  – ego → vehicle within a frame
  • ego_to_pedestrian – ego → pedestrian within a frame
  • env_to_env      – continuity between environment nodes across frames

Node/edge features are mostly loaded from a Parquet file, with some
auxiliary NPZ / JSON inputs.  This is a **first‑pass scaffold**: all data
extraction logic is stubbed out with TODOs so we can fill in feature
engineering later.
"""

def build_graph_json(parquet_path: str | Path, aux_paths: Dict[str, str | Path],
                     image_lookup: Callable[[str], str], output_path: Optional[str | Path] = None,
                     graph_id: str | None = None) -> Dict[str, Any]:
    
    parquet_path = Path(parquet_path)
    graph_id = graph_id or parquet_path.stem

    df = load_and_restore_parquet(parquet_path)
    aux_data = _load_aux_data(aux_paths)

    graph: Dict[str, Any] = {
        "nodes": {"ego": [], "vehicle": [], "pedestrian": [], "environment": []},
        "edges": {
            "ego_to_ego": [],
            "ego_to_vehicle": [],
            "ego_to_pedestrian": [],
            "env_to_env": [],
            "ego_to_environment": [],
        },
        "metadata": {"graph_id": graph_id, "source_files": {
            "parquet": str(parquet_path),
            **{k: str(v) for k, v in aux_paths.items()},
        }},
    }

    for idx, row in df.iterrows():
        frame_id = str(idx)

        ego_id = f"ego_{frame_id}"
        ego_features = {
            "latitude": row["vehicle_latitude"],
            "longitude": row["vehicle_longitude"],
            "speed": row["vehicle_speed"],
            "heading": row["vehicle_heading"],
            "heading_error": row["vehicle_heading_error"],
            "accel_x": row["vehicle_acceleration_x"],
            "accel_y": row["vehicle_acceleration_y"],
            "gas": row["action_gas_pedal"],
            "brake": row["action_brake_pedal"],
            "steering": row["action_steering_angle"],
            "gear": row["action_gear"],
            "turn_signal": row["action_turn_signal"],
            "turning_behavior": row["turning_behavior"],
            "front_cam": image_lookup(frame_id),
        }
        graph["nodes"]["ego"].append({"id": ego_id, "features": ego_features})

        env_id = f"env_{frame_id}"
        env_features = {
            "month": row["month"],
            "day_of_week": row["day_of_week"],
            "time_of_day": row["time_of_day"],
            "road_type": row["road_type"],
            "road_name": row["road_name"],
            "maxspeed": row["maxspeed"],
            "lanes": row["lanes"],
            "surface": row["surface"],
            "oneway": row["oneway"],
            "width": row["width"],
            "sidewalk": row["sidewalk"],
            "bicycle": row["bicycle"],
            "bridge": row["bridge"],
            "tunnel": row["tunnel"],
            "traffic_controls": row["traffic_controls"],
            "traffic_features": row["traffic_features"],
            "landuse": row["landuse"],
            "is_narrow": row["is_narrow"],
            "is_unlit": row["is_unlit"],
            "bike_friendly": row["bike_friendly"],
        }
        graph["nodes"]["environment"].append({"id": env_id, "features": env_features})

        frame_annotations = aux_data.get(str(idx), {})['annotations']
        for obj in frame_annotations:
            track_id = obj.get("track_id")
            obj_class = obj.get("attributes", {}).get("class")

            if not track_id or not obj_class:
                continue

            node_id = f"{track_id}_{frame_id}"

            if obj_class.lower() == "car":
                graph["nodes"]["vehicle"].append({"id": node_id, "features": {}})
            elif obj_class.lower() == "pedestrian":
                graph["nodes"]["pedestrian"].append({"id": node_id, "features": {}})

    _add_ego_continuity_edges(graph)
    _add_env_continuity_edges(graph)
    _add_ego_entity_edges(graph, aux_data)
    _add_ego_env_edges(graph)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(graph, f, indent=2)

    return graph

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_ego_continuity_edges(graph: Dict[str, Any]) -> None:
    ego_nodes = graph["nodes"]["ego"]
    for n1, n2 in zip(ego_nodes, ego_nodes[1:]):
        coord1 = (n1["features"]["latitude"], n1["features"]["longitude"])
        coord2 = (n2["features"]["latitude"], n2["features"]["longitude"])
        dist = float(geodesic(coord1, coord2).meters)
        graph["edges"]["ego_to_ego"].append({
            "source": n1["id"],
            "target": n2["id"],
            "features": {"distance_m": dist},
        })


def _add_env_continuity_edges(graph: Dict[str, Any]) -> None:
    """Link consecutive environment nodes using Δt computed from time_of_day."""
    env_nodes = graph["nodes"]["environment"]

    def parse_time(tstr: str) -> float:
        #try:
            dt = datetime.strptime(tstr, "%H:%M:%S")
            return dt.hour * 3600 + dt.minute * 60 + dt.second
        #except Exception:
        #    return np.nan

    for n1, n2 in zip(env_nodes, env_nodes[1:]):
        t1 = parse_time(n1["features"]["time_of_day"])
        t2 = parse_time(n2["features"]["time_of_day"])
        delta_t = float(t2 - t1) if not np.isnan(t1) and not np.isnan(t2) else np.nan
        graph["edges"]["env_to_env"].append({
            "source": n1["id"],
            "target": n2["id"],
            "features": {"delta_t": delta_t},
        })


def _add_ego_entity_edges(graph: Dict[str, Any], aux: Dict[str, Any]) -> None:
    ego_by_frame = {n["id"].split("_")[1]: n for n in graph["nodes"]["ego"]}

    for frame_id, ego_node in ego_by_frame.items():
        frame_objs = aux.get(str(frame_id), {})['annotations']

        for ent in frame_objs:
            if ent['attributes']['class'] == 'car':
                dist = ent['attributes']['depth_stats']['mean_depth']
                graph["edges"]["ego_to_vehicle"].append({
                    "source": ego_node["id"],
                    "target": f'{ent["track_id"]}_{frame_id}',
                    "features": {"distance": dist},
                })
            else:
                dist = ent['attributes']['depth_stats']['mean_depth']
                graph["edges"]["ego_to_pedestrian"].append({
                    "source": ego_node["id"],
                    "target": f'{ent["track_id"]}_{frame_id}',
                    "features": {"distance": dist},
                })


def _add_ego_env_edges(graph: Dict[str, Any]) -> None:
    ego_nodes = graph["nodes"]["ego"]
    env_nodes = graph["nodes"]["environment"]
    env_by_frame = {n["id"].split("_")[1]: n for n in env_nodes}
    for ego_node in ego_nodes:
        ego_id = ego_node["id"].split("_")[1]
        env_node = env_by_frame.get(ego_id)
        if env_node:
            graph["edges"]["ego_to_environment"].append({
                "source": ego_node["id"],
                "target": env_node["id"],
                "features": {},
            })


def _load_aux_data(aux_paths: Dict[str, Path | str]) -> Dict[str, Any]:
    loaded: Dict[str, Any] = {}
    for name, path in aux_paths.items():
        path = Path(path)
        if path.suffix == ".npz":
            loaded[name] = np.load(path)
        elif path.suffix == ".json":
            with path.open() as f:
                loaded[name] = json.load(f)
        else:
            raise ValueError(f"Unsupported aux file: {path}")
    return loaded



class EgoSequenceDataset(Dataset):
    def __init__(self, json_path, feature_keys, num_steps=5, skip_short=True, norm_stats=None):
        with open(json_path, 'r') as f:
            data = json.load(f)

        self.feature_keys = feature_keys
        self.num_steps = num_steps
        self.skip_short = skip_short
        self.ego_nodes = data['nodes']['ego']
        self.norm = norm_stats
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        total_frames = len(self.ego_nodes)
        if self.skip_short and total_frames < self.num_steps + 1:
            return []

        for i in range(total_frames - self.num_steps):
            samples.append((i, i + self.num_steps))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start, target_idx = self.samples[idx]
        input_seq = self.ego_nodes[start: start + self.num_steps]
        target = self.ego_nodes[target_idx]

        #x = np.array([
        #    [frame['features'][k] for k in self.feature_keys]
        #    for frame in input_seq
        #], dtype=np.float32)
        x = np.array([
            [(frame['features'][k] - self.norm['feat_mean'][i]) / self.norm['feat_std'][i]
            for i, k in enumerate(self.feature_keys)]
            for frame in input_seq
        ], dtype=np.float32)

        #y = np.array([
        #    target['features']['latitude'],
        #    target['features']['longitude']
        #], dtype=np.float32)
        lat = (target['features']['latitude'] - self.norm['lat_mean']) / self.norm['lat_std']
        lon = (target['features']['longitude'] - self.norm['lon_mean']) / self.norm['lon_std']
        y = np.array([lat, lon], dtype=np.float32)

        return sequence_to_pyg_graph(x, y)

class EgoDirectoryDataset(Dataset):
    def __init__(self, directory, feature_keys, num_steps=5, skip_short=True, norm_stats=None):
        self.feature_keys = feature_keys
        self.datasets = []
        for fname in sorted(os.listdir(directory)):
            if not fname.endswith(".json"):
                continue
            full_path = os.path.join(directory, fname)
            ds = EgoSequenceDataset(full_path, feature_keys, num_steps, skip_short, norm_stats)
            if len(ds) > 0:
                self.datasets.append(ds)

        self.index_map = []
        for ds_idx, ds in enumerate(self.datasets):
            for sample_idx in range(len(ds)):
                self.index_map.append((ds_idx, sample_idx))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.index_map[idx]
        return self.datasets[ds_idx][sample_idx]
    
def sequence_to_pyg_graph(x_seq, y):
    num_nodes = x_seq.shape[0]
    edge_index = torch.tensor([
        [i for i in range(num_nodes - 1)],
        [i + 1 for i in range(num_nodes - 1)]
    ], dtype=torch.long)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # bidirectional

    return Data(
        x=torch.tensor(x_seq, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y, dtype=torch.float).view(1, -1)
    )

def get_normalization_stats(directory_dataset):
    feature_keys = directory_dataset.feature_keys
    feat_vals = []
    lat_vals = []
    lon_vals = []

    for episode in directory_dataset.datasets:
        for node in episode.ego_nodes:
            features = node["features"]
            feat_vals.append([features[k] for k in feature_keys])
            lat_vals.append(features["latitude"])
            lon_vals.append(features["longitude"])

    feat_vals = np.array(feat_vals)
    lat_vals = np.array(lat_vals)
    lon_vals = np.array(lon_vals)

    return {
        "feat_mean": feat_vals.mean(axis=0),
        "feat_std": feat_vals.std(axis=0) + 1e-8,  # avoid divide by zero
        "lat_mean": lat_vals.mean(),
        "lat_std": lat_vals.std() + 1e-8,
        "lon_mean": lon_vals.mean(),
        "lon_std": lon_vals.std() + 1e-8
    }

def split_dataset(dataset, train_ratio=0.8, seed=42):
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    test_len = total_len - train_len
    return random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(seed))

def create_dataloaders(directory, feature_keys, num_steps=5, skip_short=True, batch_size=32, train_ratio=0.8):
    raw_dataset = EgoDirectoryDataset(directory, feature_keys, num_steps, skip_short=True)
    norm_stats = get_normalization_stats(raw_dataset)
    full_dataset = EgoDirectoryDataset(directory, feature_keys, num_steps, skip_short=True, norm_stats=norm_stats)
    train_set, test_set = split_dataset(full_dataset, train_ratio=train_ratio)

    train_loader = PyGDataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = PyGDataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_set, test_set

def json_graph_to_networkx(graph_data, frame_idx=0):
    G = nx.DiGraph()
    node_mapping = {}
    existing_ids = set()

    # Add nodes safely
    for node_type, node_list in graph_data.get("nodes", {}).items():
        for i, node in enumerate(node_list):
            original_id = node.get("id", f"unknown_{len(node_mapping)}")

            # Make IDs unique across duplicates
            unique_id = original_id
            while unique_id in existing_ids:
                unique_id = f"{original_id}_{i}"
            existing_ids.add(unique_id)

            node_id = f"{unique_id}_f{frame_idx}"
            node_mapping[original_id] = node_id
            
            features = dict(node.get("features", {}))
            features.update({"id": node_id, "type": node_type})
            
            G.add_node(node_id, **features)

    # Add edges, only if nodes exist
    for edge_type, edge_list in graph_data.get("edges", {}).items():
        for edge in edge_list:
            src = edge.get("source")
            tgt = edge.get("target")
            src_id = node_mapping.get(src)
            tgt_id = node_mapping.get(tgt)

            # Skip invalid edges
            if src_id is None or tgt_id is None:
                continue

            attributes = dict(edge.get("features", {}))
            attributes["interaction"] = edge_type
            G.add_edge(src_id, tgt_id, **attributes)

    return G, node_mapping

def combined_graph_viewer(graphs_by_frame: dict):
    type_color_map = {
        'vehicle': '#1f77b4',
        'pedestrian': '#ff7f0e',
        'environment': '#2ca02c',
        'ego': '#FF0000'
    }

    G_combined = nx.DiGraph()
    framewise_nodes = []

    # Combine all frames
    for frame_idx, (frame_key, graph_data) in enumerate(graphs_by_frame.items()):
        G_frame, node_mapping = json_graph_to_networkx(graph_data, frame_idx)
        G_combined.update(G_frame)
        framewise_nodes.append(node_mapping)

    # Temporal edges
    for i in range(len(framewise_nodes) - 1):
        for orig_id in framewise_nodes[i]:
            if orig_id in framewise_nodes[i + 1]:
                G_combined.add_edge(
                    framewise_nodes[i][orig_id],
                    framewise_nodes[i + 1][orig_id],
                    interaction="temporal"
                )

    # Render graph
    cyto = ipycytoscape.CytoscapeWidget()
    cyto.graph.add_graph_from_networkx(G_combined)

    # Node styling
    for node in cyto.graph.nodes:
        node.data['label'] = node.data.get('id', '?')
        node.data['tooltip'] = '\n'.join(f"{k}: {v}" for k, v in node.data.items() if k != 'id')
        node_type = node.data.get('type', 'unknown')
        node.data['color'] = type_color_map.get(node_type, '#d3d3d3')

    # Edge styling
    for edge in cyto.graph.edges:
        edge.data['tooltip'] = '\n'.join(f"{k}: {v}" for k, v in edge.data.items() if k not in ['source', 'target'])

    # Apply style
    cyto.set_style([
        {'selector': 'node',
         'style': {
             'label': 'data(label)',
             'background-color': 'data(color)',
             'width': '25',
             'height': '25',
             'font-size': '8px'
         }},
        {'selector': 'edge',
         'style': {
             'label': 'data(interaction)',
             'width': 1,
             'line-color': '#ccc',
             'target-arrow-color': '#ccc',
             'target-arrow-shape': 'triangle',
             'font-size': '6px'
         }}
    ])

    return cyto
