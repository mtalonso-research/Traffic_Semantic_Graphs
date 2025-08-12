from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Any, Optional
import json
import pandas as pd
import numpy as np
import os
from geopy.distance import geodesic
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
import math
import pandas as pd
import ast

import networkx as nx
import ipycytoscape
import networkx as nx
import torch
from torch.utils.data import Dataset, random_split
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from tqdm import tqdm

from functions.utils_L2D import load_and_restore_parquet

def graph_to_dfs(graph_dict):
    # --- NODES ---
    node_rows = []
    for node_type, node_list in graph_dict['nodes'].items():
        for node in node_list:
            row = {'id': node['id'], 'type': node_type}
            row.update(node.get('features', {}))  # Flatten features
            node_rows.append(row)
    nodes_df = pd.DataFrame(node_rows)

    # --- EDGES ---
    edge_rows = []
    for edge_type, edge_list in graph_dict['edges'].items():
        for edge in edge_list:
            row = {'source': edge['source'], 'target': edge['target'], 'type': edge_type}
            row.update(edge.get('features', {}))  # Flatten features
            edge_rows.append(row)
    edges_df = pd.DataFrame(edge_rows)

    return nodes_df, edges_df

def generate_graphs(min_ep,max_ep=-1,
                    source_data_dir='../data/processed/L2D',
                    processed_frame_dir='../data/processed_frames/L2D',
                    output_dir='../data/graphical/L2D'):
    
    if max_ep == -1: max_ep = min_ep + 1

    for ep_num in tqdm(range(min_ep,max_ep)):
        try:
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
        except: print(f'Problems with episode {ep_num}')

def directory_lookup(ep_num,source_data_dir,processed_frame_dir):
    parquet_path = os.path.join(source_data_dir,f'episode_{ep_num:06d}.parquet')
    df = load_and_restore_parquet(parquet_path)
    aux_path = {}
    for frame_idx in range(len(df)):
        aux_path[str(frame_idx)] = os.path.join(processed_frame_dir,f"Episode{ep_num:06d}/front_left_Annotations",f'frame_{frame_idx:05d}.json')
    return parquet_path,aux_path

def image_lookup(frame_idx: str) -> str:
    return f"L2D_data/camera/chunk-000/episode_000000/observation.images.front_left/frame_{frame_idx:05}.jpg"

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

        # --- Ego node (unchanged) ---
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

        # --- Environment node (unchanged) ---
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

        # --- Entities: vehicles & pedestrians (UPDATED) ---
        frame_record = aux_data.get(str(idx), {}) or {}
        frame_annotations = frame_record.get("annotations") or []

        for obj in frame_annotations:
            attrs = (obj.get("attributes") or {})
            obj_class = (attrs.get("class") or "").lower()
            track_id = obj.get("track_id")
            if not track_id or not obj_class:
                continue

            node_id = f"{track_id}_{frame_id}"

            if obj_class == "car":
                # Pull directly from this frame's aux JSON; no computations.
                speed_info = (attrs.get("speed_info") or {})

                veh_features = {
                    "speed_ms":          speed_info.get("speed_ms"),
                    "speed_kmh":         speed_info.get("speed_kmh"),
                    "velocity_ms":       speed_info.get("velocity_vector_ms"),   # [vx, vy, vz]
                    "velocity_kmh":      speed_info.get("velocity_vector_kmh"),  # [vx, vy, vz]
                    # If acceleration vector exists in the file, include it; otherwise omit.
                    "accel_ms2":         speed_info.get("acceleration_vector_ms")
                                          or speed_info.get("accel_vector_ms")
                                          or speed_info.get("acceleration_ms2"),
                    # Useful metadata passthroughs
                    "vector_dimension":  speed_info.get("vector_dimension"),
                    "coordinate_system": speed_info.get("coordinate_system"),
                    "relative_to_ego":   speed_info.get("relative_to_ego"),
                }
                # Drop None values to keep JSON tidy
                veh_features = {k: v for k, v in veh_features.items() if v is not None}

                graph["nodes"]["vehicle"].append({"id": node_id, "features": veh_features})

            elif obj_class == "pedestrian":
                graph["nodes"]["pedestrian"].append({"id": node_id, "features": {}})

            # ignore other classes for now

    # --- Edges (unchanged) ---
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
def _node_lookup(graph: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Return {ntype: {node_id: node_dict}} so we can update node features in-place."""
    return {nt: {n["id"]: n for n in graph["nodes"].get(nt, [])}
            for nt in ["ego", "vehicle", "pedestrian", "environment"]}

def _add_ego_continuity_edges(graph: Dict[str, Any]) -> None:
    ego_nodes = graph["nodes"]["ego"]
    ego_map = {n["id"]: n for n in ego_nodes}  # for in-place updates

    for n1, n2 in zip(ego_nodes, ego_nodes[1:]):
        coord1 = (n1["features"]["latitude"], n1["features"]["longitude"])
        coord2 = (n2["features"]["latitude"], n2["features"]["longitude"])
        dist = float(geodesic(coord1, coord2).meters)

        # copy edge feature onto the target ego node (frame t+1)
        n2["features"]["prev_step_distance_m"] = dist

        # keep the edge for structure (feature can stay or be emptied later)
        graph["edges"]["ego_to_ego"].append({
            "source": n1["id"],
            "target": n2["id"],
            "features": {"distance_m": dist},
        })

def _add_env_continuity_edges(graph: Dict[str, Any]) -> None:
    env_nodes = graph["nodes"]["environment"]

    def parse_time(tstr: str) -> Optional[int]:
        try:
            dt = datetime.strptime(str(tstr), "%H:%M:%S")
            return dt.hour * 3600 + dt.minute * 60 + dt.second
        except Exception:
            return None

    for n1, n2 in zip(env_nodes, env_nodes[1:]):
        t1 = parse_time(n1["features"].get("time_of_day"))
        t2 = parse_time(n2["features"].get("time_of_day"))
        if t1 is None or t2 is None:
            delta_t = float("nan")
        else:
            delta_t = float((t2 - t1) % 86400)  # midnight-safe

        # copy edge feature onto the target env node (frame t+1)
        n2["features"]["delta_t_prev_s"] = delta_t

        graph["edges"]["env_to_env"].append({
            "source": n1["id"],
            "target": n2["id"],
            "features": {"delta_t": delta_t},
        })

def _add_ego_entity_edges(graph: Dict[str, Any], aux: Dict[str, Any]) -> None:
    # lookups for quick in-place feature updates
    veh_map = {n["id"]: n for n in graph["nodes"]["vehicle"]}
    ped_map = {n["id"]: n for n in graph["nodes"]["pedestrian"]}
    ego_by_frame = {n["id"].rsplit("_", 1)[-1]: n for n in graph["nodes"]["ego"]}

    for frame_id, ego_node in ego_by_frame.items():
        frame = (aux.get(str(frame_id), {}) or {})
        annos = (frame.get("annotations") or [])

        for ent in annos:
            attrs = (ent.get("attributes") or {})
            cls   = (attrs.get("class") or "").lower()
            tid   = ent.get("track_id")
            if not tid:
                continue

            # distance lives in depth_stats.mean_depth (guarded)
            ds = attrs.get("depth_stats") or {}
            dist = ds.get("mean_depth", None)
            if not isinstance(dist, (int, float)):
                continue
            dist = float(dist)

            nid = f"{tid}_{frame_id}"

            if cls == "car":
                # copy edge feature onto vehicle node
                node = veh_map.get(nid)
                if node is not None:
                    prev = node["features"].get("dist_to_ego", None)
                    node["features"]["dist_to_ego"] = min(prev, dist) if isinstance(prev, (int, float)) else dist

                graph["edges"]["ego_to_vehicle"].append({
                    "source": ego_node["id"],
                    "target": nid,
                    "features": {"distance": dist},
                })

            elif cls == "pedestrian":
                # copy edge feature onto pedestrian node
                node = ped_map.get(nid)
                if node is not None:
                    prev = node["features"].get("dist_to_ego", None)
                    node["features"]["dist_to_ego"] = min(prev, dist) if isinstance(prev, (int, float)) else dist

                graph["edges"]["ego_to_pedestrian"].append({
                    "source": ego_node["id"],
                    "target": nid,
                    "features": {"distance": dist},
                })
            # ignore other classes

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

# -------------------------
# Utility: feature handling
# -------------------------

_NUMERIC_TYPES = (int, float, np.integer, np.floating)

def _is_numeric(x) -> bool:
    return isinstance(x, _NUMERIC_TYPES)

def _expand_vector(name: str, vec: List[float]) -> Dict[str, float]:
    # default to x,y,z suffixes for len=3, otherwise index suffixes
    if len(vec) == 3:
        suffixes = ["_x", "_y", "_z"]
    else:
        suffixes = [f"_{i}" for i in range(len(vec))]
    return {name + s: float(v) for s, v in zip(suffixes, vec)}

# Which keys to treat as vectors if present (we’ll expand them when expand_vectors=True)
_VECTOR_KEYS_BY_TYPE = {
    "vehicle": ["velocity_ms", "velocity_kmh", "accel_ms2"],
    "ego": [],
    "pedestrian": [],
    "environment": [],
}

def _discover_feature_schema(episode_graphs: List[Dict[str, Any]],
                             include_nodes: List[str],
                             feature_keys: Optional[Dict[str, Union[str, List[str]]]],
                             expand_vectors: bool) -> Dict[str, List[str]]:

    schema: Dict[str, List[str]] = {}
    for ntype in include_nodes:
        want = None if feature_keys is None else feature_keys.get(ntype, 'all')
        if isinstance(want, list) and want:
            schema[ntype] = want[:]
            continue

        # else discover from data
        keys = set()
        for g in episode_graphs:
            for node in g["nodes"].get(ntype, []):
                feats = node.get("features", {})
                # numeric scalars
                for k, v in feats.items():
                    if _is_numeric(v):
                        keys.add(k)
                # expand known vectors
                if expand_vectors:
                    for vk in _VECTOR_KEYS_BY_TYPE.get(ntype, []):
                        if vk in feats and isinstance(feats[vk], (list, tuple)):
                            expanded = _expand_vector(vk, feats[vk])
                            keys.update(expanded.keys())
        schema[ntype] = sorted(keys)
    return schema

# -------------------------
# Episode parsing / caching
# -------------------------

def _parse_episode_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r") as f:
        g = json.load(f)
    # quick index by frame for faster windowing
    by_frame: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    # nodes
    for ntype, nlist in g["nodes"].items():
        for n in nlist:
            nid: str = n["id"]
            # frame id is the suffix after the last underscore
            # (track_id itself may contain underscores)
            if "_" not in nid:
                continue
            frame_id = nid.rsplit("_", 1)[-1]
            by_frame.setdefault(frame_id, {}).setdefault(ntype, []).append(n)
    # edges (we’ll filter later within windows)
    return {"graph": g, "by_frame": by_frame, "num_frames": len(by_frame)}

def _frames_in_window(start: int, length: int) -> List[str]:
    return [str(i) for i in range(start, start + length)]

# -------------------------
# Main dataset
# -------------------------

class SceneSequenceDataset(Dataset):

    EDGE_MAP = {
        "ego_to_ego":         ("ego",          "ego_to_ego",         "ego"),
        "ego_to_vehicle":     ("ego",          "ego_to_vehicle",     "vehicle"),
        "ego_to_pedestrian":  ("ego",          "ego_to_pedestrian",  "pedestrian"),
        "env_to_env":         ("environment",  "env_to_env",         "environment"),
        "ego_to_environment": ("ego",          "ego_to_environment", "environment"),
    }

    def __init__(
        self,
        directory: Union[str, Path],
        num_steps: int = 5,
        skip_short: bool = True,
        include_nodes: Union[str, List[str]] = "all",          # e.g. ['ego','vehicle']
        include_edges: Union[str, List[str]] = "all",          # e.g. ['ego_to_ego','ego_to_vehicle']
        feature_keys: Optional[Dict[str, Union[str, List[str]]]] = None,  # per-type
        expand_vectors: bool = True,
        target_spec: Union[str, Dict[str, Any]] = "none",      # placeholder; see _resolve_target
        norm: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,    # per-type {'mean':{feat:..}, 'std':{feat:..}}
        strict: bool = False,   # if True, missing features raise
        fill_missing: str = "zero",  # "zero" or "nan" for missing scalars/vector components
        episode_glob: str = "*.json",
    ):
        self.directory = Path(directory)
        self.num_steps = num_steps
        self.skip_short = skip_short
        self.target_spec = target_spec
        self.expand_vectors = expand_vectors
        self.norm = norm
        self.strict = strict
        self.fill_missing = fill_missing  # controls imputation value

        # resolve which nodes/edges to include
        all_node_types = ["ego", "vehicle", "pedestrian", "environment"]
        all_edge_types = list(self.EDGE_MAP.keys())
        self.include_nodes: List[str] = all_node_types if include_nodes == "all" else list(include_nodes)
        self.include_edges: List[str] = all_edge_types if include_edges == "all" else list(include_edges)

        # load episodes
        files = sorted(self.directory.glob(episode_glob))
        self.episodes: List[Dict[str, Any]] = []
        for fp in files:
            ep = _parse_episode_json(fp)
            # respect skip_short based on ego frames count
            T = len([k for k in ep["by_frame"].keys()])
            if self.skip_short and T < self.num_steps + 1:
                continue
            ep["path"] = str(fp)
            self.episodes.append(ep)

        # discover / lock feature schema per type
        self.feature_schema: Dict[str, List[str]] = _discover_feature_schema(
            [e["graph"] for e in self.episodes], self.include_nodes, feature_keys, self.expand_vectors
        )

        # build global index of (episode_idx, start_idx, target_idx)
        self.index: List[Tuple[int, int, int]] = []
        for epi, ep in enumerate(self.episodes):
            total_frames = len(ep["by_frame"])
            for start in range(0, total_frames - self.num_steps):
                self.index.append((epi, start, start + self.num_steps))

    def __len__(self) -> int:
        return len(self.index)

    # ------------- feature extraction (fixed-length + mask) -------------

    def _node_features_to_vec_and_mask(self, ntype: str, feats: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:

        keys = self.feature_schema.get(ntype, [])
        if not keys:
            return None

        def _default():
            return float("nan") if self.fill_missing == "nan" else 0.0

        vec: List[float] = []
        mask: List[float] = []

        for k in keys:
            have_val = False
            val: float

            # direct scalar present?
            if k in feats and _is_numeric(feats[k]):
                val = float(feats[k])
                have_val = True
            else:
                # Vector expansion case: schema key like 'velocity_ms_x' from source 'velocity_ms'
                base, sep, suffix = k.rpartition("_")
                if base in _VECTOR_KEYS_BY_TYPE.get(ntype, []) and base in feats and isinstance(feats[base], (list, tuple)):
                    idx = {"x": 0, "y": 1, "z": 2}.get(suffix, None)
                    if idx is None:
                        try:
                            idx = int(suffix)
                        except Exception:
                            idx = None
                    if idx is not None and idx < len(feats[base]) and _is_numeric(feats[base][idx]):
                        val = float(feats[base][idx])
                        have_val = True

            if not have_val:
                if self.strict:
                    raise KeyError(f"Missing feature {k} for node type {ntype}")
                val = _default()

            # normalize if stats available
            if self.norm and ntype in self.norm:
                mu = self.norm[ntype]["mean"].get(k, 0.0)
                sd = self.norm[ntype]["std"].get(k, 1.0) or 1.0
                val = (val - mu) / sd

            vec.append(val)
            mask.append(1.0 if have_val else 0.0)

        return vec, mask

    # ------------- target resolver (placeholder-friendly) -------------

    def _resolve_target(self, ep: Dict[str, Any], target_frame: int) -> Optional[torch.Tensor]:
        """
        Built-in target specs (name-only strings):
        - 'none'                    -> None
        - 'ego_next_pos'            -> tensor([lat, lon]) at t+1
        - 'ego_next_heading'        -> tensor([heading]) at t+1
        - 'ego_next_speed'          -> tensor([speed]) at t+1
        - 'ego_next_controls'       -> tensor([gas, brake, steering, gear, turn_signal]) at t+1
        - 'min_dist_vehicle'        -> tensor([min_distance]) at t+1 (1e6 if none)
        - 'min_dist_pedestrian'     -> tensor([min_distance]) at t+1 (1e6 if none)
        - 'closest_vehicle_relvel_y'-> tensor([vy]) at t+1 (0.0 if none/missing)
        - 'ego_traj_5'              -> tensor([lat1, lon1, ..., lat5, lon5]) from waypoints at t (last input frame)

        Notes:
        • For 'ego_traj_5', we read the Parquet path from graph metadata and
            extract the 'waypoints' of frame (target_frame - 1), which represent
            the next 5 positions from that frame.
        • Missing scalars default to 0.0; missing distances default to 1e6.
        """
        spec = self.target_spec
        if spec == "none":
            return None

        tgt_key = str(target_frame)
        prev_key = str(target_frame - 1)  # last frame inside the input window

        # --- helpers ---
        def _ego_feats_at(frame_key: str) -> Optional[Dict[str, Any]]:
            nodes = ep["by_frame"].get(frame_key, {}).get("ego", [])
            return nodes[0]["features"] if nodes else None

        def _min_edge_distance(edge_name: str, frame_key: str) -> float:
            # Find edges of type edge_name whose source/target are both at frame_key
            dists: List[float] = []
            for e in ep["graph"]["edges"].get(edge_name, []):
                s_id = e["source"]; d_id = e["target"]
                if s_id.endswith(f"_{frame_key}") and d_id.endswith(f"_{frame_key}"):
                    dist = e.get("features", {}).get("distance", None)
                    if isinstance(dist, (int, float)):
                        dists.append(float(dist))
            return min(dists) if dists else 1e6  # large default if none

        def _closest_vehicle_relvel_y(frame_key: str) -> float:
            # Find closest vehicle at this frame via ego_to_vehicle edges, then read its velocity_ms_y
            edges = []
            for e in ep["graph"]["edges"].get("ego_to_vehicle", []):
                s_id = e["source"]; d_id = e["target"]
                if s_id.endswith(f"_{frame_key}") and d_id.endswith(f"_{frame_key}"):
                    dist = e.get("features", {}).get("distance", None)
                    if isinstance(dist, (int, float)):
                        edges.append((d_id, float(dist)))
            if not edges:
                return 0.0
            # closest vehicle id
            veh_id, _ = min(edges, key=lambda x: x[1])
            # build quick index of vehicle nodes
            veh_nodes = ep["graph"]["nodes"].get("vehicle", [])
            vmap = {n["id"]: n for n in veh_nodes}
            node = vmap.get(veh_id, None)
            if not node:
                return 0.0
            feats = node.get("features", {})
            vel = feats.get("velocity_ms", None)
            if isinstance(vel, (list, tuple)) and len(vel) >= 2 and isinstance(vel[1], (int, float)):
                return float(vel[1])  # y component
            return 0.0

        # ---------------- ego_* targets at t+1 ----------------
        if spec == "ego_next_pos":
            feats = _ego_feats_at(tgt_key)
            if not feats:
                return torch.tensor([0.0, 0.0], dtype=torch.float)
            lat = float(feats.get("latitude", 0.0))
            lon = float(feats.get("longitude", 0.0))
            return torch.tensor([lat, lon], dtype=torch.float)

        if spec == "ego_next_heading":
            feats = _ego_feats_at(tgt_key)
            val = float(feats.get("heading", 0.0)) if feats else 0.0
            return torch.tensor([val], dtype=torch.float)

        if spec == "ego_next_speed":
            feats = _ego_feats_at(tgt_key)
            val = float(feats.get("speed", 0.0)) if feats else 0.0
            return torch.tensor([val], dtype=torch.float)

        if spec == "ego_next_controls":
            feats = _ego_feats_at(tgt_key) or {}
            gas   = float(feats.get("gas", 0.0))
            brake = float(feats.get("brake", 0.0))
            steer = float(feats.get("steering", 0.0))
            gear  = float(feats.get("gear", 0.0))
            tsig  = float(feats.get("turn_signal", 0.0))
            return torch.tensor([gas, brake, steer, gear, tsig], dtype=torch.float)

        # ---------------- proximity/interaction at t+1 ----------------
        if spec == "min_dist_vehicle":
            md = _min_edge_distance("ego_to_vehicle", tgt_key)
            return torch.tensor([md], dtype=torch.float)

        if spec == "min_dist_pedestrian":
            md = _min_edge_distance("ego_to_pedestrian", tgt_key)
            return torch.tensor([md], dtype=torch.float)

        if spec == "closest_vehicle_relvel_y":
            vy = _closest_vehicle_relvel_y(tgt_key)
            return torch.tensor([vy], dtype=torch.float)

        # ---------------- ego_traj_5 via Parquet waypoints at t (prev frame) ----------------
        if spec == "ego_traj_5":
            # Parquet path is recorded by your graph generator
            pq_path = ep["graph"].get("metadata", {}).get("source_files", {}).get("parquet", None)
            if not pq_path or not Path(pq_path).exists():
                # fallback to zeros if parquet missing
                return torch.zeros(10, dtype=torch.float)

            try:
                # Only read the 'waypoints' column to be fast
                df = pd.read_parquet(pq_path, columns=["waypoints"])
                wpt_str = df.iloc[int(prev_key)]["waypoints"]  # waypoints for last input frame
                # waypoints is a Python-literal string: "[{'x':..., 'y':...}, ...]"
                wpts = ast.literal_eval(wpt_str) if isinstance(wpt_str, str) else wpt_str
                # Take first 5 entries; pad if shorter
                coords = []
                for i in range(5):
                    if i < len(wpts) and isinstance(wpts[i], dict):
                        # Convention: x ~ longitude, y ~ latitude
                        lon = float(wpts[i].get("x", 0.0))
                        lat = float(wpts[i].get("y", 0.0))
                    else:
                        lat, lon = 0.0, 0.0
                    coords.extend([lat, lon])  # [lat1, lon1, lat2, lon2, ...]
                return torch.tensor(coords, dtype=torch.float)
            except Exception:
                # robust fallback
                return torch.zeros(10, dtype=torch.float)

        # Unknown spec
        return None

    # ------------- sample builder -------------

    def __getitem__(self, idx: int) -> HeteroData:
        epi, start, target = self.index[idx]
        ep = self.episodes[epi]
        window_frames = _frames_in_window(start, self.num_steps)

        data = HeteroData()
        node_id_maps: Dict[str, Dict[str, int]] = {}  # per type: global node id -> local index

        for ntype in self.include_nodes:
            nodes: List[Dict[str, Any]] = []
            for f in window_frames:
                for n in ep["by_frame"].get(f, {}).get(ntype, []):
                    nodes.append(n)

            xs: List[List[float]] = []
            ms: List[List[float]] = []
            nid_to_idx: Dict[str, int] = []
            nid_to_idx = {}
            frame_ids: List[int] = []

            for n in nodes:
                result = self._node_features_to_vec_and_mask(ntype, n.get("features", {}))
                if result is None:
                    continue
                vec, mask = result
                idx_local = len(xs)
                nid_to_idx[n["id"]] = idx_local
                xs.append(vec)
                ms.append(mask)
                # store frame id for convenience
                frame_id = n["id"].rsplit("_", 1)[-1]
                try:
                    frame_ids.append(int(frame_id))
                except Exception:
                    frame_ids.append(-1)

            if xs:
                x_arr = np.asarray(xs, dtype=np.float32)
                m_arr = np.asarray(ms, dtype=np.float32)

                data[ntype].x = torch.tensor(x_arr, dtype=torch.float)
                data[ntype].mask = torch.tensor(m_arr, dtype=torch.float)   # 1 = observed, 0 = filled
                data[ntype].frame_id = torch.tensor(frame_ids, dtype=torch.long)
                node_id_maps[ntype] = nid_to_idx

        # Edges: include chosen types; keep only edges whose endpoints are in the window & present in node maps
        for etype_name in self.include_edges:
            if etype_name not in self.EDGE_MAP:
                continue
            src_t, rel, dst_t = self.EDGE_MAP[etype_name]
            if src_t not in node_id_maps or dst_t not in node_id_maps:
                continue  # no nodes of this type in this window

            kept_src: List[int] = []
            kept_dst: List[int] = []
            for e in ep["graph"]["edges"].get(etype_name, []):
                s_id = e["source"]; d_id = e["target"]
                s_frame = s_id.rsplit("_", 1)[-1]
                d_frame = d_id.rsplit("_", 1)[-1]
                if (s_frame in window_frames) and (d_frame in window_frames):
                    si = node_id_maps[src_t].get(s_id, None)
                    di = node_id_maps[dst_t].get(d_id, None)
                    if si is not None and di is not None:
                        kept_src.append(si); kept_dst.append(di)

            if kept_src:
                edge_index = torch.tensor([kept_src, kept_dst], dtype=torch.long)
                data[(src_t, rel, dst_t)].edge_index = edge_index

        # Target
        y = self._resolve_target(ep, target)
        if y is not None:
            data.y = y.view(1, -1)

        # Meta (optional, handy for debugging)
        data.window_meta = {
            "episode_path": ep.get("path", ""),
            "start": start,
            "target": target,
            "frames": [int(f) for f in window_frames],
        }

        return data


# -------------------------
# Normalization utilities
# -------------------------

def compute_norm_stats(directory: Union[str, Path],
                       include_nodes: Union[str, List[str]] = "all",
                       feature_keys: Optional[Dict[str, Union[str, List[str]]]] = None,
                       expand_vectors: bool = True,
                       episode_glob: str = "*.json",
                       num_steps: int = 5,
                       skip_short: bool = True) -> Dict[str, Dict[str, Dict[str, float]]]:

    tmp = SceneSequenceDataset(directory, num_steps=num_steps, skip_short=skip_short,
                               include_nodes=include_nodes, include_edges="all",
                               feature_keys=feature_keys, expand_vectors=expand_vectors,
                               target_spec="none", norm=None, episode_glob=episode_glob)

    sums: Dict[str, Dict[str, float]] = {nt: {} for nt in tmp.include_nodes}
    sqs:  Dict[str, Dict[str, float]] = {nt: {} for nt in tmp.include_nodes}
    counts: Dict[str, Dict[str, int]] = {nt: {} for nt in tmp.include_nodes}

    # Iterate episodes/frames, accumulate feature stats directly from node features
    for ep in tmp.episodes:
        for ntype in tmp.include_nodes:
            keys = tmp.feature_schema.get(ntype, [])
            for frame_id, nmap in ep["by_frame"].items():
                for n in nmap.get(ntype, []):
                    feats = n.get("features", {})
                    # construct partial vec (include only present values; no imputation in stats)
                    items: List[Tuple[str, float]] = []
                    for k in keys:
                        if k in feats and _is_numeric(feats[k]):
                            items.append((k, float(feats[k])))
                        else:
                            base, _, suf = k.rpartition("_")
                            if base in _VECTOR_KEYS_BY_TYPE.get(ntype, []) and base in feats and isinstance(feats[base], (list, tuple)):
                                idx = {"x":0,"y":1,"z":2}.get(suf, None)
                                if idx is None:
                                    try: idx = int(suf)
                                    except: idx = None
                                if idx is not None and idx < len(feats[base]) and _is_numeric(feats[base][idx]):
                                    items.append((k, float(feats[base][idx])))
                    for k, v in items:
                        sums[ntype][k] = sums[ntype].get(k, 0.0) + v
                        sqs[ntype][k]  = sqs[ntype].get(k, 0.0) + v*v
                        counts[ntype][k] = counts[ntype].get(k, 0) + 1

    stats: Dict[str, Dict[str, Dict[str, float]]] = {}
    for ntype in tmp.include_nodes:
        means, stds = {}, {}
        for k, c in counts[ntype].items():
            mu = sums[ntype][k] / max(c, 1)
            var = max(sqs[ntype][k] / max(c, 1) - mu*mu, 0.0)
            sd = math.sqrt(var) if var > 0 else 1.0
            means[k] = mu; stds[k] = sd
        stats[ntype] = {"mean": means, "std": stds}
    return stats

# -------------------------
# Splitting & loaders
# -------------------------

def split_dataset(dataset: Dataset, train_ratio: float = 0.8, seed: int = 42):
    total_len = len(dataset)
    train_len = int(train_ratio * total_len)
    test_len  = total_len - train_len
    return random_split(dataset, [train_len, test_len], generator=torch.Generator().manual_seed(seed))

def create_dataloaders(
    directory: Union[str, Path],
    num_steps: int = 5,
    batch_size: int = 16,
    train_ratio: float = 0.8,
    include_nodes: Union[str, List[str]] = "all",
    include_edges: Union[str, List[str]] = "all",
    feature_keys: Optional[Dict[str, Union[str, List[str]]]] = None,
    expand_vectors: bool = True,
    target_spec: Union[str, Dict[str, Any]] = "none",
    episode_glob: str = "*.json",
    skip_short: bool = True,
    fill_missing: str = "zero",  # pass-through to dataset
):
    """
    Convenience wrapper:
      1) Build a TEMP dataset to discover schema
      2) Compute normalization on the TRAIN SPLIT ONLY (to avoid leakage)
      3) Rebuild datasets with norm applied
      4) Return PyG loaders + datasets + norm stats
    """
    # Step 1: temp dataset for indexing episodes & schema
    temp_ds = SceneSequenceDataset(directory, num_steps=num_steps, skip_short=skip_short,
                                   include_nodes=include_nodes, include_edges=include_edges,
                                   feature_keys=feature_keys, expand_vectors=expand_vectors,
                                   target_spec=target_spec, norm=None, episode_glob=episode_glob,
                                   fill_missing=fill_missing)
    # Step 2: split indices (sizes only), then compute stats on TRAIN subset’s episodes only
    train_idx, test_idx = split_dataset(temp_ds, train_ratio=train_ratio)
    # To compute stats, reuse directory; stats reflect same schema & selection.
    norm_stats = compute_norm_stats(directory, include_nodes=include_nodes, feature_keys=feature_keys,
                                    expand_vectors=expand_vectors, episode_glob=episode_glob,
                                    num_steps=num_steps, skip_short=skip_short)

    # Step 3: rebuild datasets with normalization
    full_ds = SceneSequenceDataset(directory, num_steps=num_steps, skip_short=skip_short,
                                   include_nodes=include_nodes, include_edges=include_edges,
                                   feature_keys=feature_keys, expand_vectors=expand_vectors,
                                   target_spec=target_spec, norm=norm_stats, episode_glob=episode_glob,
                                   fill_missing=fill_missing)

    # Reconstruct splits with same sizes
    train_len = len(train_idx)
    test_len  = len(full_ds) - train_len
    train_set, test_set = random_split(full_ds, [train_len, test_len], generator=torch.Generator().manual_seed(42))

    train_loader = PyGDataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader  = PyGDataLoader(test_set,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_set, test_set, norm_stats, full_ds.feature_schema


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
