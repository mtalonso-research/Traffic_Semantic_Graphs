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

from src.utils import load_and_restore_parquet

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

def generate_graphs(
    min_ep,
    max_ep=-1,
    source_data_dir='../data/processed/L2D',
    processed_frame_dir='../data/annotations/L2D',      # Filtered annotation JSONs
    output_dir='../data/graphical/L2D'):

    # Ensure target directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    if not isinstance(min_ep, list):
        if max_ep == -1: 
            max_ep = min_ep + 1
        iterable = range(min_ep,max_ep)
    else:
        iterable = min_ep

    for ep_num in tqdm(iterable):
        #try:
            parquet_path, aux_path = directory_lookup(
                ep_num,
                source_data_dir,
                processed_frame_dir
            )
            graph_dict = build_graph_json(
                parquet_path=parquet_path,
                aux_paths=aux_path,            # original per-frame JSONs
                image_lookup=image_lookup,
                graph_id=f"scene_{ep_num:06d}"
            )

            json_path = os.path.join(output_dir, f"{ep_num}_graph.json")
            with open(json_path, 'w') as f:
                json.dump(graph_dict, f, indent=2)
        #except Exception as e:
        #    print(f'Problems with episode {ep_num}: {e}')

def directory_lookup(ep_num,source_data_dir,processed_frame_dir):
    parquet_path = os.path.join(source_data_dir,f'episode_{ep_num:06d}.parquet')
    df = load_and_restore_parquet(parquet_path)
    aux_path = {}
    for frame_idx in range(len(df)):
        aux_path[str(frame_idx)] = os.path.join(processed_frame_dir, f"Episode{ep_num:06d}", f'frame_{frame_idx:05d}.json')
    return parquet_path,aux_path

def image_lookup(frame_idx: str) -> str:
    return f"L2D_data/camera/chunk-000/episode_000000/observation.images.front_left/frame_{frame_idx:05}.jpg"

########################################
from pathlib import Path
from typing import Dict, Any, Callable, Optional
import json

def build_graph_json(
    parquet_path: str | Path,
    aux_paths: Dict[str, str | Path],
    image_lookup: Callable[[str], str],
    output_path: Optional[str | Path] = None,
    graph_id: str | None = None,
) -> Dict[str, Any]:

    parquet_path = Path(parquet_path)
    graph_id = graph_id or parquet_path.stem

    df = load_and_restore_parquet(parquet_path)

    # Original per-frame JSONs (speed/depth/etc.)
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
        "metadata": {
            "graph_id": graph_id,
            "source_files": {
                "parquet": str(parquet_path),
                **{k: str(v) for k, v in (aux_paths or {}).items()},
            },
        },
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
            "month": row.get("month"),
            "day_of_week": row.get("day_of_week"),
            "time_of_day": row.get("time_of_day"),
            "conditions": row.get("observation.state.conditions"),
            "lanes": row.get("observation.state.lanes"),
            "lighting": row.get("observation.state.lighting"),
            "max_speed": row.get("observation.state.max_speed"),
            "precipitation": row.get("observation.state.precipitation"),
            "road": row.get("observation.state.road"),
            "surface": row.get("observation.state.surface"),
            "traffic_controls": row.get("traffic_controls"),
            "traffic_features": row.get("traffic_features"),
        }
        graph["nodes"]["environment"].append({"id": env_id, "features": env_features})

        # --- Entities: vehicles & pedestrians (UPDATED) ---
        frame_record = aux_data.get(frame_id, {}) or {}
        frame_annotations = frame_record.get("annotations") or []

        for obj in frame_annotations:
            attrs = (obj.get("attributes") or {})
            obj_class = (attrs.get("class") or "").lower()
            track_id = obj.get("track_id")
            if not track_id or not obj_class:
                continue

            node_id = f"{track_id}_{frame_id}"

            if track_id.startswith("Ped"):
                speed_info = (attrs.get("speed_info") or {})
                depth_stats = (attrs.get("depth_stats") or {})
                dist_to_ego = depth_stats.get("mean_depth")

                ped_features = {
                    "speed_ms":     speed_info.get("speed_ms"),
                    "speed_kmh":    speed_info.get("speed_kmh"),
                    "velocity_ms":  speed_info.get("velocity_vector_ms"),
                    "velocity_kmh": speed_info.get("velocity_vector_kmh"),
                    "accel_ms2":    speed_info.get("acceleration_vector_ms")
                                    or speed_info.get("accel_vector_ms")
                                    or speed_info.get("acceleration_ms2"),
                    "dist_to_ego": dist_to_ego,
                }

                # Prune None to keep JSON tidy
                ped_features = {k: v for k, v in ped_features.items() if v is not None}

                graph["nodes"]["pedestrian"].append({"id": node_id, "features": ped_features})
            elif obj_class in ["car", "truck", "bus", "van", "motorcycle", "bicycle"]:
                speed_info = (attrs.get("speed_info") or {})
                depth_stats = (attrs.get("depth_stats") or {})
                dist_to_ego = depth_stats.get("mean_depth")

                veh_features = {
                    "speed_ms":     speed_info.get("speed_ms"),
                    "speed_kmh":    speed_info.get("speed_kmh"),
                    "velocity_ms":  speed_info.get("velocity_vector_ms"),
                    "velocity_kmh": speed_info.get("velocity_vector_kmh"),
                    "accel_ms2":    speed_info.get("acceleration_vector_ms")
                                    or speed_info.get("accel_vector_ms")
                                    or speed_info.get("acceleration_ms2"),
                    "dist_to_ego": dist_to_ego,
                }

                # Prune None to keep JSON tidy
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

            if tid.startswith("Ped"):
                graph["edges"]["ego_to_pedestrian"].append({
                    "source": ego_node["id"],
                    "target": nid,
                    "features": {"distance": dist},
                })
            elif cls in ["car", "truck", "bus", "van", "motorcycle", "bicycle"]:
                graph["edges"]["ego_to_vehicle"].append({
                    "source": ego_node["id"],
                    "target": nid,
                    "features": {"distance": dist},
                })
            elif cls == "pedestrian":
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
        if not path.exists():
            # Warn but skip if the file is missing
            print(f"[Warning] Missing aux file: {path}")
            continue

        try:
            if path.suffix == ".npz":
                loaded[name] = np.load(path)
            elif path.suffix == ".json":
                with path.open() as f:
                    loaded[name] = json.load(f)
            else:
                print(f"[Warning] Unsupported aux file type: {path}")
        except Exception as e:
            print(f"[Warning] Failed to load {path}: {e}")
            continue

    return loaded
