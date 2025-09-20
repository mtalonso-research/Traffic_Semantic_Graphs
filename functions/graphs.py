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

from functions.utils import load_and_restore_parquet

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
    processed_frame_dir='../data/processed_frames/L2D',      # ORIGINAL JSONs
    lane_frame_dir='../data/processed_frames/L2D_lanes',                # NEW: lane JSON root 
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
        try:
            parquet_path, aux_path, lane_path = directory_lookup(
                ep_num,
                source_data_dir,
                processed_frame_dir,
                lane_frame_dir
            )
            graph_dict = build_graph_json(
                parquet_path=parquet_path,
                aux_paths=aux_path,            # original per-frame JSONs
                lane_paths=lane_path,          # NEW: lane per-frame JSONs
                image_lookup=image_lookup,
                graph_id=f"scene_{ep_num:06d}"
            )

            json_path = os.path.join(output_dir, f"{ep_num}_graph.json")
            with open(json_path, 'w') as f:
                json.dump(graph_dict, f, indent=2)
        except Exception as e:
            print(f'Problems with episode {ep_num}: {e}')

def directory_lookup(ep_num,source_data_dir,processed_frame_dir,lane_frame_dir):
    parquet_path = os.path.join(source_data_dir,f'episode_{ep_num:06d}.parquet')
    df = load_and_restore_parquet(parquet_path)
    aux_path = {}
    for frame_idx in range(len(df)):
        aux_path[str(frame_idx)] = os.path.join(processed_frame_dir,f"Episode{ep_num:06d}/front_left_Annotations",f'frame_{frame_idx:05d}.json')
    # NEW lane per-frame JSONs
    lane_path = {}
    for frame_idx in range(len(df)):
        lane_path[str(frame_idx)] = os.path.join(
            lane_frame_dir,
            f"Episode{ep_num:06d}",
            "front_left_Enhanced_LaneAnnotations",        # <-- adjust to your actual folder name
            f'frame_{frame_idx:05d}.json'
        )
    return parquet_path,aux_path,lane_path

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
    lane_paths: Optional[Dict[str, str | Path]] = None,   # NEW: separate lane JSONs
) -> Dict[str, Any]:

    parquet_path = Path(parquet_path)
    graph_id = graph_id or parquet_path.stem

    df = load_and_restore_parquet(parquet_path)

    # Original per-frame JSONs (speed/depth/etc.)
    aux_data = _load_aux_data(aux_paths)

    # NEW: lane per-frame JSONs (lane detection + vehicle lane classification)
    lane_data = _load_aux_data(lane_paths) if lane_paths else {}

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
                **({f"lane_{k}": str(v) for k, v in (lane_paths or {}).items()}),
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
        # Keep object list from the ORIGINAL aux (so we retain speed/depth/etc.)
        frame_record = aux_data.get(frame_id, {}) or {}
        frame_annotations = frame_record.get("annotations") or []

        # Lane info may exist ONLY in the new lane JSONs — index by track_id
        lane_record = lane_data.get(frame_id, {}) or {}
        vlc = lane_record.get("vehicle_lane_classification") or {}
        lane_index = { (v.get("track_id") or ""): v for v in (vlc.get("vehicles") or []) }

        for obj in frame_annotations:
            attrs = (obj.get("attributes") or {})
            obj_class = (attrs.get("class") or "").lower()
            track_id = obj.get("track_id")
            if not track_id or not obj_class:
                continue

            node_id = f"{track_id}_{frame_id}"

            if obj_class == "car":
                speed_info = (attrs.get("speed_info") or {})

                # Pull lane-related fields. They may be present in the ORIGINAL attributes (rare for you),
                # but we fall back to lane_index built from the NEW lane JSONs.
                lane_entry = lane_index.get(track_id, {})  # e.g., {"lane_classification": "...", "overlap_ratio": ...}

                lane_classification = (
                    attrs.get("lane_classification")
                    or lane_entry.get("lane_classification")
                )
                lane_overlap_ratio = attrs.get("lane_overlap_ratio")
                if lane_overlap_ratio is None:
                    lane_overlap_ratio = lane_entry.get("overlap_ratio")
                in_ego_lane = attrs.get("in_ego_lane")
                ego_lane_available = attrs.get("ego_lane_available")

                veh_features = {
                    "speed_ms":     speed_info.get("speed_ms"),
                    "speed_kmh":    speed_info.get("speed_kmh"),
                    "velocity_ms":  speed_info.get("velocity_vector_ms"),
                    "velocity_kmh": speed_info.get("velocity_vector_kmh"),
                    "accel_ms2":    speed_info.get("acceleration_vector_ms")
                                    or speed_info.get("accel_vector_ms")
                                    or speed_info.get("acceleration_ms2"),

                    # --- NEW lane features (from lane JSONs) ---
                    "lane_classification": lane_classification,      # e.g. "in_lane", "out_of_lane_left"
                    "lane_overlap_ratio":  lane_overlap_ratio,       # float [0,1]
                    "in_ego_lane":         in_ego_lane,              # bool if provided
                    "ego_lane_available":  ego_lane_available,       # bool if provided
                }

                # Optional: include distance if the lane JSON provides it
                if lane_entry.get("distance_m") is not None:
                    veh_features["distance_m"] = lane_entry["distance_m"]

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

def json_graph_to_networkx(graph_data):
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

            node_id = f"{unique_id}"
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
    for frame_key, graph_data in graphs_by_frame.items():
        G_frame, node_mapping = json_graph_to_networkx(graph_data)
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
