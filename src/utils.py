import os
import numpy as np
import pandas as pd
import json
from PIL import Image
import matplotlib.pyplot as plt
import math
from collections import defaultdict

def _parse_t_from_id(node_id: str):
    try:
        base, t = node_id.rsplit('_', 1)
        return int(t)
    except Exception:
        return None

def _collect_by_time(nodes_list, required_type=None):
    by_t = defaultdict(list)
    for n in nodes_list:
        if required_type and n.get("features", {}).get("type") != required_type:
            # allow categories like 'bicycle' under vehicles; gate by 'type' when asked
            pass
        t = _parse_t_from_id(n["id"])
        if t is None:
            # Put into a special bucket if needed; here we skip since we score per frame
            continue
        m = {"id": n["id"], "t": t, **n}  # keep original keys; add parsed t at top-level
        by_t[t].append(m)
    return by_t

def _index_by_exact_id(nodes_list):
    return {n["id"]: n for n in nodes_list}

def extract_frames(episode: dict):
    nodes = episode["nodes"]
    meta = episode.get("metadata", {})
    n_frames_meta = meta.get("frames")

    # Ego & env are easy: one per t with exact ids ego_t / env_t
    ego_idx = _index_by_exact_id(nodes.get("ego", []))
    env_idx = _index_by_exact_id(nodes.get("environment", []))

    # Bulk agents: group by parsed timestep
    veh_by_t = _collect_by_time(nodes.get("vehicle", []))         # includes bicycles
    ped_by_t = _collect_by_time(nodes.get("pedestrian", []))
    obj_by_t = _collect_by_time(nodes.get("object", []))

    # Determine frame indices: trust metadata if present; else infer from keys
    ts = set()
    if n_frames_meta is not None:
        # Try [0..frames-1]; keep only those that actually have ego/env
        ts = set(range(n_frames_meta))
    else:
        # Infer from any of the collections
        for d in (veh_by_t, ped_by_t, obj_by_t):
            ts |= set(d.keys())
        # also consider env/ego ids of the form env_k / ego_k
        for k in env_idx.keys():
            t = _parse_t_from_id(k)
            if t is not None: ts.add(t)
        for k in ego_idx.keys():
            t = _parse_t_from_id(k)
            if t is not None: ts.add(t)

    frames = []
    for t in sorted(ts):
        ego = ego_idx.get(f"ego_{t}")
        env = env_idx.get(f"env_{t}")

        # Minimal gating: without ego, we can’t compute risk; skip
        if ego is None:
            continue

        # Vehicles: attach category (some are bicycles) for later semantics
        def _pack(lst):
            out = []
            for n in lst:
                f = n.get("features", {})
                out.append({
                    "id": n["id"],
                    "category": f.get("category", f.get("type", "unknown")),
                    "type": n.get("type", "unknown"),
                    "features": f
                })
            return out

        vehicles = _pack(veh_by_t.get(t, []))
        pedestrians = _pack(ped_by_t.get(t, []))
        objects = _pack(obj_by_t.get(t, []))

        # Time fields
        ts_raw = env.get("features", {}).get("timestamp_raw") if env else None
        time_rel = env.get("features", {}).get("time_rel_s") if env else None

        frame = {
            "t": t,
            "time_rel_s": time_rel,
            "timestamp_raw": ts_raw,
            "ego": {"id": ego["id"], "features": ego["features"]},
            "vehicles": vehicles,
            "pedestrians": pedestrians,
            "objects": objects,
            "environment": {"id": env["id"], "features": env["features"]} if env else None,
            "normalized": {
                "ego_pose": None,
                "vehicles": None,
                "pedestrians": None,
            }
        }
        frames.append(frame)

    return frames

def display_frames(frames_dir):
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    images = [Image.open(os.path.join(frames_dir, f)) for f in frame_files]
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    if len(images) == 1:
        axes = [axes]  
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def get_chunk_num(ep_num):
    if ep_num < 1000: chunk = 0
    elif ep_num < 2000: chunk = 1
    elif ep_num < 3000: chunk = 2
    elif ep_num < 4000: chunk = 3
    elif ep_num < 5000: chunk = 4
    elif ep_num < 6000: chunk = 5
    elif ep_num < 7000: chunk = 6
    elif ep_num < 8000: chunk = 7
    elif ep_num < 9000: chunk = 8
    else: chunk = 9
    return chunk

def prepare_and_save_parquet(df, path):
    json_columns = []

    # Helper: check if value needs serialization
    def needs_serialization(x):
        return isinstance(x, (list, dict))

    # Helper: safely serialize objects (including NumPy types)
    def safe_serialize(x):
        if isinstance(x, (list, dict)):
            def convert(obj):
                if isinstance(obj, np.generic):  # convert np.float32, np.int64, etc.
                    return obj.item()
                elif isinstance(obj, list):
                    return [convert(i) for i in obj]
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                else:
                    return obj
            return json.dumps(convert(x))
        else:
            return x

    # Process columns
    for col in df.columns:
        if df[col].apply(needs_serialization).any():
            df[col] = df[col].apply(safe_serialize)
            json_columns.append(col)

    # Save DataFrame to Parquet
    df.to_parquet(path, index=False)

    # Save metadata of serialized columns
    meta_path = os.path.splitext(path)[0] + "_json_cols.json"
    with open(meta_path, "w") as f:
        json.dump(json_columns, f)


    for col in df.columns:
        if df[col].apply(needs_serialization).any():
            df[col] = df[col].apply(lambda x: json.dumps(x) if needs_serialization(x) else x)
            json_columns.append(col)

    # Save DataFrame
    df.to_parquet(path, index=False)

    # Save metadata (which columns were JSON-encoded)
    meta_path = os.path.splitext(path)[0] + "_json_cols.json"
    with open(meta_path, "w") as f:
        json.dump(json_columns, f)

def load_and_restore_parquet(path):
    df = pd.read_parquet(path)

    # Load JSON metadata if it exists
    meta_path = os.path.splitext(path)[0] + "_json_cols.json"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            json_columns = json.load(f)

        # Restore each column using json.loads()
        for col in json_columns:
            df[col] = df[col].apply(
                lambda x: json.loads(x) if isinstance(x, str) and x.startswith(('[', '{')) else x
            )

    return df

def denormalize_steering_angle(z, std_estimate=45):
    """
    Estimate original driving-related angle (in degrees)
    from a normalized value z, assuming:
    - original mean = 0°
    - original std = 30–45° (default 45°)
    """
    return z * std_estimate
    
def clean_list(input_list):
    output = []
    buffer = []
    inside_quotes = False

    for item in input_list:
        if item == '[' or item == ']':
            continue
        elif item == '"':
            inside_quotes = not inside_quotes
            if not inside_quotes and buffer:
                output.append(''.join(buffer))
                buffer = []
        elif inside_quotes:
            buffer.append(item)
        else:
            output.append(item)
    
    return output

def load_episode(chunk, ep_num):
    path = f"L2D_downloaded/processed_data/chunk-{chunk:03d}/episode_{ep_num:06d}.parquet"
    try:
        df = load_and_restore_parquet(path)
        df['chunk'] = chunk
        df['episode'] = ep_num
        return df
    except Exception as e:
        #print(f"Failed to load chunk {chunk}, episode {ep_num}: {e}")
        return None
    
def normalize_width(val):
    """Normalize width values like '2.5 m' or 2.5 to float."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 2)
    if isinstance(val, str):
        try:
            return round(float(val.replace('m', '').strip()), 2)
        except:
            return None
    return None

def flatten_and_clean_values(col, series):
    values = set()
    for item in series.dropna():
        if isinstance(item, list):
            values.update(item)
        elif isinstance(item, str) and ',' in item:
            # Handle comma-separated strings like "roundabout, traffic_signal"
            split_items = [i.strip() for i in item.split(',') if i.strip()]
            values.update(split_items)
        else:
            values.add(item)
    return values

