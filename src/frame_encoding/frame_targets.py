import os
import json
import math
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm

def bin_vehicle_count(n):
    """Bins number of vehicles into 4 categories."""
    if n <= 0:
        return 0           # no vehicles
    elif n <= 2:
        return 1           # 1-2 vehicles
    elif n <= 5:
        return 2           # 3-5 vehicles
    elif n<= 10:
        return 3
    else:
        return 4           # >5 vehicles

def bin_distance(d):
    """Bins distance to closest agent (meters). None -> far bin."""
    if d is None or (isinstance(d, float) and math.isnan(d)):
        return 4           # treat missing as far
    if d <= 2.0:
        return 0
    elif d <= 5.0:
        return 1
    elif d <= 10.0:
        return 2
    elif d <= 20.0:
        return 3
    else:
        return 4

def bin_speed(s):
    """Bins ego speed (assumed in m/s). None -> 0 bin."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return 0
    if s <= 1.0:
        return 0
    elif s <= 5.0:
        return 1
    elif s <= 10.0:
        return 2
    elif s <= 15.0:
        return 3
    else:
        return 4

TURNING_MAP = {
    "left_turn": 1,
    "right_turn": 2,
    "straight": 3,
    "roundabout": 4,
    # everything else -> 0 (unknown / other)
}

def action_tag_to_category(action_tag):
    if not isinstance(action_tag, str):
        return 0
    return TURNING_MAP.get(action_tag, 0)

def _base_id(node_dict):
    """
    Try to get a stable ID for a node and strip trailing frame index.

    E.g. "vehicle_a_0" -> "vehicle_a"
    """
    raw = (
        node_dict.get("id")
        or node_dict.get("node_id")
        or node_dict.get("name")
    )
    if not isinstance(raw, str):
        return None

    parts = raw.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return raw

def extract_targets(graph_dir, tag_dir, output_dir, output_filename="targets.json"):
    """
    Iterate over all *_graph.json files in `graph_dir`, find the matching
    episode_XXXXXX.json in `tag_dir`, and extract:

      1. binned number of UNIQUE vehicles
      2. binned distance to closest vehicle
      3. binned distance to closest pedestrian
      4. binned ego speed
      5. numerical turning category

    Saves a single JSON file in `output_dir/output_filename` with:
      { "<episode_index>": [veh_count_bin, veh_dist_bin, ped_dist_bin,
                            ego_speed_bin, turn_cat], ... }
    """
    os.makedirs(output_dir, exist_ok=True)

    targets = {}

    for fname in tqdm(os.listdir(graph_dir)):
        if not fname.endswith("_graph.json"):
            continue

        graph_path = os.path.join(graph_dir, fname)

        # Extract numeric episode index from filename, e.g. "25001_graph.json"
        base = os.path.splitext(fname)[0]       # "25001_graph"
        episode_str = base.split("_")[0]        # "25001"
        try:
            episode_idx = int(episode_str)
        except ValueError:
            print(f"Skipping file with unexpected name: {fname}")
            continue

        # Build expected tag filename: "episode_025001.json"
        #tag_fname = f"episode_{episode_idx:06d}.json"
        tag_fname = f"episode_{episode_idx:06}.json"
        tag_path = os.path.join(tag_dir, tag_fname)

        if not os.path.exists(tag_path):
            print(f"Warning: no tag file for episode {episode_idx} ({tag_fname})")
            continue

        # Load graph JSON
        with open(graph_path, "r") as f:
            graph = json.load(f)

        nodes = graph.get("nodes", {})

        # --- 1) UNIQUE vehicles ---
        vehicles = nodes.get("vehicle", []) or []
        unique_vehicle_ids = set()
        veh_dists = []

        for v in vehicles:
            base_id = _base_id(v)
            if base_id is not None:
                unique_vehicle_ids.add(base_id)

            feats = v.get("features", {})
            d = feats.get("dist_to_ego", None)
            if d is not None:
                veh_dists.append(d)

        num_vehicles = len(unique_vehicle_ids)
        veh_count_bin = bin_vehicle_count(num_vehicles)

        # --- 2) Distance to closest vehicle (still min over all instances) ---
        min_veh_dist = min(veh_dists) if veh_dists else None
        veh_dist_bin = bin_distance(min_veh_dist)

        # --- 3) Distance to closest pedestrian + UNIQUE pedestrian count (if needed later) ---
        pedestrians = nodes.get("pedestrian", []) or []
        unique_ped_ids = set()
        ped_dists = []

        for p in pedestrians:
            base_id = _base_id(p)
            if base_id is not None:
                unique_ped_ids.add(base_id)

            feats = p.get("features", {})
            d = feats.get("dist_to_ego", None)
            if d is not None:
                ped_dists.append(d)

        min_ped_dist = min(ped_dists) if ped_dists else None
        ped_dist_bin = bin_distance(min_ped_dist)
        # If later you want a "pedestrian count bin", you'll have len(unique_ped_ids) here.

        # --- 4) Ego speed (mean across ego nodes) ---
        ego_nodes = nodes.get("ego", []) or []
        ego_speeds = []
        for e in ego_nodes:
            feats = e.get("features", {})
            s = feats.get("speed", None)
            if s is not None:
                ego_speeds.append(s)
        mean_ego_speed = sum(ego_speeds) / len(ego_speeds) if ego_speeds else None
        ego_speed_bin = bin_speed(mean_ego_speed)

        # --- 5) Turning behavior from tag file ---
        with open(tag_path, "r") as f:
            tags = json.load(f)
        action_tag = tags.get("action_tag", None)
        turn_cat = action_tag_to_category(action_tag)

        targets[str(episode_idx)] = [
            veh_count_bin,
            veh_dist_bin,
            ped_dist_bin,
            ego_speed_bin,
            turn_cat,
        ]

    out_path = os.path.join(output_dir, output_filename)
    with open(out_path, "w") as f:
        json.dump(targets, f, indent=2)

    print(f"Saved targets for {len(targets)} episodes to {out_path}")

def visualize_target_distribution(targets_path):
    with open(targets_path, "r") as f:
        targets = json.load(f)

    # convert dict -> list of vectors
    vectors = list(targets.values())

    # transpose to get each position separately
    # e.g. pos[0] = list of all veh_count_bin values
    positions = list(zip(*vectors))

    names = [
        "veh_count_bin",
        "veh_dist_bin",
        "ped_dist_bin",
        "ego_speed_bin",
        "turn_cat"
    ]

    for i, vals in enumerate(positions):
        print(f"\n=== {names[i]} ===")
        c = Counter(vals)
        for k in sorted(c.keys()):
            print(f"  {k}: {c[k]}")

        # quick histogram
        plt.figure(figsize=(4,3))
        plt.hist(vals, bins=range(min(vals), max(vals)+2), rwidth=0.8)
        plt.title(names[i])
        plt.xlabel("bin value")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()    