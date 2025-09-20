import os
import json
import math
import datetime as dt
from datetime import datetime
from pathlib import Path
import calendar
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import shutil

# -------------------------------------------------------------
# Utility helpers
# -------------------------------------------------------------

def kmh_to_ms(speed_kmh: float) -> float:
    """Convert km/h → m/s. Return 0.0 if input is None."""
    if speed_kmh is None:
        return 0.0
    return float(speed_kmh) * (1000.0 / 3600.0)

def compute_speed_from_velocity(vx: float, vy: float, vz: float) -> float:
    """Compute magnitude of velocity vector."""
    return math.sqrt((vx or 0.0)**2 + (vy or 0.0)**2 + (vz or 0.0)**2)

def compute_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance in 2D. Returns 0.0 if any input missing."""
    if None in (x1, y1, x2, y2):
        return 0.0
    return math.hypot(x1 - x2, y1 - y2)

def extract_time_features(timestamp_raw: int):
    """
    Given a raw timestamp (µs since epoch), return dict with month, day_of_week, time_of_day (all numeric).
    """
    if timestamp_raw is None:
        return {"month": -1, "day_of_week": -1, "time_of_day": -1}

    dt_obj = dt.datetime.utcfromtimestamp(timestamp_raw / 1e6)
    return {
        "month": dt_obj.month,  # 1–12
        "day_of_week": dt_obj.weekday(),  # Monday=0
        "time_of_day": dt_obj.hour * 3600 + dt_obj.minute * 60 + dt_obj.second,  # seconds since midnight
    }

# -------------------------------------------------------------
# Main processing function
# -------------------------------------------------------------

def standardize_graph(data: dict, dataset: str) -> dict:
    """
    Standardize a single graph dict according to dataset rules.
    dataset: "l2d" or "nuplan"
    """
    nodes_out = {"ego": [], "vehicle": [], "pedestrian": [], "object": [], "environment": []}

    # ------------------ Ego ------------------
    for ego in data.get("nodes", {}).get("ego", []):
        feats = ego.get("features", {})
        if dataset == "l2d":
            speed = kmh_to_ms(feats.get("speed"))
            vx, vy, vz = -1.0, -1.0, -1.0
        elif dataset == "nuplan":
            vx = feats.get("vx", 0.0)
            vy = feats.get("vy", 0.0)
            vz = feats.get("vz", 0.0)
            speed = compute_speed_from_velocity(vx, vy, vz)
        else:
            raise ValueError(f"Unknown dataset {dataset}")

        nodes_out["ego"].append({
            "id": ego.get("id"),
            "features": {
                "speed": speed,
                "vx": vx,
                "vy": vy,
                "vz": vz,
            }
        })

    # Need ego position for distance_to_ego (nuplan only)
    ego_x, ego_y = None, None
    if dataset == "nuplan":
        ego_feats = data.get("nodes", {}).get("ego", [{}])[0].get("features", {})
        ego_x, ego_y = ego_feats.get("x"), ego_feats.get("y")

    # ------------------ Vehicles ------------------
    edge_distances = {}
    if dataset == "l2d":
        for edge in data.get("edges", {}).get("ego_to_vehicle", []):
            edge_distances[(edge["source"], edge["target"])] = edge.get("features", {}).get("distance", 0.0)

    for veh in data.get("nodes", {}).get("vehicle", []):
        feats = veh.get("features", {})
        if dataset == "l2d":
            speed = feats.get("speed", 0.0)
            vx = feats.get("velocity_dx", 0.0)
            vy = feats.get("velocity_dy", 0.0)
            vz = feats.get("velocity_dz", 0.0) if "velocity_dz" in feats else 0.0
            dist = next((d for (src, tgt), d in edge_distances.items() if tgt == veh.get("id")), 0.0)
        else:  # nuplan
            vx = feats.get("vx", 0.0)
            vy = feats.get("vy", 0.0)
            vz = feats.get("vz", 0.0)
            speed = compute_speed_from_velocity(vx, vy, vz)
            x, y = feats.get("x"), feats.get("y")
            dist = compute_distance(x, y, ego_x, ego_y)

        nodes_out["vehicle"].append({
            "id": veh.get("id"),
            "features": {
                "speed": speed,
                "vx": vx,
                "vy": vy,
                "vz": vz,
                "distance_to_ego": dist
            }
        })

    # ------------------ Pedestrians ------------------
    for ped in data.get("nodes", {}).get("pedestrian", []):
        feats = ped.get("features", {})
        if dataset == "l2d":
            dist = feats.get("distance_to_ego", 0.0)
        else:
            x, y = feats.get("x"), feats.get("y")
            dist = compute_distance(x, y, ego_x, ego_y)

        nodes_out["pedestrian"].append({
            "id": ped.get("id"),
            "features": {"distance_to_ego": dist}
        })

    # ------------------ Objects ------------------
    for obj in data.get("nodes", {}).get("object", []):
        feats = obj.get("features", {})
        if dataset == "l2d":
            dist = feats.get("distance_to_ego", 0.0)
        else:
            x, y = feats.get("x"), feats.get("y")
            dist = compute_distance(x, y, ego_x, ego_y)

        nodes_out["object"].append({
            "id": obj.get("id"),
            "features": {"distance_to_ego": dist}
        })

    # ------------------ Environment ------------------
    for env in data.get("nodes", {}).get("environment", []):
        feats = env.get("features", {})
        if dataset == "l2d":
            # --- Month (e.g., "January") ---
            month_str = feats.get("month", "")
            if isinstance(month_str, str):
                try:
                    month = list(calendar.month_name).index(month_str)
                except ValueError:
                    month = -1
            else:
                month = -1

            # --- Day of week (e.g., "Monday") ---
            dow_str = feats.get("day_of_week", "")
            if isinstance(dow_str, str):
                try:
                    # Monday=0, Sunday=6
                    day_of_week = list(calendar.day_name).index(dow_str)
                except ValueError:
                    day_of_week = -1
            else:
                day_of_week = -1

            # --- Time of day (e.g., "11:34:38") ---
            tod_str = feats.get("time_of_day", "")
            if isinstance(tod_str, str):
                try:
                    tod = datetime.strptime(tod_str, "%H:%M:%S").time()
                    time_of_day = tod.hour * 3600 + tod.minute * 60 + tod.second
                except ValueError:
                    time_of_day = -1
            else:
                time_of_day = -1

        else:
            ts_raw = feats.get("timestamp_raw")
            time_feats = extract_time_features(ts_raw)
            month = time_feats["month"]
            day_of_week = time_feats["day_of_week"]
            time_of_day = time_feats["time_of_day"]

        nodes_out["environment"].append({
            "id": env.get("id"),
            "features": {
                "month": month,
                "day_of_week": day_of_week,
                "time_of_day": time_of_day
            }
        })

    return {
        "nodes": nodes_out,
        "edges": data.get("edges", {}),
        "metadata": data.get("metadata", {})
    }

# -------------------------------------------------------------
# Batch runner
# -------------------------------------------------------------

def process_dataset(src_dir: str, dst_dir: str, dataset: str, overwrite=False):
    """
    Process all .json files in src_dir into dst_dir.
    dataset: "l2d" or "nuplan"
    """
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    processed, skipped, errors = 0, 0, 0

    for p in tqdm(sorted(src.glob("*.json"))):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)

            out_graph = standardize_graph(data, dataset)

            out_path = dst / p.name
            if out_path.exists() and not overwrite:
                skipped += 1
                continue

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out_graph, f, indent=2)

            processed += 1

        except Exception as e:
            errors += 1
            print(f"[ERROR] {p.name}: {e}")

    print(f"Done {dataset}. Wrote: {processed}, skipped: {skipped}, errors: {errors}")


def copy_files(file_list, src_dir, dest_dir):
    """
    Copy a list of files from src_dir to dest_dir without deleting originals.
    
    Args:
        file_list (list): list of file names (not full paths).
        src_dir (str): directory where the files currently reside.
        dest_dir (str): directory where files should be copied to.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir, exist_ok=True)

    for fname in tqdm(file_list):
        src_path = os.path.join(src_dir, fname)
        dest_path = os.path.join(dest_dir, fname)

        if not os.path.exists(src_path):
            print(f"Skipping {fname} (not found in source dir)")
            continue

        try:
            shutil.copy2(src_path, dest_path)  # preserves metadata
        except Exception as e:
            print(f"Failed to copy {fname}: {e}")


def find_non_uncertain_files(directory_path: str):
    """
    Return a list of JSON filenames where 'unknown' or 'uncertain'
    is NOT the only action tag.
    
    Excludes files with only ['unknown'] or only ['uncertain'].
    Includes files where:
      - neither tag is present
      - one of them is present along with other tags
    """
    result_files = []

    for filename in os.listdir(directory_path):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(directory_path, filename)

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {filename} (error: {e})")
            continue

        # Normalize action tags
        tags = []
        if "action_tags" in data:
            tags = data["action_tags"]
        elif "action_tag" in data:
            tags = [data["action_tag"]]

        if isinstance(tags, str):  # edge case: single string
            tags = [tags]

        # Check condition
        if tags in [["unknown"], ["uncertain"]]:
            continue  # skip: only unknown/uncertain
        else:
            result_files.append(filename)

    return result_files


def find_non_stationary_files(directory_path: str):
    """
    Return a list of JSON filenames where 'stationary' is NOT the only action tag.
    Includes cases where:
      - 'stationary' is absent
      - 'stationary' is present with other tags
    Excludes files where the only tag is ['stationary'].
    """
    result_files = []

    for filename in os.listdir(directory_path):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(directory_path, filename)

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {filename} (error: {e})")
            continue

        # Normalize action tags
        tags = []
        if "action_tags" in data:
            tags = data["action_tags"]
        elif "action_tag" in data:
            tags = [data["action_tag"]]

        if isinstance(tags, str):  # edge case: single string
            tags = [tags]

        # Check condition
        if tags and tags == ["stationary"]:
            continue  # skip: only stationary
        else:
            result_files.append(filename)

    return result_files


def clean_json_tags(directory_path: str):
    """
    Go through all JSON files in a directory, apply substitutions to tag values,
    and overwrite the files in place.
    
    Substitutions:
      - "unknown" -> "uncertain"
      - "go_straight" -> "straight"
      - "pedestrian_ara" -> "ped_area"
      - "ped_crossing" -> "ped_area"
    """
    substitutions = {
        "unknown": "uncertain",
        "go_straight": "straight",
        "pedestrian_ara": "ped_area",
        "ped_crossing": "ped_area",
    }

    for filename in tqdm(os.listdir(directory_path)):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(directory_path, filename)

        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"Skipping {filename} (error: {e})")
            continue

        modified = False

        # Traverse all keys/values in the JSON
        for key, value in data.items():
            if isinstance(value, str):
                if value in substitutions:
                    data[key] = substitutions[value]
                    modified = True
            elif isinstance(value, list):
                new_list = []
                for item in value:
                    if isinstance(item, str) and item in substitutions:
                        new_list.append(substitutions[item])
                        modified = True
                    else:
                        new_list.append(item)
                data[key] = new_list

        if modified:
            try:
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=4)
                #print(f"Cleaned: {filename}")
            except Exception as e:
                print(f"Failed to write {filename} (error: {e})")

def filter_nuplan_episodes(origin_tag_path,origin_graph_path,graph_path,tag_path):
    non_stationary_list = find_non_stationary_files(origin_tag_path)
    non_unceretain_list = find_non_uncertain_files(origin_tag_path)
    final_list = list(set(non_stationary_list) & set(non_unceretain_list))
    copy_files(final_list,origin_tag_path,tag_path)
    copy_files(final_list,origin_graph_path,graph_path)
