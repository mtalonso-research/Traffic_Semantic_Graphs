import json
import os
from typing import List
import sqlite3
import pandas as pd
from tqdm import tqdm
import ast

def process_tags(raw_tags):
    """
    Processes a list of raw tags and categorizes them.

    Args:
        raw_tags (list): A list of raw tags.

    Returns:
        dict: A dictionary of processed tags.
    """
    # Step 1: Initialize processed tags dictionary
    processed = {
        "action_tag": "unmarked",
        "traffic_control_tag": "unmarked",
        "road_feature_tags": [],
    }

    # Step 2: Define keywords for each tag category
    action_keywords = {
        "straight": ["straight"],
        "left_turn": ["left_turn", "left"],
        "right_turn": ["right_turn", "right"],
        "u_turn": ["u_turn"],
        "lane_change": ["lane_change"],
    }

    traffic_control_keywords = {
        "traffic_signal": ["traffic_signal", "traffic_light"],
        "stop_sign": ["stop_sign", "stop"],
        "roundabout": ["roundabout"],
        "yield": ["yield"],
    }

    road_feature_keywords = {
        "pedestrian_area": ["pedestrian_area", "pedestrian"],
        "crosswalk": ["crosswalk"],
        "on_ramp": ["on_ramp", "ramp"],
        "off_ramp": ["off_ramp"],
    }

    # Step 3: Process each raw tag
    for tag in raw_tags:
        tag_lower = tag.lower()

        # Step 3a: Process action tag
        for action, keywords in action_keywords.items():
            if any(keyword in tag_lower for keyword in keywords):
                processed["action_tag"] = action
                break

        # Step 3b: Process traffic control tag
        for control, keywords in traffic_control_keywords.items():
            if any(keyword in tag_lower for keyword in keywords):
                processed["traffic_control_tag"] = control
                break

        # Step 3c: Process road feature tags
        for feature, keywords in road_feature_keywords.items():
            if any(keyword in tag_lower for keyword in keywords):
                processed["road_feature_tags"].append(feature)

    # Step 4: Remove duplicates from road_feature_tags
    processed["road_feature_tags"] = list(set(processed["road_feature_tags"]))

    return processed

def extract_environment_tags(graph_file_path):
    """
    Extracts environment tags from a graph file.

    Args:
        graph_file_path (str): The path to the graph file.

    Returns:
        list: A list of environment tags.
    """
    # Step 1: Initialize environment tags list
    env_tags = []

    if not os.path.exists(graph_file_path):
        return env_tags

    # Step 2: Load graph data
    with open(graph_file_path, "r") as f:
        graph_data = json.load(f)

    env_nodes = graph_data.get("nodes", {}).get("environment", [])
    if not env_nodes:
        return env_tags

    # Step 3: Extract features from the first environment node
    features = env_nodes[0].get("features", {})

    # Step 4: Extract day/night tag
    if "is_daylight" in features:
        env_tags.append("day" if features["is_daylight"] else "night")

    # Step 5: Extract weather tag
    if "weather_description" in features:
        weather = features["weather_description"].lower()
        if "rain" in weather:
            env_tags.append("rain")
        elif "cloudy" in weather or "overcast" in weather:
            env_tags.append("cloudy")
        elif "snow" in weather:
            env_tags.append("snow")
        elif "fog" in weather:
            env_tags.append("fog")

    # Step 6: Extract weekend/weekday tag
    if "day_of_week" in features:
        day = features["day_of_week"].lower()
        if day in ["saturday", "sunday"]:
            env_tags.append("weekend")
        else:
            env_tags.append("weekday")

    # Step 7: Extract winter conditions tag
    if "month" in features:
        month = features["month"]
        if isinstance(month, str):
            month = month.lower()
            if month in ["december", "january", "february"]:
                env_tags.append("winter_conditions_possible")
        elif isinstance(month, int):
            if month in [12, 1, 2]:
                env_tags.append("winter_conditions_possible")

    return list(set(env_tags))


def extract_tags(data_root, output_dir, graph_dir):
    """
    Extracts scenario tags from nuPlan databases and saves them as JSON files.

    Args:
        data_root (str): The root directory of the nuPlan dataset.
        output_dir (str): The directory where the tag JSON files will be saved.
        graph_dir (str): The directory where the graph JSON files and file_mapping.csv are located.
    """
    # Step 1: Load mapping file
    mapping_file = os.path.join(graph_dir, "file_mapping.csv")
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found at {mapping_file}")
        return

    mappings = pd.read_csv(mapping_file)
    os.makedirs(output_dir, exist_ok=True)

    # Step 2: Process each mapping
    for _, row in tqdm(mappings.iterrows(), total=len(mappings)):
        json_file_name = row["json_file_name"]
        db_name = row["db_name"]
        scene_token_str = row["scene_token"]

        try:
            scene_token_bytes = ast.literal_eval(scene_token_str)
            scene_token_hex = scene_token_bytes.hex()
        except (ValueError, SyntaxError):
            scene_token_hex = scene_token_str
        except:
            print(f'Could not Process {json_file_name}')
            continue

        db_path = os.path.join(data_root, db_name)
        if not os.path.exists(db_path):
            print(f"Warning: Database file not found: {db_path}")
            continue

        # Step 3: Connect to database and get raw tags
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            scene_token_bytes = bytes.fromhex(scene_token_hex)
            cursor.execute("SELECT type FROM scenario_tag WHERE lidar_pc_token IN (SELECT token FROM lidar_pc WHERE scene_token=?)", (scene_token_bytes,))
            raw_tags = [row[0] for row in cursor.fetchall()]
        except:
            print(f'Could not Process {json_file_name}')
            continue

        conn.close()

        if raw_tags:
            # Step 4: Process raw tags
            processed_tags = process_tags(raw_tags)

            # Step 5: Extract environment tags
            graph_file_path = os.path.join(graph_dir, json_file_name)
            environment_tags = extract_environment_tags(graph_file_path)

            # Step 6: Combine all tags
            output_data = {
                "raw_tags": list(set(raw_tags)),
                "action_tag": processed_tags["action_tag"],
                "traffic_control_tag": processed_tags["traffic_control_tag"],
                "road_feature_tags": processed_tags["road_feature_tags"],
                "environment_tags": environment_tags,
            }

            # Step 7: Save tags to JSON file
            output_filename = os.path.join(output_dir, json_file_name)
            with open(output_filename, "w") as f:
                json.dump(output_data, f, indent=4)
                