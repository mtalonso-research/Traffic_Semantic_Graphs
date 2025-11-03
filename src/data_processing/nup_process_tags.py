import json
import os
from typing import List
import sqlite3
import pandas as pd
from tqdm import tqdm

def extract_tags(data_root: str, output_dir: str, graph_dir: str):
    """
    Extracts scenario tags from nuPlan databases and saves them as JSON files.

    :param data_root: The root directory of the nuPlan dataset.
    :param output_dir: The directory where the tag JSON files will be saved.
    :param graph_dir: The directory where the graph JSON files and file_mapping.csv are located.
    """
    mapping_file = os.path.join(graph_dir, "file_mapping.csv")
    if not os.path.exists(mapping_file):
        print(f"Error: Mapping file not found at {mapping_file}")
        return

    mappings = pd.read_csv(mapping_file)

    for _, row in tqdm(mappings.iterrows(), total=len(mappings)):
        json_file_name = row["json_file_name"]
        db_name = row["db_name"]
        scene_token_hex = row["scene_token"]

        db_path = os.path.join(data_root, db_name)
        if not os.path.exists(db_path):
            print(f"Warning: Database file not found: {db_path}")
            continue

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get all scenario tags for the scene
        scene_token_bytes = bytes.fromhex(scene_token_hex)
        cursor.execute("SELECT type FROM scenario_tag WHERE lidar_pc_token IN (SELECT token FROM lidar_pc WHERE scene_token=?)", (scene_token_bytes,))
        tags = [row[0] for row in cursor.fetchall()]

        conn.close()

        if tags:
            output_filename = os.path.join(output_dir, json_file_name)
            with open(output_filename, "w") as f:
                json.dump({"tags": tags}, f, indent=4)