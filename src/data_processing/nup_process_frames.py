#!/usr/bin/env python3
"""
Build episode directories from NuPlan DB + mapping CSV.

For each row in file_mapping.csv:
  - graph_name (e.g. 0_graph.json)
  - scene_name (e.g. scene-0019)
  - db_file (e.g. 2021.05.12.22.00.38_veh-35_01008_01518.db)

we:
  - open the corresponding .db
  - extract ordered front-camera frames for that scene
  - copy them from vegas_frames into EpisodeXXXXXX/frame_YYYYY.jpg
"""

import os
import csv
import sqlite3
from types import SimpleNamespace
from pathlib import Path
from tqdm import tqdm
import shutil
import re


# =======================
# CONFIG
# =======================

# CSV with graph ↔ scene ↔ db filename mapping
MAPPING_CSV = Path("data/graphical/nuplan_las_vegas/file_mapping.csv")

# Root where all jpgs live; DB filenames are relative to this root
VEGAS_FRAMES_ROOT = Path("data/raw/NuPlan/vegas_frames")

# Root where you want your Episode directories to be created
EPISODE_OUTPUT_ROOT = Path("data/raw/NuPlan/vegas_episodes")

# Root directory containing all .db files
DB_ROOT = Path("data/raw/NuPlan/train_las_vegas/nuplan-v1.1/train")

# Camera channel to use (as in the camera.channel column)
CAMERA_VIEW = "CAM_F0"


# =======================
# CORE FUNCTIONS
# =======================

def load_scene_front_images(db_path: Path, scene_name: str, camera_view: str):
    """
    Given a DB file and a scene name, return a list of image rows (SimpleNamespace)
    for the chosen camera view, ordered by timestamp.

    Each returned row has attributes: token, camera_token, timestamp, filename.

    IMPORTANT CHANGE:
    - Scene time window is now [MIN(timestamp), MAX(timestamp)] from lidar_pc
      for that scene, instead of [scene_start, MAX(image.timestamp)].
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Load cameras
    cursor.execute("SELECT token, channel FROM camera;")
    camera_rows = cursor.fetchall()
    cameras = [SimpleNamespace(token=r[0], channel=r[1]) for r in camera_rows]

    # Pick the camera we care about
    front_camera = next(
        (cam for cam in cameras if cam.channel == camera_view),
        None,
    )
    if front_camera is None:
        conn.close()
        raise ValueError(f"Camera view '{camera_view}' not found in {db_path}")

    # Get scene token from scene table by name
    cursor.execute("SELECT token FROM scene WHERE name = ?;", (scene_name,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Scene '{scene_name}' not found in {db_path}")
    scene_token = row[0]

    # Get start/end timestamps for this scene from lidar_pc
    cursor.execute(
        """
        SELECT MIN(timestamp), MAX(timestamp)
        FROM lidar_pc
        WHERE scene_token = ?;
        """,
        (scene_token,),
    )
    ts_min, ts_max = cursor.fetchone()
    if ts_min is None or ts_max is None:
        conn.close()
        raise ValueError(
            f"No lidar_pc rows for scene '{scene_name}' (scene_token={scene_token}) in {db_path}"
        )

    # All front-camera images in [ts_min, ts_max], ordered
    cursor.execute(
        """
        SELECT token, camera_token, timestamp, filename_jpg
        FROM image
        WHERE camera_token = ?
          AND timestamp BETWEEN ? AND ?
        ORDER BY timestamp
        """,
        (front_camera.token, ts_min, ts_max),
    )
    image_rows = cursor.fetchall()
    conn.close()

    front_images = [
        SimpleNamespace(token=r[0], camera_token=r[1], timestamp=r[2], filename=r[3])
        for r in image_rows
    ]

    if not front_images:
        raise ValueError(
            f"No front-camera images found for scene '{scene_name}' in {db_path}"
        )

    return front_images


def graph_name_to_episode_dir(graph_name: str) -> str:
    """
    Map something like '0_graph.json' -> 'Episode000000'.

    Assumes the graph_name starts with an integer index, e.g. '123_graph.json'.
    """
    m = re.match(r"(\d+)_graph", graph_name)
    if not m:
        raise ValueError(f"Could not parse graph index from graph_name='{graph_name}'")
    idx = int(m.group(1))
    return f"Episode{idx:06d}"


def build_episode_for_row(graph_name: str, scene_name: str, db_file: str,
                          cache: dict):
    """
    For one mapping row: open the DB, get front camera frames for the scene,
    and copy/rename them into the appropriate Episode directory.

    `cache` is a dict to reuse (db_path, scene_name) → front_images so we don't
    re-query DBs when multiple graphs map to the same scene.
    """
    db_path = DB_ROOT / db_file

    if not db_path.exists():
        raise FileNotFoundError(f"DB file not found: {db_path}")

    cache_key = (db_path, scene_name)
    if cache_key in cache:
        front_images = cache[cache_key]
    else:
        front_images = load_scene_front_images(db_path, scene_name, CAMERA_VIEW)
        cache[cache_key] = front_images

    # Build output episode dir name
    episode_dir_name = graph_name_to_episode_dir(graph_name)
    episode_dir = EPISODE_OUTPUT_ROOT / episode_dir_name
    episode_dir.mkdir(parents=True, exist_ok=True)

    # Copy + rename frames (still copies; originals untouched)
    for idx, img in enumerate(front_images):
        # filename from DB is relative to VEGAS_FRAMES_ROOT
        src = VEGAS_FRAMES_ROOT / img.filename
        if not src.exists():
            raise FileNotFoundError(f"Frame file not found on disk: {src}")

        dst_name = f"frame_{idx:05d}.jpg"
        dst = episode_dir / dst_name
        shutil.copy2(src, dst)

    print(
        f"Built {episode_dir} with {len(front_images)} frames "
        f"for graph '{graph_name}' / scene '{scene_name}'"
    )


# =======================
# MAIN
# =======================

if not MAPPING_CSV.exists():
    raise FileNotFoundError(f"Mapping CSV not found: {MAPPING_CSV}")

if not VEGAS_FRAMES_ROOT.exists():
    raise FileNotFoundError(f"Vegas frames root not found: {VEGAS_FRAMES_ROOT}")

if not DB_ROOT.exists():
    raise FileNotFoundError(f"DB root directory not found: {DB_ROOT}")

EPISODE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Read mapping CSV
with MAPPING_CSV.open("r", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Small cache so we don't re-read DB for repeated (db, scene) combos
cache = {}

start_indx = 747
for i,row in enumerate(rows[start_indx:]):
    try:
        graph_name = row["json_file_name"]
        scene_name = row["scene_name"]
        db_file = row["db_name"]

        build_episode_for_row(graph_name, scene_name, db_file, cache)
    except: print(f'ERROR PROCESSING EPISODE {start_indx+i}')