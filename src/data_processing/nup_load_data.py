import os, json, sqlite3, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- NuPlan API imports for ego features ---
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

MAX_VEHICLES_PER_FRAME = None

# ---------------- Utilities ----------------
def infer_time_scale(ts):
    ts = np.asarray(ts, dtype="int64")
    if ts.size < 3: return 1e6
    d = np.diff(np.sort(ts)); d = d[d>0]
    if d.size == 0: return 1e6
    med = float(np.median(d))
    for s in (1e9,1e6,1e3,1.0):     # ns, µs, ms, s
        if 0.02 <= med/s <= 0.5:    # ~2–50 Hz
            return s
    return 1e6

def nearest_row_by_time(df_sorted, t_col, t):
    i = np.searchsorted(df_sorted[t_col].values, t)
    i = max(0, min(len(df_sorted)-1, i))
    return df_sorted.iloc[i]

def downsample_frames(frames, t_col="t_rel", step_s=1.0, tol=1e-9):
    """
    Keep first row, then rows with time difference >= step_s.
    step_s <= 0 disables downsampling.
    """
    if frames.empty:
        return frames
    fr = frames.sort_values(t_col).reset_index(drop=True)
    if not step_s or step_s <= 0:
        return fr
    keep_idx = []
    last_t = -np.inf
    for i, t in enumerate(fr[t_col].values):
        if t - last_t >= step_s - tol:
            keep_idx.append(i)
            last_t = t
    return fr.iloc[keep_idx].reset_index(drop=True)

def is_vehicle_category(cat: str) -> bool:
    if not isinstance(cat, str):
        return False
    c = cat.lower()
    vehicle_keywords = [
        'vehicle', 'car', 'truck', 'bus', 'trailer', 'van', 'pickup',
        'motorcycle', 'bicycle', 'scooter', 'moped'
    ]
    return any(k in c for k in vehicle_keywords)

def is_pedestrian_category(cat: str) -> bool:
    if not isinstance(cat, str):
        return False
    c = cat.lower()
    # nuPlan typically uses "human.pedestrian.*"
    return ('pedestrian' in c) or ('person' in c) or ('human.' in c and 'pedestrian' in c)

def nearest_ego_state(ts_us, ego_lookup, tol=5e5):
    if not ego_lookup:
        return None
    # Find closest timestamp in lookup within tolerance
    k = min(ego_lookup.keys(), key=lambda k: abs(k - ts_us))
    return ego_lookup[k] if abs(k - ts_us) <= tol else None

# ---------------- Helper to locate NuPlan data root ----------------
def _find_nuplan_data_root(db_path: str) -> str:
    """
    Given a full path to a .db file, try to locate the directory that contains 'nuplan-v1.1/<split>/*.db'.
    Returns the parent that contains 'nuplan-v1.1'.
    Example:
      db_path = /.../train_pittsburgh/nuplan-v1.1/train/2021.08....db
      -> returns /.../train_pittsburgh
    """
    p = Path(db_path).resolve()
    for parent in p.parents:
        # detect pattern .../<root>/nuplan-v1.1/<split>
        if parent.name in ("train", "val", "mini", "test") and parent.parent.name == "nuplan-v1.1":
            return str(parent.parent.parent)  # go up from .../nuplan-v1.1/<split> to <root>
    # fallback: three levels up
    return str(Path(db_path).resolve().parents[2])

# ---------------- Dummy Worker for NuPlan builder ----------------
class DummyWorker:
    def __init__(self):
        self.number_of_threads = 0
    def map(self, fn, inputs):
        return [fn(x) for x in inputs]
    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

# ---------------- Core pipeline ----------------
def process_db(DB_PATH: str, out_dir_root, sample_step_s: float):
    """
    Simplified: export one JSON per scene (no episode logic). Pedestrians are their own node type.
    Returns a list of mapping dictionaries for generated files.
    """
    db_stem = Path(DB_PATH).stem
    db_name = Path(DB_PATH).name
    OUT_DIR = os.path.join(out_dir_root, db_stem)
    os.makedirs(OUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    scenes = pd.read_sql_query("SELECT token, name FROM scene ORDER BY ROWID;", conn)
    if scenes.empty:
        print(f"[{db_stem}] No scenes found.")
        conn.close()
        return [] # Return empty list

    lb_cols   = pd.read_sql_query("PRAGMA table_info(lidar_box);", conn)['name'].tolist()
    ego_cols  = pd.read_sql_query("PRAGMA table_info(ego_pose);", conn)['name'].tolist()
    
    # Setup NuPlan API (for ego lookup)
    nuplan_root = _find_nuplan_data_root(DB_PATH)
    builder = NuPlanScenarioBuilder(
        data_root=nuplan_root,
        sensor_root=nuplan_root,
        db_files=[str(Path(DB_PATH).expanduser().resolve())],
        map_version="nuplan-maps-v1.0",
        map_root=None,
    )
    scenario_filter = ScenarioFilter(
        scenario_types=None, scenario_tokens=None, log_names=None, map_names=None,
        num_scenarios_per_type=None, limit_total_scenarios=None, timestamp_threshold_s=0,
        ego_displacement_minimum_m=0, expand_scenarios=False,
        remove_invalid_goals=False, shuffle=False,
    )
    worker = DummyWorker()
    scenarios = builder.get_scenarios(scenario_filter, worker=worker)

    # Build ego lookup (timestamp_us → ego_state)
    ego_lookup = {}
    for sc in scenarios:
        if hasattr(sc, "get_number_of_iterations"):
            for i in range(sc.get_number_of_iterations()):
                ego = sc.get_ego_state_at_iteration(i)
                ego_lookup[int(ego.time_point.time_us)] = ego
        else:
            for ego in sc.get_ego_state_iter():
                ego_lookup[int(ego.time_point.time_us)] = ego

    written_mappings = [] # Changed from written_paths

    for _, sc in scenes.iterrows():
        SCENE_TOKEN = sc["token"]
        SCENE_NAME  = sc.get("name", None)

        frames = pd.read_sql_query("""
            SELECT token AS lidar_pc_token, timestamp
            FROM lidar_pc
            WHERE scene_token = ?
            ORDER BY timestamp
        """, conn, params=(SCENE_TOKEN,))
        if frames.empty:
            continue

        t_scale = infer_time_scale(frames["timestamp"])
        t0_raw = int(frames["timestamp"].min())
        frames["t_rel"] = (frames["timestamp"] - t0_raw) / t_scale
        frames_ds = downsample_frames(frames, "t_rel", step_s=sample_step_s)

        ego = pd.read_sql_query(f"""
            SELECT ep.token AS ego_pose_token, ep.timestamp,
                   ep.x, ep.y, ep.z,
                   { 'ep.vx, ep.vy, ep.vz' if all(c in ego_cols for c in ['vx','vy','vz']) else 'NULL AS vx, NULL AS vy, NULL AS vz'}
            FROM ego_pose ep
            JOIN lidar_pc lp ON ep.token = lp.ego_pose_token
            WHERE lp.scene_token = ?
            ORDER BY ep.timestamp
        """, conn, params=(SCENE_TOKEN,))
        ego["t_rel"] = (ego["timestamp"] - t0_raw) / t_scale
        ego_sorted = ego.sort_values("t_rel").reset_index(drop=True)

        boxes = pd.read_sql_query(f"""
            SELECT lb.lidar_pc_token, lb.track_token,
                   lp.timestamp AS frame_timestamp,
                   lb.x, lb.y, lb.z, lb.yaw,
                   { 'lb.vx, lb.vy, lb.vz' if all(c in lb_cols for c in ['vx','vy','vz']) else 'NULL AS vx, NULL AS vy, NULL AS vz' },
                   c.name AS category
            FROM lidar_box lb
            JOIN lidar_pc lp  ON lb.lidar_pc_token = lp.token
            JOIN track tr     ON lb.track_token    = tr.token
            JOIN category c   ON tr.category_token = c.token
            WHERE lp.scene_token = ?
            ORDER BY lp.timestamp, lb.track_token
        """, conn, params=(SCENE_TOKEN,))
        boxes["t_rel"] = (boxes["frame_timestamp"] - t0_raw) / t_scale
        boxes_sorted = boxes.sort_values(["t_rel", "track_token"]).reset_index(drop=True)

        # ---- Graph construction per scene ----
        ego_nodes, env_nodes = [], []
        veh_nodes, ped_nodes, obj_nodes = [], [], []
        ego_edges, env_edges = [], []
        ego_env_edges, ego_veh_edges, ego_ped_edges, ego_obj_edges = [], [], [], []

        for i_f, frow in frames_ds.iterrows():
            t = float(frow["t_rel"])
            lpt = frow["lidar_pc_token"]
            ts_us = int(frow["timestamp"])

            ego_state = nearest_ego_state(ts_us, ego_lookup)
            e_sql = nearest_row_by_time(ego_sorted, "t_rel", t) if not ego_sorted.empty else pd.Series(dtype=float)

            ego_features = {
                "x": float(ego_state.rear_axle.x) if ego_state else float(e_sql.get("x")) if pd.notna(e_sql.get("x", np.nan)) else None,
                "y": float(ego_state.rear_axle.y) if ego_state else float(e_sql.get("y")) if pd.notna(e_sql.get("y", np.nan)) else None,
                "z": float(getattr(ego_state.rear_axle, "z", np.nan)) if ego_state and pd.notna(getattr(ego_state.rear_axle, "z", np.nan)) else float(e_sql.get("z")) if pd.notna(e_sql.get("z", np.nan)) else None,
                "vx": float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x) if ego_state else float(e_sql.get("vx")) if pd.notna(e_sql.get("vx", np.nan)) else None,
                "vy": float(ego_state.dynamic_car_state.rear_axle_velocity_2d.y) if ego_state else float(e_sql.get("vy")) if pd.notna(e_sql.get("vy", np.nan)) else None,
                "vz": float(e_sql.get("vz")) if pd.notna(e_sql.get("vz", np.nan)) else None,
                "ax": float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x) if ego_state else None,
                "ay": float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.y) if ego_state else None,
                "heading": float(ego_state.rear_axle.heading) if ego_state else None,
                "tire_angle": float(ego_state.tire_steering_angle) if ego_state else None,
                "type": "ego",
            }
            ego_nodes.append({"id": f"ego_{i_f}", "features": ego_features})

            env_nodes.append({
                "id": f"env_{i_f}",
                "features": {"timestamp_raw": ts_us, "time_rel_s": t}
            })
            if i_f > 0:
                ego_edges.append({"source": f"ego_{i_f-1}", "target": f"ego_{i_f}", "features": {}})
                env_edges.append({"source": f"env_{i_f-1}", "target": f"env_{i_f}", "features": {}})
            ego_env_edges.append({"source": f"ego_{i_f}", "target": f"env_{i_f}", "features": {}})

            subset = boxes_sorted[boxes_sorted["lidar_pc_token"] == lpt]
            if MAX_VEHICLES_PER_FRAME:
                subset = subset.head(MAX_VEHICLES_PER_FRAME)

            veh_counter, ped_counter, obj_counter = 0, 0, 0
            for _, ar in subset.iterrows():
                cat = str(ar["category"])
                if is_pedestrian_category(cat):
                    base_type = "pedestrian"
                elif is_vehicle_category(cat):
                    base_type = "vehicle"
                else:
                    base_type = "object"

                if base_type == "vehicle":
                    vid = f"vehicle_{chr(97 + veh_counter)}_{i_f}"
                    veh_counter += 1
                    veh_nodes.append({
                        "id": vid,
                        "features": {
                            "x": float(ar["x"]) if pd.notna(ar["x"]) else None,
                            "y": float(ar["y"]) if pd.notna(ar["y"]) else None,
                            "z": float(ar["z"]) if pd.notna(ar["z"]) else None,
                            "vx": float(ar["vx"]) if pd.notna(ar["vx"]) else None,
                            "vy": float(ar["vy"]) if pd.notna(ar["vy"]) else None,
                            "vz": float(ar["vz"]) if pd.notna(ar["vz"]) else None,
                            "yaw": float(ar["yaw"]) if pd.notna(ar["yaw"]) else None,
                            "category": ar["category"],
                            "type": "vehicle"
                        }
                    })
                    ego_veh_edges.append({"source": f"ego_{i_f}", "target": vid, "features": {}})

                elif base_type == "pedestrian":
                    pid = f"pedestrian_{chr(97 + ped_counter)}_{i_f}"
                    ped_counter += 1
                    ped_nodes.append({
                        "id": pid,
                        "features": {
                            "x": float(ar["x"]) if pd.notna(ar["x"]) else None,
                            "y": float(ar["y"]) if pd.notna(ar["y"]) else None,
                            "z": float(ar["z"]) if pd.notna(ar["z"]) else None,
                            "vx": float(ar["vx"]) if pd.notna(ar["vx"]) else None,
                            "vy": float(ar["vy"]) if pd.notna(ar["vy"]) else None,
                            "vz": float(ar["vz"]) if pd.notna(ar["vz"]) else None,
                            "yaw": float(ar["yaw"]) if pd.notna(ar["yaw"]) else None,
                            "category": ar["category"],
                            "type": "pedestrian"
                        }
                    })
                    ego_ped_edges.append({"source": f"ego_{i_f}", "target": pid, "features": {}})

                else:  # object
                    oid = f"object_{chr(97 + obj_counter)}_{i_f}"
                    obj_counter += 1
                    obj_nodes.append({
                        "id": oid,
                        "features": {
                            "x": float(ar["x"]) if pd.notna(ar["x"]) else None,
                            "y": float(ar["y"]) if pd.notna(ar["y"]) else None,
                            "z": float(ar["z"]) if pd.notna(ar["z"]) else None,
                            "vx": float(ar["vx"]) if pd.notna(ar["vx"]) else None,
                            "vy": float(ar["vy"]) if pd.notna(ar["vy"]) else None,
                            "vz": float(ar["vz"]) if pd.notna(ar["vz"]) else None,
                            "yaw": float(ar["yaw"]) if pd.notna(ar["yaw"]) else None,
                            "category": ar["category"],
                            "type": "object"
                        }
                    })
                    ego_obj_edges.append({"source": f"ego_{i_f}", "target": oid, "features": {}})

        graph = {
            "nodes": {
                "ego": ego_nodes,
                "vehicle": veh_nodes,
                "pedestrian": ped_nodes,
                "object": obj_nodes,
                "environment": env_nodes
            },
            "edges": {
                "ego_to_ego": ego_edges,
                "ego_to_vehicle": ego_veh_edges,
                "ego_to_pedestrian": ego_ped_edges,
                "ego_to_object": ego_obj_edges,
                "env_to_env": env_edges,
                "ego_to_environment": ego_env_edges
            },
            "metadata": {
                "graph_id": f"{db_stem[:8]}_scene_{str(SCENE_TOKEN)[:8]}",
                "db_file": db_name,
                "scene_token": str(SCENE_TOKEN),
                "scene_name": SCENE_NAME,
                "t_start": float(frames_ds["t_rel"].min()),
                "t_end": float(frames_ds["t_rel"].max()),
                "frames": int(len(frames_ds)),
            },
        }

        out_path = os.path.join(OUT_DIR, f"{graph['metadata']['graph_id']}.json")
        with open(out_path, "w") as f:
            json.dump(graph, f, indent=2)
        
        # --- MODIFICATION: Store mapping info ---
        written_mappings.append({
            "json_file_name": Path(out_path).name,
            "db_name": db_name,
            "scene_token": str(SCENE_TOKEN),
            "scene_name": SCENE_NAME,
            "temp_path": out_path  # Store the full path for later moving
        })
        # --- END MODIFICATION ---

    conn.close()
    return written_mappings # Return the list of mappings

# ---------------- I/O helpers ----------------
# --- MODIFICATION: Removed rename_jsons_in_dir function ---
# It will be replaced by inline logic in load_data

def load_data(db_dir, out_dir='../data/graphical/nuplan', time_idx=1, file_min=0, file_max=None):
    """
    db_dir: directory that contains 'nuplan-v1.1/<split>/*.db' (e.g., ../data/raw/NuPlan/train_pittsburgh)
    time_idx: sampling step in seconds at export time.
              Use 0 or negative to DISABLE downsampling (use every frame).
              Use 1 for ~1 Hz (default).
    file_min, file_max: integer range of DB indices to process, based on sorted order.
                        Example: file_min=10, file_max=19 processes the 10th–19th DBs.
                        file_max=None means "to the end".
    """
    # --- MODIFICATION: Initialize mapping list ---
    total_count = 0
    all_mappings = []
    # --- END MODIFICATION ---

    os.makedirs(out_dir, exist_ok=True)
    db_paths = sorted([str(p) for p in Path(db_dir).rglob("*.db")])
    n_total = len(db_paths)
    if not db_paths:
        raise RuntimeError(f"No .db files found under: {db_dir}")

    # clamp the bounds safely
    if file_min < 0:
        file_min = 0
    if file_max is None or file_max > n_total:
        file_max = n_total

    db_subset = db_paths[file_min:file_max]
    print(f"Found {n_total} DB files under {db_dir}")
    print(f"Processing subset {file_min}:{file_max} ({len(db_subset)} DBs)")

    failed = []

    for idx, db in enumerate(tqdm(db_subset, desc="DB processing", unit="db", initial=file_min)):
        try:
            print(f"\n[{file_min + idx}] Processing DB: {db}")
            # --- MODIFICATION: Collect mappings ---
            mappings_from_db = process_db(db, out_dir, sample_step_s=time_idx)
            all_mappings.extend(mappings_from_db)
            total_count += len(mappings_from_db)
            # --- END MODIFICATION ---
        except FileNotFoundError as e:
            print(f"⚠️ Skipping {db} — file not found: {e}")
            failed.append((db, str(e)))
        except sqlite3.DatabaseError as e:
            print(f"⚠️ Skipping {db} — database error: {e}")
            failed.append((db, str(e)))
        except Exception as e:
            print(f"⚠️ Skipping {db} due to unexpected error: {type(e).__name__}: {e}")
            failed.append((db, str(e)))

    # --- MODIFICATION: Replace flattening logic to use the mapping list ---
    print(f"\nFlattening {len(all_mappings)} JSON files...")
    for mapping in tqdm(all_mappings, desc="Flattening files"):
        src = mapping['temp_path']
        base_name = mapping['json_file_name']
        dst = os.path.join(out_dir, base_name)

        if not os.path.exists(src):
            print(f"Warning: Source file {src} not found during flattening. Skipping.")
            mapping['flat_path'] = None # Mark as failed
            continue

        if os.path.exists(dst):
            base, ext = os.path.splitext(base_name)
            i = 1
            while os.path.exists(dst):
                dst_name = f"{base}_{i}{ext}"
                dst = os.path.join(out_dir, dst_name)
                i += 1
            # Update the filename in the mapping if it changed
            mapping['json_file_name'] = dst_name
        
        shutil.move(src, dst)
        mapping['flat_path'] = dst # Store the new, flattened path
    # --- END MODIFICATION ---

    # Remove now-empty directories
    for root, dirs, files in os.walk(out_dir, topdown=False):
        if root != out_dir and not files and not dirs:
            try:
                os.rmdir(root)
            except OSError as e:
                print(f"Warning: Could not remove directory {root}: {e}")

    # --- MODIFICATION: Replace rename_jsons_in_dir with inline logic & CSV generation ---
    print("\nRenaming files and generating mapping CSV...")
    final_csv_data = []
    
    # Filter out any files that failed to flatten
    valid_mappings = [m for m in all_mappings if m.get('flat_path') and os.path.exists(m['flat_path'])]
    
    # Sort by the flattened file path to replicate original rename behavior
    valid_mappings.sort(key=lambda m: m['flat_path'])

    for idx, mapping in enumerate(tqdm(valid_mappings, desc="Renaming files")):
        old_path = mapping['flat_path']
        new_filename = f"{idx}_graph.json"
        new_path = os.path.join(out_dir, new_filename)
        
        try:
            os.rename(old_path, new_path)
            # Add to CSV data *only* if rename was successful
            final_csv_data.append({
                "json_file_name": new_filename,
                "db_name": mapping['db_name'],
                "scene_name": mapping['scene_name'],
                "scene_token": mapping['scene_token']
            })
        except OSError as e:
            print(f"Warning: Could not rename {old_path} to {new_path}: {e}")

    # Save the CSV
    if final_csv_data:
        df = pd.DataFrame(final_csv_data)
        # Ensure column order
        csv_path = os.path.join(out_dir, "file_mapping.csv")
        df.to_csv(csv_path, index=False, columns=["json_file_name", "db_name", "scene_name", "scene_token"])
        print(f"✅ Mapping CSV saved to {csv_path}")
    else:
        print("No valid files were processed, CSV not generated.")
    # --- END MODIFICATION ---

    print(f"\n✅ Successfully processed {total_count} scenes (skipped {len(failed)} broken DBs).")
    if failed:
        print("Failed DBs:")
        for db, err in failed:
            print(f"  {db}: {err}")
    
    return total_count # Return the total count as before