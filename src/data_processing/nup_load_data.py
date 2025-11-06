import os, json, sqlite3, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

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

def extract_and_generate_graphs(DB_PATH: str, out_dir_root, sample_step_s: float):
    """
    Extracts data from a nuPlan .db file, generates graph structures, and saves them as JSON files.
    This function does NOT require the nuPlan API.
    """
    db_stem = Path(DB_PATH).stem
    db_name = Path(DB_PATH).name
    OUT_DIR = os.path.join(out_dir_root, db_stem)
    os.makedirs(OUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    scenes = pd.read_sql_query("SELECT token, name FROM scene ORDER BY ROWID;", conn)
    if scenes.empty:
        conn.close()
        return []

    lb_cols = pd.read_sql_query("PRAGMA table_info(lidar_box);", conn)['name'].tolist()
    ego_cols = pd.read_sql_query("PRAGMA table_info(ego_pose);", conn)['name'].tolist()

    written_mappings = []

    for _, sc in scenes.iterrows():
        SCENE_TOKEN = sc["token"]
        SCENE_NAME = sc.get("name", None)

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

        ego_nodes, env_nodes = [], []
        veh_nodes, ped_nodes, obj_nodes = [], [], []
        ego_edges, env_edges = [], []
        ego_env_edges, ego_veh_edges, ego_ped_edges, ego_obj_edges = [], [], [], []

        for i_f, frow in frames_ds.iterrows():
            t = float(frow["t_rel"])
            lpt = frow["lidar_pc_token"]
            ts_us = int(frow["timestamp"])

            e_sql = nearest_row_by_time(ego_sorted, "t_rel", t) if not ego_sorted.empty else pd.Series(dtype=float)

            ego_features = {
                "x": float(e_sql.get("x")) if pd.notna(e_sql.get("x")) else None,
                "y": float(e_sql.get("y")) if pd.notna(e_sql.get("y")) else None,
                "z": float(e_sql.get("z")) if pd.notna(e_sql.get("z")) else None,
                "vx": float(e_sql.get("vx")) if pd.notna(e_sql.get("vx")) else None,
                "vy": float(e_sql.get("vy")) if pd.notna(e_sql.get("vy")) else None,
                "vz": float(e_sql.get("vz")) if pd.notna(e_sql.get("vz")) else None,
                "ax": None,
                "ay": None,
                "heading": None,
                "tire_angle": None,
                "type": "ego",
            }
            ego_nodes.append({"id": f"ego_{i_f}", "features": ego_features})

            env_nodes.append({
                "id": f"env_{i_f}",
                "features": {"timestamp_raw": ts_us, "time_rel_s": t}
            })
            if i_f > 0:
                ego_edges.append({"source": f"ego_{{i_f-1}}", "target": f"ego_{{i_f}}", "features": {}})
                env_edges.append({"source": f"env_{{i_f-1}}", "target": f"env_{{i_f}}", "features": {}})
            ego_env_edges.append({"source": f"ego_{{i_f}}", "target": f"env_{{i_f}}", "features": {}})

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
                    ego_veh_edges.append({"source": f"ego_{{i_f}}", "target": vid, "features": {}})

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
                    ego_ped_edges.append({"source": f"ego_{{i_f}}", "target": pid, "features": {}})

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
                    ego_obj_edges.append({"source": f"ego_{{i_f}}", "target": oid, "features": {}})

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
                "graph_id": f"{db_stem[:8]}_scene_{SCENE_TOKEN.hex()[:16]}",
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

        written_mappings.append({
            "json_file_name": Path(out_path).name,
            "db_name": db_name,
            "scene_token": str(SCENE_TOKEN),
            "scene_name": SCENE_NAME,
            "temp_path": out_path
        })

    conn.close()
    return written_mappings

def enrich_graphs_with_api_data(mappings: list, db_dir: str, out_dir: str):
    from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
    from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

    class DummyWorker:
        def __init__(self):
            self.number_of_threads = 0
        def map(self, fn, inputs):
            return [fn(x) for x in inputs]
        def submit(self, fn, *args, **kwargs):
            return fn(*args, **kwargs)

    """
    Enriches previously generated graph JSON files with data from the nuPlan API.
    """
    if not mappings:
        print("No mappings provided for enrichment. Skipping API enrichment.")
        return

    # Group mappings by db_name to initialize NuPlanScenarioBuilder once per DB
    db_to_mappings = {}
    for mapping in mappings:
        db_name = mapping['db_name']
        if db_name not in db_to_mappings:
            db_to_mappings[db_name] = []
        db_to_mappings[db_name].append(mapping)

    for db_name, db_mappings in tqdm(db_to_mappings.items(), desc="Enriching DBs"):
        # Find the full path to the DB file
        db_path = None
        for p in Path(db_dir).rglob(db_name):
            db_path = str(p.resolve())
            break
        if not db_path:
            print(f"Warning: Could not find DB file {db_name}. Skipping enrichment for its graphs.")
            continue

        nuplan_root = _find_nuplan_data_root(db_path)
        builder = NuPlanScenarioBuilder(
            data_root=nuplan_root,
            sensor_root=nuplan_root,
            db_files=[str(Path(db_path).expanduser().resolve())],
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

        # Build ego lookup (timestamp_us → ego_state) for the current DB
        ego_lookup = {}
        for sc in scenarios:
            if hasattr(sc, "get_number_of_iterations"):
                for i in range(sc.get_number_of_iterations()):
                    ego = sc.get_ego_state_at_iteration(i)
                    ego_lookup[int(ego.time_point.time_us)] = ego
            else:
                for ego in sc.get_ego_state_iter():
                    ego_lookup[int(ego.time_point.time_us)] = ego

        for mapping in db_mappings:
            json_path = mapping['flat_path'] # Use the flattened path
            if not json_path or not os.path.exists(json_path):
                print(f"Warning: JSON file not found at {json_path}. Skipping enrichment.")
                continue

            with open(json_path, "r") as f:
                graph = json.load(f)

            # Enrich ego nodes
            for ego_node in graph['nodes']['ego']:
                # Assuming ego_node id is like "ego_0", "ego_1", etc.
                # We need the corresponding timestamp from the environment node
                # This requires a bit of a lookup or a change in how timestamps are stored
                # For now, let's assume we can get the timestamp from the env node
                # This part might need adjustment based on exact graph structure
                env_node_id = ego_node['id'].replace("ego_", "env_")
                env_node = next((n for n in graph['nodes']['environment'] if n['id'] == env_node_id), None)

                if env_node and 'timestamp_raw' in env_node['features']:
                    ts_us = env_node['features']['timestamp_raw']
                    ego_state = nearest_ego_state(ts_us, ego_lookup)

                    if ego_state:
                        ego_node['features']['ax'] = float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x)
                        ego_node['features']['ay'] = float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.y)
                        ego_node['features']['heading'] = float(ego_state.rear_axle.heading)
                        ego_node['features']['tire_angle'] = float(ego_state.tire_steering_angle)
                        # Update x, y, z, vx, vy, vz if they were None or from SQL only
                        ego_node['features']['x'] = float(ego_state.rear_axle.x)
                        ego_node['features']['y'] = float(ego_state.rear_axle.y)
                        ego_node['features']['z'] = float(getattr(ego_state.rear_axle, "z", np.nan)) if pd.notna(getattr(ego_state.rear_axle, "z", np.nan)) else None
                        ego_node['features']['vx'] = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x)
                        ego_node['features']['vy'] = float(ego_state.dynamic_car_state.rear_axle_velocity_2d.y)
                    else:
                        # If ego_state not found, ensure API-specific fields are None
                        ego_node['features']['ax'] = None
                        ego_node['features']['ay'] = None
                        ego_node['features']['heading'] = None
                        ego_node['features']['tire_angle'] = None

            with open(json_path, "w") as f:
                json.dump(graph, f, indent=2)



# ---------------- I/O helpers ----------------
# --- MODIFICATION: Removed rename_jsons_in_dir function ---
# It will be replaced by inline logic in load_data

def extract_and_flatten_graphs(db_dir, out_dir='../data/graphical/nuplan', time_idx=1, file_min=0, file_max=None):
    total_count = 0
    all_mappings = []

    os.makedirs(out_dir, exist_ok=True)
    db_paths = sorted([str(p) for p in Path(db_dir).rglob("*.db")])
    n_total = len(db_paths)
    if not db_paths:
        raise RuntimeError(f"No .db files found under: {db_dir}")

    if file_min < 0:
        file_min = 0
    if file_max is None or file_max > n_total:
        file_max = n_total

    db_subset = db_paths[file_min:file_max]

    failed = []

    for idx, db in enumerate(tqdm(db_subset, desc="DB processing (extraction)", unit="db", initial=file_min)):
        try:
            mappings_from_db = extract_and_generate_graphs(db, out_dir, sample_step_s=time_idx)
            all_mappings.extend(mappings_from_db)
            total_count += len(mappings_from_db)
        except FileNotFoundError as e:
            print(f"⚠️ Skipping {db} — file not found: {e}")
            failed.append((db, str(e)))
        except sqlite3.DatabaseError as e:
            print(f"⚠️ Skipping {db} — database error: {e}")
            failed.append((db, str(e)))
        except Exception as e:
            print(f"⚠️ Skipping {db} due to unexpected error: {type(e).__name__}: {e}")
            failed.append((db, str(e)))

    for mapping in tqdm(all_mappings, desc="Flattening files"):
        src = mapping['temp_path']
        base_name = mapping['json_file_name']
        dst = os.path.join(out_dir, base_name)

        if not os.path.exists(src):
            print(f"Warning: Source file {src} not found during flattening. Skipping.")
            mapping['flat_path'] = None
            continue

        if os.path.exists(dst):
            base, ext = os.path.splitext(base_name)
            i = 1
            while os.path.exists(dst):
                dst_name = f"{base}_{i}{ext}"
                dst = os.path.join(out_dir, dst_name)
                i += 1
            mapping['json_file_name'] = dst_name
        
        shutil.move(src, dst)
        mapping['flat_path'] = dst

    # Save mappings for the next step
    mappings_path = os.path.join(out_dir, "temp_mappings.json")
    with open(mappings_path, "w") as f:
        json.dump(all_mappings, f)

    print(f"\n✅ Successfully extracted and flattened {total_count} scenes.")
    if failed:
        print("Failed DBs:")
        for db, err in failed:
            print(f"  {db}: {err}")

def enrich_and_finalize_graphs(db_dir, out_dir='../data/graphical/nuplan'):
    mappings_path = os.path.join(out_dir, "temp_mappings.json")
    if not os.path.exists(mappings_path):
        print("Error: temp_mappings.json not found. Please run the extraction step first.")
        return

    with open(mappings_path, "r") as f:
        all_mappings = json.load(f)

    print("\n--- Step 2: Enriching graphs with NuPlan API data ---")
    valid_mappings_for_enrichment = [m for m in all_mappings if m.get('flat_path') and os.path.exists(m['flat_path'])]
    enrich_graphs_with_api_data(valid_mappings_for_enrichment, db_dir, out_dir)

    for root, dirs, files in os.walk(out_dir, topdown=False):
        if root != out_dir and not files and not dirs:
            try:
                os.rmdir(root)
            except OSError as e:
                print(f"Warning: Could not remove directory {root}: {e}")

    print("\n--- Step 3: Renaming files and generating mapping CSV ---")
    final_csv_data = []
    
    valid_mappings_for_final_processing = [m for m in valid_mappings_for_enrichment if m.get('flat_path') and os.path.exists(m['flat_path'])]
    
    valid_mappings_for_final_processing.sort(key=lambda m: m['flat_path'])

    for idx, mapping in enumerate(tqdm(valid_mappings_for_final_processing, desc="Renaming files")):
        old_path = mapping['flat_path']
        new_filename = f"{idx}_graph.json"
        new_path = os.path.join(out_dir, new_filename)
        
        try:
            os.rename(old_path, new_path)
            final_csv_data.append({
                "json_file_name": new_filename,
                "db_name": mapping['db_name'],
                "scene_name": mapping['scene_name'],
                "scene_token": mapping['scene_token']
            })
        except OSError as e:
            print(f"Warning: Could not rename {old_path} to {new_path}: {e}")

    if final_csv_data:
        df = pd.DataFrame(final_csv_data)
        csv_path = os.path.join(out_dir, "file_mapping.csv")
        df.to_csv(csv_path, index=False, columns=["json_file_name", "db_name", "scene_name", "scene_token"])
        print(f"✅ Mapping CSV saved to {csv_path}")
    else:
        print("No valid files were processed, CSV not generated.")

    # Clean up the temporary mappings file
    os.remove(mappings_path)

    print(f"\n✅ Successfully enriched and finalized graphs.")
