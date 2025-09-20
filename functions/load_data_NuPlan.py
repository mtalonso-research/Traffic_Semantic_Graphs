import os, json, sqlite3, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- NuPlan API imports for ego features ---
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter

# --- Config (you can override via load_data(...)) ---
EXPAND_INSTANT_SECONDS = 0.0     # not used for episode creation below, left here if you later want to enable
MIN_DURATION_S         = 4*3.0   # default: 12s minimum episode length
MAX_VEHICLES_PER_FRAME = None    # None = include all agents; set an int to cap per frame

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

def run_segments(vals):
    v = np.asarray(vals, dtype=object)
    if len(v)==0: return []
    starts = np.flatnonzero(np.r_[True, v[1:] != v[:-1]])
    ends   = np.r_[starts[1:], len(v)] - 1
    return list(zip(starts, ends, v[starts]))

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

def is_vehicle_category(cat):
    """Robust vehicle detector based on category string."""
    if not isinstance(cat, str):
        return False
    c = cat.lower()
    vehicle_keywords = [
        'vehicle', 'car', 'truck', 'bus', 'trailer', 'van', 'pickup',
        'motorcycle', 'bicycle', 'scooter', 'moped'
    ]
    return any(k in c for k in vehicle_keywords)

# ---------------- Helper to locate NuPlan data root ----------------
def _find_nuplan_data_root(db_path: str) -> str:
    """
    Given a full path to a .db file, try to locate the directory that contains 'nuplan-v1.1/<split>/...db'.
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
        # Some older devkit paths check for this attribute
        self.number_of_threads = 0
    def map(self, fn, inputs):
        return [fn(x) for x in inputs]
    def submit(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

# ---------------- Core pipeline ----------------
def process_db(DB_PATH: str, out_dir_root, sample_step_s: float):
    """
    sample_step_s: seconds between exported frames (<=0 to disable downsampling).
    """
    db_stem = Path(DB_PATH).stem
    OUT_DIR = os.path.join(out_dir_root, db_stem)
    os.makedirs(OUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    scenes = pd.read_sql_query("SELECT token, name FROM scene ORDER BY ROWID;", conn)
    if scenes.empty:
        print(f"[{db_stem}] No scenes found.")
        conn.close()
        return 0

    lb_cols   = pd.read_sql_query("PRAGMA table_info(lidar_box);", conn)['name'].tolist()
    ego_cols  = pd.read_sql_query("PRAGMA table_info(ego_pose);", conn)['name'].tolist()
    has_stags = (cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='scenario_tag';").fetchone()[0] == 1)

    # ---------------- Setup NuPlan API for ego (builder per-DB) ----------------
    nuplan_root = _find_nuplan_data_root(DB_PATH)  # e.g., ../data/raw/NuPlan/train_pittsburgh
    builder = NuPlanScenarioBuilder(
        data_root=nuplan_root,
        sensor_root=nuplan_root,
        db_files=[str(Path(DB_PATH).expanduser().resolve())],  # restrict to the current DB filename
        map_version="nuplan-maps-v1.0",
        map_root=None,                  # no maps required for ego features we need
    )
    # ScenarioFilter signature varies across versions; these values keep everything
    scenario_filter = ScenarioFilter(
        scenario_types=None,
        scenario_tokens=None,
        log_names=None,
        map_names=None,
        num_scenarios_per_type=None,
        limit_total_scenarios=None,
        timestamp_threshold_s=0,
        ego_displacement_minimum_m=0,
        expand_scenarios=False,
        remove_invalid_goals=False,
        shuffle=False,
    )
    worker = DummyWorker()
    scenarios = builder.get_scenarios(scenario_filter, worker=worker)

    # Build ego lookup: ABSOLUTE timestamp (µs) -> EgoState
    ego_lookup = {}
    for sc in scenarios:
        # Older devkit API: iterate by index
        if hasattr(sc, "get_number_of_iterations"):
            for i in range(sc.get_number_of_iterations()):
                ego = sc.get_ego_state_at_iteration(i)
                ego_lookup[int(ego.time_point.time_us)] = ego
        else:
            # If newer API, fallback to iterator
            for ego in sc.get_ego_state_iter():
                ego_lookup[int(ego.time_point.time_us)] = ego

    written_paths = []

    for _, sc in scenes.iterrows():
        SCENE_TOKEN = sc['token']
        SCENE_NAME  = sc.get('name', None)

        # Frames (full timeline)
        frames_full = pd.read_sql_query("""
            SELECT token AS lidar_pc_token, timestamp
            FROM lidar_pc
            WHERE scene_token = ?
            ORDER BY timestamp
        """, conn, params=(SCENE_TOKEN,))
        if frames_full.empty:
            continue

        t_scale = infer_time_scale(frames_full['timestamp'])
        t0_raw  = int(frames_full['timestamp'].min())
        frames_full['t_rel'] = (frames_full['timestamp'] - t0_raw) / t_scale
        # Downsampled view for export only
        frames_ds = downsample_frames(frames_full, t_col="t_rel", step_s=sample_step_s)

        # Ego poses from SQL (kept for fallback values; heading/steering come from API)
        ego = pd.read_sql_query(f"""
            SELECT ep.token AS ego_pose_token, ep.timestamp,
                   ep.x, ep.y, ep.z,
                   { 'ep.vx, ep.vy, ep.vz' if all(c in ego_cols for c in ['vx','vy','vz']) else 'NULL AS vx, NULL AS vy, NULL AS vz'}
            FROM ego_pose ep
            JOIN lidar_pc lp ON ep.token = lp.ego_pose_token
            WHERE lp.scene_token = ?
            ORDER BY ep.timestamp
        """, conn, params=(SCENE_TOKEN,))
        ego['t_rel'] = (ego['timestamp'] - t0_raw) / t_scale
        ego_sorted = ego.sort_values('t_rel').reset_index(drop=True)

        # Agents
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
        boxes['t_rel'] = (boxes['frame_timestamp'] - t0_raw) / t_scale
        boxes_sorted = boxes.sort_values(['t_rel','track_token']).reset_index(drop=True)

        # scenario_tag
        if has_stags:
            st = pd.read_sql_query("""
                SELECT lidar_pc_token, agent_track_token, type
                FROM scenario_tag
                WHERE lidar_pc_token IN (SELECT token FROM lidar_pc WHERE scene_token = ?)
                ORDER BY lidar_pc_token
            """, conn, params=(SCENE_TOKEN,))
        else:
            st = pd.DataFrame(columns=['lidar_pc_token','agent_track_token','type'])

        # Build maps for labels
        frame_tag_map = {}
        agent_tag_map = {}
        if not st.empty:
            for _, r in st.iterrows():
                lpt, typ, att = r['lidar_pc_token'], r['type'], r['agent_track_token']
                if pd.isna(att):
                    frame_tag_map.setdefault(lpt, []).append(typ)
                else:
                    agent_tag_map.setdefault((lpt, att), []).append(typ)

        # -------- Frame episodes (FEP) with your rules on the DENSE timeline --------
        episodes_f = []
        if not st.empty:
            # Build per-frame label (primary) on frames_full (dense), None if no frame-level tags
            per_frame_full = frames_full[['lidar_pc_token','t_rel']].copy()
            def choose_primary(tags):
                if not tags: return None
                tags = [t for t in tags if isinstance(t, str)]
                if not tags: return None
                # prioritize "state-like" tags
                for t in tags:
                    tl = t.lower()
                    if tl.startswith('on_') or tl.startswith('traversing_') or tl.startswith('stationary') or tl.startswith('following_'):
                        return t
                return tags[0]  # fallback to first
            per = st[st['agent_track_token'].isna()].groupby('lidar_pc_token')['type'].apply(list) if not st.empty else pd.Series(dtype=object)
            per = per.to_dict()
            per_frame_full['label'] = [ choose_primary(per.get(tok, [])) for tok in per_frame_full['lidar_pc_token'] ]

            labels = per_frame_full['label'].tolist()
            times  = per_frame_full['t_rel'].to_numpy(float)

            N = len(times)
            i = 0
            while i < N:
                # Find first tagged frame at/after i
                j_tag = None
                for j in range(i, N):
                    if labels[j] is not None:
                        j_tag = j
                        break
                if j_tag is None:
                    # no more tags → no further episodes (requires at least one tag)
                    break

                # Find end of this tag: the next index where label becomes None
                lab0 = labels[j_tag]
                j_end = j_tag
                k = j_tag + 1
                while k < N and labels[k] is not None:
                    # We treat "any tag present" as tagged span; episode label is lab0 (first tag seen)
                    j_end = k
                    k += 1

                t_start = times[i]
                t_tag   = times[j_tag]
                t_tag_end = times[j_end]

                if (t_tag - t_start) < MIN_DURATION_S:
                    target_end_time = max(t_start + MIN_DURATION_S, t_tag_end)
                else:
                    target_end_time = t_tag_end

                # Clip to scene end
                target_end_time = min(target_end_time, times[-1])

                # Convert target_end_time to index e (last idx with t <= target_end_time)
                e = int(np.searchsorted(times, target_end_time, side='right') - 1)
                if e <= i:
                    # Ensure progress even in degenerate cases
                    e = min(i+1, N-1)

                # Record episode with label = label at first tagged frame
                episodes_f.append({'label': lab0, 't0': float(t_start), 't1': float(times[e])})

                # Next episode starts after e
                i = e + 1

        # -------- Agent episodes (unchanged; time spans from dense timeline) --------
        episodes_a = []
        st_agent = st[st['agent_track_token'].notna()].copy()
        if not st_agent.empty:
            st_agent = st_agent.merge(frames_full[['lidar_pc_token','t_rel']], on='lidar_pc_token', how='left')
            for track, g in st_agent.sort_values(['agent_track_token','t_rel']).groupby('agent_track_token'):
                seq = g['type'].tolist()
                idx = g.index.to_list()
                for rs, re, lab in run_segments(seq):
                    sidx, eidx = idx[rs], idx[re]
                    t0 = float(g.loc[sidx,'t_rel']); t1 = float(g.loc[eidx,'t_rel'])
                    if (t1 - t0) < MIN_DURATION_S:
                        # Enforce min duration by extending to t0+MIN if possible
                        t1 = min(t1 if t1 >= t0 + MIN_DURATION_S else t0 + MIN_DURATION_S, float(frames_full['t_rel'].iloc[-1]))
                        if t1 <= t0:  # guard
                            continue
                    episodes_a.append({'track_token': track, 'label': lab, 't0': t0, 't1': t1})

        # -------- Export helper (uses downsampled frames for graph density control) --------
        def export_episode(t0, t1, label, kind, idx, track_token=None):
            fr = frames_ds[(frames_ds['t_rel'] >= t0) & (frames_ds['t_rel'] <= t1)].copy().reset_index(drop=True)
            if fr.empty:
                return None

            ego_nodes, env_nodes, veh_nodes, obj_nodes = [], [], [], []
            ego_edges, env_edges, ego_env_edges, ego_veh_edges = [], [], [], []

            for i_f, frow in fr.iterrows():
                t   = float(frow['t_rel'])
                lpt = frow['lidar_pc_token']

                frame_labels = frame_tag_map.get(lpt, [])
                frame_label_primary = frame_labels[0] if frame_labels else None

                # --- Ego features from NuPlan API lookup (preferred) ---
                ego_state = ego_lookup.get(int(frow['timestamp']))

                # Fallback to nearest SQL ego row for position/velocity if lookup misses
                e_sql = nearest_row_by_time(ego_sorted, 't_rel', t) if not ego_sorted.empty else pd.Series(dtype=float)

                ego_features = {
                    "x": float(ego_state.rear_axle.x) if ego_state else (float(e_sql.get('x', np.nan)) if pd.notna(e_sql.get('x', np.nan)) else None),
                    "y": float(ego_state.rear_axle.y) if ego_state else (float(e_sql.get('y', np.nan)) if pd.notna(e_sql.get('y', np.nan)) else None),
                    "z": float(getattr(ego_state.rear_axle, 'z', np.nan)) if ego_state and pd.notna(getattr(ego_state.rear_axle, 'z', np.nan)) else (float(e_sql.get('z', np.nan)) if pd.notna(e_sql.get('z', np.nan)) else None),
                    "vx": float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x) if ego_state else (float(e_sql.get('vx', np.nan)) if pd.notna(e_sql.get('vx', np.nan)) else None),
                    "vy": float(ego_state.dynamic_car_state.rear_axle_velocity_2d.y) if ego_state else (float(e_sql.get('vy', np.nan)) if pd.notna(e_sql.get('vy', np.nan)) else None),
                    "vz": float(e_sql.get('vz', np.nan)) if pd.notna(e_sql.get('vz', np.nan)) else None,
                    # New fields from API:
                    "ax": float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.x) if ego_state else None,
                    "ay": float(ego_state.dynamic_car_state.rear_axle_acceleration_2d.y) if ego_state else None,
                    "heading": float(ego_state.rear_axle.heading) if ego_state else None,
                    "tire_angle": float(ego_state.tire_steering_angle) if ego_state else None,
                    "type": "ego",
                }
                ego_nodes.append({"id": f"ego_{i_f}", "features": ego_features})

                env_nodes.append({
                    "id": f"env_{i_f}",
                    "features": {
                        "timestamp_raw": int(frow['timestamp']),
                        "time_rel_s": t,
                        "frame_label": frame_label_primary,
                        "frame_labels_all": frame_labels
                    }
                })
                if i_f > 0:
                    ego_edges.append({"source": f"ego_{i_f-1}", "target": f"ego_{i_f}", "features": {}})
                    env_edges.append({"source": f"env_{i_f-1}", "target": f"env_{i_f}", "features": {}})
                ego_env_edges.append({"source": f"ego_{i_f}", "target": f"env_{i_f}", "features": {}})

                subset = boxes_sorted[boxes_sorted['lidar_pc_token'] == lpt]
                if kind == 'AEP' and track_token is not None:
                    subset = subset[subset['track_token'] == track_token]
                if MAX_VEHICLES_PER_FRAME:
                    subset = subset.head(MAX_VEHICLES_PER_FRAME)

                # Per-frame letter counters
                veh_counter, obj_counter = 0, 0
                for _, ar in subset.iterrows():
                    base_type = "vehicle" if is_vehicle_category(ar['category']) else "object"
                    if base_type == "vehicle":
                        vid = f"vehicle_{chr(97 + veh_counter)}_{i_f}"  # a,b,c...
                        veh_counter += 1
                        veh_nodes.append({
                            "id": vid,
                            "features": {
                                "x": float(ar['x']) if pd.notna(ar['x']) else None,
                                "y": float(ar['y']) if pd.notna(ar['y']) else None,
                                "z": float(ar['z']) if pd.notna(ar['z']) else None,
                                "vx": float(ar['vx']) if pd.notna(ar['vx']) else None,
                                "vy": float(ar['vy']) if pd.notna(ar['vy']) else None,
                                "vz": float(ar['vz']) if pd.notna(ar['vz']) else None,
                                "yaw": float(ar['yaw']) if pd.notna(ar['yaw']) else None,
                                "category": ar['category'],
                                "type": "vehicle",
                                "tags": agent_tag_map.get((lpt, ar['track_token']), [])
                            }
                        })
                        ego_veh_edges.append({"source": f"ego_{i_f}", "target": vid, "features": {}})
                    else:
                        oid = f"object_{chr(97 + obj_counter)}_{i_f}"
                        obj_counter += 1
                        obj_nodes.append({
                            "id": oid,
                            "features": {
                                "x": float(ar['x']) if pd.notna(ar['x']) else None,
                                "y": float(ar['y']) if pd.notna(ar['y']) else None,
                                "z": float(ar['z']) if pd.notna(ar['z']) else None,
                                "vx": float(ar['vx']) if pd.notna(ar['vx']) else None,
                                "vy": float(ar['vy']) if pd.notna(ar['vy']) else None,
                                "vz": float(ar['vz']) if pd.notna(ar['vz']) else None,
                                "yaw": float(ar['yaw']) if pd.notna(ar['yaw']) else None,
                                "category": ar['category'],
                                "type": "object",
                                "tags": agent_tag_map.get((lpt, ar['track_token']), [])
                            }
                        })

            graph = {
                "nodes": {
                    "ego": ego_nodes,
                    "vehicle": veh_nodes,
                    "object": obj_nodes,
                    "environment": env_nodes
                },
                "edges": {
                    "ego_to_ego": ego_edges,
                    "ego_to_vehicle": ego_veh_edges,
                    "env_to_env": env_edges,
                    "ego_to_environment": ego_env_edges
                },
                "metadata": {
                    "graph_id": f"{db_stem[:8]}_scene_{str(SCENE_TOKEN)[:8]}_{kind.lower()}_{idx:04d}",
                    "db_file": Path(DB_PATH).name,
                    "scene_token": str(SCENE_TOKEN),
                    "scene_name": SCENE_NAME,
                    "episode_kind": kind,          # FEP or AEP
                    "episode_label": label,
                    "t_start": float(t0),
                    "t_end": float(t1),
                    "frames": int(len(fr)),
                    **({"track_token": str(track_token)} if track_token is not None else {})
                }
            }
            out_path = os.path.join(OUT_DIR, f"{graph['metadata']['graph_id']}.json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(graph, f, indent=2)
            return out_path

        # Export all episodes
        for i_ep, ep in enumerate(episodes_f):
            p = export_episode(ep['t0'], ep['t1'], ep['label'], kind='FEP', idx=i_ep, track_token=None)
            if p: written_paths.append(p)
        for i_ep, ep in enumerate(episodes_a):
            p = export_episode(ep['t0'], ep['t1'], ep['label'], kind='AEP', idx=i_ep, track_token=ep['track_token'])
            if p: written_paths.append(p)

    conn.close()
    return len(written_paths)

# ---------------- I/O helpers ----------------
def rename_jsons_in_dir(directory):
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    json_files.sort()
    for idx, filename in enumerate(json_files):
        old_path = os.path.join(directory, filename)
        new_filename = f"{idx}_graph.json"
        new_path = os.path.join(directory, new_filename)
        os.rename(old_path, new_path)

def load_data(db_dir, out_dir='../data/graphical/nuplan', time_idx=1, max_files=None):
    """
    db_dir: directory that contains 'nuplan-v1.1/<split>/*.db' (e.g., ../data/raw/NuPlan/train_pittsburgh)
    time_idx: sampling step in seconds at export time.
              Use 0 or negative to DISABLE downsampling (use every frame).
              Use 1 for ~1 Hz (default).
    """
    total = 0
    os.makedirs(out_dir, exist_ok=True)
    # RECURSIVE search to support nested layout nuplan-v1.1/train/*.db
    db_paths = sorted([str(p) for p in Path(db_dir).rglob("*.db")])
    if not db_paths:
        raise RuntimeError(f"No .db files found under: {db_dir}")
    for db in tqdm(db_paths[:max_files]):
        total += process_db(db, out_dir, sample_step_s=time_idx)

    # Flatten scene subdirs into out_dir
    for root, dirs, files in os.walk(out_dir):
        if root == out_dir:
            continue
        for file in files:
            if file.endswith(".json"):
                src = os.path.join(root, file)
                dst = os.path.join(out_dir, file)
                if os.path.exists(dst):
                    base, ext = os.path.splitext(file)
                    i = 1
                    while os.path.exists(dst):
                        dst = os.path.join(out_dir, f"{base}_{i}{ext}")
                        i += 1
                shutil.move(src, dst)

    # Remove now-empty directories
    for root, dirs, files in os.walk(out_dir, topdown=False):
        if root != out_dir and not files and not dirs:
            os.rmdir(root)

    # Normalize names
    rename_jsons_in_dir(out_dir)
    return total
