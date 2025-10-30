import os
import json
import math
from datetime import datetime
import re
from collections import defaultdict
from tqdm import tqdm

def ego_processing_l2d(input_dir, output_dir):
    """
    Processes all JSON files in input_dir, applying ego-node transformations:
    1. Converts heading (deg, compass CW from North) → radians (-π, π) math-style.
    2. Computes world-frame vx, vy from speed and heading.
    3. Rotates body-frame ax, ay to world-frame for consistency.
    4. Converts lat/lon to local x, y (meters, relative to first ego node).
    5. Converts speed from km/h → m/s.
    6. Keeps only standardized, world-frame features.

    Saves processed JSONs to output_dir with same filenames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def latlon_to_xy(lat, lon, lat0, lon0):
        k_y = 111_320.0
        k_x = 111_320.0 * math.cos(math.radians(lat0))
        x = (lon - lon0) * k_x
        y = (lat - lat0) * k_y
        return x, y

    def heading_deg_to_rad(hdg_deg):
        # Converts from Compass (CW from North) to Math angle (CCW from East/X-axis)
        hdg_rad_unwrapped = math.radians(90 - hdg_deg)
        # Wrap the angle to the [-pi, pi] interval
        hdg_rad = (hdg_rad_unwrapped + math.pi) % (2 * math.pi) - math.pi
        return hdg_rad

    for fname in tqdm(os.listdir(input_dir),desc='Ego Vehicle'):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        egos = data.get("nodes", {}).get("ego", [])
        if not egos:
            continue

        # FIX 3: Use .get() for robust access to prevent crashes
        first_features = egos[0].get("features", {})
        lat0 = first_features.get("latitude", 0.0)
        lon0 = first_features.get("longitude", 0.0)

        for ego in egos:
            feat = ego.get("features", {})

            heading_deg = feat.get("heading", 90.0) # Default to North if missing
            heading_rad = heading_deg_to_rad(heading_deg)

            # FIX 1: Unconditionally convert speed from km/h to m/s
            speed_kph = feat.get("speed", 0.0)
            speed_ms = speed_kph / 3.6

            # Velocity components (calculated in World Frame)
            cos_h = math.cos(heading_rad)
            sin_h = math.sin(heading_rad)
            vx_world = speed_ms * cos_h
            vy_world = speed_ms * sin_h

            # FIX 2: Rotate acceleration from Body Frame to World Frame
            ax_body = feat.get("accel_x", 0.0)
            ay_body = feat.get("accel_y", 0.0)
            ax_world = ax_body * cos_h - ay_body * sin_h
            ay_world = ax_body * sin_h + ay_body * cos_h

            # Local x, y (in meters)
            lat = feat.get("latitude", lat0)
            lon = feat.get("longitude", lon0)
            x, y = latlon_to_xy(lat, lon, lat0, lon0)

            ego["features"] = {
                "vx": vx_world,
                "vy": vy_world,
                "ax": ax_world,
                "ay": ay_world,
                "heading": heading_rad,
                "x": x,
                "y": y,
                "speed": speed_ms,
                "steering": feat.get("steering", 0.0),
            }
            ego["type"] = feat.get("type")

        outpath = os.path.join(output_dir, fname)
        with open(outpath, "w") as f:
            json.dump(data, f, indent=2)

    print("✅ All files processed and saved to:", output_dir)

def ego_processing_nup(input_dir, output_dir):
    """
    Processes JSON files with ego node features.
    Steps:
      1. Normalize x/y coordinates to be relative to the first ego node (scene-centric).
      2. If heading is missing, estimate it from the direction of travel.
      3. Rotate body-frame (vx, vy) and (ax, ay) to the world frame using heading.
      4. Keep only standardized, world-frame features.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in tqdm(os.listdir(input_dir),desc='Ego Vehicle'):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        egos = data.get("nodes", {}).get("ego", [])
        if not egos or len(egos) < 2:
            continue

        first_ego_features = egos[0].get("features", {})
        x0 = first_ego_features.get("x", 0.0)
        y0 = first_ego_features.get("y", 0.0)
        
        last_known_heading = None

        for i, ego_node in enumerate(egos):
            feat = ego_node["features"]
            heading = feat.get("heading")
            heading_is_estimated = False

            if heading is None:
                heading_is_estimated = True
                current_pos = (feat.get("x") or 0.0, feat.get("y") or 0.0)
                
                if i > 0:
                    prev_feat = egos[i-1]["features"]
                    prev_pos = (prev_feat.get("x") or 0.0, prev_feat.get("y") or 0.0)
                    dx = current_pos[0] - prev_pos[0]
                    dy = current_pos[1] - prev_pos[1]
                else:
                    next_feat = egos[i+1]["features"]
                    next_pos = (next_feat.get("x") or 0.0, next_feat.get("y") or 0.0)
                    dx = next_pos[0] - current_pos[0]
                    dy = next_pos[1] - current_pos[1]
                
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    heading = math.atan2(dy, dx)
                elif last_known_heading is not None:
                    heading = last_known_heading
                else:
                    heading = 0.0
                    print(f"⚠️ Warning: Cannot estimate heading for stationary start in {fname}. Defaulting to 0.")
            
            if heading is not None:
                last_known_heading = heading

            # --- FIX: Ensure motion vectors are floats, not None ---
            vx_body = feat.get("vx") or 0.0
            vy_body = feat.get("vy") or 0.0
            ax_body = feat.get("ax") or 0.0
            ay_body = feat.get("ay") or 0.0
            # ----------------------------------------------------
            
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            vx_world = vx_body * cos_h - vy_body * sin_h
            vy_world = vx_body * sin_h + vy_body * cos_h
            ax_world = ax_body * cos_h - ay_body * sin_h
            ay_world = ax_body * sin_h + ay_body * cos_h

            speed = math.sqrt(vx_body**2 + vy_body**2)
            x_abs = feat.get("x") or 0.0
            y_abs = feat.get("y") or 0.0
            x_rel = x_abs - x0
            y_rel = y_abs - y0

            ego_node["features"] = {
                "vx": vx_world,
                "vy": vy_world,
                "ax": ax_world,
                "ay": ay_world,
                "heading": heading,
                "heading_is_estimated": heading_is_estimated,
                "x": x_rel,
                "y": y_rel,
                "speed": speed,
                "steering": feat.get("tire_angle") or 0.0,
            }
            ego_node["type"] = feat.get("type")

        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(data, f, indent=2)

    print("✅ All files processed and saved to:", output_dir)

def env_processing_l2d(input_dir):
    """
    Processes 'environment' nodes in JSON scene files.
    Transforms features as follows:
      - month: string name → int (1–12)
      - day_of_week: string name → int (0–6, Sunday=0)
      - time_of_day: 'HH:MM:SS' → seconds since midnight
      - precipitation: string → float
      - lighting: 'Day'/'Night' → daylight boolean
      - conditions: mapped to int {0=clear, 1=overcast, 2=rain, 3=snow, 4=fog/haze}
    Keeps only: month, day, time, conditions, precipitation, daylight.

    The JSONs are modified **in place** (same filenames).
    """

    # month and weekday mappings
    month_map = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12
    }
    day_map = {
        "sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
        "thursday": 4, "friday": 5, "saturday": 6
    }
    condition_map = {
        "clear": 0,
        "clouds": 1, "overcast": 1,
        "rain": 2, "drizzle": 2,
        "snow": 3,
        "fog": 4, "haze": 4, "mist": 4
    }

    for fname in tqdm(os.listdir(input_dir),desc='Environment'):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        envs = data.get("nodes", {}).get("environment", [])
        if not envs:
            continue

        for env in envs:
            feat = env["features"]

            # month number
            month = month_map.get(str(feat.get("month", "")).lower(), 0)

            # day of week number
            day = day_map.get(str(feat.get("day_of_week", "")).lower(), 0)

            # time in seconds
            time_str = str(feat.get("time_of_day", "00:00:00"))
            try:
                h, m, s = [int(x) for x in time_str.split(":")]
                time = h * 3600 + m * 60 + s
            except Exception:
                time = 0

            # precipitation
            try:
                precipitation = float(feat.get("precipitation", 0.0))
            except ValueError:
                precipitation = 0.0

            # daylight (true if lighting is 'Day')
            lighting = str(feat.get("lighting", "")).lower()
            daylight = lighting.startswith("day")

            # conditions to int
            cond = str(feat.get("conditions", "")).lower()
            conditions = condition_map.get(cond, 1 if "cloud" in cond else 0)

            # keep only the reduced feature set
            env["features"] = {
                "month": month,
                "day": day,
                "time": time,
                "conditions": conditions,
                "precipitation": precipitation,
                "daylight": daylight
            }

        # overwrite the file in place
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

    print("✅ Environment processing complete.")

def env_processing_nup(input_dir):
    """
    Processes 'environment' nodes from NUP-format JSONs and converts them
    into the standardized reduced form:
        month, day, time, conditions, precipitation, daylight

    - Maps WMO weather codes / descriptions into 0=clear, 1=overcast,
      2=rain, 3=snow, 4=fog/haze.
    - Converts day_of_week to int (0–6, Sunday=0).
    - Converts time_of_day to seconds since midnight.
    - Substitutes missing values with None.
    - Saves JSONs in place.
    """

    day_map = {
        "sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
        "thursday": 4, "friday": 5, "saturday": 6
    }

    def parse_time_seconds(time_str):
        try:
            h, m, s = [int(x) for x in str(time_str).split(":")]
            return h * 3600 + m * 60 + s
        except Exception:
            return None

    def map_weather_to_condition(code=None, desc=None):
        if code is not None:
            # prioritize code-based mapping
            if code in [0, 1]:
                return 0  # clear
            elif code in [2, 3]:
                return 1  # overcast
            elif code in list(range(51, 68)) + list(range(80, 83)) + list(range(95, 100)):
                return 2  # rain
            elif code in list(range(71, 78)) + [85, 86]:
                return 3  # snow
            elif code in [45, 48]:
                return 4  # fog/haze
            else:
                return 1  # default to overcast
        elif desc:
            desc = str(desc).lower()
            if "clear" in desc:
                return 0
            elif any(k in desc for k in ["cloud", "overcast"]):
                return 1
            elif any(k in desc for k in ["rain", "drizzle", "thunder"]):
                return 2
            elif "snow" in desc:
                return 3
            elif any(k in desc for k in ["fog", "haze", "mist"]):
                return 4
            else:
                return 1
        else:
            return None

    for fname in tqdm(os.listdir(input_dir),desc='Environment'):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        envs = data.get("nodes", {}).get("environment", [])
        if not envs:
            continue

        for env in envs:
            feat = env.get("features", {})

            month = feat.get("month", None)
            day_str = feat.get("day_of_week", None)
            day = day_map.get(str(day_str).lower(), None) if day_str else None
            time = parse_time_seconds(feat.get("time_of_day"))
            precipitation = feat.get("precipitation_mm", None)
            if precipitation is not None:
                try:
                    precipitation = float(precipitation)
                except Exception:
                    precipitation = None
            daylight = feat.get("is_daylight", None)

            code = feat.get("weather_code", None)
            desc = feat.get("weather_description", None)
            conditions = map_weather_to_condition(code, desc)

            env["features"] = {
                "month": month,
                "day": day,
                "time": time,
                "conditions": conditions,
                "precipitation": precipitation,
                "daylight": daylight
            }

        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

    print("✅ NUP environment processing complete.")

def veh_processing_l2d(input_dir, annotation_root, hfov_deg=90, time_step_s=3.0):
    """
    Processes vehicle nodes by recalculating motion from estimated positions.
    1. Pass 1: For each vehicle track, calculate all scene-centric (x, y) positions.
    2. Pass 2: Use the history of positions to calculate a new, consistent
       world-frame (vx, vy) vector for each frame using finite differences.
    3. Speed is recalculated from the new (vx, vy) vector.
    """

    # ... (helper functions compute_xy_from_bbox, _episode_dir_for remain the same) ...
    def compute_xy_from_bbox(bbox, dist, img_w, hfov_deg):
        if not bbox or dist is None: raise ValueError("Missing bbox or distance.")
        x_min, _, w, _ = bbox
        u_center = x_min + w / 2.0
        cx = img_w / 2.0
        bearing = -((u_center - cx) / cx) * math.radians(hfov_deg / 2.0)
        x_rel, y_rel = dist * math.cos(bearing), dist * math.sin(bearing)
        return x_rel, y_rel

    def _episode_dir_for(graph_filename, metadata_graph_id, annotation_root):
        m = re.search(r'(\d+)', graph_filename)
        id_digits = m.group(1) if m else None
        if not id_digits and metadata_graph_id:
            m2 = re.search(r'(\d+)', str(metadata_graph_id))
            id_digits = m2.group(1) if m2 else None
        if not id_digits: raise ValueError(f"Could not extract id from '{graph_filename}' or '{metadata_graph_id}'")
        eid = id_digits.zfill(6)
        ep_no_underscore, ep_with_underscore = f"Episode{eid}", f"Episode_{eid}"
        if os.path.isdir(os.path.join(annotation_root, ep_no_underscore)): return ep_no_underscore
        if os.path.isdir(os.path.join(annotation_root, ep_with_underscore)): return ep_with_underscore
        raise FileNotFoundError(f"Could not find annotation folder for episode {eid}")
    # -----------------------------------------------------------------------------

    for fname in tqdm(os.listdir(input_dir),desc='Vehicles'):
        if not fname.endswith("_graph.json"): continue
        graph_path = os.path.join(input_dir, fname)
        with open(graph_path, "r") as f: data = json.load(f)

        vehicles = data.get("nodes", {}).get("vehicle", [])
        if not vehicles: continue

        egos = data.get("nodes", {}).get("ego", [])
        ego_pos_lookup = {}
        for ego_node in egos:
            try:
                frame_idx = int(ego_node["id"].split("_")[-1])
                ego_feat = ego_node.get("features", {})
                if "x" in ego_feat and "y" in ego_feat: ego_pos_lookup[frame_idx] = (ego_feat["x"], ego_feat["y"])
            except (ValueError, IndexError): continue

        metadata_graph_id = data.get("metadata", {}).get("graph_id")
        episode_dir = _episode_dir_for(fname, metadata_graph_id, annotation_root)
        ann_base = os.path.join(annotation_root, episode_dir, "front_left_Annotations")

        grouped = defaultdict(list)
        for v in vehicles:
            parts = v["id"].split("_")
            grouped["_".join(parts[:-1])].append((int(parts[-1]), v))

        for track_prefix, veh_list in grouped.items():
            # Sort the track by frame index to ensure correct order
            veh_list.sort(key=lambda item: item[0])

            # --- PASS 1: Calculate and store all scene-centric positions for the track ---
            track_positions = []
            for frame_idx, veh_node in veh_list:
                feat = veh_node.get("features", {})
                annot_path = os.path.join(ann_base, f"frame_{frame_idx:05}.json")
                if not os.path.exists(annot_path): continue # Skip if annotation is missing

                with open(annot_path, "r") as f: annot_data = json.load(f)
                track_entry = next((ann for ann in annot_data.get("annotations", []) if ann.get("track_id") == track_prefix), None)
                if not track_entry: continue

                dist = feat.get("dist_to_ego") or track_entry.get("attributes", {}).get("depth_stats", {}).get("mean_depth")
                bbox = track_entry.get("bbox")
                if dist is None or not bbox: continue

                img_w = annot_data["images"][0]["width"]
                x_rel_ego, y_rel_ego = compute_xy_from_bbox(bbox, dist, img_w, hfov_deg)

                x_ego_scene, y_ego_scene = ego_pos_lookup.get(frame_idx, (0.0, 0.0))
                x_scene, y_scene = x_rel_ego + x_ego_scene, y_rel_ego + y_ego_scene

                track_positions.append({
                    "x": x_scene, "y": y_scene, "dist_to_ego": dist, "original_node": veh_node
                })

            # --- PASS 2: Calculate velocities based on the stored positions ---
            num_frames = len(track_positions)
            if num_frames < 2: # Cannot calculate velocity for single-frame tracks
                if num_frames == 1: # Set velocity to zero for single frame
                     track_positions[0]['original_node']['features'] = {
                        "x": track_positions[0]['x'], "y": track_positions[0]['y'],
                        "vx": 0.0, "vy": 0.0, "speed": 0.0,
                        "dist_to_ego": track_positions[0]['dist_to_ego'],
                    }
                continue

            for i in range(num_frames):
                current = track_positions[i]
                
                if i == 0: # First frame: use forward difference
                    next_frame = track_positions[i+1]
                    vx = (next_frame['x'] - current['x']) / time_step_s
                    vy = (next_frame['y'] - current['y']) / time_step_s
                elif i == num_frames - 1: # Last frame: use backward difference
                    prev_frame = track_positions[i-1]
                    vx = (current['x'] - prev_frame['x']) / time_step_s
                    vy = (current['y'] - prev_frame['y']) / time_step_s
                else: # Middle frames: use central difference for more stability
                    next_frame = track_positions[i+1]
                    prev_frame = track_positions[i-1]
                    vx = (next_frame['x'] - prev_frame['x']) / (2 * time_step_s)
                    vy = (next_frame['y'] - prev_frame['y']) / (2 * time_step_s)
                
                speed = math.hypot(vx, vy)

                # Update the original node's features with the newly calculated motion
                current['original_node']['features'] = {
                    "x": current['x'], "y": current['y'],
                    "vx": vx, "vy": vy, "speed": speed,
                    "dist_to_ego": current['dist_to_ego'],
                }
                current['original_node']["type"] = current["type"]

        with open(graph_path, "w") as f:
            json.dump(data, f, indent=2)

    print("🏁 Vehicle node processing complete.")

def veh_processing_nup(input_dir, raw_dir):
    """
    Processes NUP vehicle nodes using a consistent scene-centric frame.
    - All coordinates are made relative to the EGO'S STARTING POSITION.
    - Computes speed from vx, vy.
    - Keeps only {x, y, vx, vy, speed, dist_to_ego} in the final features.
    """
    for fname in tqdm(os.listdir(input_dir),desc='Vehicles'):
        if not fname.endswith("_graph.json"):
            continue

        graph_path = os.path.join(input_dir, fname)
        raw_path = os.path.join(raw_dir, fname)

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw file for {fname}: {raw_path}")

        with open(graph_path, "r") as f:
            proc_data = json.load(f)
        with open(raw_path, "r") as f:
            raw_data = json.load(f)

        ego_nodes_raw = raw_data.get("nodes", {}).get("ego", [])
        vehicles = proc_data.get("nodes", {}).get("vehicle", [])

        if not ego_nodes_raw or not vehicles:
            continue

        # --- START OF CHANGE ---

        # Get the reference coordinates from the FIRST ego node only
        first_ego_features = ego_nodes_raw[0].get("features", {})
        x0 = first_ego_features.get("x", 0.0)
        y0 = first_ego_features.get("y", 0.0)

        # Build a lookup for the ego vehicle's x, y position at each frame
        ego_lookup = {}
        for ego_node in ego_nodes_raw:
            ego_id = ego_node["id"]
            m = re.search(r"(\d+)$", ego_id)
            if not m:
                continue
            idx = int(m.group(1))
            f_raw = ego_node.get("features", {})
            x, y = f_raw.get("x"), f_raw.get("y")
            if x is not None and y is not None:
                ego_lookup[idx] = (x, y)
        
        # --- END OF CHANGE ---

        for v in vehicles:
            vid = v["id"]
            parts = re.split(r"[_\-]", vid)
            try:
                frame_idx = int(parts[-1])
            except (ValueError, IndexError):
                continue

            ego_coords_current = ego_lookup.get(frame_idx) # Ego's position now
            if not ego_coords_current:
                raise ValueError(f"No ego coordinates found for frame {frame_idx} in {fname}")

            x_e_current, y_e_current = ego_coords_current
            f = v.get("features", {})

            x_v_abs = f.get("x")
            y_v_abs = f.get("y")
            if x_v_abs is None or y_v_abs is None:
                raise ValueError(f"No x/y for vehicle {vid} in {fname}")

            # --- START OF CHANGE ---

            # Project vehicle coordinates relative to the scene origin (ego's start)
            x_rel_scene = x_v_abs - x0
            y_rel_scene = y_v_abs - y0

            # Calculate distance to ego using their absolute positions in this frame
            dist_to_ego = math.sqrt((x_v_abs - x_e_current)**2 + (y_v_abs - y_e_current)**2)
            
            # --- END OF CHANGE ---

            vx = f.get("vx")
            vy = f.get("vy")
            speed = None
            if vx is not None and vy is not None:
                speed = math.sqrt(vx**2 + vy**2)

            v["features"] = {
                "x": x_rel_scene, # Use the new scene-relative coordinates
                "y": y_rel_scene, # Use the new scene-relative coordinates
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "dist_to_ego": dist_to_ego # This is still correct
            }
            v["type"] = f.get("type")

        with open(graph_path, "w") as f:
            json.dump(proc_data, f, indent=2)

        print(f"✅ Processed {fname}")

    print("🏁 Vehicle node NUP processing complete.")

def ped_processing_l2d(input_dir):
    """
    Moves pedestrian nodes (IDs starting with 'Ped') out of the 'vehicle' category
    into 'pedestrian' within each *_graph.json file.

    Parameters
    ----------
    input_dir : str
        Directory containing *_graph.json files to process.
    """

    for fname in tqdm(os.listdir(input_dir),desc='Pedestrians'):
        if not fname.endswith("_graph.json"):
            continue

        graph_path = os.path.join(input_dir, fname)
        with open(graph_path, "r") as f:
            data = json.load(f)

        nodes = data.get("nodes", {})
        vehicles = nodes.get("vehicle", [])
        pedestrians = nodes.get("pedestrian", [])

        moved = []
        kept = []

        for v in vehicles:
            vid = v.get("id", "")
            if vid.startswith("Ped"):
                moved.append(v)
            else:
                kept.append(v)

        # Update node lists
        if moved:
            nodes["vehicle"] = kept
            nodes["pedestrian"] = pedestrians + moved
            data["nodes"] = nodes

            with open(graph_path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            pass

    print("🏁 Pedestrian node reclassification complete.")

def ped_processing_nup(input_dir, raw_dir):
    """
    Processes NUP pedestrian nodes using a consistent scene-centric frame.
    - All coordinates are made relative to the EGO'S STARTING POSITION.
    - Computes speed from vx, vy.
    - Keeps only {x, y, vx, vy, speed, dist_to_ego} in the final features.
    """

    for fname in tqdm(os.listdir(input_dir),desc='Pedestrians'):
        if not fname.endswith("_graph.json"):
            continue

        graph_path = os.path.join(input_dir, fname)
        raw_path = os.path.join(raw_dir, fname)

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw file for {fname}: {raw_path}")

        with open(graph_path, "r") as f:
            proc_data = json.load(f)
        with open(raw_path, "r") as f:
            raw_data = json.load(f)

        ego_nodes_raw = raw_data.get("nodes", {}).get("ego", [])
        pedestrians = proc_data.get("nodes", {}).get("pedestrian", [])

        if not ego_nodes_raw or not pedestrians:
            continue
        
        # Get the reference coordinates from the FIRST ego node only
        first_ego_features = ego_nodes_raw[0].get("features", {})
        x0 = first_ego_features.get("x", 0.0)
        y0 = first_ego_features.get("y", 0.0)

        # Build a lookup for the ego vehicle's x, y position at each frame
        ego_lookup = {}
        for ego_node in ego_nodes_raw:
            ego_id = ego_node["id"]
            m = re.search(r"(\d+)$", ego_id)
            if not m:
                continue
            idx = int(m.group(1))
            f_raw = ego_node.get("features", {})
            x, y = f_raw.get("x"), f_raw.get("y")
            if x is not None and y is not None:
                ego_lookup[idx] = (x, y)

        # Process each pedestrian node
        for p in pedestrians:
            pid = p["id"]
            parts = re.split(r"[_\-]", pid)
            try:
                frame_idx = int(parts[-1])
            except (ValueError, IndexError):
                continue

            ego_coords_current = ego_lookup.get(frame_idx)
            if not ego_coords_current:
                raise ValueError(f"No ego coordinates found for frame {frame_idx} in {fname}")

            x_e_current, y_e_current = ego_coords_current
            f = p.get("features", {})

            x_p_abs = f.get("x")
            y_p_abs = f.get("y")
            if x_p_abs is None or y_p_abs is None:
                raise ValueError(f"No x/y for pedestrian {pid} in {fname}")

            # Project pedestrian coordinates relative to the scene origin (ego's start)
            x_rel_scene = x_p_abs - x0
            y_rel_scene = y_p_abs - y0

            # Calculate distance to ego using their absolute positions in this frame
            dist_to_ego = math.sqrt((x_p_abs - x_e_current)**2 + (y_p_abs - y_e_current)**2)

            vx = f.get("vx")
            vy = f.get("vy")
            speed = None
            if vx is not None and vy is not None:
                speed = math.sqrt(vx**2 + vy**2)

            p["features"] = {
                "x": x_rel_scene,
                "y": y_rel_scene,
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "dist_to_ego": dist_to_ego
            }
            p["type"] = f.get("type")

        with open(graph_path, "w") as f:
            json.dump(proc_data, f, indent=2)

    print("🏁 Pedestrian node NUP processing complete.")

def obj_processing_nup(input_dir, raw_dir):
    """
    Processes NUP object nodes using a consistent scene-centric frame.
    - All coordinates are made relative to the EGO'S STARTING POSITION.
    - Encodes category as integer.
    - Keeps only {x, y, dist_to_ego, category} in the final features.
    """

    category_map = {
        "czone_sign": 0,
        "traffic_cone": 1,
        "generic_object": 2,
        "barrier": 3
    }

    for fname in tqdm(os.listdir(input_dir),desc='Objects'):
        if not fname.endswith("_graph.json"):
            continue

        graph_path = os.path.join(input_dir, fname)
        raw_path = os.path.join(raw_dir, fname)

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw file for {fname}: {raw_path}")

        with open(graph_path, "r") as f:
            proc_data = json.load(f)
        with open(raw_path, "r") as f:
            raw_data = json.load(f)

        ego_nodes_raw = raw_data.get("nodes", {}).get("ego", [])
        objects = proc_data.get("nodes", {}).get("object", [])

        if not ego_nodes_raw or not objects:
            continue
        
        # Get the reference coordinates from the FIRST ego node only
        first_ego_features = ego_nodes_raw[0].get("features", {})
        x0 = first_ego_features.get("x", 0.0)
        y0 = first_ego_features.get("y", 0.0)

        # Build a lookup for the ego vehicle's x, y position at each frame
        ego_lookup = {}
        for ego_node in ego_nodes_raw:
            ego_id = ego_node["id"]
            m = re.search(r"(\d+)$", ego_id)
            if not m:
                continue
            idx = int(m.group(1))
            f_raw = ego_node.get("features", {})
            x, y = f_raw.get("x"), f_raw.get("y")
            if x is not None and y is not None:
                ego_lookup[idx] = (x, y)

        # Process each object node
        for o in objects:
            oid = o["id"]
            parts = re.split(r"[_\-]", oid)
            try:
                frame_idx = int(parts[-1])
            except (ValueError, IndexError):
                continue

            ego_coords_current = ego_lookup.get(frame_idx)
            if not ego_coords_current:
                raise ValueError(f"No ego coordinates found for frame {frame_idx} in {fname}")

            x_e_current, y_e_current = ego_coords_current
            f = o.get("features", {})

            x_o_abs = f.get("x")
            y_o_abs = f.get("y")
            if x_o_abs is None or y_o_abs is None:
                raise ValueError(f"No x/y for object {oid} in {fname}")

            # Project object coordinates relative to the scene origin (ego's start)
            x_rel_scene = x_o_abs - x0
            y_rel_scene = y_o_abs - y0

            # Calculate distance to ego using their absolute positions in this frame
            dist_to_ego = math.sqrt((x_o_abs - x_e_current)**2 + (y_o_abs - y_e_current)**2)

            # Category mapping
            cat_str = f.get("category", "generic_object")
            cat_idx = category_map.get(cat_str, 2)

            o["features"] = {
                "x": x_rel_scene,
                "y": y_rel_scene,
                "dist_to_ego": dist_to_ego,
                "category": cat_idx
            }
            o["type"] = f.get("type")

        with open(graph_path, "w") as f:
            json.dump(proc_data, f, indent=2)

    print("🏁 Object node NUP processing complete.")

