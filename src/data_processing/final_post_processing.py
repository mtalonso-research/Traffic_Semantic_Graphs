import os
import json
import math
from datetime import datetime
import re
from collections import defaultdict
from tqdm import tqdm

EARTH_RADIUS_M = 6371000.0
def ego_processing_l2d(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    file_list = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for filename in tqdm(file_list, desc="Ego"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Skipping file {filename} due to an error: {e}")
            continue

        ego_frames = data.get('nodes', {}).get('ego', [])

        if not ego_frames:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            continue
            
        # --- 1. Initialization ---
        origin_lat_deg = ego_frames[0]['features']['latitude']
        origin_lon_deg = ego_frames[0]['features']['longitude']
        origin_lat_rad = math.radians(origin_lat_deg)

        processed_ego_frames = []
        for frame_data in ego_frames:
            features = frame_data['features']
            
            # --- 2. Position Conversion (Lat/Lon -> ENU x, y) ---
            lat_rad = math.radians(features['latitude'])
            lon_rad = math.radians(features['longitude'])
            
            x = EARTH_RADIUS_M * (lon_rad - math.radians(origin_lon_deg)) * math.cos(origin_lat_rad)
            y = EARTH_RADIUS_M * (lat_rad - math.radians(origin_lat_deg))
            
            # --- 3. Speed Conversion (km/h -> m/s) ---
            speed_ms = features['speed'] / 3.6
            
            # --- 4. Heading Conversion (Compass Degrees -> ENU Radians) ---
            heading_compass_deg = features['heading']
            heading_enu_deg = (450.0 - heading_compass_deg) % 360.0
            heading_enu_rad = math.radians(heading_enu_deg)
            # Normalize to the [-pi, pi] range
            heading_normalized_rad = (heading_enu_rad + math.pi) % (2 * math.pi) - math.pi

            # --- 5. Velocity Calculation (ENU vx, vy) ---
            vx = speed_ms * math.cos(heading_normalized_rad)
            vy = speed_ms * math.sin(heading_normalized_rad)
            
            # --- 6. Acceleration Transformation (Vehicle Frame -> ENU Frame) ---
            accel_longitudinal = features['accel_x']
            accel_lateral = features['accel_y']
            
            ax_enu = accel_longitudinal * math.cos(heading_normalized_rad) - accel_lateral * math.sin(heading_normalized_rad)
            ay_enu = accel_longitudinal * math.sin(heading_normalized_rad) + accel_lateral * math.cos(heading_normalized_rad)
            
            # --- 7. Assemble Final Features ---
            processed_features = {
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'ax': ax_enu,
                'ay': ay_enu,
                'heading': heading_normalized_rad,
                'speed': speed_ms,
                'steering': features['steering'] # Pass this value through directly
            }
            
            new_frame = {
                'id': frame_data['id'],
                'features': processed_features
            }
            processed_ego_frames.append(new_frame)

        data['nodes']['ego'] = processed_ego_frames

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

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

            vx_body = feat.get("vx") or 0.0
            vy_body = feat.get("vy") or 0.0
            ax_body = feat.get("ax") or 0.0
            ay_body = feat.get("ay") or 0.0
            
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)

            if not heading_is_estimated:
                speed = math.sqrt(vx_body**2 + vy_body**2)
                vx_world = vx_body * cos_h - vy_body * sin_h
                vy_world = vx_body * sin_h + vy_body * cos_h
                ax_world = ax_body * cos_h - ay_body * sin_h
                ay_world = ax_body * sin_h + ay_body * cos_h
            else:
                if last_known_heading is not None:
                    heading = last_known_heading
                else:
                    heading = 0.0

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

HFOV_DEG = 100.0  # Assumed Horizontal Field of View in degrees
IMAGE_WIDTH_PX = 1920.0 # Standard image width for calculating pixel offset

def veh_processing_l2d(input_dir, annotation_root):
    """
    Processes other vehicle data in JSON files by converting their ego-centric
    features into an absolute ENU coordinate system.

    This function reads files from an input directory, assumes the 'ego' nodes
    are already processed, and uses corresponding annotation files to calculate
    absolute position and velocity for other vehicles. The files are modified in-place.

    Args:
        input_dir (str): Path to the directory with JSON files (with processed ego nodes).
        annotation_root (str): Root path for the annotation files directory structure.
    """
    hfov_rad = math.radians(HFOV_DEG)
    image_center_px = IMAGE_WIDTH_PX / 2.0
    
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for filename in tqdm(file_list, desc="Processing Other Vehicles"):
        file_path = os.path.join(input_dir, filename)

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Skipping file {filename} due to an error: {e}")
            continue

        vehicle_frames = data.get('nodes', {}).get('vehicle', [])
        ego_frames = data.get('nodes', {}).get('ego', [])

        if not vehicle_frames or not ego_frames:
            continue # No vehicles to process or no ego reference data

        # --- 1. Pre-computation and Setup ---
        ego_lookup = {frame['id']: frame['features'] for frame in ego_frames}

        try:
            parquet_path = data['metadata']['source_files']['parquet']
            match = re.search(r'episode_(\d+)', parquet_path)
            if not match:
                print(f"Warning: Could not find episode number in {filename}. Skipping.")
                continue
            ep_num = int(match.group(1))
        except KeyError:
            print(f"Warning: Could not find parquet metadata in {filename}. Skipping.")
            continue
            
        processed_vehicle_frames = []
        # --- 2. Main Processing Loop ---
        for vehicle in vehicle_frames:
            try:
                parts = vehicle['id'].split('_')
                frame_num = int(parts[-1])
                track_id = "_".join(parts[:-1]) 

                # Find the corresponding ego data for this frame
                ego_id = f'ego_{frame_num}'
                ego_features = ego_lookup.get(ego_id)
                if not ego_features:
                    print('COULD NOT FIND')
                    continue 

                # Load the corresponding annotation file
                annotation_path = os.path.join(
                    annotation_root,
                    f'Episode{ep_num:06d}',
                    'front_left_Annotations',
                    f'frame_{frame_num:05d}.json'
                )
                with open(annotation_path, 'r') as f_ann:
                    annotation_data = json.load(f_ann)
                
                # Find the vehicle's specific annotation using its track_id
                vehicle_annotation = next((ann for ann in annotation_data['annotations'] if ann['track_id'] == track_id), None)
                if not vehicle_annotation:
                    continue # Skip if no matching annotation is found

                # --- 3. Perform Calculations ---
                ego_heading_rad = ego_features['heading']
                dist_to_ego = vehicle['features']['dist_to_ego']
                
                # --- Position Calculation ---
                bbox = vehicle_annotation['bbox'] # [x_min, y_min, width, height]
                bbox_center_px = bbox[0] + (bbox[2] / 2.0)
                pixel_offset = bbox_center_px - image_center_px
                
                # Positive angle is left, negative is right
                relative_angle = -(pixel_offset / IMAGE_WIDTH_PX) * hfov_rad
                global_angle = ego_heading_rad + relative_angle
                
                x = ego_features['x'] + dist_to_ego * math.cos(global_angle)
                y = ego_features['y'] + dist_to_ego * math.sin(global_angle)
                
                # --- Velocity Calculation ---
                v_rel_lat, v_rel_lon = vehicle['features']['velocity_ms'][0], vehicle['features']['velocity_ms'][1]
                
                cos_h, sin_h = math.cos(ego_heading_rad), math.sin(ego_heading_rad)
                
                # Rotate the relative velocity vector into the global ENU frame
                v_rot_x = v_rel_lon * cos_h - v_rel_lat * sin_h
                v_rot_y = v_rel_lon * sin_h + v_rel_lat * cos_h
                
                # Add the ego's absolute velocity
                vx = ego_features['vx'] + v_rot_x
                vy = ego_features['vy'] + v_rot_y
                
                speed = math.sqrt(vx**2 + vy**2)

                # --- 4. Assemble Final Features ---
                processed_features = {
                    'x': x,
                    'y': y,
                    'vx': vx,
                    'vy': vy,
                    'speed': speed,
                    'dist_to_ego': dist_to_ego
                }
                
                processed_vehicle_frames.append({'id': vehicle['id'], 'features': processed_features, 'type': 'vehicle'})

            except (KeyError, IndexError, FileNotFoundError) as e:
                continue

        data['nodes']['vehicle'] = processed_vehicle_frames
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    print(f"✅ All vehicle data processed and files updated in: {input_dir}")

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

            # Project vehicle coordinates relative to the scene origin (ego's start)
            x_rel_scene = x_v_abs - x0
            y_rel_scene = y_v_abs - y0

            # Calculate distance to ego using their absolute positions in this frame
            dist_to_ego = math.sqrt((x_v_abs - x_e_current)**2 + (y_v_abs - y_e_current)**2)
            

            vx = f.get("vx")
            vy = f.get("vy")
            speed = None
            if vx is not None and vy is not None:
                speed = math.sqrt(vx**2 + vy**2)

            v["features"] = {
                "x": x_rel_scene, 
                "y": y_rel_scene, 
                "vx": vx,
                "vy": vy,
                "speed": speed,
                "dist_to_ego": dist_to_ego 
            }
            v["type"] = f.get("type")

        with open(graph_path, "w") as f:
            json.dump(proc_data, f, indent=2)

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

