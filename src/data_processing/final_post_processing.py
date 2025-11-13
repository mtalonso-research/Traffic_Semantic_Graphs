import os
import json
import math
from datetime import datetime, date
import holidays
import re
from collections import defaultdict
from tqdm import tqdm

EARTH_RADIUS_M = 6371000.0
def ego_processing_l2d(input_dir, output_dir):
    """
    Processes ego vehicle data from L2D dataset.
    
    This function reads raw ego vehicle data, converts coordinates from Latitude/Longitude
    to a local East-North-Up (ENU) frame, transforms velocities and accelerations
    into this frame, and saves the processed data to new JSON files.

    Args:
        input_dir (str): The directory containing the input JSON files.
        output_dir (str): The directory where the processed JSON files will be saved.
    """
    # Step 1: Initialization
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
            
        # Step 2: Set Scene Origin
        # The first frame's location is used as the origin for the local ENU coordinate system.
        origin_lat_deg = ego_frames[0]['features']['latitude']
        origin_lon_deg = ego_frames[0]['features']['longitude']
        origin_lat_rad = math.radians(origin_lat_deg)

        processed_ego_frames = []
        # Step 3: Process Each Ego Frame
        for frame_data in ego_frames:
            features = frame_data['features']
            
            # Step 3a: Position Conversion (Lat/Lon -> ENU x, y)
            lat_rad = math.radians(features['latitude'])
            lon_rad = math.radians(features['longitude'])
            
            x = EARTH_RADIUS_M * (lon_rad - math.radians(origin_lon_deg)) * math.cos(origin_lat_rad)
            y = EARTH_RADIUS_M * (lat_rad - math.radians(origin_lat_deg))
            
            # Step 3b: Speed Conversion (km/h -> m/s)
            speed_ms = features['speed'] / 3.6
            
            # Step 3c: Heading Conversion (Compass Degrees -> ENU Radians)
            heading_compass_deg = features['heading']
            heading_enu_deg = (450.0 - heading_compass_deg) % 360.0
            heading_enu_rad = math.radians(heading_enu_deg)
            heading_normalized_rad = (heading_enu_rad + math.pi) % (2 * math.pi) - math.pi

            # Step 3d: Velocity Calculation (ENU vx, vy)
            vx = speed_ms * math.cos(heading_normalized_rad)
            vy = speed_ms * math.sin(heading_normalized_rad)
            
            # Step 3e: Acceleration Transformation (Vehicle Body Frame -> ENU Frame)
            accel_longitudinal = features['accel_x']
            accel_lateral = features['accel_y']
            
            ax_enu = accel_longitudinal * math.cos(heading_normalized_rad) - accel_lateral * math.sin(heading_normalized_rad)
            ay_enu = accel_longitudinal * math.sin(heading_normalized_rad) + accel_lateral * math.cos(heading_normalized_rad)
            
            # Step 3f: Assemble Final Features
            processed_features = {
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'ax': ax_enu,
                'ay': ay_enu,
                'heading': heading_normalized_rad,
                'speed': speed_ms,
                'steering': features['steering']
            }
            
            new_frame = {
                'id': frame_data['id'],
                'features': processed_features
            }
            processed_ego_frames.append(new_frame)

        # Step 4: Update and Save Data
        data['nodes']['ego'] = processed_ego_frames
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=4)

    print("✅ All files processed and saved to:", output_dir)

def ego_processing_nup(input_dir, output_dir):
    """
    Processes ego vehicle data from the nuPlan dataset.

    This function normalizes coordinates relative to the first frame, transforms velocities
    and accelerations from the vehicle's body frame to the world frame, and estimates
    heading where it's missing.

    Args:
        input_dir (str): The directory containing the input JSON files.
        output_dir (str): The directory where the processed JSON files will be saved.
    """
    # Step 1: Initialization
    os.makedirs(output_dir, exist_ok=True)

    for fname in tqdm(os.listdir(input_dir), desc='Ego Vehicle'):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        egos = data.get("nodes", {}).get("ego", [])
        if not egos or len(egos) < 2:
            continue

        # Step 2: Set Scene Origin
        first_ego_features = egos[0].get("features", {})
        x0 = first_ego_features.get("x", 0.0)
        y0 = first_ego_features.get("y", 0.0)
        
        last_known_heading = None

        # Step 3: Process each ego node
        for i, ego_node in enumerate(egos):
            feat = ego_node["features"]
            heading = feat.get("heading")
            heading_is_estimated = False

            # Step 3a: Estimate Heading if missing
            if heading is None:
                heading_is_estimated = True
                current_pos = (feat.get("x") or 0.0, feat.get("y") or 0.0)
                
                # Use displacement to estimate heading
                if i > 0:
                    prev_feat = egos[i-1]["features"]
                    prev_pos = (prev_feat.get("x") or 0.0, prev_feat.get("y") or 0.0)
                    dx = current_pos[0] - prev_pos[0]
                    dy = current_pos[1] - prev_pos[1]
                else: # For the first frame, use the next frame
                    next_feat = egos[i+1]["features"]
                    next_pos = (next_feat.get("x") or 0.0, next_feat.get("y") or 0.0)
                    dx = next_pos[0] - current_pos[0]
                    dy = next_pos[1] - current_pos[1]
                
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    heading = math.atan2(dy, dx)
                elif last_known_heading is not None:
                    heading = last_known_heading
                else: # Fallback for stationary start
                    heading = 0.0
                    print(f"⚠️ Warning: Cannot estimate heading for stationary start in {fname}. Defaulting to 0.")
            
            if heading is not None:
                last_known_heading = heading

            # Step 3b: Transform Velocity and Acceleration to World Frame
            vx_body = feat.get("vx") or 0.0
            vy_body = feat.get("vy") or 0.0
            ax_body = feat.get("ax") or 0.0
            ay_body = feat.get("ay") or 0.0
            
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)

            speed = math.sqrt(vx_body**2 + vy_body**2)
            vx_world = vx_body * cos_h - vy_body * sin_h
            vy_world = vx_body * sin_h + vy_body * cos_h
            ax_world = ax_body * cos_h - ay_body * sin_h
            ay_world = ax_body * sin_h + ay_body * cos_h

            # Step 3c: Normalize Coordinates
            x_abs = feat.get("x") or 0.0
            y_abs = feat.get("y") or 0.0
            x_rel = x_abs - x0
            y_rel = y_abs - y0

            # Step 3d: Assemble Final Features
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

        # Step 4: Write Processed Data
        with open(os.path.join(output_dir, fname), "w") as f:
            json.dump(data, f, indent=2)

    print("✅ All files processed and saved to:", output_dir)

def env_processing_l2d(input_dir):
    """
    Processes and standardizes environment data from the L2D dataset.

    This function maps categorical environment features (like month, day of the week,
    weather conditions) to numerical values and extracts other relevant information.

    Args:
        input_dir (str): The directory containing the input JSON files to be updated in-place.
    """
    # Step 1: Initialization of Mapping Dictionaries
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

    # Step 2: Process Each File
    for fname in tqdm(os.listdir(input_dir), desc='Environment'):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        envs = data.get("nodes", {}).get("environment", [])
        if not envs:
            continue

        # Step 3: Process each environment node
        for env in envs:
            feat = env["features"]

            # Step 3a: Map Textual Features to Numerical Values
            month = month_map.get(str(feat.get("month", "")).lower(), 0)
            day = day_map.get(str(feat.get("day_of_week", "")).lower(), 0)
            cond_str = str(feat.get("conditions", "")).lower()
            conditions = condition_map.get(cond_str, 1 if "cloud" in cond_str else 0)

            # Step 3b: Parse Time of Day to Seconds
            time_str = str(feat.get("time_of_day", "00:00:00"))
            try:
                h, m, s = [int(x) for x in time_str.split(":")]
                time = h * 3600 + m * 60 + s
            except Exception:
                time = 0

            # Step 3c: Sanitize and Extract Other Features
            try:
                precipitation = float(feat.get("precipitation", 0.0))
            except ValueError:
                precipitation = 0.0
            
            lighting = str(feat.get("lighting", "")).lower()
            daylight = lighting.startswith("day")

            # Step 3d: Assemble Final Features
            env["features"] = {
                "month": month,
                "day": day,
                "time": time,
                "conditions": conditions,
                "precipitation": precipitation,
                "daylight": daylight,
                "weekend": day in [5, 6],
                "holiday": False # L2D data does not provide holiday info
            }

        # Step 4: Update File In-Place
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

    print("✅ Environment processing complete.")

def env_processing_nup(input_dir):
    """
    Processes and standardizes environment data from the nuPlan dataset.

    This function extracts date and location from metadata, determines holidays,
    and maps weather conditions to a standardized numerical format.

    Args:
        input_dir (str): The directory containing the input JSON files to be updated in-place.
    """
    # Step 1: Initialization
    day_map = {
        "sunday": 0, "monday": 1, "tuesday": 2, "wednesday": 3,
        "thursday": 4, "friday": 5, "saturday": 6
    }

    # Helper function to parse time string into seconds
    def parse_time_seconds(time_str):
        try:
            h, m, s = [int(x) for x in str(time_str).split(":")]
            return h * 3600 + m * 60 + s
        except Exception:
            return None

    # Helper function to map weather codes/descriptions to a category
    def map_weather_to_condition(code=None, desc=None):
        if code is not None:
            if code in [0, 1]: return 0 # Clear
            elif code in [2, 3]: return 1 # Clouds
            elif code in list(range(51, 68)) + list(range(80, 83)) + list(range(95, 100)): return 2 # Rain
            elif code in list(range(71, 78)) + [85, 86]: return 3 # Snow
            elif code in [45, 48]: return 4 # Fog
            else: return 1 # Default to clouds
        elif desc:
            desc = str(desc).lower()
            if "clear" in desc: return 0
            elif any(k in desc for k in ["cloud", "overcast"]): return 1
            elif any(k in desc for k in ["rain", "drizzle", "thunder"]): return 2
            elif "snow" in desc: return 3
            elif any(k in desc for k in ["fog", "haze", "mist"]): return 4
            else: return 1 # Default to clouds
        return None

    # Step 2: Process Each File
    for fname in tqdm(os.listdir(input_dir), desc='Environment'):
        if not fname.endswith(".json"):
            continue

        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r") as f:
            data = json.load(f)

        # Step 3: Extract Metadata for Holiday Calculation
        log_name = data.get('metadata', {}).get('log_name', '')
        year, month, day_of_month = (None, None, None)
        if log_name:
            try:
                date_parts = log_name.split('.')[:3]
                year, month, day_of_month = int(date_parts[0]), int(date_parts[1]), int(date_parts[2])
            except (ValueError, IndexError): pass

        location = data.get('metadata', {}).get('location', '')
        country, state = ('', '')
        if 'us' in location:
            country, state = 'US', location.split('-')[1].upper() if '-' in location else ''
        elif 'sg' in location:
            country = 'SG'

        holiday_calendar = None
        if country == 'US' and state: holiday_calendar = holidays.US(state=state)
        elif country == 'SG': holiday_calendar = holidays.SG()

        envs = data.get("nodes", {}).get("environment", [])
        if not envs:
            continue

        # Step 4: Process each environment node
        for env in envs:
            feat = env.get("features", {})

            # Step 4a: Parse and Map Features
            month_feat = feat.get("month", None)
            day_str = feat.get("day_of_week", None)
            day = day_map.get(str(day_str).lower(), None) if day_str else None
            time = parse_time_seconds(feat.get("time_of_day"))
            conditions = map_weather_to_condition(feat.get("weather_code"), feat.get("weather_description"))
            
            try:
                precipitation = float(feat.get("precipitation_mm")) if feat.get("precipitation_mm") is not None else None
            except (ValueError, TypeError):
                precipitation = None

            # Step 4b: Determine Weekend and Holiday
            weekend = day in [5, 6]
            holiday = False
            if year and month and day_of_month and holiday_calendar:
                try:
                    holiday = date(year, month, day_of_month) in holiday_calendar
                except ValueError: pass

            # Step 4c: Assemble Final Features
            env["features"] = {
                "month": month_feat,
                "day": day,
                "time": time,
                "conditions": conditions,
                "precipitation": precipitation,
                "daylight": feat.get("is_daylight", None),
                "weekend": weekend,
                "holiday": holiday
            }

        # Step 5: Update File In-Place
        with open(fpath, "w") as f:
            json.dump(data, f, indent=2)

    print("✅ NUP environment processing complete.")

HFOV_DEG = 100.0
IMAGE_WIDTH_PX = 1920.0

def veh_processing_l2d(input_dir, annotation_root):
    """
    Processes vehicle data from the L2D dataset.

    This function calculates the world coordinates (x, y) and velocities (vx, vy) for
    other vehicles. It uses camera intrinsics and ego motion to project vehicles
    from the image plane into the 3D world.

    Args:
        input_dir (str): The directory containing the graph JSON files to be updated.
        annotation_root (str): The root directory of the corresponding annotation files.
    """
    # Step 1: Initialization
    hfov_rad = math.radians(HFOV_DEG)
    image_center_px = IMAGE_WIDTH_PX / 2.0
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Step 2: Process Each File
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
            continue

        # Step 3: Pre-computation and Setup
        ego_lookup = {frame['id']: frame['features'] for frame in ego_frames}
        try:
            parquet_path = data['metadata']['source_files']['parquet']
            match = re.search(r'episode_(\d+)', parquet_path)
            if not match: continue
            ep_num = int(match.group(1))
        except KeyError: continue
            
        processed_vehicle_frames = []
        # Step 4: Main Processing Loop for Each Vehicle
        for vehicle in vehicle_frames:
            try:
                # Step 4a: Find Corresponding Ego and Annotation Data
                parts = vehicle['id'].split('_')
                frame_num = int(parts[-1])
                track_id = "_".join(parts[:-1]) 

                ego_features = ego_lookup.get(f'ego_{frame_num}')
                if not ego_features: continue 

                annotation_path = os.path.join(annotation_root, f'Episode{ep_num:06d}', f'frame_{frame_num:05d}.json')
                with open(annotation_path, 'r') as f_ann:
                    annotation_data = json.load(f_ann)
                
                vehicle_annotation = next((ann for ann in annotation_data['annotations'] if ann['track_id'] == track_id), None)
                if not vehicle_annotation: continue

                # Step 4b: Calculate Position in World Coordinates
                ego_heading_rad = ego_features['heading']
                dist_to_ego = vehicle['features']['dist_to_ego']
                
                bbox = vehicle_annotation['bbox']
                bbox_center_px = bbox[0] + (bbox[2] / 2.0)
                pixel_offset = bbox_center_px - image_center_px
                
                relative_angle = -(pixel_offset / IMAGE_WIDTH_PX) * hfov_rad
                global_angle = ego_heading_rad + relative_angle
                
                x = ego_features['x'] + dist_to_ego * math.cos(global_angle)
                y = ego_features['y'] + dist_to_ego * math.sin(global_angle)
                
                # Step 4c: Transform Relative Velocity to World Frame
                v_rel_lat, v_rel_lon, v_rel_alt = vehicle['features']['velocity_ms']
                cos_h, sin_h = math.cos(ego_heading_rad), math.sin(ego_heading_rad)
                
                v_rot_x = v_rel_lon * cos_h - v_rel_lat * sin_h
                v_rot_y = v_rel_lon * sin_h + v_rel_lat * cos_h
                
                vx = ego_features['vx'] + v_rot_x
                vy = ego_features['vy'] + v_rot_y
                speed = math.sqrt(vx**2 + vy**2)

                # Step 4d: Assemble Final Features
                processed_features = {
                    'x': x, 'y': y, 'vx': vx, 'vy': vy,
                    'speed': speed, 'dist_to_ego': dist_to_ego,
                    'lane_classification': vehicle['features'].get('lane_classification', -1)
                }
                processed_vehicle_frames.append({'id': vehicle['id'], 'features': processed_features, 'type': 'vehicle'})

            except (KeyError, IndexError, FileNotFoundError):
                continue
        
        # Step 5: Update and Save Data
        data['nodes']['vehicle'] = processed_vehicle_frames
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    print(f"✅ All vehicle data processed and files updated in: {input_dir}")

def veh_processing_nup(input_dir, raw_dir):
    """
    Processes vehicle data from the nuPlan dataset.

    This function converts absolute world coordinates to be relative to the scene's
    origin (the ego vehicle's starting position) and calculates the distance to the
    ego vehicle at each timestep.

    Args:
        input_dir (str): The directory containing the processed graph JSON files.
        raw_dir (str): The directory containing the raw (unprocessed) graph JSON files.
    """
    # Step 1: Initialization
    for fname in tqdm(os.listdir(input_dir), desc='Vehicles'):
        if not fname.endswith("_graph.json"):
            continue

        graph_path = os.path.join(input_dir, fname)
        raw_path = os.path.join(raw_dir, fname)
        if not os.path.exists(raw_path): continue

        with open(graph_path, "r") as f: proc_data = json.load(f)
        with open(raw_path, "r") as f: raw_data = json.load(f)

        ego_nodes_raw = raw_data.get("nodes", {}).get("ego", [])
        vehicles = proc_data.get("nodes", {}).get("vehicle", [])
        if not ego_nodes_raw or not vehicles: continue

        # Step 2: Set Scene Origin and Create Ego Position Lookup
        x0 = ego_nodes_raw[0].get("features", {}).get("x", 0.0)
        y0 = ego_nodes_raw[0].get("features", {}).get("y", 0.0)

        ego_lookup = {}
        for ego_node in ego_nodes_raw:
            m = re.search(r"(\d+)$", ego_node["id"])
            if m:
                idx = int(m.group(1))
                f_raw = ego_node.get("features", {})
                if "x" in f_raw and "y" in f_raw:
                    ego_lookup[idx] = (f_raw["x"], f_raw["y"])

        # Step 3: Process each vehicle node
        for v in vehicles:
            # Step 3a: Find Corresponding Ego Position
            m = re.search(r"(\d+)$", v["id"])
            if not m: continue
            frame_idx = int(m.group(1))
            
            ego_coords_current = ego_lookup.get(frame_idx)
            if not ego_coords_current: continue

            # Step 3b: Calculate Scene-Relative Position and Distance to Ego
            f = v.get("features", {})
            x_v_abs, y_v_abs = f.get("x"), f.get("y")
            if x_v_abs is None or y_v_abs is None: continue

            x_rel_scene = x_v_abs - x0
            y_rel_scene = y_v_abs - y0
            
            x_e_current, y_e_current = ego_coords_current
            dist_to_ego = math.sqrt((x_v_abs - x_e_current)**2 + (y_v_abs - y_e_current)**2)
            
            # Step 3c: Calculate Speed
            vx, vy = f.get("vx"), f.get("vy")
            speed = math.sqrt(vx**2 + vy**2) if vx is not None and vy is not None else None

            # Step 3d: Assemble Final Features
            v["features"] = {
                "x": x_rel_scene, "y": y_rel_scene, "vx": vx, "vy": vy,
                "speed": speed, "dist_to_ego": dist_to_ego 
            }
            v["type"] = f.get("type")

        # Step 4: Update and Save Data
        with open(graph_path, "w") as f:
            json.dump(proc_data, f, indent=2)

    print("🏁 Vehicle node NUP processing complete.")

def ped_processing_l2d(input_dir, annotation_root):
    """
    Processes pedestrian data from the L2D dataset.

    This function calculates the world coordinates (x, y) and velocities (vx, vy) for
    pedestrians, mirroring the logic used for vehicles. It projects pedestrians
    from the image plane into the 3D world based on ego motion.

    Args:
        input_dir (str): The directory containing the graph JSON files to be updated.
        annotation_root (str): The root directory of the corresponding annotation files.
    """
    # Step 1: Initialization
    hfov_rad = math.radians(HFOV_DEG)
    image_center_px = IMAGE_WIDTH_PX / 2.0
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Step 2: Process Each File
    for filename in tqdm(file_list, desc="Processing Pedestrians"):
        file_path = os.path.join(input_dir, filename)

        try:
            with open(file_path, 'r') as f: data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Skipping file {filename} due to an error: {e}")
            continue

        pedestrian_frames = data.get('nodes', {}).get('pedestrian', [])
        ego_frames = data.get('nodes', {}).get('ego', [])
        if not pedestrian_frames or not ego_frames: continue

        # Step 3: Pre-computation and Setup
        ego_lookup = {frame['id']: frame['features'] for frame in ego_frames}
        try:
            parquet_path = data['metadata']['source_files']['parquet']
            match = re.search(r'episode_(\d+)', parquet_path)
            if not match: continue
            ep_num = int(match.group(1))
        except KeyError: continue
            
        processed_pedestrian_frames = []
        # Step 4: Main Processing Loop for Each Pedestrian
        for pedestrian in pedestrian_frames:
            try:
                # Step 4a: Find Corresponding Ego and Annotation Data
                parts = pedestrian['id'].split('_')
                frame_num = int(parts[-1])
                track_id = "_".join(parts[:-1]) 

                ego_features = ego_lookup.get(f'ego_{frame_num}')
                if not ego_features: continue 

                annotation_path = os.path.join(annotation_root, f'Episode{ep_num:06d}', f'frame_{frame_num:05d}.json')
                with open(annotation_path, 'r') as f_ann:
                    annotation_data = json.load(f_ann)
                
                pedestrian_annotation = next((ann for ann in annotation_data['annotations'] if ann['track_id'] == track_id), None)
                if not pedestrian_annotation: continue

                # Step 4b: Calculate Position in World Coordinates
                ego_heading_rad = ego_features['heading']
                dist_to_ego = pedestrian['features']['dist_to_ego']
                
                bbox = pedestrian_annotation['bbox']
                bbox_center_px = bbox[0] + (bbox[2] / 2.0)
                pixel_offset = bbox_center_px - image_center_px
                
                relative_angle = -(pixel_offset / IMAGE_WIDTH_PX) * hfov_rad
                global_angle = ego_heading_rad + relative_angle
                
                x = ego_features['x'] + dist_to_ego * math.cos(global_angle)
                y = ego_features['y'] + dist_to_ego * math.sin(global_angle)
                
                # Step 4c: Transform Relative Velocity to World Frame
                v_rel_lat, v_rel_lon, v_rel_alt = pedestrian['features'].get('velocity_ms', [0, 0, 0])
                cos_h, sin_h = math.cos(ego_heading_rad), math.sin(ego_heading_rad)
                
                v_rot_x = v_rel_lon * cos_h - v_rel_lat * sin_h
                v_rot_y = v_rel_lon * sin_h + v_rel_lat * cos_h
                
                vx = ego_features['vx'] + v_rot_x
                vy = ego_features['vy'] + v_rot_y
                speed = math.sqrt(vx**2 + vy**2)

                # Step 4d: Assemble Final Features
                processed_features = {
                    'x': x, 'y': y, 'vx': vx, 'vy': vy,
                    'speed': speed, 'dist_to_ego': dist_to_ego
                }
                processed_pedestrian_frames.append({'id': pedestrian['id'], 'features': processed_features, 'type': 'pedestrian'})

            except (KeyError, IndexError, FileNotFoundError):
                continue

        # Step 5: Update and Save Data
        data['nodes']['pedestrian'] = processed_pedestrian_frames
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    print(f"✅ All pedestrian data processed and files updated in: {input_dir}")

def ped_processing_nup(input_dir, raw_dir):
    """
    Processes pedestrian data from the nuPlan dataset.

    This function converts absolute world coordinates to be relative to the scene's
    origin and calculates the distance to the ego vehicle at each timestep.

    Args:
        input_dir (str): The directory containing the processed graph JSON files.
        raw_dir (str): The directory containing the raw (unprocessed) graph JSON files.
    """
    # Step 1: Initialization
    for fname in tqdm(os.listdir(input_dir), desc='Pedestrians'):
        if not fname.endswith("_graph.json"):
            continue

        graph_path = os.path.join(input_dir, fname)
        raw_path = os.path.join(raw_dir, fname)
        if not os.path.exists(raw_path): continue

        with open(graph_path, "r") as f: proc_data = json.load(f)
        with open(raw_path, "r") as f: raw_data = json.load(f)

        ego_nodes_raw = raw_data.get("nodes", {}).get("ego", [])
        pedestrians = proc_data.get("nodes", {}).get("pedestrian", [])
        if not ego_nodes_raw or not pedestrians: continue
        
        # Step 2: Set Scene Origin and Create Ego Position Lookup
        x0 = ego_nodes_raw[0].get("features", {}).get("x", 0.0)
        y0 = ego_nodes_raw[0].get("features", {}).get("y", 0.0)

        ego_lookup = {}
        for ego_node in ego_nodes_raw:
            m = re.search(r"(\d+)$", ego_node["id"])
            if m:
                idx = int(m.group(1))
                f_raw = ego_node.get("features", {})
                if "x" in f_raw and "y" in f_raw:
                    ego_lookup[idx] = (f_raw["x"], f_raw["y"])

        # Step 3: Process each pedestrian node
        for p in pedestrians:
            # Step 3a: Find Corresponding Ego Position
            m = re.search(r"(\d+)$", p["id"])
            if not m: continue
            frame_idx = int(m.group(1))

            ego_coords_current = ego_lookup.get(frame_idx)
            if not ego_coords_current: continue

            # Step 3b: Calculate Scene-Relative Position and Distance to Ego
            f = p.get("features", {})
            x_p_abs, y_p_abs = f.get("x"), f.get("y")
            if x_p_abs is None or y_p_abs is None: continue

            x_rel_scene = x_p_abs - x0
            y_rel_scene = y_p_abs - y0

            x_e_current, y_e_current = ego_coords_current
            dist_to_ego = math.sqrt((x_p_abs - x_e_current)**2 + (y_p_abs - y_e_current)**2)

            # Step 3c: Calculate Speed
            vx, vy = f.get("vx"), f.get("vy")
            speed = math.sqrt(vx**2 + vy**2) if vx is not None and vy is not None else None

            # Step 3d: Assemble Final Features
            p["features"] = {
                "x": x_rel_scene, "y": y_rel_scene, "vx": vx, "vy": vy,
                "speed": speed, "dist_to_ego": dist_to_ego
            }
            p["type"] = f.get("type")

        # Step 4: Update and Save Data
        with open(graph_path, "w") as f:
            json.dump(proc_data, f, indent=2)

    print("🏁 Pedestrian node NUP processing complete.")

def obj_processing_nup(input_dir, raw_dir):
    """
    Processes generic object data from the nuPlan dataset.

    This function converts absolute world coordinates to be relative to the scene's
    origin, calculates distance to ego, and maps object categories to numerical IDs.

    Args:
        input_dir (str): The directory containing the processed graph JSON files.
        raw_dir (str): The directory containing the raw (unprocessed) graph JSON files.
    """
    # Step 1: Initialization
    category_map = {
        "czone_sign": 0, "traffic_cone": 1,
        "generic_object": 2, "barrier": 3
    }

    for fname in tqdm(os.listdir(input_dir), desc='Objects'):
        if not fname.endswith("_graph.json"):
            continue

        graph_path = os.path.join(input_dir, fname)
        raw_path = os.path.join(raw_dir, fname)
        if not os.path.exists(raw_path): continue

        with open(graph_path, "r") as f: proc_data = json.load(f)
        with open(raw_path, "r") as f: raw_data = json.load(f)

        ego_nodes_raw = raw_data.get("nodes", {}).get("ego", [])
        objects = proc_data.get("nodes", {}).get("object", [])
        if not ego_nodes_raw or not objects: continue
        
        # Step 2: Set Scene Origin and Create Ego Position Lookup
        x0 = ego_nodes_raw[0].get("features", {}).get("x", 0.0)
        y0 = ego_nodes_raw[0].get("features", {}).get("y", 0.0)

        ego_lookup = {}
        for ego_node in ego_nodes_raw:
            m = re.search(r"(\d+)$", ego_node["id"])
            if m:
                idx = int(m.group(1))
                f_raw = ego_node.get("features", {})
                if "x" in f_raw and "y" in f_raw:
                    ego_lookup[idx] = (f_raw["x"], f_raw["y"])

        # Step 3: Process each object node
        for o in objects:
            # Step 3a: Find Corresponding Ego Position
            m = re.search(r"(\d+)$", o["id"])
            if not m: continue
            frame_idx = int(m.group(1))

            ego_coords_current = ego_lookup.get(frame_idx)
            if not ego_coords_current: continue

            # Step 3b: Calculate Scene-Relative Position and Distance to Ego
            f = o.get("features", {})
            x_o_abs, y_o_abs = f.get("x"), f.get("y")
            if x_o_abs is None or y_o_abs is None: continue

            x_rel_scene = x_o_abs - x0
            y_rel_scene = y_o_abs - y0

            x_e_current, y_e_current = ego_coords_current
            dist_to_ego = math.sqrt((x_o_abs - x_e_current)**2 + (y_o_abs - y_e_current)**2)

            # Step 3c: Map Category to Index
            cat_str = f.get("category", "generic_object")
            cat_idx = category_map.get(cat_str, 2)

            # Step 3d: Assemble Final Features
            o["features"] = {
                "x": x_rel_scene, "y": y_rel_scene,
                "dist_to_ego": dist_to_ego, "category": cat_idx
            }
            o["type"] = f.get("type")

        # Step 4: Update and Save Data
        with open(graph_path, "w") as f:
            json.dump(proc_data, f, indent=2)

    print("🏁 Object node NUP processing complete.")
    