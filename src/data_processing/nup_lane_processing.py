import os
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import sqlite3
import traceback

from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from shapely.geometry import Point as ShapelyPoint
from src.utils import DummyWorker

def get_map_name_from_db(db_path: str, scene_token: str) -> str:
    """
    Connects to the nuPlan DB and retrieves the map name (location) for a given scene_token.
    """
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.cursor()
            scene_token_bytes = bytes.fromhex(scene_token)
            
            cursor.execute("SELECT log_token FROM scene WHERE token = ?", (scene_token_bytes,))
            log_token_result = cursor.fetchone()
            if not log_token_result:
                return None
            
            log_token_bytes = log_token_result[0]
            
            cursor.execute("SELECT location FROM log WHERE token = ?", (log_token_bytes,))
            location_result = cursor.fetchone()
            
            return location_result[0] if location_result else None
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

def process_lanes(json_dir: str, db_dir: str, map_dir: str, city_name: str, episodes: list = None):
    """
    Processes lane information for vehicle nodes in nuPlan graphs.
    """
    file_mapping_path = Path(json_dir) / "file_mapping.csv"
    if not file_mapping_path.exists():
        print(f"Error: {file_mapping_path} not found. Please run the enrichment step first.")
        return

    file_mapping_df = pd.read_csv(file_mapping_path)

    if episodes is not None:
        episodes_str = [str(e) for e in episodes]
        file_mapping_df = file_mapping_df[file_mapping_df['json_file_name'].str.split('_').str[0].isin(episodes_str)]

    if file_mapping_df.empty:
        print("No episodes found to process after filtering. Exiting.")
        return

    # Create a cache for map_api objects to avoid reloading for the same map
    map_api_cache = {}

    for episode_idx, group in tqdm(file_mapping_df.groupby(file_mapping_df['json_file_name'].str.split('_').str[0]), desc="Processing Episodes"):
        first_row = group.iloc[0]
        db_name = first_row['db_name']
        scene_token = first_row['scene_token']
        db_path = str(Path(db_dir) / db_name)
        
        map_name = get_map_name_from_db(db_path, scene_token)
        if not map_name:
            print(f"Warning: Could not find map name for scene {scene_token} in {db_name}. Skipping.")
            continue

        # Correct map name 
        if map_name == 'las_vegas': map_name = 'us-nv-las-vegas-strip'
        elif map_name == 'boston': map_name = 'us-ma-boston'
        elif map_name == 'pitssburgh': map_name = 'us-pa-pittsburgh-hazelwood'
        elif map_name == 'singapore': map_name = 'sg-one-north'
        #else: print('MAP NAME NOT RECOGNIZED')

        if map_name in map_api_cache:
            map_api = map_api_cache[map_name]
        else:
            print(f"Loading map API for '{map_name}'...")
            map_dir_abs = str(Path(map_dir).resolve())
            map_api = get_maps_api(map_dir_abs, "nuplan-maps-v1.0", map_name)
            map_api_cache[map_name] = map_api

        for _, row in tqdm(group.iterrows(), desc=f"Processing graphs for Ep. {episode_idx}", total=len(group), leave=False):
            json_path = os.path.join(json_dir, row['json_file_name'])
            with open(json_path, 'r') as f:
                graph = json.load(f)

            nodes_by_frame = {}
            for node_type in graph['nodes']:
                for node in graph['nodes'][node_type]:
                    if '_' in node['id']:
                        frame = int(node['id'].split('_')[-1])
                        if frame not in nodes_by_frame:
                            nodes_by_frame[frame] = []
                        nodes_by_frame[frame].append(node)

            for frame, nodes in nodes_by_frame.items():
                ego_node = next((n for n in nodes if n['features']['type'] == 'ego'), None)
                if not ego_node or ego_node['features']['x'] is None:
                    continue

                ego_heading = ego_node['features']['heading'] if ego_node['features']['heading'] is not None else 0.0
                ego_pos = StateSE2(ego_node['features']['x'], ego_node['features']['y'], ego_heading)
                
                try:
                    proximal_lanes_dict = map_api.get_proximal_map_objects(ego_pos, 20.0, [SemanticMapLayer.LANE])
                    proximal_lanes = proximal_lanes_dict.get(SemanticMapLayer.LANE, [])

                    if not proximal_lanes:
                        continue

                    ego_point = ShapelyPoint(ego_pos.x, ego_pos.y)
                    ego_lane = min(proximal_lanes, key=lambda lane: lane.polygon.distance(ego_point))
                    
                    if not ego_lane:
                        continue
                    
                    ego_node['features']['lane_id'] = ego_lane.id
                    
                except Exception as e:
                    print(f"Debug: An error occurred during ego lane processing: {e}")
                    print(traceback.format_exc())
                    continue

                for vehicle_node in (n for n in nodes if n['features'].get('type') == 'vehicle' and n['features']['x'] is not None):
                    veh_pos = StateSE2(vehicle_node['features']['x'], vehicle_node['features']['y'], 0.0)
                    
                    try:
                        proximal_veh_lanes_dict = map_api.get_proximal_map_objects(veh_pos, 5.0, [SemanticMapLayer.LANE])
                        proximal_veh_lanes = proximal_veh_lanes_dict.get(SemanticMapLayer.LANE, [])

                        if not proximal_veh_lanes:
                            vehicle_node['features']['lane_id'] = None
                            vehicle_node['features']['lane_classification'] = -1
                            continue

                        veh_point = ShapelyPoint(veh_pos.x, veh_pos.y)
                        veh_lane = min(proximal_veh_lanes, key=lambda lane: lane.polygon.distance(veh_point))

                        if not veh_lane:
                            vehicle_node['features']['lane_id'] = None
                            vehicle_node['features']['lane_classification'] = -1
                            continue
                        
                        vehicle_node['features']['lane_id'] = veh_lane.id

                        if veh_lane.id == ego_lane.id:
                            classification = 1 # Same lane
                        elif veh_lane.is_same_roadblock(ego_lane):
                            if veh_lane.is_left_of(ego_lane):
                                classification = 0 # Left lane
                            elif veh_lane.is_right_of(ego_lane):
                                classification = 2 # Right lane
                            else:
                                classification = 4 # Same roadblock, not left/right
                        else:
                            classification = 3 # Different roadblock
                        
                        vehicle_node['features']['lane_classification'] = classification
                    
                    except Exception as e:
                        print(f"Debug: An error occurred during vehicle lane processing: {e}")
                        print(traceback.format_exc())
                        vehicle_node['features']['lane_classification'] = -1

            with open(json_path, 'w') as f:
                json.dump(graph, f, indent=2)

    print("✅ Finished processing lanes.")