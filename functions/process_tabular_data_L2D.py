import pandas as pd
import overpy
import time
import warnings
from tqdm import tqdm
import os
import json
import glob
from functions.utils import prepare_and_save_parquet, get_chunk_num

def expand_columns(df):
    # Expand vehicle state
    vehicle_fields = [
        "speed", "heading", "heading_error",
        "latitude", "longitude", "altitude",
        "acceleration_x", "acceleration_y"
    ]
    for i, name in enumerate(vehicle_fields):
        df[f'vehicle_{name}'] = df['observation.state.vehicle'][i]

    # Restructure waypoints into a list of dicts
    waypoints = []
    for point in df['observation.state.waypoints']:
        waypoints.append({"x": point[0], "y": point[1]})
    df['waypoints'] = waypoints

    # Expand timestamp
    df['obs_unix_timestamp'] = df['observation.state.timestamp']

    # Expand continuous actions
    cont_actions = ['gas_pedal', 'brake_pedal', 'steering_angle']
    for i, name in enumerate(cont_actions):
        df[f'action_{name}'] = df['action.continuous'][i]

    # Expand discrete actions
    df['action_gear'] = df['action.discrete'][0]
    df['action_turn_signal'] = df['action.discrete'][1]

    return df

def get_osm_features(lat, lon, time_sleep=1):
    api = overpy.Overpass()
    time.sleep(time_sleep)  # prevent Overpass rate-limit errors
    errors = 0

    query = f"""
    [out:json][timeout:25];
    (
      way(around:30,{lat},{lon})["highway"];
      node(around:30,{lat},{lon})["highway"="crossing"];
      node(around:30,{lat},{lon})["highway"="stop"];
      node(around:30,{lat},{lon})["highway"="give_way"];
      node(around:30,{lat},{lon})["highway"="traffic_signals"];
      node(around:30,{lat},{lon})["traffic_calming"];
      node(around:30,{lat},{lon})["railway"="level_crossing"];
      way(around:30,{lat},{lon})["junction"="roundabout"];
      way(around:100,{lat},{lon})["landuse"];
      node(around:100,{lat},{lon})["place"];
    );
    out body;
    >;
    out skel qt;
    """

    try:
        result = api.query(query)
    except Exception as e:
        #print(f"Query failed at lat={lat}, lon={lon}: {e}")
        errors = 1
        return {}, errors

    # Initialize outputs
    road = {}
    traffic_controls = set()
    traffic_features = set()
    amenities = set()
    shops = set()
    landuse = set()
    places = set()
    derived_flags = {}

    for way in result.ways:
        if "highway" in way.tags and not road:
            road = {
                "road_type": way.tags.get("highway"),
                "road_name": way.tags.get("name"),
                "maxspeed": way.tags.get("maxspeed"),
                "lanes": way.tags.get("lanes"),
                "surface": way.tags.get("surface"),
                "oneway": way.tags.get("oneway"),
                "width": way.tags.get("width"),
                "sidewalk": way.tags.get("sidewalk"),
                "bicycle": way.tags.get("bicycle"),
                "bridge": way.tags.get("bridge", "no"),
                "tunnel": way.tags.get("tunnel", "no")
            }
        if "landuse" in way.tags:
            landuse.add(way.tags["landuse"])
        if way.tags.get("junction") == "roundabout":
            traffic_controls.add("roundabout")

    for node in result.nodes:
        if node.tags.get("highway") == "crossing":
            traffic_features.add("pedestrian_crossing")
        elif node.tags.get("highway") == "bus_stop":
            traffic_features.add("bus_stop")
        elif node.tags.get("highway") == "stop":
            traffic_controls.add("stop_sign")
        elif node.tags.get("highway") == "give_way":
            traffic_controls.add("yield_sign")
        elif node.tags.get("highway") == "traffic_signals":
            traffic_controls.add("traffic_signal")
        if "traffic_calming" in node.tags:
            traffic_controls.add(f"traffic_calming:{node.tags['traffic_calming']}")
        if node.tags.get("railway") == "level_crossing":
            traffic_controls.add("railway_crossing")

    if road:
        width_val = road.get("width")
        derived_flags = {
            "is_narrow": float(width_val.replace('m', '').strip()) < 5.0 if isinstance(width_val, str) and width_val.strip() else None,
            "is_unlit": road.get("lit") == "no",
            "bike_friendly": road.get("bicycle") in ["yes", "designated"],
        }

    flat_features = {**road}
    flat_features.update({
        "traffic_controls": ", ".join(sorted(traffic_controls)) if traffic_controls else None,
        "traffic_features": ", ".join(sorted(traffic_features)) if traffic_features else None,
        "landuse": ", ".join(sorted(landuse)) if landuse else None,
    })
    flat_features.update(derived_flags)

    return flat_features, errors

def enrich_dataframe_with_osm_tags(df, lat_col="lat", lon_col="lon", time_sleep=1, verbose=True):
    enriched_rows = []
    error_counter = 0

    for idx, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        if verbose: print(f"Processing row {idx} — lat: {lat}, lon: {lon}")
        enriched, errors = get_osm_features(lat, lon, time_sleep)
        error_counter += errors
        enriched_rows.append(enriched)

    enriched_df = pd.DataFrame(enriched_rows)
    return pd.concat([df.reset_index(drop=True), enriched_df.reset_index(drop=True)], axis=1), error_counter

def preprocess_df_columns(df):
    df = df.apply(expand_columns,axis=1)

    # Convert the UNIX timestamp to datetime
    df['obs_datetime'] = pd.to_datetime(df['obs_unix_timestamp'])

    # Extract components
    df['month'] = df['obs_datetime'].dt.month_name()
    df['day_of_week'] = df['obs_datetime'].dt.day_name()
    df['time_of_day'] = df['obs_datetime'].dt.strftime('%H:%M:%S')

    do = {'policy':df['task.policy'].iloc[0],'instructions':df['task.instructions'].iloc[0]}

    df = df.drop(columns=[
        "observation.state.vehicle",
        "observation.state.waypoints",
        "observation.state.timestamp",
        "action.continuous",
        "action.discrete",
        'obs_unix_timestamp',
        'obs_datetime',
        'timestamp',
        'frame_index',
        'episode_index',
        'index',
        'task_index',
        'task.policy',
        'task.instructions'
    ])
    
    return df, do

#===============================================================
#         NEED TO EDIT TURNING BEHAVIOR FUNCTION HERE
#===============================================================
def classify_turning_with_lane_change(angle_sum, turn_signal):
    if pd.isna(angle_sum) and (pd.isna(turn_signal) or turn_signal == "none"):
        return "unknown"

    # LANE CHANGE detection (signal + small angle + decent speed)
    if 3 <= abs(angle_sum) <= 30:
        if turn_signal == 1:
            return "lane change left"
        elif turn_signal == 2:
            return "lane change right"

    # TURN classification
    if -30 <= angle_sum <= 30:
        angle_label = "straight"
    elif 30 < angle_sum <= 135:
        angle_label = "turning right"
    elif angle_sum > 135:
        angle_label = "u-turn right"
    elif -135 <= angle_sum < -30:
        angle_label = "turning left"
    elif angle_sum < -135:
        angle_label = "u-turn left"
    else:
        angle_label = "unknown"

    # Signal-based override or inconsistency
    #if turn_signal == 1:
    #    if "right" in angle_label:
    #        return "inconsistent (left signal, right turn)"
    #    elif angle_label in ["straight", "unknown"]:
    #        return "turning left (signal)"
    #elif turn_signal == 2:
    #    if "left" in angle_label:
    #        return "inconsistent (right signal, left turn)"
    #    elif angle_label in ["straight", "unknown"]:
    #        return "turning right (signal)"
    
    return angle_label
def compute_turning_behavior_with_lane_change(df, angle_col='angle_change', signal_col='action_turn_signal'):
    # Normalize angle
    df['norm_angle'] = df[angle_col] * 180

    # Apply classifier
    df['turning_behavior'] = df.apply(
        lambda row: classify_turning_with_lane_change(
            row['norm_angle'],
            row.get(signal_col, None),
        ),
        axis=1
    )
    df.drop(columns=['norm_angle'],inplace=True)

    return df

def process_tabular_data(min_ep,max_ep=-1,n_sec=3,
                         source_dir='../data/raw/L2D/tabular',
                         output_dir_processed='../data/processed/L2D',
                         output_dir_tags='../data/semantic_tags/L2D',
                         overwrite=False, process_columns=True, 
                         process_osm=True, process_turning=True,
                         time_sleep=1):
    
    if not isinstance(min_ep, list):
        if max_ep == -1: 
            max_ep = min_ep + 1
        iterable = range(min_ep,max_ep)
    else:
        iterable = min_ep

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        pbar = tqdm(iterable)
        for ep_num in pbar:
            chunk = get_chunk_num(ep_num)
            source_parquet = os.path.join(source_dir,f"episode_{ep_num:06d}.parquet")
            output_parquet = os.path.join(output_dir_processed,f"episode_{ep_num:06d}.parquet")
            output_json = os.path.join(output_dir_tags,f"episode_{ep_num:06d}.json")

            if os.path.exists(output_parquet) and os.path.exists(output_json) and not overwrite:
                continue

            #try:
            df = pd.read_parquet(source_parquet, engine="pyarrow").astype("object")
            if process_columns:
                df, do = preprocess_df_columns(df)
            if process_osm:
                df, error_counter = enrich_dataframe_with_osm_tags(
                    df, 'vehicle_latitude', 'vehicle_longitude', time_sleep, verbose=False
                )
                error_ratio = f"{error_counter} / {len(df)}"
                pbar.set_postfix({"errors": error_ratio})
            if process_turning:
                df = compute_turning_behavior_with_lane_change(
                    df, angle_col='action_steering_angle', signal_col='action_turn_signal'
                )

            os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
            prepare_and_save_parquet(df, output_parquet)

            os.makedirs(os.path.dirname(output_json), exist_ok=True)
            with open(output_json, 'w') as f:
                json.dump(do, f, indent=4)

            #except Exception as e:
                #print(f"Error processing episode {ep_num} in chunk {chunk}: {e}")

def process_tabular_split(min_ep, max_ep=-1,
                          source_dir="../data/raw/L2D/tabular",
                          output_dir="../data/processed/L2D/tabular",
                          overwrite=False):
    """
    Iterate over shard files (parquets) indexed by min_ep:max_ep,
    split all episodes inside each shard, and save per-episode parquet files.
    """
    # Resolve iterable
    if not isinstance(min_ep, list):
        if max_ep == -1:
            max_ep = min_ep + 1
        iterable = range(min_ep, max_ep)
    else:
        iterable = min_ep

    # Collect shard files
    parquet_files = sorted(glob.glob(os.path.join(source_dir, "*.parquet")))
    os.makedirs(output_dir, exist_ok=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        pbar = tqdm(iterable, desc="Splitting episodes (tabular)")
        for file_idx in pbar:
            if file_idx >= len(parquet_files):
                continue  # skip out-of-range
            file_path = parquet_files[file_idx]
            df = pd.read_parquet(file_path)

            # Group by episode and save
            for ep_id, ep_df in df.groupby("episode_index"):
                out_path = os.path.join(output_dir, f"episode_{ep_id:06d}.parquet")
                if os.path.exists(out_path) and not overwrite:
                    continue
                ep_df.sort_values("frame_index").to_parquet(out_path, index=False)

