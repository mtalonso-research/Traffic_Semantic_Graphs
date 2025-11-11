import pandas as pd
import overpy
import time
import warnings
from tqdm import tqdm
import os
import json
import glob
from src.utils import prepare_and_save_parquet, get_chunk_num

def expand_columns(df):
    """
    Expands the columns of a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to expand.
        
    Returns:
        pd.DataFrame: The expanded DataFrame.
    """
    # Step 1: Expand vehicle state
    vehicle_fields = [
        "speed", "heading", "heading_error",
        "latitude", "longitude", "altitude",
        "acceleration_x", "acceleration_y"
    ]
    for i, name in enumerate(vehicle_fields):
        df[f'vehicle_{name}'] = df['observation.state.vehicle'][i]

    # Step 2: Restructure waypoints into a list of dicts
    waypoints = []
    for point in df['observation.state.waypoints']:
        waypoints.append({"x": point[0], "y": point[1]})
    df['waypoints'] = waypoints

    # Step 3: Expand timestamp
    df['obs_unix_timestamp'] = df['observation.state.timestamp']

    # Step 4: Expand continuous actions
    cont_actions = ['gas_pedal', 'brake_pedal', 'steering_angle']
    for i, name in enumerate(cont_actions):
        df[f'action_{name}'] = df['action.continuous'][i]

    # Step 5: Expand discrete actions
    df['action_gear'] = df['action.discrete'][0]
    df['action_turn_signal'] = df['action.discrete'][1]

    return df

import overpy
import time

def get_osm_features(lat, lon, time_sleep=1):
    """
    Gets OpenStreetMap features for a given latitude and longitude.
    
    Args:
        lat (float): The latitude.
        lon (float): The longitude.
        time_sleep (int, optional): The number of seconds to sleep between requests. Defaults to 1.
        
    Returns:
        tuple: A tuple containing a dictionary of OSM features and the number of errors.
    """
    # Step 1: Initialize the API and error counter
    api = overpy.Overpass()
    time.sleep(time_sleep)
    errors = 0

    # Step 2: Build the query
    query = f"""
    [out:json][timeout:25];
    (
      node(around:30,{lat},{lon})["highway"="crossing"];
      node(around:30,{lat},{lon})["highway"="stop"];
      node(around:30,{lat},{lon})["highway"="give_way"];
      node(around:30,{lat},{lon})["highway"="traffic_signals"];
      node(around:30,{lat},{lon})["highway"="bus_stop"];
      node(around:30,{lat},{lon})["railway"="level_crossing"];
      node(around:30,{lat},{lon})["traffic_calming"];
      way(around:30,{lat},{lon})["junction"="roundabout"];
    );
    out body;
    >;
    out skel qt;
    """

    # Step 3: Execute the query
    try:
        result = api.query(query)
    except Exception as e:
        errors = 1
        return {"traffic_controls": [], "traffic_features": []}, errors

    traffic_controls = set()
    traffic_features = set()

    # Step 4: Process the ways
    for way in result.ways:
        if way.tags.get("junction") == "roundabout":
            traffic_controls.add("roundabout")

    # Step 5: Process the nodes
    for node in result.nodes:
        highway_tag = node.tags.get("highway")
        railway_tag = node.tags.get("railway")

        if highway_tag == "crossing":
            traffic_features.add("pedestrian_crossing")
        elif highway_tag == "bus_stop":
            traffic_features.add("bus_stop")
        elif highway_tag == "stop":
            traffic_controls.add("stop_sign")
        elif highway_tag == "give_way":
            traffic_controls.add("yield_sign")
        elif highway_tag == "traffic_signals":
            traffic_controls.add("traffic_signal")

        if "traffic_calming" in node.tags:
            tc = node.tags["traffic_calming"]
            traffic_controls.add(f"traffic_calming:{tc}")

        if railway_tag == "level_crossing":
            traffic_controls.add("railway_crossing")

    # Step 6: Return the features
    flat_output = {
        "traffic_controls": sorted(list(traffic_controls)),
        "traffic_features": sorted(list(traffic_features))
    }

    return flat_output, errors


def enrich_dataframe_with_osm_tags(df, lat_col="lat", lon_col="lon", time_sleep=1, verbose=True):
    """
    Enriches a DataFrame with OpenStreetMap tags.
    
    Args:
        df (pd.DataFrame): The DataFrame to enrich.
        lat_col (str, optional): The name of the latitude column. Defaults to "lat".
        lon_col (str, optional): The name of the longitude column. Defaults to "lon".
        time_sleep (int, optional): The number of seconds to sleep between requests. Defaults to 1.
        verbose (bool, optional): Whether to print verbose output. Defaults to True.
        
    Returns:
        tuple: A tuple containing the enriched DataFrame and the number of errors.
    """
    # Step 1: Initialize the enriched rows and error counter
    enriched_rows = []
    error_counter = 0

    # Step 2: Process each row
    for idx, row in df.iterrows():
        lat = row[lat_col]
        lon = row[lon_col]
        if verbose: print(f"Processing row {idx} — lat: {lat}, lon: {lon}")
        enriched, errors = get_osm_features(lat, lon, time_sleep)
        error_counter += errors
        enriched_rows.append(enriched)

    # Step 3: Return the enriched DataFrame
    enriched_df = pd.DataFrame(enriched_rows)
    return pd.concat([df.reset_index(drop=True), enriched_df.reset_index(drop=True)], axis=1), error_counter

def preprocess_df_columns(df):
    """
    Preprocesses the columns of a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Step 1: Expand the columns
    df = df.apply(expand_columns,axis=1)

    timestamp_unit = 'ns' 
    local_timezone = 'Europe/Berlin' 

    # Step 2: Convert the timestamp to a datetime object
    df['obs_datetime'] = pd.to_datetime(
        df['observation.state.timestamp'], 
        unit=timestamp_unit,
        utc=True
    )
    df['obs_datetime_local'] = df['obs_datetime'].dt.tz_convert(local_timezone)
    df['month'] = df['obs_datetime_local'].dt.month_name()
    df['day_of_week'] = df['obs_datetime_local'].dt.day_name()
    df['time_of_day'] = df['obs_datetime_local'].dt.strftime('%H:%M:%S')

    # Step 3: Drop the original columns
    df = df.drop(columns=[
        "observation.state.vehicle",
        "observation.state.waypoints",
        "observation.state.timestamp",
        "action.continuous",
        "action.discrete",
        'obs_datetime',
        'timestamp',
        'frame_index',
        'episode_index',
        'index',
        'task_index',
        'task.policy',
        'task.instructions'
    ])
    
    return df

def classify_turning_with_lane_change(angle_sum, turn_signal):
    """
    Classifies the turning behavior of a vehicle.
    
    Args:
        angle_sum (float): The sum of the steering angles.
        turn_signal (int): The turn signal.
        
    Returns:
        str: The turning behavior.
    """
    if pd.isna(angle_sum) and (pd.isna(turn_signal) or turn_signal == "none"):
        return "unknown"

    if 3 <= abs(angle_sum) <= 30:
        if turn_signal == 1:
            return "lane change left"
        elif turn_signal == 2:
            return "lane change right"

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
    
    return angle_label
def compute_turning_behavior_with_lane_change(df, angle_col='angle_change', signal_col='action_turn_signal'):
    """
    Computes the turning behavior of a vehicle.
    
    Args:
        df (pd.DataFrame): The DataFrame to compute the turning behavior for.
        angle_col (str, optional): The name of the angle column. Defaults to 'angle_change'.
        signal_col (str, optional): The name of the signal column. Defaults to 'action_turn_signal'.
        
    Returns:
        pd.DataFrame: The DataFrame with the turning behavior.
    """
    # Step 1: Normalize the angle
    df['norm_angle'] = df[angle_col] * 180

    # Step 2: Apply the classifier
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
    """
    Processes the tabular data.
    
    Args:
        min_ep (int or list): The minimum episode number to process, or a list of episode numbers.
        max_ep (int, optional): The maximum episode number to process. Defaults to -1.
        n_sec (int, optional): The number of seconds of data to process. Defaults to 3.
        source_dir (str, optional): The directory containing the source data. Defaults to '../data/raw/L2D/tabular'.
        output_dir_processed (str, optional): The directory where the processed data will be saved. Defaults to '../data/processed/L2D'.
        output_dir_tags (str, optional): The directory where the tags will be saved. Defaults to '../data/semantic_tags/L2D'.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
        process_columns (bool, optional): Whether to process the columns. Defaults to True.
        process_osm (bool, optional): Whether to process the OpenStreetMap data. Defaults to True.
        process_turning (bool, optional): Whether to process the turning behavior. Defaults to True.
        time_sleep (int, optional): The number of seconds to sleep between requests. Defaults to 1.
    """
    # Step 1: Initialize the iterable
    if not isinstance(min_ep, list):
        if max_ep == -1: 
            max_ep = min_ep + 1
        iterable = range(min_ep,max_ep)
    else:
        iterable = min_ep

    # Step 2: Process each episode
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

            df = pd.read_parquet(source_parquet, engine="pyarrow").astype("object")
            if process_columns:
                df = preprocess_df_columns(df)
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

def process_tabular_split(min_ep, max_ep=-1,
                          source_dir="../data/raw/L2D/tabular",
                          output_dir="../data/processed/L2D/tabular",
                          overwrite=False):
    """
    Splits the tabular data into per-episode parquet files.
    
    Args:
        min_ep (int or list): The minimum episode number to process, or a list of episode numbers.
        max_ep (int, optional): The maximum episode number to process. Defaults to -1.
        source_dir (str, optional): The directory containing the source data. Defaults to "../data/raw/L2D/tabular".
        output_dir (str, optional): The directory where the processed data will be saved. Defaults to "../data/processed/L2D/tabular".
        overwrite (bool, optional): Whether to overwrite existing files. Defaults to False.
    """
    # Step 1: Initialize the iterable
    if not isinstance(min_ep, list):
        if max_ep == -1:
            max_ep = min_ep + 1
        iterable = range(min_ep, max_ep)
    else:
        iterable = min_ep

    # Step 2: Collect the parquet files
    parquet_files = sorted(glob.glob(os.path.join(source_dir, "*.parquet")))
    os.makedirs(output_dir, exist_ok=True)

    # Step 3: Process each parquet file
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        pbar = tqdm(iterable, desc="Splitting episodes (tabular)")
        for file_idx in pbar:
            if file_idx >= len(parquet_files):
                continue
            file_path = parquet_files[file_idx]
            df = pd.read_parquet(file_path)

            for ep_id, ep_df in df.groupby("episode_index"):
                out_path = os.path.join(output_dir, f"episode_{ep_id:06d}.parquet")
                if os.path.exists(out_path) and not overwrite:
                    continue
                ep_df.sort_values("frame_index").to_parquet(out_path, index=False)