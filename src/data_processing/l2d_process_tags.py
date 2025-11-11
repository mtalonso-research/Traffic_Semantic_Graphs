import pandas as pd
import json
import warnings
from tqdm import tqdm
import os
from src.utils import load_and_restore_parquet

def filter_compensating_turns(turns):
    """
    Filters out compensating turns from a list of turns.
    
    Args:
        turns (list): A list of turns.
        
    Returns:
        pd.Series: A pandas Series of filtered turns.
    """
    # Step 1: Initialize the filtered list
    filtered = []
    i = 0
    while i < len(turns):
        current = turns[i].lower()

        # Step 2: Look ahead to check for a compensating turn
        if i + 1 < len(turns):
            next_turn = turns[i + 1].lower()
            if (
                ('lane change left' in current and 'turning right' in next_turn) or
                ('lane change right' in current and 'turning left' in next_turn)
            ):
                filtered.append(current)
                i += 2
                continue

        # Step 3: Add the current turn to the filtered list
        filtered.append(current)
        i += 1

    return pd.Series(filtered)

def detect_speed_decreased_during_turn(df, direction='right', window=5):
    """
    Detects if the speed decreased during a turn.
    
    Args:
        df (pd.DataFrame): The DataFrame to check.
        direction (str, optional): The direction of the turn. Defaults to 'right'.
        window (int, optional): The window size to check for speed decrease. Defaults to 5.
        
    Returns:
        bool: True if the speed decreased, False otherwise.
    """
    # Step 1: Get the turning behavior and speeds
    turning = df.get('turning_behavior', pd.Series(dtype=str)).dropna().astype(str).reset_index(drop=True)
    speeds = df.get('vehicle_speed', pd.Series(dtype=float)).reset_index(drop=True)

    # Step 2: Find the index of the first matching turn
    turn_idx = turning[turning.str.contains(f'turning {direction}', case=False)].index

    if len(turn_idx) == 0:
        return True

    idx = turn_idx[0]
    start = max(0, idx - window)
    end = min(len(speeds), idx + window + 1)

    before = speeds[start:idx]
    after = speeds[idx:end]

    if before.empty or after.empty:
        return True

    # Step 3: Check if the speed decreased
    avg_before = before.mean()
    avg_after = after.mean()

    return avg_after < avg_before

def slowed_down_after_straight(df, direction='right'):
    """
    Detects if the vehicle slowed down after going straight.
    
    Args:
        df (pd.DataFrame): The DataFrame to check.
        direction (str, optional): The direction of the turn. Defaults to 'right'.
        
    Returns:
        bool: True if the vehicle slowed down, False otherwise.
    """
    # Step 1: Get the turning behavior and speeds
    turning = df.get('turning_behavior', pd.Series(dtype=str)).dropna().astype(str).reset_index(drop=True)
    speeds = df.get('vehicle_speed', pd.Series(dtype=float)).reset_index(drop=True)

    # Step 2: Check for a transition from straight to turning
    for i in range(1, len(turning)):
        prev = turning[i - 1].lower()
        curr = turning[i].lower()

        if 'straight' in prev and f'turning {direction}' in curr:
            speed_before = speeds[i - 1]
            speed_after = speeds[i]
            return speed_after < speed_before
        
    if turning[0].lower() == f'turning {direction}' and len(speeds) > 1:
        return speeds[1] <= speeds[0]

    return True

def assign_action_tag(df):
    """
    Assigns an action tag to a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to assign the tag to.
        
    Returns:
        str: The action tag.
    """
    def safe_str_contains(series, keyword):
        return series.dropna().astype(str).str.contains(keyword, case=False)

    # Step 1: Get the road type, turning behavior, and gear
    road_type = df.get('road_type', pd.Series(dtype=str))
    raw_turning = df.get('turning_behavior', pd.Series(dtype=str)).dropna().astype(str).tolist()
    turning_behavior = filter_compensating_turns(raw_turning)

    action_gear = df.get('action_gear', pd.Series(dtype=str)).dropna().astype(str).tolist()
    u_turn_detected = any('u-turn' in turn.lower() for turn in turning_behavior)
    gear_has_one = any('1' in gear for gear in action_gear)
    final_speed = df.get('vehicle_speed', pd.Series(dtype=float)).dropna().values[-1] if not df.get('vehicle_speed', pd.Series()).empty else None
    speed_threshold = 2.0
    road_type_contains_link = safe_str_contains(road_type, '_link').any()

    # Step 2: Assign the action tag based on priority
    if df.astype(str).apply(lambda col: col.str.contains('roundabout', case=False, na=False)).any().any():
        return 'roundabout'
        
    if gear_has_one:
        if u_turn_detected:
            if final_speed is not None and final_speed < speed_threshold:
                return 'parking'
            elif final_speed is not None and final_speed >= speed_threshold:
                return '3-point-turn'
        else:
            return 'backing_up'
        
    left_turns = safe_str_contains(turning_behavior, 'turning left').sum()
    right_turns = safe_str_contains(turning_behavior, 'turning right').sum()
    total_turns = left_turns + right_turns

    if road_type_contains_link: 
        return 'merge'

    if total_turns > 0:
        ratio = abs(left_turns - right_turns) / total_turns
        if ratio < 0.5 and final_speed < speed_threshold: 
            return 'parking'
        elif ratio < 0.5 and final_speed >= speed_threshold: 
            return 'uncertain'
        elif left_turns > right_turns:
                return 'left_turn'
        elif right_turns > left_turns:
                return 'right_turn'

    return 'straight'

def assign_traffic_control_tag(df):
    """
    Assigns a traffic control tag to a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to assign the tag to.
        
    Returns:
        str: The traffic control tag.
    """
    # Step 1: Define the priority order of traffic controls
    priority_order = ['traffic_signal', 'stop_sign', 'roundabout', 'yield_sign']
    col = df.get('traffic_controls', pd.Series(dtype=str)).dropna()

    found_controls = set()

    # Step 2: Find all traffic controls in the DataFrame
    for item in col:
        if isinstance(item, list):
            entries = item
        elif isinstance(item, str):
            entries = [v.strip() for v in item.split(',')]
        else:
            continue

        for val in entries:
            val_clean = str(val).strip().lower()
            found_controls.add(val_clean)

    # Step 3: Return the highest-priority control found
    for control in priority_order:
        if control in found_controls:
            return control

    return 'unmarked'

def assign_road_features(df):
    """
    Assigns road feature tags to a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to assign the tags to.
        
    Returns:
        list: A list of road feature tags.
    """
    # Step 1: Initialize the tags set
    tags = set()

    # Step 2: Assign tags based on road type, speed zones, traffic features, etc.
    if df.get('road_type', pd.Series(dtype=str)).dropna().str.contains('motorway', case=False).any():
        tags.add('motorway')

    maxspeeds = df.get('maxspeed', pd.Series(dtype=str)).dropna().astype(str)
    numeric_speeds = []

    for val in maxspeeds:
        try:
            val_clean = float(val.replace('km/h', '').strip())
            numeric_speeds.append(val_clean)
        except:
            continue

    if numeric_speeds:
        avg_speed = sum(numeric_speeds) / len(numeric_speeds)
        if avg_speed >= 80:
            tags.add('high_speed_zone')
        elif avg_speed <= 30:
            tags.add('low_speed_zone')

    traffic_features = df.get('traffic_features', pd.Series(dtype=str)).dropna()
    found_pedestrian_tags = False
    for item in traffic_features:
        if isinstance(item, list):
            entries = item
        elif isinstance(item, str):
            entries = [v.strip() for v in item.split(',')]
        else:
            continue
        for val in entries:
            val_clean = str(val).lower()
            if 'crossing' in val_clean or 'bus_stop' in val_clean or 'pedestrian' in val_clean:
                found_pedestrian_tags = True
                break
    if found_pedestrian_tags:
        tags.add('pedestrian_area')

    if df.get('is_narrow', pd.Series(dtype=bool)).dropna().astype(bool).any():
        tags.add('narrow_road')

    if df.get('is_unlit', pd.Series(dtype=bool)).dropna().astype(bool).any():
        tags.add('unlit_road')

    tunnel_vals = df.get('tunnel', pd.Series(dtype=str)).dropna().str.lower()
    if tunnel_vals.isin({'yes', 'building_passage', 'culvert'}).any():
        tags.add('tunnel')

    bridge_vals = df.get('bridge', pd.Series(dtype=str)).dropna().str.lower()
    if bridge_vals.isin({'yes'}).any():
        tags.add('bridge')

    lanes = df.get('lanes', pd.Series(dtype=str)).dropna()
    for val in lanes:
        try:
            if int(str(val).strip()) > 3:
                tags.add('multilane_road')
                break
        except:
            continue

    landuse = df.get('landuse', pd.Series(dtype=str)).dropna().str.lower()
    if landuse.str.contains('residential').any():
        tags.add('urban')
    elif landuse.str.contains('farmland|forest|meadow|grass').any():
        tags.add('rural')

    sidewalk = df.get('sidewalk', pd.Series(dtype=str)).dropna().str.lower()
    if not sidewalk.empty and not sidewalk.str.contains('no').all():
        tags.add('sidewalk_present')

    if "railway_crossing" in df["traffic_controls"]:
        tags.add('railway_crossing')

    if df.get('bike_friendly', pd.Series(dtype=bool)).dropna().astype(bool).any():
        tags.add('bike_friendly')

    return sorted(tags)

def assign_environmental_tags(df):
    """
    Assigns environmental tags to a DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to assign the tags to.
        
    Returns:
        list: A list of environmental tags.
    """
    # Step 1: Initialize the tags set
    tags = set()

    # Step 2: Get the datetime fields
    months = df.get('month', pd.Series(dtype=str)).dropna().str.lower().unique()
    days = df.get('day_of_week', pd.Series(dtype=str)).dropna().str.lower().unique()
    
    if 'hour' in df.columns:
        hours = df['hour'].dropna().astype(int)
    elif 'time_of_day' in df.columns:
        hours = pd.to_datetime(df['time_of_day'], errors='coerce').dt.hour.dropna()
    else:
        hours = pd.Series(dtype=int)

    if len(hours) == 0:
        return ['unknown_time']

    # Step 3: Assign tags based on the datetime fields
    winter_months = {'november', 'december', 'january', 'february', 'march'}
    if any(m in winter_months for m in months):
        tags.add('winter_conditions_possible')

    if ((hours < 6) | (hours >= 20)).any():
        tags.add('night_time')

    if any(d not in {'saturday', 'sunday'} for d in days):
        if ((hours >= 7) & (hours <= 9)).any() or ((hours >= 16) & (hours <= 18)).any():
            tags.add('rush_hour')

    if any(d in {'saturday', 'sunday'} for d in days):
        tags.add('weekend')

    if 'night_time' in tags or 'winter_conditions_possible' in tags:
        tags.add('low_visibility_possible')

    if ((hours >= 10) & (hours <= 15)).any():
        tags.add('off_peak_hours')

    if 'observation.state.conditions' in df.columns:
        conditions = (
            df['observation.state.conditions']
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )
        for cond in conditions:
            cond_clean = cond.strip()
            if cond_clean:
                tags.add(cond_clean)

    if 'observation.state.lighting' in df.columns:
        conditions = (
            df['observation.state.lighting']
            .dropna()
            .astype(str)
            .str.lower()
            .unique()
        )
        for cond in conditions:
            cond_clean = cond.strip()
            if cond_clean:
                tags.add(cond_clean)

    return sorted(tags)

def add_data_tags(min_ep, max_ep=-1,
                  data_dir='../data/processed/L2D',
                  tags_dir='../data/semantic_tags/L2D'):
    """
    Adds data tags to the data.
    
    Args:
        min_ep (int or list): The minimum episode number to process, or a list of episode numbers.
        max_ep (int, optional): The maximum episode number to process. Defaults to -1.
        data_dir (str, optional): The directory containing the data. Defaults to '../data/processed/L2D'.
        tags_dir (str, optional): The directory where the tags will be saved. Defaults to '../data/semantic_tags/L2D'.
    """
    # Step 1: Initialize the tags directory and iterable
    os.makedirs(tags_dir, exist_ok=True)

    if not isinstance(min_ep, list):
        if max_ep == -1:
            max_ep = min_ep + 1
        iterable = range(min_ep, max_ep)
    else:
        iterable = min_ep

    # Step 2: Process each episode
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        for ep_num in tqdm(iterable, desc="Generating semantic tags"):

            data_parquet = os.path.join(data_dir, f"episode_{ep_num:06d}.parquet")
            output_json = os.path.join(tags_dir, f"episode_{ep_num:06d}.json")

            if not os.path.exists(data_parquet):
                print(f"⚠️ Missing parquet for episode {ep_num}, skipping.")
                continue

            try:
                df = load_and_restore_parquet(data_parquet)
            except Exception as e:
                print(f"⚠️ Could not load parquet for episode {ep_num}: {e}")
                continue

            do = {
                "episode_index": int(ep_num),
                "action_tag": None,
                "traffic_control_tag": None,
                "road_feature_tags": [],
                "environment_tags": [],
            }

            try:
                do["action_tag"] = assign_action_tag(df)
            except Exception as e:
                print(f"⚠️ Action tag failed for episode {ep_num}: {e}")
                do["action_tag"] = "none"

            try:
                do["traffic_control_tag"] = assign_traffic_control_tag(df)
            except Exception as e:
                print(f"⚠️ Traffic control tag failed for episode {ep_num}: {e}")
                do["traffic_control_tag"] = "none"

            try:
                do["road_feature_tags"] = assign_road_features(df)
            except Exception as e:
                print(f"⚠️ Road features failed for episode {ep_num}: {e}")
                do["road_feature_tags"] = []

            try:
                do["environment_tags"] = assign_environmental_tags(df)
            except Exception as e:
                print(f"⚠️ Environment tags failed for episode {ep_num}: {e}")
                do["environment_tags"] = []

            try:
                with open(output_json, "w") as f:
                    json.dump(do, f, indent=4)
            except Exception as e:
                print(f"❌ Failed to write JSON for episode {ep_num}: {e}")
