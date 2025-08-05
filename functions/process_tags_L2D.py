import pandas as pd
import json
import warnings
from tqdm import tqdm
import os
from functions.utils_L2D import load_and_restore_parquet

def filter_compensating_turns(turns):
    filtered = []
    i = 0
    while i < len(turns):
        current = turns[i].lower()

        # Look ahead to check for a compensating turn
        if i + 1 < len(turns):
            next_turn = turns[i + 1].lower()
            if (
                ('lane change left' in current and 'turning right' in next_turn) or
                ('lane change right' in current and 'turning left' in next_turn)
            ):
                # Keep current, skip next
                filtered.append(current)
                i += 2  # skip the next turn
                continue

        # No compensation detected → just add current
        filtered.append(current)
        i += 1

    return pd.Series(filtered)

def detect_speed_decreased_during_turn(df, direction='right', window=5):
    turning = df.get('turning_behavior', pd.Series(dtype=str)).dropna().astype(str).reset_index(drop=True)
    speeds = df.get('vehicle_speed', pd.Series(dtype=float)).reset_index(drop=True)

    # Find the index of the first matching turn
    turn_idx = turning[turning.str.contains(f'turning {direction}', case=False)].index

    if len(turn_idx) == 0:
        return True  # assume true turn if not found

    idx = turn_idx[0]  # take the first one
    start = max(0, idx - window)
    end = min(len(speeds), idx + window + 1)

    before = speeds[start:idx]
    after = speeds[idx:end]

    if before.empty or after.empty:
        return True  # not enough data, assume true turn

    avg_before = before.mean()
    avg_after = after.mean()

    return avg_after < avg_before  # True if speed decreased

def slowed_down_after_straight(df, direction='right'):
    turning = df.get('turning_behavior', pd.Series(dtype=str)).dropna().astype(str).reset_index(drop=True)
    speeds = df.get('vehicle_speed', pd.Series(dtype=float)).reset_index(drop=True)

    for i in range(1, len(turning)):
        prev = turning[i - 1].lower()
        curr = turning[i].lower()

        # Look for transition from 'straight' to the specified direction
        if 'straight' in prev and f'turning {direction}' in curr:
            speed_before = speeds[i - 1]
            speed_after = speeds[i]
            return speed_after < speed_before  # True if slowing down
        
    # 🧨 Handle edge case: turning is first behavior
    if turning[0].lower() == f'turning {direction}' and len(speeds) > 1:
        return speeds[1] <= speeds[0]  # slow down? → true turn

    # If no such transition found, assume deceleration (be conservative)
    return True

def assign_action_tag(df):
    def safe_str_contains(series, keyword):
        return series.dropna().astype(str).str.contains(keyword, case=False)

    road_type = df.get('road_type', pd.Series(dtype=str))
    raw_turning = df.get('turning_behavior', pd.Series(dtype=str)).dropna().astype(str).tolist()
    turning_behavior = filter_compensating_turns(raw_turning)

    action_gear = df.get('action_gear', pd.Series(dtype=str)).dropna().astype(str).tolist()
    u_turn_detected = any('u-turn' in turn.lower() for turn in turning_behavior)
    gear_has_one = any('1' in gear for gear in action_gear)
    final_speed = df.get('vehicle_speed', pd.Series(dtype=float)).dropna().values[-1] if not df.get('vehicle_speed', pd.Series()).empty else None
    speed_threshold = 2.0  # you can adjust this
    road_type_contains_link = safe_str_contains(road_type, '_link').any()

    # 🥇 Priority 1: Roundabout
    if df.astype(str).apply(lambda col: col.str.contains('roundabout', case=False, na=False)).any().any():
        return 'roundabout'
        
    # 🥉 Priority 3: Parking / 3-point-turn / Backing up
    if gear_has_one:
        if u_turn_detected:
            if final_speed is not None and final_speed < speed_threshold:
                return 'parking'
            elif final_speed is not None and final_speed >= speed_threshold:
                return '3-point-turn'
        else:
            return 'backing_up'
        
    # 🥈 Priority 2: Uncertain (both left and right turns with similar count)
    left_turns = safe_str_contains(turning_behavior, 'turning left').sum()
    right_turns = safe_str_contains(turning_behavior, 'turning right').sum()
    total_turns = left_turns + right_turns

    # 🥉 Priority 3: Merge
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
    # Define controls in order of priority
    priority_order = ['traffic_signal', 'stop_sign', 'roundabout', 'yield_sign']
    col = df.get('traffic_controls', pd.Series(dtype=str)).dropna()

    found_controls = set()

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

    # Return the highest-priority control found
    for control in priority_order:
        if control in found_controls:
            return control

    return 'unmarked'

def assign_road_features(df):
    tags = set()

    # Motorway
    if df.get('road_type', pd.Series(dtype=str)).dropna().str.contains('motorway', case=False).any():
        tags.add('motorway')

    # Speed zones
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

    # Pedestrian area
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

    # is_narrow
    if df.get('is_narrow', pd.Series(dtype=bool)).dropna().astype(bool).any():
        tags.add('narrow_road')

    # is_unlit
    if df.get('is_unlit', pd.Series(dtype=bool)).dropna().astype(bool).any():
        tags.add('unlit_road')

    # Bridge or tunnel
    tunnel_vals = df.get('tunnel', pd.Series(dtype=str)).dropna().str.lower()
    if tunnel_vals.isin({'yes', 'building_passage', 'culvert'}).any():
        tags.add('tunnel')

    bridge_vals = df.get('bridge', pd.Series(dtype=str)).dropna().str.lower()
    if bridge_vals.isin({'yes'}).any():
        tags.add('bridge')

    # Multi-lane
    lanes = df.get('lanes', pd.Series(dtype=str)).dropna()
    for val in lanes:
        try:
            if int(str(val).strip()) > 3:
                tags.add('multilane_road')
                break
        except:
            continue

    # Landuse-based (urban/rural)
    landuse = df.get('landuse', pd.Series(dtype=str)).dropna().str.lower()
    if landuse.str.contains('residential').any():
        tags.add('urban')
    elif landuse.str.contains('farmland|forest|meadow|grass').any():
        tags.add('rural')

    # Sidewalk
    sidewalk = df.get('sidewalk', pd.Series(dtype=str)).dropna().str.lower()
    if not sidewalk.empty and not sidewalk.str.contains('no').all():
        tags.add('sidewalk_present')

    if "railway_crossing" in df["traffic_controls"]:
        tags.add('railway_crossing')

    # Bike friendly
    if df.get('bike_friendly', pd.Series(dtype=bool)).dropna().astype(bool).any():
        tags.add('bike_friendly')

    return sorted(tags)

def assign_environmental_tags(df):
    tags = set()

    # Handle datetime fields
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

    # 1. Winter conditions possible (Nov–Mar)
    winter_months = {'november', 'december', 'january', 'february', 'march'}
    if any(m in winter_months for m in months):
        tags.add('winter_conditions_possible')

    # 2. Night time (defined here as 20:00–6:00)
    if ((hours < 6) | (hours >= 20)).any():
        tags.add('night_time')

    # 3. Rush hour (typically 7–9 AM and 16–18 PM weekdays)
    if any(d not in {'saturday', 'sunday'} for d in days):
        if ((hours >= 7) & (hours <= 9)).any() or ((hours >= 16) & (hours <= 18)).any():
            tags.add('rush_hour')

    # 4. Weekend
    if any(d in {'saturday', 'sunday'} for d in days):
        tags.add('weekend')

    # 5. Low visibility conditions (bonus idea: night or winter = likely)
    if 'night_time' in tags or 'winter_conditions_possible' in tags:
        tags.add('low_visibility_possible')

    # 6. Off-peak hours (optional)
    if ((hours >= 10) & (hours <= 15)).any():
        tags.add('off_peak_hours')

    return sorted(tags)

def add_data_tags(min_ep,max_ep=-1,
                  data_dir='../data/processed/L2D',
                  tags_dir='../data/semantic_tags/L2D'):

    if max_ep == -1: max_ep = min_ep + 1
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=UserWarning)
        for ep_num in tqdm(range(min_ep, max_ep)):

            data_parquet = os.path.join(data_dir,f"episode_{ep_num:06d}.parquet")
            output_json = os.path.join(tags_dir,f"episode_{ep_num:06d}.json")

            try:
                df = load_and_restore_parquet(data_parquet)
                with open(output_json, 'r') as f:
                    do = json.load(f)

                try: do['action_tag'] = assign_action_tag(df)
                except: do['action_tag'] = 'none'
                try: do['traffic_control_tag'] = assign_traffic_control_tag(df)
                except: do['traffic_control_tag'] = 'none'
                try: do['road_feature_tags'] = assign_road_features(df)
                except: do['road_feature_tags'] = []
                try: do['environment_tags'] = assign_environmental_tags(df)
                except: do['environment_tags'] = []

                with open(output_json, 'w') as f:
                    json.dump(do, f, indent=4)

            except: print(f'Trouble processing episode: {ep_num}')