import os
import json
import argparse
from tqdm import tqdm

# ==========================================================================================
# WMO Weather Map
# ==========================================================================================
WMO_WEATHER_MAP = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog",
    48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle", 61: "Slight rain",
    63: "Moderate rain", 65: "Heavy rain", 66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall", 77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers", 95: "Thunderstorm (slight or moderate)",
    96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail",
}

# ==========================================================================================
# UNIFIED TAG MAPPING
# ==========================================================================================
UNIFIED_TAG_MAPPING = {
    # Maneuver
    "maneuver_turn": [
        "left_turn", "right_turn", "starting_left_turn", "starting_right_turn",
        "starting_high_speed_turn", "starting_low_speed_turn", "starting_protected_cross_turn",
        "starting_unprotected_cross_turn", "starting_protected_noncross_turn", "starting_unprotected_noncross_turn",
    ],
    "maneuver_straight": [
        "straight", "following_lane_without_lead", "following_lane_with_lead", "following_lane_with_slow_lead",
        "starting_straight_stop_sign_intersection_traversal", "starting_straight_traffic_light_intersection_traversal",
    ],
    # "maneuver_lane_change": ["changing_lane", "changing_lane_to_left", "changing_lane_to_right", "changing_lane_with_lead", "changing_lane_with_trail"],
    # "maneuver_u_turn": ["3-point-turn", "starting_u_turn"],
    # "maneuver_roundabout": ["roundabout"],
    "maneuver_parking": ["parking", "on_carpark"],
    # "maneuver_backing_up": ["backing_up"],

    # Traffic Control
    "traffic_control_signal": [
        "traffic_signal", "accelerating_at_traffic_light", "accelerating_at_traffic_light_with_lead",
        "accelerating_at_traffic_light_without_lead", "on_stopline_traffic_light", "on_traffic_light_intersection",
        "stationary_at_traffic_light_with_lead", "stationary_at_traffic_light_without_lead", "stopping_at_traffic_light_with_lead",
        "stopping_at_traffic_light_without_lead", "traversing_traffic_light_intersection",
    ],
    "traffic_control_stop_sign": [
        "stop_sign", "accelerating_at_stop_sign", "accelerating_at_stop_sign_no_crosswalk",
        "on_all_way_stop_intersection", "on_stopline_stop_sign", "stopping_at_stop_sign_no_crosswalk",
        "stopping_at_stop_sign_with_lead", "stopping_at_stop_sign_without_lead",
    ],
    # "traffic_control_yield": ["yield_sign"],
    "traffic_control_uncontrolled": ["unmarked"],

    # Road Features
    "road_feature_pedestrian_crossing": [
        "pedestrian_area", "accelerating_at_crosswalk", "on_stopline_crosswalk",
        "stationary_at_crosswalk", "stopping_at_crosswalk", "traversing_crosswalk", "waiting_for_pedestrian_to_cross",
    ],
    # "road_feature_narrow_lane": ["traversing_narrow_lane"],

    # Environment
    "environment_day": ["day", "sunrise", "dusk"],
    "environment_night": ["night_time"],
    "environment_clear": ["clear", "Clear sky", "Mainly clear"],
    "environment_clouds": ["clouds", "Partly cloudy", "Overcast"],
    # "environment_rain": [
    #     "rain", "drizzle", "Light drizzle", "Moderate drizzle", "Dense drizzle", "Slight rain",
    #     "Moderate rain", "Heavy rain", "Slight rain showers", "Moderate rain showers", "Violent rain showers",
    # ],
    # "environment_snow": [
    #     "snow", "winter_conditions_possible", "Slight snow fall", "Moderate snow fall", "Heavy snow fall",
    #     "Snow grains", "Slight snow showers", "Heavy snow showers",
    # ],
    "environment_low_visibility": ["low_visibility_possible", "haze", "Fog", "Depositing rime fog"],
    "environment_freezing_rain": ["Light freezing drizzle", "Dense freezing drizzle", "Light freezing rain", "Heavy freezing rain"],
    "environment_thunderstorm": ["Thunderstorm (slight or moderate)", "Thunderstorm with slight hail", "Thunderstorm with heavy hail"],
    # Balancing Tags
    "balance_veh_count_none": ["no_vehicles"],
    "balance_veh_count_low": ["1_vehicle", "2-3_vehicles"],
    "balance_veh_count_high": ["4-7_vehicles", "8+_vehicles"],
    "balance_veh_dist_very_close": ["vehicle_very_close"],
    "balance_veh_dist_close": ["vehicle_close"],
    "balance_veh_dist_far": ["vehicle_near", "vehicle_medium", "vehicle_far"],
    "balance_ped_present": ["ped_nearby"],
    "balance_ped_absent": ["no_ped_nearby"],
    "balance_ego_speed_low": ["ego_stopped_or_slow", "ego_slow_speed"],
    "balance_ego_speed_high": ["ego_medium_speed", "ego_high_speed", "ego_very_high_speed"],
    # "balance_is_straight": ["going_straight"],
    # "balance_is_turning": ["turning"],
}

def decode_balance_tags(balance_vector):
    """
    Decodes a balance tag vector into a list of raw tag strings.
    """
    if not balance_vector or len(balance_vector) != 5:
        return []

    tags = []
    
    # Vehicle count
    veh_count_bin = balance_vector[0]
    if veh_count_bin == 0: tags.append("no_vehicles")
    elif veh_count_bin == 1: tags.append("1_vehicle")
    elif veh_count_bin == 2: tags.append("2-3_vehicles")
    elif veh_count_bin == 3: tags.append("4-7_vehicles")
    else: tags.append("8+_vehicles")

    # Vehicle distance
    veh_dist_bin = balance_vector[1]
    if veh_dist_bin == 0: tags.append("vehicle_very_close")
    elif veh_dist_bin == 1: tags.append("vehicle_close")
    elif veh_dist_bin == 2: tags.append("vehicle_near")
    elif veh_dist_bin == 3: tags.append("vehicle_medium")
    else: tags.append("vehicle_far")
        
    # Pedestrian distance
    ped_dist_bin = balance_vector[2]
    if ped_dist_bin == 0: tags.append("no_ped_nearby")
    else: tags.append("ped_nearby")

    # Ego speed
    ego_speed_bin = balance_vector[3]
    if ego_speed_bin == 0: tags.append("ego_stopped_or_slow")
    elif ego_speed_bin == 1: tags.append("ego_slow_speed")
    elif ego_speed_bin == 2: tags.append("ego_medium_speed")
    elif ego_speed_bin == 3: tags.append("ego_high_speed")
    else: tags.append("ego_very_high_speed")

    # Turn category
    # turn_cat = balance_vector[4]
    # if turn_cat == 0: tags.append("going_straight")
    # else: tags.append("turning")
        
    return tags

def process_tags(l2d_tag_dir: str, nuplan_tag_dir: str, nuplan_graph_dir: str, output_file: str, l2d_balance_file: str, nuplan_balance_file: str):
    """
    Processes and unifies raw tag files from L2D and NuPlan datasets,
    saving the result to a single JSON file.
    """
    print("Starting unified tag processing...")

    reverse_mapping = {raw_tag: unified_tag for unified_tag, raw_tags in UNIFIED_TAG_MAPPING.items() for raw_tag in raw_tags}
    unified_tags = {}

    # --- Load Balance Tags ---
    with open(l2d_balance_file, 'r') as f:
        l2d_balance_tags = json.load(f)
    with open(nuplan_balance_file, 'r') as f:
        nuplan_balance_tags = json.load(f)

    # --- Process L2D Tags ---
    print(f"Processing L2D directory: {l2d_tag_dir}")
    l2d_files = [f for f in os.listdir(l2d_tag_dir) if f.endswith('.json')]
    for filename in tqdm(l2d_files, desc="L2D Tags"):
        filepath = os.path.join(l2d_tag_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            l2d_raw_tags = []
            if data.get("action_tag"): l2d_raw_tags.append(data["action_tag"])
            if data.get("traffic_control_tag"): l2d_raw_tags.append(data["traffic_control_tag"])
            if data.get("road_feature_tags"): l2d_raw_tags.extend(data["road_feature_tags"])
            if data.get("environment_tags"): l2d_raw_tags.extend(data["environment_tags"])

            # Add balance tags
            episode_num = str(int(os.path.splitext(filename)[0].split('_')[-1]))
            if episode_num in l2d_balance_tags:
                l2d_raw_tags.extend(decode_balance_tags(l2d_balance_tags[episode_num]))

            processed_tags = {reverse_mapping[tag] for tag in l2d_raw_tags if tag in reverse_mapping}
            
            if processed_tags:
                episode_id = f"L2D_{os.path.splitext(filename)[0]}"
                unified_tags[episode_id] = sorted(list(processed_tags))

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not process L2D file {filepath}. Error: {e}")

    # --- Process NuPlan Tags ---
    print(f"Processing NuPlan directory: {nuplan_tag_dir}")
    nuplan_files = [f for f in os.listdir(nuplan_tag_dir) if f.endswith('.json')]
    for filename in tqdm(nuplan_files, desc="NuPlan Tags"):
        tag_filepath = os.path.join(nuplan_tag_dir, filename)
        graph_filepath = os.path.join(nuplan_graph_dir, filename)
        try:
            with open(tag_filepath, 'r') as f:
                tag_data = json.load(f)

            nuplan_raw_tags = tag_data.get("raw_tags", [])

            # Add environment tags from corresponding graph file
            if os.path.exists(graph_filepath):
                with open(graph_filepath, 'r') as f:
                    graph_data = json.load(f)
                
                env_nodes = graph_data.get("nodes", {}).get("environment", [])
                if env_nodes:
                    env_features = env_nodes[0].get("features", {})
                    
                    # Weather conditions
                    weather_code = env_features.get("conditions")
                    if weather_code is not None and weather_code in WMO_WEATHER_MAP:
                        nuplan_raw_tags.append(WMO_WEATHER_MAP[weather_code])
                    
                    # Day/Night and Low Visibility (from night)
                    is_daylight = env_features.get("daylight", True)
                    if not is_daylight:
                        nuplan_raw_tags.append("night_time")
                        nuplan_raw_tags.append("low_visibility_possible")
                    else:
                        nuplan_raw_tags.append("day")

                    # Low Visibility (from winter)
                    month = env_features.get("month")
                    winter_months = [11, 12, 1, 2, 3] # November to March
                    if month in winter_months:
                        nuplan_raw_tags.append("winter_conditions_possible")
                        if "low_visibility_possible" not in nuplan_raw_tags:
                            nuplan_raw_tags.append("low_visibility_possible")

                    # Rush hour / Off-peak
                    if env_features.get("weekend") is True:
                        nuplan_raw_tags.append("weekend")
                    else:
                        time_seconds = env_features.get("time")
                        if time_seconds is not None:
                            hour = time_seconds / 3600
                            if (hour >= 7 and hour <= 9) or (hour >= 16 and hour <= 18):
                                nuplan_raw_tags.append("rush_hour")
                            elif (hour >= 10 and hour <= 15):
                                nuplan_raw_tags.append("off_peak_hours")

            # Add balance tags
            episode_num = os.path.splitext(filename)[0].replace('_graph', '')
            if episode_num in nuplan_balance_tags:
                nuplan_raw_tags.extend(decode_balance_tags(nuplan_balance_tags[episode_num]))

            # Default to "uncontrolled" if no other traffic control tag is present
            traffic_control_raw_tags = {
                tag for unified, raw_tags in UNIFIED_TAG_MAPPING.items() 
                if unified.startswith("traffic_control") and unified != "traffic_control_uncontrolled"
                for tag in raw_tags
            }
            if not any(tag in traffic_control_raw_tags for tag in nuplan_raw_tags):
                nuplan_raw_tags.append("unmarked")

            processed_tags = {reverse_mapping[tag] for tag in nuplan_raw_tags if tag in reverse_mapping}

            if processed_tags:
                episode_id = f"NuPlan_{os.path.splitext(filename)[0]}"
                unified_tags[episode_id] = sorted(list(processed_tags))

        except (json.JSONDecodeError, TypeError) as e:
            print(f"Warning: Could not process NuPlan file {tag_filepath}. Error: {e}")

    # --- Save the unified tags ---
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_file, 'w') as f:
        json.dump(unified_tags, f, indent=4)

    print(f"\n✅ Unified tag processing complete.")
    print(f"Processed {len(unified_tags)} total episodes.")
    print(f"Unified tags saved to: {output_file}")


if __name__ == "__main__":
    L2D_TAG_DIR = 'data/training_data/L2D/tags/'
    NUPLAN_TAG_DIR = 'data/training_data/NuPlan/tags/'
    NUPLAN_GRAPH_DIR = 'data/training_data/NuPlan/graphs/'
    L2D_BALANCE_FILE = 'data/training_data/L2D/balance_tags.json'
    NUPLAN_BALANCE_FILE = 'data/training_data/NuPlan/balance_tags.json'
    OUTPUT_FILE = 'data/training_data/unified_tags.json'
    
    process_tags(
        l2d_tag_dir=L2D_TAG_DIR,
        nuplan_tag_dir=NUPLAN_TAG_DIR,
        nuplan_graph_dir=NUPLAN_GRAPH_DIR,
        output_file=OUTPUT_FILE,
        l2d_balance_file=L2D_BALANCE_FILE,
        nuplan_balance_file=NUPLAN_BALANCE_FILE
    )
