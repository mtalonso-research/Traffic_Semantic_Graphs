import os
import argparse

from src.data_processing.nup_load_data import load_data
from src.data_processing.nup_process_jsons import (
    add_latlon_to_graphs,
    enrich_weather_features,
    replace_weather_code_with_description,
    add_temporal_features,
)

# --- 1. Add boolean arguments for each processing step ---
parser = argparse.ArgumentParser(description="Process NuPlan data.")
parser.add_argument("--city", type=str, default="boston", help="City to process (boston or pittsburgh).")
parser.add_argument("--file_min", type=int, default=0, help="Minimum DB file index to process (inclusive).")
parser.add_argument("--file_max", type=lambda x: None if x.lower() == "none" else int(x), default=None, help="Maximum DB file index to process (exclusive). Use 'none' for all after file_min.")
parser.add_argument("--episodes", type=lambda x: None if x.lower() == "none" else int(x), nargs='+', default=None, help="Episodes to Process.")

# Boolean flags for sub-parts, default to False
parser.add_argument("--load", action='store_true', help="Run only the data loading step.")
parser.add_argument("--latlon", action='store_true', help="Run only the lat/lon addition step.")
parser.add_argument("--weather", action='store_true', help="Run only the weather enrichment step.")
parser.add_argument("--weather_codes", action='store_true', help="Run only the weather code replacement step.")
parser.add_argument("--temporal", action='store_true', help="Run only the temporal feature addition step.")

args = parser.parse_args()

# --- 2. Logic to handle the default "all true" or "specific true" case ---
processing_steps = ['load', 'latlon', 'weather', 'weather_codes', 'temporal']
# Check if any specific step was requested by the user
any_step_selected = any(getattr(args, step) for step in processing_steps)

# If no specific step was selected, default all to True
if not any_step_selected:
    for step in processing_steps:
        setattr(args, step, True)

# --- 3. Update the function to accept the boolean flags ---
def default_nuplan_processing(city, file_min=0, file_max=None, episodes=None, run_load=True, run_latlon=True, run_weather=True, run_weather_codes=True, run_temporal=True):
    db_dir = f"./data/raw/NuPlan/train_{city}"
    out_dir = f"./data/graphical/nuplan_{city}"
    os.makedirs(out_dir, exist_ok=True)

    if run_load:
        print("========== Load Data ==========")
        load_data(db_dir=db_dir, out_dir=out_dir, time_idx=3, file_min=file_min, file_max=file_max)

    if run_latlon:
        print("========== Add Latitude / Longitude ==========")
        add_latlon_to_graphs(json_dir=out_dir, out_dir=out_dir, map_region=city, episodes=episodes)

    if run_weather:
        print("========== Enrich Weather Features ==========")
        enrich_weather_features(json_dir=out_dir, out_dir=out_dir, episodes=episodes, sleep_s=0.15)

    if run_weather_codes:
        print("========== Replace Weather Codes ==========")
        replace_weather_code_with_description(json_dir=out_dir, out_dir=out_dir, remove_numeric=False)

    if run_temporal:
        print("========== Add Temporal Features ==========")
        add_temporal_features(json_dir=out_dir, out_dir=out_dir)

    print(f"✅ Finished processing NuPlan data for {city}.")


if __name__ == "__main__":
    # --- 4. Pass the arguments to the function ---
    default_nuplan_processing(
        city=args.city,
        file_min=args.file_min,
        file_max=args.file_max,
        episodes=args.episodes,
        run_load=args.load,
        run_latlon=args.latlon,
        run_weather=args.weather,
        run_weather_codes=args.weather_codes,
        run_temporal=args.temporal
    )