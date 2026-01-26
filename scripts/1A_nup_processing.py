import os
import argparse
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.nup_load_data import extract_and_flatten_graphs, enrich_and_finalize_graphs
from src.data_processing.nup_process_jsons import (
    add_latlon_to_graphs,
    enrich_weather_features,
    replace_weather_code_with_description,
    add_temporal_features,
)
from src.data_processing.nup_process_tags import extract_tags
from src.data_processing.nup_process_edges import process_edges
from src.data_processing.nup_lane_processing import process_lanes

import glob

def default_nuplan_processing(city, file_min=0, file_max=None, episodes=None, run_extract=True, run_enrich=True, run_lanes=True, run_latlon=True, run_weather=True, run_weather_codes=True, run_temporal=True, run_tags=True, run_edges=True):

    db_dir = f"./data/raw/NuPlan/train_{city}/nuplan-v1.1/train"
    map_dir = f"./data/raw/NuPlan/maps/nuplan-maps-v1.0/maps"
    graph_dir = f"./data/graphical/nuplan_{city}"
    tag_dir = f"./data/semantic_tags/nuplan_{city}"
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(tag_dir, exist_ok=True)

    if run_extract:
        print("========== Extract and Flatten Graphs ==========")
        extract_and_flatten_graphs(db_dir=db_dir, out_dir=graph_dir, time_idx=3, file_min=file_min, file_max=file_max)

    if run_enrich:
        print("========== Enrich and Finalize Graphs ==========")
        enrich_and_finalize_graphs(db_dir=db_dir, out_dir=graph_dir)

    if run_lanes:
        print("========== Process Lanes ==========")
        process_lanes(json_dir=graph_dir, db_dir=db_dir, map_dir=map_dir, city_name=city, episodes=episodes)

    if run_latlon:
        print("========== Add Latitude / Longitude ==========")
        add_latlon_to_graphs(json_dir=graph_dir, out_dir=graph_dir, map_region=city, episodes=episodes)

    if run_weather:
        print("========== Enrich Weather Features ==========")
        enrich_weather_features(json_dir=graph_dir, out_dir=graph_dir, episodes=episodes, sleep_s=0.15)

    if run_weather_codes:
        print("========== Replace Weather Codes ==========")
        replace_weather_code_with_description(json_dir=graph_dir, out_dir=graph_dir, remove_numeric=False)

    if run_temporal:
        print("========== Add Temporal Features ==========")
        add_temporal_features(json_dir=graph_dir, out_dir=graph_dir, city=city)

    if run_tags:
        print("========== Extract Semantic Tags ==========")
        extract_tags(data_root=db_dir, output_dir=tag_dir, graph_dir=graph_dir)

    if run_edges:
        print("========== Process Edges ==========")
        process_edges(json_dir=graph_dir)

    print(f"✅ Finished processing NuPlan data for {city}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NuPlan data.")
    parser.add_argument("--city", type=str, default="boston", help="City to process (boston or pittsburgh).")
    parser.add_argument("--file_min", type=int, default=0, help="Minimum DB file index to process (inclusive).")
    parser.add_argument("--file_max", type=lambda x: None if x.lower() == "none" else int(x), default=None, help="Maximum DB file index to process (exclusive). Use 'none' for all after file_min.")
    parser.add_argument("--episodes", type=lambda x: None if x.lower() == "none" else int(x), nargs='+', default=None, help="Episodes to Process.")

    parser.add_argument("--extract", action='store_true', help="Run only the data extraction and flattening step.")
    parser.add_argument("--enrich", action='store_true', help="Run only the data enrichment and finalization step.")
    parser.add_argument("--latlon", action='store_true', help="Run only the lat/lon addition step.")
    parser.add_argument("--weather", action='store_true', help="Run only the weather enrichment step.")
    parser.add_argument("--weather_codes", action='store_true', help="Run only the weather code replacement step.")
    parser.add_argument("--temporal", action='store_true', help="Run only the temporal feature addition step.")
    parser.add_argument("--tags", action='store_true', help="Run only the tag extraction step.")
    parser.add_argument("--edges", action='store_true', help="Run only the edge processing step.")
    parser.add_argument("--lanes", action='store_true', help="Run only the lane processing step.")

    args = parser.parse_args()

    processing_steps = ['extract', 'enrich', 'lanes', 'latlon', 'weather', 'weather_codes', 'temporal', 'tags', 'edges']
    any_step_selected = any(getattr(args, step) for step in processing_steps)

    if not any_step_selected:
        for step in processing_steps:
            setattr(args, step, True)

    default_nuplan_processing(
        city=args.city,
        file_min=args.file_min,
        file_max=args.file_max,
        episodes=args.episodes,
        run_extract=args.extract,
        run_enrich=args.enrich,
        run_lanes=args.lanes,
        run_latlon=args.latlon,
        run_weather=args.weather,
        run_weather_codes=args.weather_codes,
        run_temporal=args.temporal,
        run_tags=args.tags,
        run_edges=args.edges
    )