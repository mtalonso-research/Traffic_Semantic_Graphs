import os
import json
from tqdm import tqdm

def process_and_merge_annotations(original_path, lane_path, output_path):
    """
    Reads an original annotation JSON file and a lane annotation JSON file,
    merges the lane classification data, filters out the ego-vehicle's hood,
    and saves the result to a new file.

    Args:
        original_path (str): Path to the input original annotation JSON file.
        lane_path (str): Path to the input lane annotation JSON file.
        output_path (str): Path to save the filtered annotation JSON file.
    """
    try:
        # Step 1: Read the original annotation file
        with open(original_path, 'r') as f:
            original_data = json.load(f)

        # Step 2: Read the lane annotation file and create a lookup for lane classification
        with open(lane_path, 'r') as f:
            lane_data = json.load(f)
        
        lane_classification_by_track_id = {}
        for ann in lane_data.get('annotations', []):
            track_id = ann.get('track_id')
            if track_id is not None:
                lane_classification_by_track_id[track_id] = ann.get('attributes', {}).get('lane_classification')

        # Step 3: Merge lane classification into original annotations
        for ann in original_data.get('annotations', []):
            track_id = ann.get('track_id')
            if track_id in lane_classification_by_track_id:
                if 'attributes' not in ann:
                    ann['attributes'] = {}
                ann['attributes']['lane_classification'] = lane_classification_by_track_id[track_id]

        # Step 4: Define thresholds for hood detection
        image_height = original_data['images'][0]['height']
        image_width = original_data['images'][0]['width']
        y_threshold = image_height * 0.9
        width_threshold = image_width * 0.8

        # Step 5: Filter out the hood annotation
        filtered_annotations = []
        for ann in original_data['annotations']:
            x, y, width, height = ann['bbox']
            is_car = ann.get('attributes', {}).get('class') == 'car'

            is_hood = (
                is_car and
                (y + height) > y_threshold and
                width > width_threshold
            )

            if not is_hood:
                filtered_annotations.append(ann)

        original_data['annotations'] = filtered_annotations

        # Step 6: Save the filtered annotation file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(original_data, f, indent=2)

    except FileNotFoundError:
        print(f"Warning: Annotation file not found. Original: {original_path}, Lane: {lane_path}. Skipping.")
    except Exception as e:
        print(f"Error processing files {original_path} and {lane_path}: {e}")

def process_annotations_directory(min_ep, max_ep=-1, 
                                  input_dir_original='data/processed_frames/L2D', 
                                  input_dir_lanes='data/processed_frames/L2D_lanes',
                                  output_dir='data/annotations/L2D', 
                                  original_annotations_folder_name='front_left_Annotations',
                                  lanes_annotations_folder_name='front_left_Enhanced_LaneAnnotations'):
    """
    Processes all annotation files for a given range of episodes, filtering out the
    ego-vehicle hood annotations.

    Args:
        min_ep (int): The minimum episode number to process.
        max_ep (int): The maximum episode number to process. If -1, only min_ep is processed.
        input_dir_original (str): The root directory for the original input annotations.
        input_dir_lanes (str): The root directory for the lane input annotations.
        output_dir (str): The root directory to save the filtered annotations.
        original_annotations_folder_name (str): The name of the folder containing the original annotations.
        lanes_annotations_folder_name (str): The name of the folder containing the lane annotations.
    """
    # Step 1: Determine the range of episodes to process
    if not isinstance(min_ep, list):
        if max_ep == -1: 
            max_ep = min_ep + 1
        iterable = range(min_ep,max_ep)
    else:
        iterable = min_ep

    # Step 2: Process each episode
    for ep_num in tqdm(iterable, desc="Processing Annotations"):
        episode_input_dir_original = os.path.join(input_dir_original, f"Episode{ep_num:06d}", original_annotations_folder_name)
        episode_input_dir_lanes = os.path.join(input_dir_lanes, f"Episode{ep_num:06d}", lanes_annotations_folder_name)
        episode_output_dir = os.path.join(output_dir, f"Episode{ep_num:06d}")

        if not os.path.exists(episode_input_dir_original) or not os.path.exists(episode_input_dir_lanes):
            print(f"Warning: Input directory not found for episode {ep_num}. Skipping.")
            continue

        os.makedirs(episode_output_dir, exist_ok=True)

        # Step 3: Process each annotation file in the episode
        for frame_file in os.listdir(episode_input_dir_original):
            if frame_file.endswith('.json'):
                original_path = os.path.join(episode_input_dir_original, frame_file)
                lane_path = os.path.join(episode_input_dir_lanes, frame_file)
                output_path = os.path.join(episode_output_dir, frame_file)
                process_and_merge_annotations(original_path, lane_path, output_path)