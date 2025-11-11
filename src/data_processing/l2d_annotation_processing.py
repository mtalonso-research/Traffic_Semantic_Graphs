import os
import json
from tqdm import tqdm

def filter_ego_hood(input_path, output_path):
    """
    Reads an annotation JSON file, filters out the annotation most likely
    to be the ego-vehicle's hood, and saves the result to a new file.

    Args:
        input_path (str): Path to the input annotation JSON file.
        output_path (str): Path to save the filtered annotation JSON file.
    """
    try:
        # Step 1: Read the annotation file
        with open(input_path, 'r') as f:
            data = json.load(f)

        image_height = data['images'][0]['height']
        image_width = data['images'][0]['width']

        # Step 2: Define thresholds for hood detection
        y_threshold = image_height * 0.9
        width_threshold = image_width * 0.8

        # Step 3: Filter out the hood annotation
        filtered_annotations = []
        for ann in data['annotations']:
            x, y, width, height = ann['bbox']
            is_car = ann.get('attributes', {}).get('class') == 'car'

            is_hood = (
                is_car and
                (y + height) > y_threshold and
                width > width_threshold
            )

            if not is_hood:
                filtered_annotations.append(ann)

        data['annotations'] = filtered_annotations

        # Step 4: Save the filtered annotation file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    except Exception as e:
        print(f"Error processing file {input_path}: {e}")

def process_annotations_directory(min_ep, max_ep=-1, input_dir='data/processed_frames/L2D', output_dir='data/annotations/L2D'):
    """
    Processes all annotation files for a given range of episodes, filtering out the
    ego-vehicle hood annotations.

    Args:
        min_ep (int): The minimum episode number to process.
        max_ep (int): The maximum episode number to process. If -1, only min_ep is processed.
        input_dir (str): The root directory for the input annotations.
        output_dir (str): The root directory to save the filtered annotations.
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
        episode_input_dir = os.path.join(input_dir, f"Episode{ep_num:06d}", 'front_left_Annotations')
        episode_output_dir = os.path.join(output_dir, f"Episode{ep_num:06d}")

        if not os.path.exists(episode_input_dir):
            print(f"Warning: Input directory not found for episode {ep_num}. Skipping.")
            continue

        os.makedirs(episode_output_dir, exist_ok=True)

        # Step 3: Process each annotation file in the episode
        for frame_file in os.listdir(episode_input_dir):
            if frame_file.endswith('.json'):
                input_path = os.path.join(episode_input_dir, frame_file)
                output_path = os.path.join(episode_output_dir, frame_file)
                filter_ego_hood(input_path, output_path)