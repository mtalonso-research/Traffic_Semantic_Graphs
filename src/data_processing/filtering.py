import json
import shutil
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
from src.utils import extract_frames
import os

def filter_json_files(source_dir, output_dir):
    """
    Filters JSON files based on the presence of a heading in the first frame.
    The purpose of this is to filter out episodes that do not have heading information.
    
    Args:
        source_dir (str): The directory containing the source JSON files.
        output_dir (str): The directory where the filtered JSON files will be saved.
    """
    # Step 1: Initialization
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_path.resolve()}: {e}")
        return
    
    files_processed = 0
    files_copied = 0

    # Step 2: Process each JSON file
    for file_path in tqdm(source_path.glob('*.json')):
        files_processed += 1
        try:
            with file_path.open('r', encoding='utf-8') as f:
                scene = json.load(f)
            frames = extract_frames(scene)

            heading = get_nested_value(frames, [0, 'ego', 'features', 'heading'])

            if heading is not None:
                files_copied += 1
                destination_file = output_path / file_path.name
                shutil.copy2(file_path, destination_file)
            else:
                pass

        except json.JSONDecodeError:
            print(f"  [ERROR] Skipping {file_path.name}: Invalid JSON format.")
        except (KeyError, IndexError, TypeError):
            print(f"  [ERROR] Skipping {file_path.name}: Expected structure (e.g., 'frames[0]') not found.")
        except Exception as e:
            print(f"  [ERROR] Skipping {file_path.name}: An unexpected error occurred: {e}")

    print(f"\n--- Filter Complete ---")
    print(f"Total files scanned: {files_processed}")
    print(f"Files copied:      {files_copied}")
    print(f"Files skipped:     {files_processed - files_copied}")

def get_nested_value(data, keys):
    """
    Retrieves a nested value from a dictionary or list.
    
    Args:
        data (dict): The dictionary or list to retrieve the value from.
        keys (list): A list of keys or indices to traverse the data structure.
        
    Returns:
        Optional[Any]: The nested value, or None if the value cannot be found.
    """
    current_level = data
    for key in keys:
        if isinstance(current_level, dict):
            current_level = current_level.get(key)
        elif isinstance(current_level, list):
            try:
                current_level = current_level[key]
            except (IndexError, TypeError):
                return None
        else:
            return None
        
        if current_level is None:
            return None
            
    return current_level

def filter_episodes_by_frame_count(input_dir, output_dir, min_frames):
    """
    Filters graph JSON files based on the number of frames in the episode.
    
    Args:
        input_dir (str): The directory containing the input JSON files.
        output_dir (str): The directory where the filtered JSON files will be saved.
        min_frames (int): The minimum number of frames required for an episode to be kept.
    """
    # Step 1: Initialization
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    files_to_process = [f for f in input_path.glob('*.json')] 

    files_processed = 0
    files_copied = 0

    # Step 2: Process each JSON file
    for file_path in tqdm(files_to_process, desc="Filtering by frame count"):
        files_processed += 1
        try:
            with file_path.open('r') as f:
                graph_data = json.load(f)
            
            num_frames = len(graph_data.get('nodes', {}).get('ego', []))

            if num_frames >= min_frames:
                destination_file = output_path / file_path.name
                if not destination_file.exists():
                    files_copied += 1
                    shutil.copy2(file_path, destination_file)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Skipping file {file_path.name} due to error: {e}")

    print(f"\n--- Frame Count Filter Complete ---")
    print(f"Total files scanned: {files_processed}")
    print(f"Files copied (>= {min_frames} frames): {files_copied}")
    print(f"Files skipped (< {min_frames} frames or already exist): {files_processed - files_copied}")
